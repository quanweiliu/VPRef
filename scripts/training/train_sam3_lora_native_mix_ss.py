#!/usr/bin/env python3
"""SAM3 LoRA Training Script

Validation Strategy (Following SAM3):
  - During training: Only compute validation LOSS (fast, no metrics)
  - After training: Run validate_sam3_lora.py for full metrics (mAP, cgF1) with NMS

This approach significantly speeds up training by avoiding expensive metric computation
during each epoch, while still monitoring overfitting via validation loss.

Multi-GPU Training:
  Single GPU:
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu

  Multi-GPU with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu
"""

import os
import sys
sys.path.append("/home/icclab/Documents/lqw/sam3_finetune_lora")  # Add project root to path
sys.path.append("/home/icclab/Documents/lqw/sam3_finetune_lora/scripts")

import argparse
import json
import math
from pathlib import Path

import torch
# Distributed training imports
import torch.distributed as dist
import yaml
from PIL import Image as PILImage
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
# from torchvision.transforms import v2
from tqdm import tqdm

from lora_layers import (
    LoRAConfig,
    apply_lora_to_model,
    count_parameters,
    save_lora_weights,
)
from sam3.model.model_misc import SAM3Output

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
# from sam3.train.data.sam3_image_dataset import (
#     Datapoint,
#     FindQueryLoaded,
#     Image,
#     InferenceMetadata,
#     Object,
# )
from sam3.train.loss.loss_fns import CORE_LOSS_KEY, Boxes, IABCEMdetr, Masks
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.masks_ops import rle_encode  # For encoding masks to RLE format
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

# Dataset backends (can also be selected via config['training']['dataset'])
from data.isprs_dataset import ISPRSRefDataset
from data.isprs_dataset_ss import ISPRSRefDataset_SS
from data.isprs_dataset_mix import ISPRSRefDataset_mix
from data.isprs_dataset_mix_ss import ISPRSRefDataset_MixSS
from data.COCODataset import COCOSegmentDataset, DirectCOCODataset
from tools.utils import setup_distributed, cleanup_distributed, patch_dynamic_rope
from tools.utils import is_main_process, get_world_size, get_rank, print_rank0


class SAM3TrainerNative:
    def __init__(self, config_path, multi_gpu=False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

            # Create output directory
            out_dir = Path(self.config["output"]["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(os.path.join(out_dir, 'config.yaml'), 'w', encoding='utf-8') as fid:
                # 使用 yaml.dump 将字典写回文件
                yaml.dump(self.config, fid, Dumper=yaml.Dumper, default_flow_style=False, allow_unicode=True)
        
        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Model
        print_rank0("Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            checkpoint_path="/home/icclab/Documents/lqw/sam3/weights/sam3.pt",  # Set to None to load from HF
            load_from_HF=False,  # Tries to download from HF if checkpoint_path is None
            bpe_path="/home/icclab/Documents/lqw/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=False,
        )

        # Patch RoPE for dynamic resolution support (enables non-1008 inputs)
        n_patched = patch_dynamic_rope(self.model)
        print_rank0(f"Dynamic RoPE patch applied to {n_patched} attention layers")

        # Apply LoRA
        print_rank0("Applying LoRA...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        stats = count_parameters(self.model)
        print_rank0(
            f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)"
        )

        self.model.to(self.device)

        # Wrap model with DDP if multi-GPU
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # Frozen params (requires_grad=False) don't need this flag
            )
            print_rank0("Model wrapped with DistributedDataParallel")

        # Store reference to unwrapped model for accessing custom methods
        self._unwrapped_model = self.model.module if self.multi_gpu else self.model

        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=self.config["training"]["weight_decay"],
        )

        # LR scheduler will be created in train() once dataloader length is known
        self.lr_scheduler = None

        # Matcher & Loss
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        # Loss weights tuned for small-dataset LoRA fine-tuning (not massive pre-training).
        loss_fns = [
            Boxes(weight_dict={"loss_bbox": 2.0, "loss_giou": 1.0}),
            IABCEMdetr(
                pos_weight=5.0,
                weight_dict={"loss_ce": 5.0, "presence_loss": 2.0},
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=100,
            ),
            Masks(
                weight_dict={
                    "loss_mask": 5.0,
                    "loss_dice": 5.0,
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False,
            ),
        ]

        # Create one-to-many matcher for auxiliary outputs
        o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)

        # Use Sam3LossWrapper for proper loss computation
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",  # Use local normalization (no distributed training)
            normalize_by_valid_object_num=False,
        )

    def train(self):
        # Get data directory from config
        data_dir = self.config["training"]["data_dir"]
        dataset_type = self.config["training"].get("dataset", "coco")

        # Load datasets
        print_rank0(f"\nLoading training data from {data_dir} (dataset={dataset_type})...")

        if dataset_type == "isprsRef":
            variant = self.config["training"].get("variant", "standard")
            print("current variant:", variant)
            train_ds = ISPRSRefDataset_MixSS(
                data_dir=data_dir, 
                unlabeled_dir="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/",
                # unlabeled_mask_path="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/unlabeled_valid_standard_mix",
                # unlabeled_text_path="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/output_phrase_val_standard.txt",
                unlabeled_mask_path="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/unlabeled_valid_standard_mixf",
                unlabeled_text_path="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/output_phrase_val_standard.txt",
                split="train", 
                variant=variant, 
                resolution=480
            )
            has_validation = True
            val_ds = ISPRSRefDataset_MixSS(
                data_dir=data_dir, split="valid", variant=variant, resolution=480
            )

            # for input_batch in val_ds:

                # img = input_batch.images[0]
                # q = input_batch.find_queries[0]
                # obj = img.objects[0]

                # print(f"\n--- Sample {i} ---")
                # print(f"  Image tensor: {img.data.shape}")
                # print(f"  Query: {q.query_text[:60]}...")
                # print(f"  Mask: {obj.segment.shape}, sum={obj.segment.sum().item():.0f} px")
                # print(f"  Bbox (CxCyWH): {obj.bbox.tolist()}")
                # print(f"  Original size: {q.inference_metadata.original_size}")

        else:
            # Default: COCO format
            train_ds = COCOSegmentDataset(data_dir=data_dir, split="train")
            has_validation = False
            val_ds = None

            try:
                print_rank0(f"\nLoading validation data from {data_dir}...")
                val_ds = COCOSegmentDataset(data_dir=data_dir, split="valid")
                if len(val_ds) > 0:
                    has_validation = True
                    print_rank0(f"Found validation data: {len(val_ds)} images")
                else:
                    print_rank0("Validation dataset is empty.")
                    val_ds = None
            except Exception as e:
                print_rank0(f"Could not load validation data: {e}")
                val_ds = None

            if not has_validation:
                val_ds = None

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None

        if self.multi_gpu:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=get_rank(), shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds, num_replicas=self.world_size, rank=get_rank(), shuffle=False
                )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=True,
        )

        if has_validation:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=self.config["training"].get("num_workers", 0),
                pin_memory=True,
            )
        else:
            val_loader = None

        self.model.train()

        # Weights from a standard SAM config roughly
        # weight_dict = {
        #     "loss_ce": 2.0,
        #     "loss_bbox": 5.0,
        #     "loss_giou": 2.0,
        #     "loss_mask": 5.0,
        #     "loss_dice": 5.0,
        # }

        epochs = self.config["training"]["num_epochs"]
        best_val_loss = float("inf")
        print_rank0(f"Starting training for {epochs} epochs...")

        if has_validation:
            print_rank0(
                f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}"
            )
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("⚠️  No validation data found - training without validation")

        if self.multi_gpu:
            print_rank0(
                f"Effective batch size: {self.config['training']['batch_size']} x {self.world_size} = {self.config['training']['batch_size'] * self.world_size}"
            )

        # Helper to move BatchedDatapoint to device
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            if isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            if isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            if hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        # Create output directory
        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── LR Scheduler: warmup + cosine annealing ──
        total_steps = epochs * len(train_loader)
        print("total_steps: ", total_steps)
        warmup_steps = self.config["training"].get("warmup_steps", 500)
        warmup_steps = min(warmup_steps, total_steps // 2)  # cap at 50% of total
        print_rank0(f"LR schedule: warmup={warmup_steps} steps, total={total_steps} steps "
                     f"(~{warmup_steps / max(1, len(train_loader)):.1f} epochs)")

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        for epoch in range(epochs):
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Track training losses for this epoch
            train_losses = []

            # Only show progress bar on rank 0
            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}", disable=not is_main_process()
            )
            for batch_dict in pbar:
                input_batch = batch_dict["input"]
                input_batch = move_to_device(input_batch, self.device)

                # Forward pass
                # outputs_list is SAM3Output, we need to pass the whole thing to loss_wrapper
                outputs_list = self.model(input_batch)

                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget (one per stage)
                find_targets = [
                    self._unwrapped_model.back_convert(target)
                    for target in input_batch.find_targets
                ]

                # Move targets to device
                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                # Add matcher indices to outputs (required by Sam3LossWrapper)
                # Use SAM3Output.iteration_mode to properly iterate over outputs
                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        # stage_targets is a single target dict, replicate for all steps
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            # Compute indices for main output
                            outputs["indices"] = self.matcher(outputs, targets)

                            # Also add indices to auxiliary outputs if they exist
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                # Compute loss using Sam3LossWrapper
                # This handles num_boxes calculation and proper weighting
                loss_dict = self.loss_wrapper(outputs_list, find_targets)

                # Extract total loss
                total_loss = loss_dict[CORE_LOSS_KEY]

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                # Track training loss
                train_losses.append(total_loss.item())
                pbar.set_postfix({"loss": total_loss.item()})

            # Calculate average training loss for this epoch
            avg_train_loss = (
                sum(train_losses) / len(train_losses) if train_losses else 0.0
            )

            # Validation (only compute loss - no metrics, like SAM3)
            if has_validation and val_loader is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    val_pbar = tqdm(
                        val_loader, desc="Validation", disable=not is_main_process()
                    )

                    for batch_dict in val_pbar:
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)

                        # Forward pass
                        outputs_list = self.model(input_batch)

                        # Prepare targets
                        find_targets = [
                            self._unwrapped_model.back_convert(target)
                            for target in input_batch.find_targets
                        ]

                        # Move targets to device
                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        # Add matcher indices to outputs (required by Sam3LossWrapper)
                        with SAM3Output.iteration_mode(
                            outputs_list,
                            iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(
                                outputs_iter, find_targets
                            ):
                                stage_targets_list = [stage_targets] * len(
                                    stage_outputs
                                )
                                for outputs, targets in zip(
                                    stage_outputs, stage_targets_list
                                ):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(
                                                aux_out, targets
                                            )

                        # Compute loss using Sam3LossWrapper
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]

                        val_losses.append(total_loss.item())
                        val_pbar.set_postfix({"val_loss": total_loss.item()})

                avg_val_loss = sum(val_losses) / len(val_losses)

                # Synchronize val_loss across all processes for consistent best model selection
                if self.multi_gpu:
                    val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                print_rank0(
                    f"\nEpoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )

                # Save models based on validation loss (only on rank 0)
                if is_main_process():
                    # Get underlying model from DDP wrapper
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(
                        model_to_save, str(out_dir / "last_lora_weights.pt")
                    )

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_lora_weights(
                            model_to_save, str(out_dir / "best_lora_weights.pt")
                        )
                        print(f"✓ New best model saved (val_loss: {avg_val_loss:.6f})")

                    # Log to file
                    with open(out_dir / "val_stats.json", "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "epoch": epoch + 1,
                                    "train_loss": avg_train_loss,
                                    "val_loss": avg_val_loss,
                                }
                            )
                            + "\n"
                        )

                torch.cuda.empty_cache()

                # Back to training mode
                self.model.train()
            else:
                # No validation - just save model each epoch (only on rank 0)
                if is_main_process():
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(
                        model_to_save, str(out_dir / "last_lora_weights.pt")
                    )

        # Synchronize before final save
        if self.multi_gpu:
            dist.barrier()

        # Final save (only on rank 0)
        if is_main_process():
            if has_validation:
                print(f"\n{'=' * 80}")
                print("✅ Training complete!")
                print(f"{'=' * 80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print("  - best_lora_weights.pt (best validation loss)")
                print("  - last_lora_weights.pt (last epoch)")
                print("\n📊 To compute full metrics (mAP, cgF1) with NMS:")
                print("   python validate_sam3_lora.py \\")
                print("     --config <config_path> \\")
                print(f"     --weights {out_dir}/best_lora_weights.pt \\")
                print("     --val_data_dir <data_dir>/valid")
                print(f"{'=' * 80}")
            else:
                # If no validation, copy last to best
                import shutil

                last_path = out_dir / "last_lora_weights.pt"
                best_path = out_dir / "best_lora_weights.pt"
                if last_path.exists():
                    shutil.copy(last_path, best_path)

                print(f"\n{'=' * 80}")
                print("✅ Training complete!")
                print(f"{'=' * 80}")
                print(f"\nModels saved to {out_dir}:")
                print("  - best_lora_weights.pt (copy of last epoch)")
                print("  - last_lora_weights.pt (last epoch)")
                print(
                    "\nℹ️  No validation data - consider adding data/valid/ for better model selection"
                )
                print(f"{'=' * 80}")

        # Cleanup distributed training
        if self.multi_gpu:
            cleanup_distributed()


def launch_distributed_training(args):
    """Launch training with multiple GPUs using torchrun subprocess."""
    import subprocess
    import sys

    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching distributed training on GPUs: {devices}")
    print(f"Number of processes: {num_gpus}")

    # Build the command
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port",
        str(args.master_port),
        sys.argv[0],  # This script
        "--config",
        args.config,
        "--device",
        *map(str, devices),
        "--_launched_by_torchrun",  # Internal flag to indicate we're in subprocess
    ]

    # Set environment variable for visible devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str

    # Run the subprocess
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU (default GPU 0):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Single GPU (specific GPU):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 1

  Multi-GPU (GPUs 0 and 1):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1

  Multi-GPU (GPUs 0, 2, 3):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 2 3

  Multi-GPU (all 4 GPUs):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1 2 3
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_lora_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID(s) to use. Single value for single GPU, multiple values for multi-GPU. "
        "Example: --device 0 (single GPU), --device 0 1 2 (3 GPUs)",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun)",
    )
    parser.add_argument(
        "--_launched_by_torchrun",
        action="store_true",
        help=argparse.SUPPRESS,  # Hidden argument for internal use
    )
    args = parser.parse_args()

    # Determine if multi-GPU training is requested
    num_devices = len(args.device)
    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ

    if num_devices > 1 and not is_torchrun_subprocess:
        # Multi-GPU requested but not yet in torchrun - launch it
        launch_distributed_training(args)
    else:
        # Single GPU or already in torchrun subprocess
        multi_gpu = num_devices > 1 and is_torchrun_subprocess

        if not multi_gpu and num_devices == 1:
            # Single GPU mode - set the device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
            print(f"Using single GPU: {args.device[0]}")

        trainer = SAM3TrainerNative(args.config, multi_gpu=multi_gpu)
        trainer.train()



# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 scripts/training/train_sam3_lora_native.py --config configs/vaihingen_lora_config.yaml --device 0
# torchrun --nproc_per_node 2 scripts/training/train_sam3_lora_native_mix_ss.py --config configs/vaihingen_lora_config.yaml --device 0 1
# torchrun --nproc_per_node 2 scripts/training/train_sam3_lora_native_mix_ss.py --config configs/potsdam_lora_config.yaml --device 0 1


