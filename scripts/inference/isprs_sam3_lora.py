#!/usr/bin/env python3
"""
Validation script for SAM3 LoRA model
Loads saved weights and runs validation with detailed debugging
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [0]))
print('using GPU %s' % ','.join(map(str, [0])))

import sys
# Add project root and training directory for local imports
_SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(_SCRIPT_DIR, "../.."))
sys.path.append(os.path.join(_SCRIPT_DIR, "../training"))
sys.path.append("/home/icclab/Documents/lqw/sam3_finetune_lora/scripts")

import argparse
import yaml
import shutil
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import logging
import numpy as np
from PIL import Image as PILImage
import contextlib
import gc

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights, count_parameters

from torchvision.transforms import v2

# Import evaluation modules
from sam3.eval.cgf1_eval import CGF1Evaluator, COCOCustom
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
from sam3.train.masks_ops import rle_encode

# Import SAM3's NMS
# from sam3.perflib.nms import nms_masks

from data.RefSegRS import RefSegRSDataset
from data.isprs_dataset import ISPRSRefDataset
from data.rrsid import RRSISDDataset
from data.COCODataset import COCOSegmentDataset, DirectCOCODataset
from tools.utils import setup_distributed, cleanup_distributed, patch_dynamic_rope
from tools.utils import merge_overlapping_masks, get_world_size, get_rank, print_rank0
from tools.utils import compute_miou, convert_predictions_to_coco_format, create_coco_gt_from_dataset
from tools.utils import convert_predictions_to_coco_format_original_res, create_coco_gt_from_dataset_original_res


def move_to_device(obj, device):
    """Recursively move objects to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            val = getattr(obj, field)
            setattr(obj, field, move_to_device(val, device))
        return obj
    return obj


def validate(config_path, weights_path, val_data_dir, num_samples=None,
             prob_threshold=0.3, nms_iou=0.7, merge_cracks=False, merge_iou=0.15,
             use_base_model=False, dataset_type="coco",
             vai_data_dir=None, vai_split="valid", vai_variant="standard"):
    """Run validation with full metrics (mAP, cgF1) and SAM3 NMS

    Args:
        config_path: Path to config file (for LoRA settings only). Not required if use_base_model=True.
        weights_path: Path to LoRA weights. Not required if use_base_model=True.
        val_data_dir: Direct path to validation data directory containing _annotations.coco.json
                      (e.g., /workspace/data2/valid)
        num_samples: Optional limit for number of samples (for debugging)
        use_base_model: If True, use original SAM3 model without LoRA (default: False)

    Example (with LoRA):
        validate(
            config_path="configs/full_lora_config.yaml",
            weights_path="outputs/sam3_lora_full/best_lora_weights.pt",
            val_data_dir="/workspace/data2/valid"
        )

    Example (base SAM3 model):
        validate(
            config_path=None,
            weights_path=None,
            val_data_dir="/workspace/data2/valid",
            use_base_model=True
        )
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger=logging.getLogger("test")

    # Build model
    print("\nBuilding SAM3 model...")
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        checkpoint_path="/home/icclab/Documents/lqw/sam3/weights/sam3.pt",
        load_from_HF=False,
        bpe_path="/home/icclab/Documents/lqw/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False
    )

    # Patch RoPE for dynamic resolution support (enables non-1008 inputs)
    n_patched = patch_dynamic_rope(model)
    print(f"Dynamic RoPE patch applied to {n_patched} attention layers")
    
    # Load config for batch_size and other settings
    if use_base_model:
        # Use original SAM3 model without LoRA
        print("Using original SAM3 model (no LoRA)")
        stats = count_parameters(model)
        print(f"Total params: {stats['total_parameters']:,}")
        # Use default batch_size for base model
        batch_size = 1
    else:
        # Apply LoRA and load weights
        if config_path is None or weights_path is None:
            raise ValueError("config_path and weights_path are required when use_base_model=False")

        # Apply LoRA
        print("Applying LoRA configuration...")
        lora_cfg = config["lora"]
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
        model = apply_lora_to_model(model, lora_config)

        # Load weights
        print(f"\nLoading LoRA weights from {weights_path}...")
        load_lora_weights(model, weights_path)

        stats = count_parameters(model)
        print(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        # Get batch_size from config
        batch_size = config["training"]["batch_size"]

    model.to(device)
    model.eval()


    if vai_data_dir is None:
        raise ValueError("vai_data_dir is required when dataset_type='vaihingen'")
    # Load validation data
    if dataset_type == "isprsRef":
        print(f"\nLoading Vaihingen validation data from {vai_data_dir}...")
        val_ds = ISPRSRefDataset(
            data_dir=vai_data_dir,
            split=vai_split,
            variant=vai_variant,
            resolution=480
        )
    elif dataset_type == "refsegrs":
        print(f"\nLoading refsegrs validation data from {vai_data_dir}...")
        val_ds = RefSegRSDataset(
            data_dir=vai_data_dir, 
            split=vai_split, 
            resolution=800)
    
    elif dataset_type == "rrsisd":
        print(f"\nLoading rrsisd validation data from {vai_data_dir}...")
        val_ds = RRSISDDataset(
            data_dir=vai_data_dir, 
            split=vai_split, 
            resolution=480)

        # for input_batch in val_ds:

        #     img = input_batch.images[0]
        #     q = input_batch.find_queries[0]
        #     obj = img.objects[0]

        #     # print(f"\n--- Sample {i} ---")
        #     print(f"  Image tensor: {img.data.shape}")
        #     print(f"  Query: {q.query_text[:60]}...")
        #     print(f"  Mask: {obj.segment.shape}, sum={obj.segment.sum().item():.0f} px")
        #     print(f"  Bbox (CxCyWH): {obj.bbox.tolist()}")
        #     print(f"  Original size: {q.inference_metadata.original_size}")
    
    else:
        print(f"\nLoading validation data from {val_data_dir}...")

        # Load COCO annotations directly
        from pathlib import Path
        ann_file = Path(val_data_dir) / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        # Create a simple dataset class that loads from the directory directly
        val_ds = DirectCOCODataset(val_data_dir)

    if num_samples:
        print(f"\n[INFO] Limiting validation to {num_samples} samples for debugging")

    def collate_fn(batch):
        return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,  # Enable parallel data loading
        pin_memory=True  # Faster GPU transfer
    )

    # Run validation
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)

    all_image_ids = []
    all_coco_predictions = []
    CHUNK_SIZE = 100  # Process 100 images per chunk to limit memory
    first_batch_scores = None  # For debug when no predictions survive NMS

    # Use automatic mixed precision for faster inference
    use_amp = device.type == 'cuda'

    chunk_predictions = []
    chunk_image_ids = []

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(val_loader, desc="Validation")):
            if num_samples and batch_idx * batch_size >= num_samples:
                break

            input_batch = batch_dict["input"]
            input_batch = move_to_device(input_batch, device)
            # Forward pass with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs_list = model(input_batch)
            else:
                outputs_list = model(input_batch)

            # Extract predictions
            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_stage = list(outputs_iter)[-1]
                final_outputs = final_stage[-1]

                batch_size_actual = final_outputs['pred_logits'].shape[0]

                for i in range(batch_size_actual):
                    img_id = batch_idx * batch_size + i
                    all_image_ids.append(img_id)
                    chunk_image_ids.append(img_id)
                    chunk_predictions.append({
                        'pred_logits': final_outputs['pred_logits'][i].detach().cpu(),
                        'pred_boxes': final_outputs['pred_boxes'][i].detach().cpu(),
                        'pred_masks': final_outputs['pred_masks'][i].detach().cpu()
                    })

                # Capture first batch scores for debug
                if batch_idx == 0:
                    first_batch_scores = torch.sigmoid(final_outputs['pred_logits'].detach().cpu())

            # Free GPU memory from this batch
            del input_batch, outputs_list, final_outputs, batch_dict

            # Flush chunk to compact COCO format to avoid accumulating raw tensors
            if len(chunk_predictions) >= CHUNK_SIZE:
                chunk_coco = convert_predictions_to_coco_format_original_res(
                    chunk_predictions, 
                    chunk_image_ids,
                    val_ds,
                    score_threshold=prob_threshold,
                    iou_threshold=merge_iou,
                    )
                # chunk_coco = convert_predictions_to_coco_format(
                #     chunk_predictions, 
                #     chunk_image_ids,
                #     resolution=288,
                #     prob_threshold=prob_threshold,
                #     nms_iou_threshold=nms_iou,
                #     max_detections=100,
                #     merge_cracks=merge_cracks,
                #     merge_iou_threshold=merge_iou
                # )
                all_coco_predictions.extend(chunk_coco)
                del chunk_predictions, chunk_image_ids, chunk_coco
                chunk_predictions = []
                chunk_image_ids = []
                torch.cuda.empty_cache()
                gc.collect()

    # Flush remaining predictions
    if chunk_predictions:
        chunk_coco = convert_predictions_to_coco_format_original_res(
            chunk_predictions, 
            chunk_image_ids,
            val_ds,
            score_threshold=prob_threshold,
            iou_threshold=merge_iou,
            )
        # chunk_coco = convert_predictions_to_coco_format(
        #     chunk_predictions, chunk_image_ids,
        #     resolution=288,
        #     prob_threshold=prob_threshold,
        #     nms_iou_threshold=nms_iou,
        #     max_detections=100,
        #     merge_cracks=merge_cracks,
        #     merge_iou_threshold=merge_iou
        # )
        all_coco_predictions.extend(chunk_coco)
        del chunk_predictions, chunk_image_ids, chunk_coco
        torch.cuda.empty_cache()
        gc.collect()

    coco_predictions = all_coco_predictions
    del all_coco_predictions
    print(f"\n[INFO] Collected {len(coco_predictions)} COCO-format predictions from {len(all_image_ids)} images")

    # If no predictions survived thresholding, analyze first batch scores for debugging
    if len(coco_predictions) == 0 and first_batch_scores is not None:
        print(f"\n[DEBUG] No predictions after NMS — analyzing first batch scores...")
        scores = first_batch_scores.squeeze(-1)
        print(f"[DEBUG] Raw pred_logits shape: {first_batch_scores.shape}")
        print(f"[DEBUG] Sigmoid scores: min={scores.min().item():.6f}, "
              f"max={scores.max().item():.6f}, "
              f"mean={scores.mean().item():.6f}")
        print(f"[DEBUG] Median score: {scores.median().item():.6f}")
        print(f"[DEBUG] Fraction > 0.1: {(scores > 0.1).float().mean().item():.3f}")
        print(f"[DEBUG] Fraction > 0.3: {(scores > 0.3).float().mean().item():.3f}")
        print(f"[INFO] Try: --prob-threshold 0.05 or 0.1")
        del first_batch_scores

    # Compute metrics
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)

    # Create COCO ground truth (downsampled to 288×288)
    print(f"\n[INFO] Creating ground truth from validation dataset...")
    coco_gt_dict = create_coco_gt_from_dataset_original_res(val_ds)
    # coco_gt_dict = create_coco_gt_from_dataset(
    #     val_ds,
    #     image_ids=all_image_ids,
    #     mask_resolution=288
    # )

    # Filter predictions to only include images that exist in the ground truth
    gt_image_ids = set(img['id'] for img in coco_gt_dict['images'])
    filtered_predictions = [p for p in coco_predictions if p['image_id'] in gt_image_ids]
    if len(filtered_predictions) < len(coco_predictions):
        print(f"[INFO] Skipped {len(coco_predictions) - len(filtered_predictions)} predictions with no GT")
    coco_predictions = filtered_predictions

    if merge_cracks:
        print(f"\n[INFO] Total predictions after CRACK MERGING: {len(coco_predictions)}")
    else:
        print(f"\n[INFO] Total predictions after SAM3 NMS filtering: {len(coco_predictions)}")

    if len(coco_predictions) > 0:
        print("here is true, we went into coco_predictions")
        # Save temporary files for COCO evaluation
        import tempfile

        # Create temp directory for evaluation files
        temp_dir = tempfile.mkdtemp(prefix="sam3_eval_")
        gt_file = os.path.join(temp_dir, "gt.json")
        pred_file = os.path.join(temp_dir, "pred.json")

        with open(gt_file, 'w') as f:
            json.dump(coco_gt_dict, f)
        with open(pred_file, 'w') as f:
            json.dump(coco_predictions, f)

        # Compute mAP
        print("\n" + "="*80)
        print("COCO mAP EVALUATION")
        print("="*80)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_gt = COCO(str(gt_file))
                coco_dt = coco_gt.loadRes(str(pred_file))
                coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
                coco_eval.params.useCats = False
                coco_eval.evaluate()
                coco_eval.accumulate()

        # Print mAP results
        coco_eval.summarize()

        # map_segm = coco_eval.stats[0]
        # map50_segm = coco_eval.stats[1]
        # map75_segm = coco_eval.stats[2]

        # # Compute cgF1
        # print("\n" + "="*80)
        # print("cgF1 EVALUATION")
        # print("="*80)

        # cgf1_evaluator = CGF1Evaluator(
        #     gt_path=str(gt_file),
        #     iou_type='segm',
        #     verbose=True
        # )
        # cgf1_results = cgf1_evaluator.evaluate(str(pred_file))

        # cgf1 = cgf1_results.get('cgF1_eval_segm_cgF1', 0.0)
        # cgf1_50 = cgf1_results.get('cgF1_eval_segm_cgF1@0.5', 0.0)
        # cgf1_75 = cgf1_results.get('cgF1_eval_segm_cgF1@0.75', 0.0)

        # Compute MIoU (pixel-level Mean IoU)
        print("\n" + "="*80)
        print("MIoU EVALUATION (Pixel-Level)")
        # print("="*80)

        miou_results = compute_miou(coco_gt_dict, coco_predictions)

        # # Print summary
        # print("\n" + "="*80)
        # print("FINAL RESULTS")
        # print("="*80)
        # print(f"mAP (IoU 0.50:0.95): {map_segm:.4f}")
        # print(f"mAP@50: {map50_segm:.4f}")
        # print(f"mAP@75: {map75_segm:.4f}")
        # print(f"cgF1 (IoU 0.50:0.95): {cgf1:.4f}")
        # print(f"cgF1@50: {cgf1_50:.4f}")
        # print(f"cgF1@75: {cgf1_75:.4f}")
        # print(f"---")
        # print(f"mIoU: {miou_results['miou']:.4f}")
        # print(f"IoU (object): {miou_results['iou_object']:.4f}")
        # print(f"IoU (background): {miou_results['iou_background']:.4f}")
        # print("="*80)

        results_str = '\n'
        results_str += f"mIoU: {100*miou_results['miou']:.4f}\n " \
                    f"oIoU: {100*miou_results['oiou']:.4f}\n " \
                    f"IoU (object): {100*miou_results['iou_object']:.4f}\n " \
                    f"IoU (background): {100*miou_results['iou_background']:.4f}"

        logger.info('Final results:')
        logger.info(results_str)

        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    else:
        print("\n[ERROR] No predictions generated! Cannot compute metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone validation script for SAM3 LoRA model with full metrics (mAP, cgF1) and SAM3 NMS"
    )

    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Path to config file (for LoRA settings). Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        # required=True,
        default=None,
        help="Direct path to validation data directory containing _annotations.coco.json (e.g., /workspace/data2/valid)"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use original SAM3 model without LoRA (for baseline comparison)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit validation to N samples (for debugging)"
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for filtering predictions (default: 0.3)"
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable aggressive merging of overlapping segments (recommended for crack detection)"
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.15,
        help="IoU threshold for merging overlapping predictions (default: 0.15, lower = more aggressive)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="coco",
        choices=["coco", "isprsRef", "refsegrs", "rrsisd"],
        help="Dataset type: 'coco' (COCO format) or 'vaihingen' (VaihingenRef)"
    )
    parser.add_argument(
        "--vai-data-dir",
        type=str,
        default=None,
        help="Path to VaihingenRef dataset root (required when --dataset-type=vaihingen)"
    )
    parser.add_argument(
        "--vai-split",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Vaihingen dataset split (default: valid)"
    )
    parser.add_argument(
        "--vai-variant",
        type=str,
        default="standard",
        choices=["concept", "simple", "standard", "complex"],
        help="Vaihingen text variant (default: standard)"
    )
    args = parser.parse_args()
    args.config = os.path.join(args.file_path, "config.yaml")
    args.weights = os.path.join(args.file_path, "best_lora_weights.pt")
    
    # Validate argument combinations
    if not args.use_base_model:
        if args.config is None or args.weights is None:
            parser.error("--config and --weights are required when not using --use-base-model")

    logging.basicConfig(level=logging.INFO, \
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", \
                        datefmt="%Y-%m-%d %H:%M:%S", \
                        handlers=[
                            logging.FileHandler(os.path.join(args.file_path, "results.log"), mode="a"),  # 用于文件保存
                            logging.StreamHandler()   # 用于在 terminal 中的文件打印
                        ],
                        force=True  # <--- 关键：强制覆盖之前的配置
                        )
    
    validate(
        config_path=args.config,
        weights_path=args.weights,
        val_data_dir=args.val_data_dir,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        merge_cracks=args.merge,
        merge_iou=args.merge_iou,
        use_base_model=args.use_base_model,
        dataset_type=args.dataset_type,
        vai_data_dir=args.vai_data_dir,
        vai_split=args.vai_split,
        vai_variant=args.vai_variant,
    )
    
    logging.shutdown()


# python scripts/inference/isprs_sam3_lora.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_standard_newSet2 --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split valid --vai-variant standard --use-base-model
# python scripts/inference/isprs_sam3_lora.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_standard2222222 --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef --vai-split test --vai-variant standard --use-base-model
# python scripts/inference/isprs_sam3_lora.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_simple_newSet_ss --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split valid --vai-variant simple --use-base-model
# python scripts/inference/isprs_sam3_lora.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_simple_newSet --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef --vai-split test --vai-variant standard --use-base-model
# python scripts/inference/isprs_sam3_lora.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_mix --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef --vai-split test --vai-variant standard --use-base-model
# python scripts/inference/isprs_sam3_lora.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_mix_ssf --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split test --vai-variant standard --use-base-model

# python scripts/inference/isprs_sam3_lora.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_refsegrs --dataset-type refsegrs --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/RefSegRS --vai-split test --use-base-model
# python scripts/inference/isprs_sam3_lora.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_rrsid --dataset-type rrsisd --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/RRSISD --vai-split test



