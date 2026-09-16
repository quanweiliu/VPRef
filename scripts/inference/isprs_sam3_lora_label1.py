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
import cv2
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

from data.isprs_dataset import ISPRSRefDataset, SUFFIX_TO_CLASS
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


# Per-class score thresholds derived from calibration analysis.
# Building/Car: well-calibrated, accurate even at moderate scores.
# Impervious_surface/LowVeg: moderately calibrated, need stricter filtering.
# Tree: poorly calibrated (top-1 accuracy only 38.7%), skipped entirely.
CLASS_THRESHOLDS = {
    "Building":             0.8,
    "Car":                  0.8,
    "Impervious_surface":   0.9,
    "LowVeg":               0.9,
    "Tree":                 None,   # None = skip this class entirely
}

# Keyword mapping to infer class from the referring expression (text query).
# The text query is the model's INPUT condition, not a label — this is legitimate.
# Order matters: longer/more-specific phrases match first.
TEXT_TO_CLASS = [
    ("Impervious_surface", ["paved surface", "road", "street", "impervious",
                             "intersection", "pavement"]),
    ("Building",           ["detached house", "house", "building", "apartment"]),
    ("Car",                ["parking lot", "parked", "cars", "car", "vehicle",
                             "vehicles"]),
    ("Tree",               ["wooded patch", "wooded", "trees", "tree",
                             "forest", "woods"]),
    ("LowVeg",             ["low vegetation", "lawn", "grass", "meadow",
                             "vegetation", "green area"]),
]


def _resolve_class_from_query_text(query_text: str) -> str:
    """Infer class name from the text query via keyword matching (case-insensitive)."""
    text_lower = query_text.lower()
    for class_name, keywords in TEXT_TO_CLASS:
        for kw in keywords:
            if kw in text_lower:
                return class_name
    return "unknown"


def labelling(model, dataset, output_dir, score_threshold=0.5, device=None):
    """
    Generate pseudo masks from SAM3 model for self-training.

    For each sample, runs SAM3 inference, selects the single highest-confidence
    mask, checks it against a per-class score threshold (CLASS_THRESHOLDS),
    upsamples to original resolution, and saves as PNG.

    Classes with threshold=None (e.g. Tree) are skipped entirely.

    Args:
        model: SAM3 model (should already be on device and in eval mode)
        dataset: ISPRSRefDataset or DirectCOCODataset
        output_dir: Directory to save pseudo mask PNGs
        score_threshold: Fallback threshold for datasets without per-class mapping
        device: torch device (auto-detect if None)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Build dataloader with batch_size=1 for clean 1:1 sample→output mapping
    def collate_fn(batch):
        return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Detect dataset type for filename extraction
    has_samples_attr = hasattr(dataset, 'samples')
    print("has_samples_attr", has_samples_attr)  # Debug: check if dataset has 'samples' attribute
    # 如果 dataset 内部定义了 samples 变量或 samples() 方法，返回 True。
    # 如果没有找到，则返回 False（且不会报错）。

    skip_stats = {}  # {class_name: skip_count}
    save_stats = {}  # {class_name: save_count}

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(dataloader, desc="Generating pseudo masks")):
            input_batch = batch_dict["input"]
            input_batch = move_to_device(input_batch, device)

            outputs_list = model(input_batch)

            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_stage = list(outputs_iter)[-1]
                final_outputs = final_stage[-1]

            # batch_size=1: one sample, one query group
            pred_logits = final_outputs['pred_logits'][0]  # [num_queries, 1]
            pred_masks = final_outputs['pred_masks'][0]    # [num_queries, H_mask, W_mask]
            scores = torch.sigmoid(pred_logits).squeeze(-1)  # [num_queries]

            # ---- Debug: print score distribution for first 5 images ----
            # if batch_idx < 5:
            #     print(f"\n[DEBUG] Image {batch_idx}: {len(scores)} total queries")
            #     print(f"  Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
            #     print(f"  Mean/median: {scores.mean().item():.4f} / {scores.median().item():.4f}")
            #     print(f"  Valid (>{score_threshold}): {valid.sum().item()}/{len(scores)}")
            #     if len(scores) <= 15:
            #         print(f"  All scores: {[f'{s:.4f}' for s in scores.tolist()]}")
            #     else:
            #         top8 = scores.topk(min(8, len(scores)))
            #         print(f"  Top-8: {[f'{s:.4f}' for s in top8.values.tolist()]}")
            # ----------------------------------------------------------------

            # Get original image size (before resize/crop)
            datapoint = dataset[batch_idx]
            orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

            # --- Resolve class and threshold ---
            if has_samples_attr:
                query_text = dataset.samples[batch_idx][2]
                class_name = _resolve_class_from_query_text(query_text)
                class_threshold = CLASS_THRESHOLDS.get(class_name, score_threshold)
            else:
                class_name = "unknown"
                class_threshold = score_threshold

            # Skip classes marked as None
            if class_threshold is None:
                skip_stats[class_name] = skip_stats.get(class_name, 0) + 1
                del input_batch, outputs_list, final_outputs, batch_dict
                torch.cuda.empty_cache()
                continue

            # Take the single highest-confidence mask
            best_score, best_idx = scores.max(dim=0)
            if best_score < class_threshold:
                skip_stats[class_name] = skip_stats.get(class_name, 0) + 1
                del input_batch, outputs_list, final_outputs, batch_dict
                torch.cuda.empty_cache()
                continue

            best_mask_logits = pred_masks[best_idx]
            best_mask_prob = torch.sigmoid(best_mask_logits)

            # Upsample to original resolution
            # 仅限单张掩码 [H, W]
            mask_upsampled = torch.nn.functional.interpolate(
                best_mask_prob.unsqueeze(0).unsqueeze(0).float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            binary_masks = mask_upsampled > 0.5
            pseudo_mask = binary_masks.cpu().numpy().astype(np.uint8) * 255

            # --- Resolve output filename ---
            if has_samples_attr:
                # ISPRSRefDataset: samples[idx] = (img_path, mask_path, query_text)
                mask_path = dataset.samples[batch_idx][1]
                filename = Path(mask_path).stem + '.tif'
            else:
                # DirectCOCODataset / fallback: use COCO image_id
                coco_img_id = datapoint.find_queries[0].inference_metadata.coco_image_id
                filename = f"{coco_img_id}.tif"
            # print("filename", filename)

            cv2.imwrite(os.path.join(output_dir, filename), pseudo_mask)
            save_stats[class_name] = save_stats.get(class_name, 0) + 1

            # Free GPU memory
            del input_batch, outputs_list, final_outputs, batch_dict
            torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\nPseudo masks saved to {output_dir}/")
    print(f"  Total dataset samples: {len(dataset)}")
    print(f"  Per-class breakdown:")
    for cls_name in sorted(set(list(save_stats.keys()) + list(skip_stats.keys()))):
        saved = save_stats.get(cls_name, 0)
        skipped = skip_stats.get(cls_name, 0)
        threshold = CLASS_THRESHOLDS.get(cls_name, score_threshold)
        if threshold is None:
            note = "(class disabled)"
        else:
            note = f"(score > {threshold})"
        print(f"    {cls_name:>25s}: saved {saved:4d}, skipped {skipped:4d}  {note}")
    print(f"  Total saved: {sum(save_stats.values())}, total skipped: {sum(skip_stats.values())}")


def check_calibration(model, dataset, device=None, num_bins=10, iou_threshold=0.5):
    """
    Check if SAM3 sigmoid scores are well-calibrated against actual mask quality.

    For each sample, runs inference and records every query's sigmoid score
    along with its IoU against the ground-truth mask. Then groups predictions
    into confidence bins and reports:
      - Expected accuracy (mean confidence in bin)
      - Actual accuracy (fraction of masks with IoU > threshold)
      - ECE (Expected Calibration Error)

    Args:
        model: SAM3 model (should already be on device and in eval mode)
        dataset: ISPRSRefDataset with ground-truth masks
        device: torch device (auto-detect if None)
        num_bins: Number of confidence bins (default: 10)
        iou_threshold: IoU threshold for a "correct" prediction (default: 0.5)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    def collate_fn(batch):
        return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    all_scores = []       # sigmoid scores of every query
    all_ious_max = []     # best IoU against GT for every query
    all_ious_best = []    # IoU of the single best-scoring mask per sample
    all_scores_best = []  # score of the single best-scoring mask per sample
    all_sample_classes = []  # class name per sample (for per-class breakdown)

    has_samples = hasattr(dataset, 'samples')

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(dataloader, desc="Calibration check")):
            input_batch = batch_dict["input"]
            input_batch = move_to_device(input_batch, device)

            outputs_list = model(input_batch)

            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_stage = list(outputs_iter)[-1]
                final_outputs = final_stage[-1]

            pred_logits = final_outputs['pred_logits'][0]  # [num_queries, 1]
            pred_masks = final_outputs['pred_masks'][0]    # [num_queries, H_mask, W_mask]
            scores = torch.sigmoid(pred_logits).squeeze(-1)  # [num_queries]

            # Get GT mask from batch
            if input_batch.find_targets is None or len(input_batch.find_targets) == 0:
                continue
            gt_targets = input_batch.find_targets[0]
            if gt_targets.segments is None or gt_targets.is_valid_segment is None:
                continue
            # Take the first valid GT mask (batch_size=1, single object per sample)
            valid_mask = gt_targets.is_valid_segment[0].item()
            if not valid_mask:
                continue
            gt_mask = gt_targets.segments[0]  # [H_gt, W_gt], bool

            # Resolve class name for this sample
            if has_samples:
                _, mask_path, _ = dataset.samples[batch_idx]
                fname = Path(mask_path).stem
                parts = fname.split("_")
                class_idx = parts[-1] if len(parts) >= 3 else "?"
                class_name = SUFFIX_TO_CLASS.get(class_idx, f"class_{class_idx}")
            else:
                class_name = "unknown"

            # Upsample predicted masks to GT resolution
            H_gt, W_gt = gt_mask.shape
            _, H_pred, W_pred = pred_masks.shape

            if (H_pred != H_gt) or (W_pred != W_gt):
                pred_masks_up = torch.nn.functional.interpolate(
                    pred_masks.unsqueeze(1).float(),  # [N, 1, H_pred, W_pred]
                    size=(H_gt, W_gt),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # [N, H_gt, W_gt]
            else:
                pred_masks_up = pred_masks.float()

            pred_masks_bin = pred_masks_up > 0.0  # logits > 0 → positive

            # Compute IoU for every query against the single GT mask
            n_queries = len(scores)
            for q in range(n_queries):
                pred = pred_masks_bin[q]
                intersection = (pred & gt_mask).sum().float()
                union = (pred | gt_mask).sum().float()
                iou = (intersection / union).item() if union > 0 else 0.0
                all_scores.append(scores[q].item())
                all_ious_max.append(iou)

            # Also track the best-scoring mask per sample
            best_idx = scores.argmax().item()
            all_scores_best.append(scores[best_idx].item())
            all_ious_best.append(all_ious_max[-n_queries + best_idx] if n_queries > 0 else 0.0)
            all_sample_classes.append(class_name)

            del input_batch, outputs_list, final_outputs, batch_dict
            torch.cuda.empty_cache()

    # ── Build calibration table ──
    scores_arr = np.array(all_scores)
    ious_arr = np.array(all_ious_max)
    correct_arr = (ious_arr > iou_threshold).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    print("\n" + "=" * 90)
    print("CALIBRATION ANALYSIS")
    print("=" * 90)
    print(f"  Total query predictions: {len(scores_arr)}")
    print(f"  Score range: [{scores_arr.min():.4f}, {scores_arr.max():.4f}]")
    print(f"  Score mean/median: {scores_arr.mean():.4f} / {np.median(scores_arr):.4f}")
    print(f"  Fraction with IoU > {iou_threshold}: {correct_arr.mean():.4f}")
    print(f"  IoU mean/median: {ious_arr.mean():.4f} / {np.median(ious_arr):.4f}")
    print()

    print(f"  {'Bin':>6s}  {'#Samples':>8s}  {'Conf_Mean':>10s}  {'Accuracy':>10s}  {'Gap':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")

    ece = 0.0
    total_n = len(scores_arr)

    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (scores_arr >= lo) & (scores_arr < hi)
        if b == num_bins - 1:
            in_bin = (scores_arr >= lo) & (scores_arr <= hi)  # include 1.0 in last bin
        n_bin = in_bin.sum()

        if n_bin > 0:
            conf_mean = scores_arr[in_bin].mean()
            acc = correct_arr[in_bin].mean()
            gap = conf_mean - acc
            ece += (n_bin / total_n) * abs(gap)
            print(f"  [{lo:.1f},{hi:.1f}]  {n_bin:8d}  {conf_mean:10.4f}  {acc:10.4f}  {gap:+8.4f}")
        else:
            print(f"  [{lo:.1f},{hi:.1f}]  {n_bin:8d}  {'--':>10s}  {'--':>10s}  {'--':>8s}")

    print(f"\n  ECE (Expected Calibration Error): {ece:.4f}")

    # ── Per-class breakdown ──
    if has_samples and all_sample_classes:
        unique_classes = sorted(set(all_sample_classes))
        print(f"\n  Per-class Top-1 mask quality (IoU threshold: {iou_threshold}):")
        print(f"  {'Class':>25s}  {'#Samples':>8s}  {'Score_mean':>10s}  {'IoU_mean':>10s}  {'Acc':>8s}")
        print(f"  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
        for cls_name in unique_classes:
            cls_mask = np.array([c == cls_name for c in all_sample_classes])
            n_cls = cls_mask.sum()
            if n_cls > 0:
                cls_scores = np.array(all_scores_best)[cls_mask]
                cls_ious = np.array(all_ious_best)[cls_mask]
                cls_acc = (cls_ious > iou_threshold).mean()
                print(f"  {cls_name:>25s}  {n_cls:8d}  {cls_scores.mean():10.4f}  {cls_ious.mean():10.4f}  {cls_acc:8.4f}")

    # ── Best-mask analysis ──
    scores_best_arr = np.array(all_scores_best)
    ious_best_arr = np.array(all_ious_best)
    print(f"\n  Top-1 mask statistics (n={len(scores_best_arr)} samples):")
    print(f"    Score mean/median: {scores_best_arr.mean():.4f} / {np.median(scores_best_arr):.4f}")
    print(f"    IoU   mean/median: {ious_best_arr.mean():.4f} / {np.median(ious_best_arr):.4f}")
    print(f"    IoU > {iou_threshold}: {(ious_best_arr > iou_threshold).mean():.4f}")

    # ── Score distribution histogram (text-based) ──
    print(f"\n  Score distribution of all queries:")
    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (scores_arr >= lo) & (scores_arr < hi)
        if b == num_bins - 1:
            in_bin = (scores_arr >= lo) & (scores_arr <= hi)
        n_bin = in_bin.sum()
        pct = n_bin / total_n * 100
        bar = "#" * max(1, int(pct))
        print(f"  [{lo:.1f},{hi:.1f}] {n_bin:6d} ({pct:5.1f}%) {bar}")

    # ── Recommendation ──
    print(f"\n  ── Recommendation ──")
    if ece < 0.05:
        print(f"  Scores are well-calibrated (ECE={ece:.4f}). No threshold adjustment needed.")
    elif ece < 0.10:
        print(f"  Scores are moderately calibrated (ECE={ece:.4f}). Minor threshold tuning may help.")
    else:
        print(f"  Scores are poorly calibrated (ECE={ece:.4f}).")
        print(f"  Consider per-class score thresholds rather than a single global value.")
        # Suggest a threshold where accuracy is still acceptable
        for b in range(num_bins - 1, -1, -1):
            in_bin = (scores_arr >= bin_edges[b]) & (scores_arr <= 1.0)
            if in_bin.sum() > 10:
                acc = correct_arr[in_bin].mean()
                if acc > 0.7:
                    print(f"  Suggested threshold ~{bin_edges[b]:.2f} "
                          f"(accuracy={acc:.3f} for scores >= {bin_edges[b]:.2f})")
                    break

    print("=" * 90)

    return {
        "ece": ece,
        "scores": scores_arr,
        "ious": ious_arr,
        "correct": correct_arr,
        "scores_best": scores_best_arr,
        "ious_best": ious_best_arr,
    }


def setup_model_data_configure(config_path, weights_path, val_data_dir, 
             use_base_model=False, dataset_type="coco",
             vai_data_dir=None, vai_split="valid", 
             vai_variant="standard", device=None):
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

        # Load config (only needed for LoRA settings)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

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
        batch_size = config["training"]["batch_size"]*4

    model.to(device)
    model.eval()

    # Load validation data
    if dataset_type == "isprsRef":
        if vai_data_dir is None:
            raise ValueError("vai_data_dir is required when dataset_type='vaihingen'")
        print(f"\nLoading Vaihingen validation data from {vai_data_dir}...")
        val_ds = ISPRSRefDataset(
            data_dir=vai_data_dir,
            split=vai_split,
            variant=vai_variant,
            resolution=480
        )

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

    return val_ds, val_loader, model, batch_size


def validate(val_ds, val_loader, model, batch_size, num_samples=None, \
             prob_threshold=0.3, nms_iou=0.7, merge_cracks=False, merge_iou=0.15, \
             device=None):
    
    logger=logging.getLogger("test")

    all_image_ids = []
    all_coco_predictions = []
    CHUNK_SIZE = 100  # Process 100 images per chunk to limit memory
    first_batch_scores = None  # For debug when no predictions survive NMS

    # Use automatic mixed precision for faster inference
    use_amp = device.type == 'cuda'

    chunk_predictions = []
    chunk_image_ids = []

    if num_samples:
        print(f"\n[INFO] Limiting validation to {num_samples} samples for debugging")

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
                # outputs_list 是 SAM3Output 类型（继承自 list），内部结构是两层嵌套列表：


                # SAM3Output([
                #     [out_dict_step_0, out_dict_step_1, ..., out_dict_step_N],   # Stage 0（通常1个stage）
                # ])
                # 外层：stages（阶段）— 单张图通常 1 个 stage
                # 内层：interactive steps — 训练时只有 1 步，eval 时可能有 0~N 步（由 num_interactive_steps_val 控制）

            # Extract predictions
            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_stage = list(outputs_iter)[-1]  # 最后一个 stage 的所有 steps
                final_outputs = final_stage[-1]    # 最后一步的 output dict

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
        "--check-calibration",
        action="store_true",
        help="Run score calibration analysis on the validation set (instead of labelling)"
    )
    parser.add_argument(
        "--calibration-score-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for considering a prediction 'correct' in calibration analysis (default: 0.5)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="coco",
        choices=["coco", "isprsRef"],
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

    if args.file_path is None:
        parser.error("--file_path is required (points to output dir with config.yaml and best_lora_weights.pt)")

    args.config = os.path.join(args.file_path, "config.yaml")
    args.weights = os.path.join(args.file_path, "best_lora_weights.pt")
    
    # Validate argument combinations
    if not args.use_base_model:
        if not os.path.exists(args.config) or not os.path.exists(args.weights):
            parser.error(f"config.yaml or best_lora_weights.pt not found in {args.file_path}")

    logging.basicConfig(level=logging.INFO, \
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", \
                        datefmt="%Y-%m-%d %H:%M:%S", \
                        handlers=[
                            logging.FileHandler(os.path.join(args.file_path, "results.log"), mode="a"),  # 用于文件保存
                            logging.StreamHandler()   # 用于在 terminal 中的文件打印
                        ],
                        force=True  # <--- 关键：强制覆盖之前的配置
                        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_ds, val_loader, model, batch_size = setup_model_data_configure(
        config_path=args.config,
        weights_path=args.weights,
        val_data_dir=args.val_data_dir,
        use_base_model=args.use_base_model,
        dataset_type=args.dataset_type,
        vai_data_dir=args.vai_data_dir,
        vai_split=args.vai_split,
        vai_variant=args.vai_variant,
        device=device
    )

    validate(
        val_ds,
        val_loader,
        model,
        batch_size,
        num_samples=None,
        merge_iou=args.merge_iou,
        device=device
    )

    if args.check_calibration:
        check_calibration(
            model=model,
            dataset=val_ds,
            device=device,
            iou_threshold=args.calibration_score_threshold,
        )
    else:
        labelling(
            model=model,
            dataset=val_ds,
            # output_dir="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/unlabeled_standardf",
            output_dir="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/unlabeled_valid_standard_newSf",
            score_threshold=0.5,
            device=device
        )



# python scripts/inference/isprs_sam3_lora_label.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_potsdam_concept --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split test --vai-variant concept --use-base-model
# python scripts/inference/isprs_sam3_lora_label.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_complex --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split test --vai-variant complex --use-base-model
# python scripts/inference/isprs_sam3_lora_label.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_concept --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split valid --vai-variant concept --use-base-model
# python scripts/inference/isprs_sam3_lora_label.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_standard_newSet_ss --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split test --vai-variant concept



# labelling
# python scripts/inference/isprs_sam3_lora_label.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_concept --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split valid --vai-variant concept
# python scripts/inference/isprs_sam3_lora_label.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_newSetting --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split valid --vai-variant concept
# python scripts/inference/isprs_sam3_lora_label.py --file_path /home/icclab/Documents/lqw/sam3_finetune_lora/outputs/sam3_lora_vaihingen_standard_newSet --dataset-type isprsRef --vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef --vai-split valid --vai-variant standard
