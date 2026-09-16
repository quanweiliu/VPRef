#!/usr/bin/env python3
"""
Compare LoRA vs Base model on multiple images grouped by prompt
"""

import argparse
import os
import json
import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import yaml
import pycocotools.mask as mask_utils
from pathlib import Path

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights


def load_lora_model(config_path, weights_path, device='cuda'):
    """Load SAM3 model with LoRA weights"""
    print("Loading LoRA model...")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = build_sam3_image_model(
        device=device,
        compile=False,
        checkpoint_path="/root/autodl-tmp/sam3_checkpoint/sam3.pt",  # 设置直接加载权重
        load_from_HF=False,  # Tries to download from HF if checkpoint_path is None
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=True
    )

    lora_cfg = config["lora"]
    lora_config = LoRAConfig(
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=0.0,
        target_modules=lora_cfg["target_modules"],
        apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
        apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
        apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
        apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
        apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
        apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
    )
    model = apply_lora_to_model(model, lora_config)
    load_lora_weights(model, weights_path)
    model.to(device)
    model.eval()

    return model


def load_base_model(device='cuda'):
    """Load base SAM3 model without LoRA"""
    print("Loading base model...")

    model = build_sam3_image_model(
        device=device,
        compile=False,
        checkpoint_path="/root/autodl-tmp/sam3_checkpoint/sam3.pt",  # 设置直接加载权重
        load_from_HF=False,  # Tries to download from HF if checkpoint_path is None
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=True
    )
    model.to(device)
    model.eval()

    return model


def create_datapoint(pil_image, prompt):
    """Create SAM3 datapoint"""
    w, h = pil_image.size

    sam_image = SAMImage(
        data=pil_image,
        objects=[],
        size=[h, w]
    )

    query = FindQueryLoaded(
        query_text=prompt,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=0,
            original_image_id=0,
            original_category_id=1,
            original_size=[w, h],
            object_id=0,
            frame_index=0,
        )
    )

    return Datapoint(
        find_queries=[query],
        images=[sam_image]
    )


@torch.no_grad()
def predict(model, image_path, prompt, resolution=1008, threshold=0.5, device='cuda'):
    """Run inference on image"""
    pil_image = PILImage.open(image_path).convert("RGB")
    datapoint = create_datapoint(pil_image, prompt)

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=resolution,
                max_size=resolution,
                square=True,
                consistent_transform=False
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    datapoint = transform(datapoint)

    batch = collate_fn_api([datapoint], dict_key="input")["input"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    outputs = model(batch)
    last_output = outputs[-1]

    pred_logits = last_output['pred_logits']
    pred_boxes = last_output['pred_boxes']
    pred_masks = last_output.get('pred_masks', None)

    scores = pred_logits.sigmoid()[0, :, :].max(dim=-1)[0]
    keep = scores > threshold
    num_keep = keep.sum().item()

    if num_keep == 0:
        return pil_image, None, 0

    if pred_masks is not None:
        import torch.nn.functional as F
        masks_small = pred_masks[0, keep].sigmoid() > 0.5
        orig_h, orig_w = pil_image.size[1], pil_image.size[0]
        masks_resized = F.interpolate(
            masks_small.unsqueeze(0).float(),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) > 0.5
        masks_np = masks_resized.cpu().numpy()
    else:
        masks_np = None

    return pil_image, masks_np, num_keep


def load_ground_truth_by_prompt(image_path, data_dir, target_prompt):
    """Load ground truth annotations filtered by a specific prompt"""
    image_path = Path(image_path)
    ann_file = Path(data_dir) / "_annotations.coco.json"
    
    if not ann_file.exists():
        return [], None

    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    image_name = image_path.name
    image_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_name:
            image_info = img
            break

    if image_info is None:
        return [], None

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]

    orig_h, orig_w = image_info['height'], image_info['width']
    
    # 遍历标注，只提取与目标 Prompt 匹配的 Mask
    gt_masks = []
    for ann in annotations:
        # 如果当前标注的类别不等于目标 prompt，直接跳过
        if categories.get(ann['category_id']) != target_prompt:
            continue

        segmentation = ann.get('segmentation', None)
        if segmentation:
            try:
                if isinstance(segmentation, dict):
                    mask_np = mask_utils.decode(segmentation)
                elif isinstance(segmentation, list):
                    rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                    rle = mask_utils.merge(rles)
                    mask_np = mask_utils.decode(rle)
                else:
                    continue
                gt_masks.append(mask_np)
            except Exception as e:
                print(f"Error processing annotation: {e}")

    return gt_masks, target_prompt


def create_combined_visualization(image_results, output_path, prompt):
    """Create a single large visualization with all images for a specific prompt"""
    num_images = len(image_results)

    # Create figure: 3 columns (GT, LoRA, Base) x N rows (images)
    fig, axes = plt.subplots(num_images, 3, figsize=(18, 6 * num_images))

    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for idx, result in enumerate(image_results):
        image_name = result['image_name']
        pil_image = result['pil_image']
        gt_masks = result['gt_masks']
        lora_masks = result['lora_masks']
        lora_count = result['lora_count']
        base_masks = result['base_masks']
        base_count = result['base_count']
        prompt = result['prompt']

        # Ground Truth
        axes[idx, 0].imshow(pil_image)
        if len(gt_masks) > 0:
            overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
            for mask in gt_masks:
                overlay[mask > 0] = [0, 1, 0, 0.5]  # Green
            axes[idx, 0].imshow(overlay)
        axes[idx, 0].set_title(f'GT ({len(gt_masks)})', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        # Add image name and prompt on the left
        label_text = f"{image_name}\n[ {prompt} ]"
        axes[idx, 0].text(-0.15, 0.5, label_text,
                         transform=axes[idx, 0].transAxes,
                         fontsize=12, rotation=90,
                         verticalalignment='center',
                         horizontalalignment='right',
                         fontweight='bold')

        # LoRA predictions
        axes[idx, 1].imshow(pil_image)
        if lora_count > 0 and lora_masks is not None:
            overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
            for mask in lora_masks:
                overlay[mask] = [1, 0, 0, 0.5]  # Red
            axes[idx, 1].imshow(overlay)
        axes[idx, 1].set_title(f'LoRA ({lora_count})', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')

        # Base predictions
        axes[idx, 2].imshow(pil_image)
        if base_count > 0 and base_masks is not None:
            overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
            for mask in base_masks:
                overlay[mask] = [0, 0, 1, 0.5]  # Blue
            axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Base ({base_count})', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')

    # 修改全局标题，显示当前 prompt
    plt.suptitle(f'Model Comparison - Prompt: "{prompt}"',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout()
    # 稍微增加一下左侧边距，防止字被截断
    plt.subplots_adjust(left=0.08) 
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved visualization for prompt '{prompt}' to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare LoRA vs Base on images grouped by prompt')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory with COCO annotations')
    parser.add_argument('--config', type=str, default='configs/full_lora_config.yaml',
                        help='Path to LoRA config')
    parser.add_argument('--weights', type=str, default='outputs/sam3_lora_full/best_lora_weights.pt',
                        help='Path to LoRA weights')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for comparison images')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--resolution', type=int, default=1008, help='Input resolution')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models once
    lora_model = load_lora_model(args.config, args.weights, device)
    base_model = load_base_model(device)

    # Build prompt-to-images mapping
    print("Building prompt-to-images mapping...")
    ann_file = Path(args.data_dir) / "_annotations.coco.json"
    
    if not ann_file.exists():
        print(f"Error: COCO annotation file not found at {ann_file}")
        return

    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    annotations = coco_data['annotations']
    images = {img['id']: img for img in coco_data['images']}

    # Build mapping: prompt -> list of image info
    prompt_to_images = {}
    for ann in annotations:
        category_id = ann['category_id']
        if category_id not in categories:
            continue
            
        prompt = categories[category_id]
        image_id = ann['image_id']
        if image_id not in images:
            continue
            
        image_info = images[image_id]
        if prompt not in prompt_to_images:
            prompt_to_images[prompt] = []
        
        # Avoid duplicate images for the same prompt
        if image_info not in prompt_to_images[prompt]:
            prompt_to_images[prompt].append(image_info)

    # Filter images that exist in the image directory
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found at {image_dir}")
        return

    # Get list of actual image files in the directory
    image_files = {f.name for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}}
    
    # Filter the mapping to only include images that exist in the directory
    filtered_prompt_to_images = {}
    for prompt, image_list in prompt_to_images.items():
        filtered_images = []
        for img_info in image_list:
            if img_info['file_name'] in image_files:
                filtered_images.append(img_info)
        
        if filtered_images:  # Only include prompts that have at least one image
            filtered_prompt_to_images[prompt] = filtered_images

    print(f"Found {len(filtered_prompt_to_images)} prompts with images:")
    for prompt, img_list in filtered_prompt_to_images.items():
        print(f"  {prompt}: {len(img_list)} images")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each prompt
    for prompt, image_list in filtered_prompt_to_images.items():
        print(f"\nProcessing prompt: {prompt} ({len(image_list)} images)")
        
        results = []
        
        # Process each image for this prompt
        for img_idx, image_info in enumerate(image_list, 1):
            image_path = image_dir / image_info['file_name']
            image_name = image_info['file_name']
            
            print(f"  [{img_idx}/{len(image_list)}] Processing: {image_name}")

            # Load ground truth for this specific prompt
            gt_masks, _ = load_ground_truth_by_prompt(image_path, args.data_dir, prompt)
            print(f"    GT: {len(gt_masks)} masks")

            # Run LoRA model
            pil_image, lora_masks, lora_count = predict(
                lora_model, image_path, prompt, args.resolution, args.threshold, device
            )
            print(f"    LoRA: {lora_count} detections")

            # Run Base model
            _, base_masks, base_count = predict(
                base_model, image_path, prompt, args.resolution, args.threshold, device
            )
            print(f"    Base: {base_count} detections")

            results.append({
                'image_name': image_name,
                'pil_image': pil_image,
                'gt_masks': gt_masks,
                'lora_masks': lora_masks,
                'lora_count': lora_count,
                'base_masks': base_masks,
                'base_count': base_count,
                'prompt': prompt
            })

        # Create visualization for this prompt
        output_path = output_dir / f"comparison_{prompt}.png"
        create_combined_visualization(results, output_path, prompt)

        # Print summary for this prompt
        print(f"\nSummary for {prompt}:")
        for result in results:
            print(f"  {result['image_name']:50s} | GT:{len(result['gt_masks'])} LoRA:{result['lora_count']} Base:{result['base_count']}")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()