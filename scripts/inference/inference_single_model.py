#!/usr/bin/env python3
"""
Inference script for SAM3 models (Base or LoRA) on a directory of images
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


def create_inference_visualization(pil_image, masks, count, image_name, prompt, output_path):
    """Create visualization for a single image inference result"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Display original image
    ax.imshow(pil_image)

    # Overlay masks if any
    if count > 0 and masks is not None:
        overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
        for mask in masks:
            overlay[mask] = [1, 0, 0, 0.5]  # Red overlay for predictions
        ax.imshow(overlay)

    ax.set_title(f'Inference Result - Prompt: "{prompt}"\nDetections: {count}', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add image name at the bottom
    ax.text(0.5, -0.05, f'Image: {image_name}', 
            transform=ax.transAxes, fontsize=12, 
            horizontalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved inference result to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on images with SAM3 models')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for inference')
    parser.add_argument('--model-type', type=str, choices=['base', 'lora'], required=True, 
                        help='Type of model to use: base or lora')
    parser.add_argument('--config', type=str, default='configs/full_lora_config.yaml',
                        help='Path to LoRA config (only used if model-type is lora)')
    parser.add_argument('--weights', type=str, default='outputs/sam3_lora_full/best_lora_weights.pt',
                        help='Path to LoRA weights (only used if model-type is lora)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for inference results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--resolution', type=int, default=1008, help='Input resolution')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Using model type: {args.model_type}")
    print(f"Prompt: {args.prompt}")

    # Load the selected model
    if args.model_type == 'lora':
        model = load_lora_model(args.config, args.weights, device)
    else:
        model = load_base_model(device)

    # Get list of image files
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found at {image_dir}")
        return

    # Get list of actual image files in the directory
    image_files = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for img_idx, image_path in enumerate(image_files, 1):
        print(f"\n[{img_idx}/{len(image_files)}] Processing: {image_path.name}")

        # Run inference
        pil_image, masks, count = predict(
            model, image_path, args.prompt, args.resolution, args.threshold, device
        )
        print(f"  Detections: {count}")

        # Create visualization
        output_path = output_dir / f"inference_{args.model_type}_{image_path.stem}.png"
        create_inference_visualization(pil_image, masks, count, image_path.name, args.prompt, output_path)

    print(f"\nAll inference results saved to: {output_dir}")


if __name__ == "__main__":
    main()