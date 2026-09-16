#!/usr/bin/env python3
"""
Vaihingen Referring Segmentation Dataset for SAM3.

Wraps the VaihingenRef dataset (ISPRS Vaihingen referring segmentation)
into SAM3's Datapoint format for training with LoRA.

Each sample = (RGB image, binary mask covering ALL instances of a class,
               natural language description).

Images are 480×480. Masks are merged (no instance distinction).
Bounding boxes are auto-computed from the merged mask.
"""

import sys
sys.path.append("/home/icclab/Documents/lqw/sam3_finetune_lora")

from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms import v2

from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)

# Class name ↔ suffix mapping (from ISPRS_VaiRef.py)
# CLASS_TO_SUFFIX = {
#     "Impervious_surface": "0",
#     "Building": "1",
#     "Car": "2",
#     "Tree": "3",
#     "LowVeg": "4",
# }
# SUFFIX_TO_CLASS = {v: k for k, v in CLASS_TO_SUFFIX.items()}


class RefSegRSDataset(Dataset):
    """
    Referring Segmentation Dataset for SAM3.

    Reads the same annotation files as ISPRS_VaiRef.py but outputs
    SAM3's Datapoint format directly. Bounding boxes are computed
    automatically from the merged binary masks.

    Args:
        data_dir: Root directory of RefSegRS dataset
                  (e.g., /home/icclab/Documents/lqw/DatasetMMF/RefSegRS/)
        split: 'train', 'valid', or 'test'
        variant: None for this dataset (placeholder for future variants)
        resolution: Target image resolution (default: 1008 for SAM3)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        resolution: int = 1008,
        augment: bool = True,
        variant=None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.variant = variant
        self.augment = augment and (split == "train")  # only augment training set

        # Map split names (SAM3 uses 'valid', VaiRef files use 'val')
        split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
        vai_split = split_map.get(split, split)

        # Read annotation file
        setfile = f"output_phrase_{vai_split}.txt"
        setfile_path = self.data_dir / setfile

        if not setfile_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {setfile_path}\n"
                f"Available: {list(self.data_dir.glob('output_phrase_*.txt'))}"
            )

        self.samples = []  # List of (img_path, mask_path, query_text)

        with open(setfile_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ")
                filename = parts[0]  # e.g., "4270"
                sentence = " ".join(parts[1:])

                # Parse: "train_528_0.tif" → image="train_528", class_idx="0"
                # name_parts = filename.split("_")
                # image_name = name_parts[0] + "_" + name_parts[1]
                # class_idx = name_parts[2].split(".")[0]
                # seg_label_name = SUFFIX_TO_CLASS.get(class_idx, "object")

                img_path = self.data_dir / "images" / f"{filename}.tif"
                mask_path = (self.data_dir / "masks" / f"{filename}.tif")

                self.samples.append((str(img_path), str(mask_path), sentence))
                # print("samples", self.samples[-1])  # Debug: print the last sample added
                # print("length samples", len(self.samples))  # Debug: print total number of samples after each addition

        print(
            f"RefSegRS [{split}] loaded: {len(self.samples)} samples"
            f" (variant={variant})"
        )

        # Image transform: normalize to [-1, 1] (same as SAM3 COCO dataset)
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Spatial augmentation for training (applied to PIL image + mask jointly)
        if self.augment:
            self.aug_geometric = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                # Random 90°-multiple rotation (preserves aerial geometry)
                v2.RandomApply([
                    v2.RandomRotation(degrees=(90, 90), expand=False),   # 90°
                ], p=0.25),
                v2.RandomApply([
                    v2.RandomRotation(degrees=(180, 180), expand=False), # 180°
                ], p=0.25),
                v2.RandomApply([
                    v2.RandomRotation(degrees=(270, 270), expand=False), # 270°
                ], p=0.25),
            ])
            self.aug_color = v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        else:
            self.aug_geometric = None
            self.aug_color = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Datapoint:
        img_path, mask_path, query_text = self.samples[idx]

        # ── Load image and mask as PIL ──
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        mask_pil = PILImage.open(mask_path).convert("L")
        mask_np = np.array(mask_pil, dtype=np.uint8)
        binary_mask = (mask_np > 50).astype(np.float32)
        mask_pil_binary = PILImage.fromarray((binary_mask * 255).astype(np.uint8))

        # ── Apply shared spatial augmentation (same transform for image + mask) ──
        if self.aug_geometric is not None:
            # print(f"Applying geometric augmentation to sample (image + mask)")
            # Fix random seed so image and mask get the same flip/rotate
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed)
            pil_image = self.aug_geometric(pil_image)
            torch.manual_seed(seed)
            mask_pil_binary = self.aug_geometric(mask_pil_binary)

        # ── Color augmentation (image only) ──
        if self.aug_color is not None:
            # print(f"Applying color augmentation to sample (image only)")
            pil_image = self.aug_color(pil_image)

        # ── Resize image ──
        pil_image_resized = pil_image.resize(
            (self.resolution, self.resolution), PILImage.BILINEAR
        )
        image_tensor = self.transform(pil_image_resized)

        # ── Resize mask to model resolution ──
        mask_pil_resized = mask_pil_binary.resize(
            (self.resolution, self.resolution), PILImage.NEAREST
        )
        mask_np_resized = np.array(mask_pil_resized, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_np_resized > 127)  # [H, W] bool

        # ── Compute bounding box from mask ──
        bbox_tensor, area = self._compute_bbox_from_mask(mask_tensor)

        # ── Create SAM3 Datapoint ──
        obj = Object(
            bbox=bbox_tensor,
            area=area,
            object_id=0,
            segment=mask_tensor,
        )

        image_obj = Image(
            data=image_tensor,
            objects=[obj],
            size=(self.resolution, self.resolution),
        )

        query = FindQueryLoaded(
            query_text=query_text,
            image_id=0,  # Index into self.images list (always 0 — one image per Datapoint)
            object_ids_output=[0],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=idx,
                original_image_id=idx,
                original_category_id=0,
                original_size=(orig_h, orig_w),
                object_id=0,
                frame_index=0,
            ),
        )

        return Datapoint(
            find_queries=[query], images=[image_obj], raw_images=[pil_image]
        )

    @staticmethod
    def _compute_bbox_from_mask(mask: torch.Tensor):
        """
        Compute normalized CxCyWH bounding box from a binary mask.

        Args:
            mask: Boolean tensor [H, W]

        Returns:
            bbox: Tensor [cx, cy, w, h] normalized to [0, 1]
            area: Float area value (normalized to [0, 1])
        """
        H, W = mask.shape

        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)

        if not rows.any() or not cols.any():
            # Empty mask — zero-area box at center
            return torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32), 0.0

        y1 = torch.where(rows)[0][0].item()
        y2 = torch.where(rows)[0][-1].item()
        x1 = torch.where(cols)[0][0].item()
        x2 = torch.where(cols)[0][-1].item()

        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H

        return torch.tensor([cx, cy, bw, bh], dtype=torch.float32), float(bw * bh)


# ============================================================================
# Quick test
# ============================================================================
if __name__ == "__main__":
    data_root = "/home/icclab/Documents/lqw/DatasetMMF/RefSegRS/"

    ds = RefSegRSDataset(data_dir=data_root, split="train")
    print(f"Total samples: {len(ds)}")

    for input_batch in ds:
        img = input_batch.images[0]
        q = input_batch.find_queries[0]
        obj = img.objects[0]
        # print(f"\n--- Sample {i} ---")
        # print(f"  Image tensor: {img.data.shape}")
        # print(f"  Query: {q.query_text[:60]}...")
        # print(f"  Mask: {obj.segment.shape}, sum={obj.segment.sum().item():.0f} px")
        # print(f"  Bbox (CxCyWH): {obj.bbox.tolist()}")
        # print(f"  Original size: {q.inference_metadata.original_size}")
        break


    # for i in range(3):
    #     dp = ds[i]
    #     img = dp.images[0]
    #     q = dp.find_queries[0]
    #     obj = img.objects[0]
    #     print(f"\n--- Sample {i} ---")
    #     print(f"  Image tensor: {img.data.shape}")
    #     print(f"  Query: {q.query_text[:60]}...")
    #     print(f"  Mask: {obj.segment.shape}, sum={obj.segment.sum().item():.0f} px")
    #     print(f"  Bbox (CxCyWH): {obj.bbox.tolist()}")
    #     print(f"  Original size: {q.inference_metadata.original_size}")

    print("\n✅ RefSegRS working correctly!")
