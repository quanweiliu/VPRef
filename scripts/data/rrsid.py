#!/usr/bin/env python3
"""
RRSISD Referring Segmentation Dataset for SAM3.

Wraps the RRSISD dataset (remote sensing referring image segmentation)
into SAM3's Datapoint format for training with LoRA.

Each sample = (RGB image, binary mask of the referred object,
               natural language description).

Unlike RefSegRS/ISPRS (which use plain text annotation files), RRSISD
stores annotations in COCO style (``instances.json`` + ``refs(unc).p``).
We therefore load image / mask / text through the bundled REFER API and
then convert to SAM3's Datapoint format, exactly like RefSegRSDataset.
Bounding boxes are auto-computed from the binary masks.
"""

import sys
sys.path.append("/home/icclab/Documents/lqw/sam3_finetune_lora")
sys.path.append("/home/icclab/Documents/lqw/sam3_finetune_lora/scripts")

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
from data.refer import REFER


class RRSISDDataset(Dataset):
    """
    RRSISD Referring Segmentation Dataset wrapped for SAM3 training.

    Loads the same annotations as the original ReferSeg ``ReferDataset``
    (via the REFER API) but outputs SAM3's Datapoint format directly.
    Bounding boxes are computed automatically from the binary masks.

    Args:
        data_dir: Root directory of the RRSISD dataset
                  (e.g., /home/icclab/Documents/lqw/DatasetMMF/RRSISD/)
        split: 'train', 'valid', or 'test'
        resolution: Target image resolution (default: 480, matching source images)
        augment: Whether to apply spatial/color augmentation (train only)
        variant: Placeholder (kept for a uniform interface with other datasets)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        resolution: int = 480,
        augment: bool = True,
        variant=None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.variant = variant
        self.augment = augment and (split == "train")  # only augment training set

        # Map split names (SAM3 uses 'valid', RRSISD refs use 'val')
        split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
        ref_split = split_map.get(split, split)

        # Load RRSISD via the REFER API (COCO-style annotations + referring expressions)
        self.refer = REFER(str(self.data_dir), dataset="rrsisd", splitBy="unc")

        ref_ids = self.refer.getRefIds(split=ref_split)

        # List of (img_path, ref, query_text)
        self.samples = []
        for ref_id in ref_ids:
            ref = self.refer.Refs[ref_id]
            img_path = str(Path(self.refer.IMAGE_DIR) / ref["file_name"])
            # Each ref carries exactly one sentence in RRSISD.
            query_text = ref["sentences"][0]["raw"]
            self.samples.append((img_path, ref, query_text))

        print(
            f"RRSISD [{split}] loaded: {len(self.samples)} samples"
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
        img_path, ref, query_text = self.samples[idx]

        # ── Load image as PIL ──
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # ── Load mask from REFER (RLE-decoded binary mask) ──
        mask_np = self.refer.getMask(ref)["mask"]  # uint8 [H, W] with 0/1
        binary_mask = (mask_np > 0.5).astype(np.float32)
        mask_pil_binary = PILImage.fromarray((binary_mask * 255).astype(np.uint8))

        # ── Apply shared spatial augmentation (same transform for image + mask) ──
        if self.aug_geometric is not None:
            # Fix random seed so image and mask get the same flip/rotate
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed)
            pil_image = self.aug_geometric(pil_image)
            torch.manual_seed(seed)
            mask_pil_binary = self.aug_geometric(mask_pil_binary)

        # ── Color augmentation (image only) ──
        if self.aug_color is not None:
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
    data_root = "/home/icclab/Documents/lqw/DatasetMMF/RRSISD/"

    ds = RRSISDDataset(data_dir=data_root, split="train", resolution=480)
    print(f"Total samples: {len(ds)}")

    for input_batch in ds:
        img = input_batch.images[0]
        q = input_batch.find_queries[0]
        obj = img.objects[0]
        print(f"\n--- Sample ---")
        print(f"  Image tensor: {img.data.shape}")
        print(f"  Query: {q.query_text}")
        print(f"  Mask: {obj.segment.shape}, sum={obj.segment.sum().item():.0f} px")
        print(f"  Bbox (CxCyWH): {obj.bbox.tolist()}")
        print(f"  Original size: {q.inference_metadata.original_size}")
        break

    print("\n✅ RRSISDDataset working correctly!")
