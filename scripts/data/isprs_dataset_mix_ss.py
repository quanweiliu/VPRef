#!/usr/bin/env python3
"""
Vaihingen Referring Segmentation Dataset for SAM3 — Mixed-Version + Self-Training.

Combines two capabilities:
  1. Mixed-version text training: loads all text variants (concept/simple/standard/complex)
     aligned by mask filename, randomly samples one per epoch.
  2. Pseudo-label self-training: loads unlabeled images + predicted masks + text
     from an external source (e.g., PotsdamRef pseudo-labels) and appends them
     to the training set.

Usage:
    # Mixed-version only
    ds = ISPRSRefDataset_MixSS(data_dir=..., split="train", variant="mix")

    # Mixed-version + pseudo-labels
    ds = ISPRSRefDataset_MixSS(
        data_dir=...,
        variant="mix",
        unlabeled_dir=...,
        unlabeled_mask_path=...,
        unlabeled_text_path=...,
    )

    # Single-variant + pseudo-labels (backward compatible with _ss.py)
    ds = ISPRSRefDataset_MixSS(
        data_dir=..., variant="standard",
        unlabeled_dir=..., unlabeled_mask_path=..., unlabeled_text_path=...,
    )
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
CLASS_TO_SUFFIX = {
    "Impervious_surface": "0",
    "Building": "1",
    "Car": "2",
    "Tree": "3",
    "LowVeg": "4",
}
SUFFIX_TO_CLASS = {v: k for k, v in CLASS_TO_SUFFIX.items()}


class ISPRSRefDataset_MixSS(Dataset):
    """
    Vaihingen Referring Segmentation Dataset — Mixed-Version + Self-Training.

    Args:
        data_dir: Root directory of the labeled dataset
                  (e.g., /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/)
        split: 'train', 'valid', or 'test'
        variant: 'mix' for mixed-version training, or 'concept'/'simple'/'standard'/'complex'
                 for single-variant training.
        unlabeled_dir: Root directory of unlabeled images
                       (e.g., /home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/)
        unlabeled_mask_path: Directory of pseudo-label masks (.tif files)
        unlabeled_text_path: Annotation file mapping mask_name → text
        resolution: Target image resolution (default: 1008 for SAM3)
        augment: Whether to apply spatial + color augmentation
    """

    MIX_VERSIONS = ['concept', 'simple', 'standard', 'complex']

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        variant: str = "standard",
        resolution: int = 1008,
        augment: bool = True,
        unlabeled_dir: str = None,
        unlabeled_mask_path: str = None,
        unlabeled_text_path: str = None,
    ):
        self.data_dir = Path(data_dir)
        self.unlabeled_dir = Path(unlabeled_dir) if unlabeled_dir is not None else None
        self.resolution = resolution
        self.variant = variant
        self.augment = augment and (split == "train")

        # Map split names (SAM3 uses 'valid', VaiRef files use 'val')
        split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
        vai_split = split_map.get(split, split)

        # ============================================================
        # Part 1 — Load labeled data
        # ============================================================
        if variant == "mix":
            # ── Mixed-version: load all text variants, align by mask filename ──
            # version_sentences:  mask_name -> [sent_concept, sent_simple, sent_standard, sent_complex]
            # sample_meta:         mask_name -> (img_path, mask_path)
            version_sentences = {}
            sample_meta = {}

            for vi, ver in enumerate(self.MIX_VERSIONS):
                setfile = f"output_phrase_{vai_split}_{ver}.txt"
                setfile_path = self.data_dir / setfile

                if not setfile_path.exists():
                    raise FileNotFoundError(
                        f"Annotation file not found: {setfile_path}\n"
                        f"Available: {list(self.data_dir.glob('output_phrase_*.txt'))}")

                with open(setfile_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(" ")
                        mask_name = parts[0]  # e.g., "train_528_0.tif"
                        sentence = " ".join(parts[1:]).strip()

                        if mask_name not in version_sentences:
                            version_sentences[mask_name] = [''] * len(self.MIX_VERSIONS)
                            # Parse image + mask paths from mask_name
                            name_parts = mask_name.split("_")
                            image_name = name_parts[0] + "_" + name_parts[1]
                            class_idx = name_parts[2].split(".")[0]
                            seg_label_name = SUFFIX_TO_CLASS.get(class_idx, "object")

                            img_path = self.data_dir / "images" / f"{image_name}.tif"
                            mask_path = (
                                self.data_dir / "binary_masks" / seg_label_name / mask_name
                            )
                            sample_meta[mask_name] = (str(img_path), str(mask_path))

                        version_sentences[mask_name][vi] = sentence

            # Labeled samples: (img_path, mask_path, [list_of_sentences])
            self.labeled_samples = []
            for mask_name in sorted(sample_meta.keys()):
                img_path, mask_path = sample_meta[mask_name]
                self.labeled_samples.append(
                    (img_path, mask_path, version_sentences[mask_name])
                )

            print(
                f"  [Labeled] {len(self.labeled_samples)} samples"
                f" (variant=mix, {len(self.MIX_VERSIONS)} versions/sample)"
            )

        else:
            # ── Single-variant loading ──
            setfile = f"output_phrase_{vai_split}_{variant}.txt"
            setfile_path = self.data_dir / setfile

            if not setfile_path.exists():
                raise FileNotFoundError(
                    f"Annotation file not found: {setfile_path}\n"
                    f"Available: {list(self.data_dir.glob('output_phrase_*.txt'))}")

            self.labeled_samples = []  # (img_path, mask_path, single_sentence)

            with open(setfile_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ")
                    filename = parts[0]  # e.g., "train_528_0.tif"
                    sentence = " ".join(parts[1:])

                    # Parse: "train_528_0.tif" → image="train_528", class_idx="0"
                    name_parts = filename.split("_")
                    image_name = name_parts[0] + "_" + name_parts[1]
                    class_idx = name_parts[2].split(".")[0]
                    seg_label_name = SUFFIX_TO_CLASS.get(class_idx, "object")

                    img_path = self.data_dir / "images" / f"{image_name}.tif"
                    mask_path = (
                        self.data_dir / "binary_masks" / seg_label_name / filename
                    )

                    self.labeled_samples.append(
                        (str(img_path), str(mask_path), sentence)
                    )

            print(
                f"  [Labeled] {len(self.labeled_samples)} samples"
                f" (variant={variant})"
            )

        # ============================================================
        # Part 2 — Load pseudo-labeled data (self-training)
        # ============================================================
        self.pseudo_samples = []

        if unlabeled_mask_path is not None:
            # Build text lookup from annotation file
            text_lookup = {}
            with open(unlabeled_text_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ")
                    fname = parts[0]          # e.g., "val_0_0.tif"
                    sentence = " ".join(parts[1:])
                    text_lookup[fname] = sentence

            unlabeled_mask_path = Path(unlabeled_mask_path)
            pseudo_count = 0

            for mask_file in sorted(unlabeled_mask_path.glob("*.tif")):
                mask_filename = mask_file.name  # e.g., "val_0_0.tif"

                # (1) Look up query text from annotation file
                query_text = text_lookup.get(mask_filename)
                if query_text is None:
                    print(f"  [SKIP] {mask_filename}: not found in annotation file")
                    continue

                # (2) Derive image filename: "val_535_0.tif" → "val_535.tif"
                name_parts = mask_filename.split("_")
                image_name = "_".join(name_parts[:-1])  # "val_535"

                # Image lives in unlabeled_dir/images/
                img_path = self.unlabeled_dir / "images" / f"{image_name}.tif"
                if not img_path.exists():
                    print(f"  [SKIP] {mask_filename}: image {img_path} not found")
                    continue

                # Pseudo mask IS this file itself
                pseudo_mask_path = str(mask_file)

                self.pseudo_samples.append(
                    (str(img_path), pseudo_mask_path, query_text)
                )
                pseudo_count += 1

            print(
                f"  [Pseudo ] {pseudo_count} pseudo-labeled samples"
                f" from {unlabeled_mask_path}"
            )

        # ============================================================
        # Combined samples
        # ============================================================
        self.samples = self.labeled_samples + self.pseudo_samples

        print(
            f"VaiRefDataset_MixSS [{split}] total: {len(self.samples)} samples"
            f" (labeled={len(self.labeled_samples)}, pseudo={len(self.pseudo_samples)})"
        )

        # ============================================================
        # Transforms
        # ============================================================
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
                    v2.RandomRotation(degrees=(90, 90), expand=False),
                ], p=0.25),
                v2.RandomApply([
                    v2.RandomRotation(degrees=(180, 180), expand=False),
                ], p=0.25),
                v2.RandomApply([
                    v2.RandomRotation(degrees=(270, 270), expand=False),
                ], p=0.25),
            ])
            self.aug_color = v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        else:
            self.aug_geometric = None
            self.aug_color = None

    # ================================================================
    # Dataset interface
    # ================================================================

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Datapoint:
        img_path, mask_path, query_text = self.samples[idx]

        # ── Mixed-version: randomly sample one text variant per sample per epoch ──
        #     Labeled+mix samples store a list; pseudo/single-variant store a string.
        if isinstance(query_text, list):
            query_text = query_text[np.random.randint(0, len(query_text))]

        # ── Load image and mask as PIL ──
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        mask_pil = PILImage.open(mask_path).convert("L")
        mask_np = np.array(mask_pil, dtype=np.uint8)
        binary_mask = (mask_np > 50).astype(np.float32)
        mask_pil_binary = PILImage.fromarray((binary_mask * 255).astype(np.uint8))

        # ── Apply shared spatial augmentation (same transform for image + mask) ──
        if self.aug_geometric is not None:
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
            image_id=0,
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
    data_root = "/home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/"

    # Test 1: Mixed-version only
    print("=" * 60)
    print("Test 1: Mixed-version only")
    ds = ISPRSRefDataset_MixSS(data_dir=data_root, split="train", variant="mix")
    print(f"Total samples: {len(ds)}")
    for dp in ds:
        q = dp.find_queries[0]
        obj = dp.images[0].objects[0]
        print(f"  Query: {q.query_text[:80]}...")
        print(f"  Mask sum: {obj.segment.sum().item():.0f} px")
        print(f"  Bbox: {obj.bbox.tolist()}")
        break

    # Test 2: Mixed-version + pseudo-labels
    print("=" * 60)
    print("Test 2: Mixed-version + pseudo-labels")
    ds = ISPRSRefDataset_MixSS(
        data_dir=data_root,
        unlabeled_dir="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/",
        unlabeled_mask_path="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/unlabeled_valid/",
        unlabeled_text_path="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/output_phrase_val_concept.txt",
        split="train",
        variant="mix",
    )
    print(f"Total samples: {len(ds)}")
    for dp in ds:
        q = dp.find_queries[0]
        obj = dp.images[0].objects[0]
        print(f"  Query: {q.query_text[:80]}...")
        print(f"  Mask sum: {obj.segment.sum().item():.0f} px")
        break

    # Test 3: Single-variant + pseudo-labels (backward compatible)
    print("=" * 60)
    print("Test 3: Single-variant + pseudo-labels")
    ds = ISPRSRefDataset_MixSS(
        data_dir=data_root,
        unlabeled_dir="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/",
        unlabeled_mask_path="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/unlabeled_valid/",
        unlabeled_text_path="/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/output_phrase_val_concept.txt",
        split="train",
        variant="concept",
    )
    print(f"Total samples: {len(ds)}")
    for dp in ds:
        q = dp.find_queries[0]
        obj = dp.images[0].objects[0]
        print(f"  Query: {q.query_text[:80]}...")
        print(f"  Mask sum: {obj.segment.sum().item():.0f} px")
        break

    print("\n✅ All tests passed!")
