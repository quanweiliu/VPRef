
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import pycocotools.mask as mask_utils  # Required for RLE mask decoding in COCO dataset
from sam3.train.masks_ops import rle_encode  # For encoding masks to RLE format
from sam3.perflib.nms import nms_masks


def patch_dynamic_rope(model):
    """
    Monkey-patch SAM3's ViT Attention to support arbitrary input resolutions.

    SAM3 pre-computes RoPE (freqs_cis) for a fixed grid size (e.g., 63×63 for 1008px at patch_size=16).
    When using a different input resolution (e.g., 512 → 32×32 grid), the shapes don't match and crash.

    This patch dynamically recomputes freqs_cis during forward if the spatial dimensions differ
    from what was pre-computed — no need to re-init the model for each resolution change.

    Call right after build_sam3_image_model(), before LoRA application:
        model = build_sam3_image_model(...)
        patch_dynamic_rope(model)

    Works with both 3D (B, L, C) and 4D (B, H, W, C) attention inputs.
    """
    from sam3.model.vitdet import Attention
    import math

    patched_count = 0
    for module in model.modules():
        if isinstance(module, Attention) and module.use_rope:
            original_forward = module.forward  # bound method

            def make_forward(m, orig_fwd):
                def dynamic_forward(x):
                    # Infer H, W from input (same logic as original Attention.forward)
                    if x.ndim == 4:
                        H, W = x.shape[1], x.shape[2]
                    else:
                        s = 1 if m.cls_token else 0
                        L = x.shape[1]
                        H = W = int(math.sqrt(L - s))

                    # Recompute freqs_cis if spatial dims changed
                    expected_len = H * W + (1 if m.cls_token else 0)
                    if m.freqs_cis.shape[0] != expected_len:
                        scale_pos = 1.0
                        if m.rope_interp:
                            scale_pos = m.rope_pt_size[0] / H
                        new_freqs = m.compute_cis(
                            end_x=H, end_y=W, scale_pos=scale_pos
                        )
                        if m.cls_token:
                            t = torch.zeros(
                                m.head_dim // 2,
                                dtype=torch.float32,
                                device=new_freqs.device,
                            )
                            cls_freqs = torch.polar(torch.ones_like(t), t)[None, :]
                            new_freqs = torch.cat([cls_freqs, new_freqs], dim=0)
                        m.freqs_cis = new_freqs.to(
                            m.freqs_cis.device, dtype=m.freqs_cis.dtype
                        )

                    return orig_fwd(x)

                return dynamic_forward

            module.forward = make_forward(module, original_forward)
            patched_count += 1

    return patched_count



# ============================================================================
# 分布式训练工具函数 (Distributed Training Utilities)
# ============================================================================


def setup_distributed():
    """
    初始化分布式训练环境。

    配置 NCCL 后端，并将当前进程绑定到指定的局部 GPU (LOCAL_RANK)。

    返回:
        int: 当前进程的局部排名 (Local Rank)。

    调用说明:
        >>> local_rank = setup_distributed()
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """
    清理分布式训练资源。

    在训练结束时销毁进程组。

    调用说明:
        >>> cleanup_distributed()
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """
    检查当前是否为主进程 (Rank 0)。

    用于控制日志打印、模型保存等仅需执行一次的操作。

    返回:
        bool: 如果是主进程或非分布式环境则返回 True，否则返回 False。
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """
    获取分布式训练的总进程数 (GPU 总数)。

    返回:
        int: 进程总数。
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    获取当前进程的全局排名 (Rank)。

    返回:
        int: 当前进程的 Rank。
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_rank0(*args, **kwargs):
    """
    仅在主进程 (Rank 0) 中打印信息。

    参数:
        *args: 打印内容位置参数。
        **kwargs: 打印内容关键字参数。
    """
    if is_main_process():
        print(*args, **kwargs)



def merge_overlapping_masks(
    binary_masks: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    iou_threshold: float = 0.3,
):
    """
    合并重叠程度较高的预测掩码。

    通过 IoU 判定重叠，将属于同一物体的多个碎裂掩码合并为一个。

    参数:
        binary_masks (torch.Tensor): 二值掩码张量，形状为 [N, H, W]。
        scores (torch.Tensor): 置信度分数，形状为 [N]。
        boxes (torch.Tensor): 边界框，形状为 [N, 4]。
        iou_threshold (float, 可选): 合并的 IoU 阈值，默认 0.3。

    返回:
        tuple: 包含 (合并后的掩码, 合并后的分数, 合并后的框)。
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute IoU
            intersection = (current_mask & binary_masks[j]).sum().item()
            union = (current_mask | binary_masks[j]).sum().item()
            iou = intersection / union if union > 0 else 0

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


# def convert_predictions_to_coco_format(
#         predictions_list,
#         image_ids,
#         resolution=288,
#         score_threshold=0.0,
#         merge_overlaps=True,
#         iou_threshold=0.3,
#         debug=False,
# ):
#     """
#     将模型预测结果转换为 COCO 格式。

#     优化策略：掩码保持在模型原生输出分辨率 (288x288)，对应的 Ground Truth 也会被降采样，
#     这样可以显著加快计算速度，无需进行大图上采样。

#     参数:
#         predictions_list (list): 模型输出的预测字典列表。
#         image_ids (list): 与预测对应的图像 ID 列表。
#         resolution (int, 可选): 掩码分辨率，默认 288。
#         score_threshold (float, 可选): 置信度阈值。
#         merge_overlaps (bool, 可选): 是否合并重叠预测，默认 True。
#         iou_threshold (float, 可选): 合并 IoU 阈值，默认 0.3。
#         debug (bool, 可选): 是否打印调试信息。

#     返回:
#         list: COCO 格式的预测字典列表。
#     """
#     coco_predictions = []
#     pred_id = 0

#     for img_id, preds in zip(image_ids, predictions_list):
#         if preds is None or len(preds.get("pred_logits", [])) == 0:
#             continue

#         # 提取预测结果
#         logits = preds["pred_logits"]  # [num_queries, 1]
#         boxes = preds["pred_boxes"]  # [num_queries, 4]
#         masks = preds["pred_masks"]  # [num_queries, H, W]

#         scores = torch.sigmoid(logits).squeeze(-1)  # [num_queries]

#         # 按置信度阈值过滤
#         valid_mask = scores > score_threshold
#         num_before = len(scores)
#         scores = scores[valid_mask]
#         boxes = boxes[valid_mask]
#         masks = masks[valid_mask]

#         if debug and img_id == image_ids[0]:  # 仅对首张图进行调试打印
#             print(
#                 f"  图像 {img_id}: {num_before} 个查询 -> 过滤后剩余 {len(scores)} 个 (阈值={score_threshold})"
#             )

#         # 转换为二值掩码
#         binary_masks = (torch.sigmoid(masks) > 0.5).cpu()

#         # 合并重叠预测，避免过度分割惩罚
#         if merge_overlaps and len(binary_masks) > 0:
#             num_before_merge = len(binary_masks)
#             binary_masks, scores, boxes = merge_overlapping_masks(
#                 binary_masks, 
#                 scores.cpu(), 
#                 boxes.cpu(), 
#                 iou_threshold=iou_threshold
#             )
#             if debug and img_id == image_ids[0]:
#                 print(
#                     f"  合并前 {num_before_merge} 个 -> 合并后 {len(binary_masks)} 个 (IoU 阈值={iou_threshold})"
#                 )

#         # 编码为 RLE (原生分辨率下非常快)
#         if len(binary_masks) > 0:
#             mask_areas = binary_masks.flatten(1).sum(1)

#             if debug and img_id == image_ids[0]:
#                 print(f"  掩码形状: {binary_masks.shape}")
#                 print(
#                     f"  面积统计: 最小={mask_areas.min():.0f}, 最大={mask_areas.max():.0f}, 平均={mask_areas.float().mean():.0f}"
#                 )

#             rles = rle_encode(binary_masks)

#             for _idx, (rle, score, box) in enumerate(
#                 zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())
#             ):
#                 # 将归一化的 CxCyWH 转换为像素坐标下的 [x, y, w, h]
#                 cx, cy, w, h = box
#                 x = (cx - w / 2) * resolution
#                 y = (cy - h / 2) * resolution
#                 w = w * resolution
#                 h = h * resolution

#                 coco_predictions.append(
#                     {
#                         "image_id": int(img_id),
#                         "category_id": 1,
#                         "segmentation": rle,
#                         "bbox": [float(x), float(y), float(w), float(h)],
#                         "score": float(score),
#                         "id": pred_id,
#                     }
#                 )
#                 pred_id += 1

#     return coco_predictions



def apply_sam3_nms(pred_logits, pred_masks, pred_boxes, prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100):
    """
    Apply SAM3's standard NMS pipeline to filter predictions.

    Args:
        pred_logits: [N, 1] logits
        pred_masks: [N, H, W] mask logits
        pred_boxes: [N, 4] boxes in normalized format
        prob_threshold: Score threshold for filtering (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: IoU threshold for NMS (default: 0.7, SAM3 uses 0.5-0.7)
        max_detections: Maximum detections to keep (default: 100)

    Returns:
        Tuple of (filtered_masks, filtered_scores, filtered_boxes)
    """
    if len(pred_logits) == 0:
        return pred_masks[:0], pred_logits[:0].squeeze(-1), pred_boxes[:0]

    # Convert logits to probabilities
    pred_probs = torch.sigmoid(pred_logits).squeeze(-1)  # [N]

    # Convert mask logits to binary masks (sigmoid + threshold)
    pred_masks_sigmoid = torch.sigmoid(pred_masks)  # [N, H, W]
    pred_masks_binary = pred_masks_sigmoid > 0.5  # [N, H, W]

    # Apply SAM3's NMS
    # nms_masks expects: pred_probs [N], pred_masks [N, H, W], prob_threshold, iou_threshold
    # Returns: keep mask [N] of booleans
    keep_mask = nms_masks(
        pred_probs=pred_probs,
        pred_masks=pred_masks_binary.float(),  # NMS expects float masks
        prob_threshold=prob_threshold,
        iou_threshold=nms_iou_threshold
    )

    # Filter predictions
    filtered_masks = pred_masks_sigmoid[keep_mask]  # Keep sigmoid masks for later
    filtered_scores = pred_probs[keep_mask]
    filtered_boxes = pred_boxes[keep_mask]

    # Top-K selection by score
    if max_detections > 0 and len(filtered_scores) > max_detections:
        top_k_scores, top_k_indices = torch.topk(filtered_scores, k=max_detections, largest=True)
        filtered_masks = filtered_masks[top_k_indices]
        filtered_scores = top_k_scores
        filtered_boxes = filtered_boxes[top_k_indices]

    return filtered_masks, filtered_scores, filtered_boxes


def convert_predictions_to_coco_format(
        predictions_list, 
        image_ids, 
        resolution=288,
        prob_threshold=0.3, 
        nms_iou_threshold=0.7, 
        max_detections=100,
        merge_cracks=False, 
        merge_iou_threshold=0.15,
        debug=False,
        ):
    """
    Convert model predictions to COCO format using SAM3's NMS pipeline.

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs
        resolution: Resolution for box scaling (default: 288)
        prob_threshold: Score threshold (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: NMS IoU threshold (default: 0.7)
        max_detections: Max detections per image (default: 100)
        merge_cracks: If True, merge overlapping segments instead of NMS suppression (default: False)
        merge_iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)
    """
    coco_predictions = []
    pred_id = 0

    if merge_cracks:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using CRACK MERGING mode: prob_threshold={prob_threshold}, merge_iou={merge_iou_threshold}, max_dets={max_detections}")
        print(f"[INFO] This will MERGE overlapping crack segments instead of suppressing them")
    else:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using SAM3 NMS: prob_threshold={prob_threshold}, nms_iou={nms_iou_threshold}, max_dets={max_detections}")

    # for img_id, preds in tqdm(zip(image_ids, predictions_list), \
    #                           total=len(predictions_list), desc="Converting predictions"):
    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # 提取预测结果
        logits = preds['pred_logits']  # [N, 1]
        boxes = preds['pred_boxes']    # [N, 4]
        masks = preds['pred_masks']    # [N, H, W]

        if merge_cracks:
            # Step 1: Filter by score threshold
            pred_probs = torch.sigmoid(logits).squeeze(-1)  # [N]
            valid_mask = pred_probs > prob_threshold
            num_before = len(pred_probs)

            filtered_scores = pred_probs[valid_mask]
            filtered_boxes = boxes[valid_mask]
            filtered_masks = masks[valid_mask]

            if len(filtered_masks) > 0:
                # Step 2: Convert masks to binary
                pred_masks_sigmoid = torch.sigmoid(filtered_masks)
                pred_masks_binary = (pred_masks_sigmoid > 0.5)

                # Step 3: MERGE overlapping crack segments
                merged_masks, merged_scores, merged_boxes = merge_overlapping_masks(
                    pred_masks_binary.cpu(),
                    filtered_scores.cpu(),
                    filtered_boxes.cpu(),
                    iou_threshold=merge_iou_threshold
                )

                # Step 4: Top-K selection by score
                if max_detections > 0 and len(merged_scores) > max_detections:
                    top_k_scores, top_k_indices = torch.topk(merged_scores, k=max_detections, largest=True)
                    merged_masks = merged_masks[top_k_indices]
                    merged_scores = top_k_scores
                    merged_boxes = merged_boxes[top_k_indices]

                # Return merged results (already binary)
                filtered_masks = merged_masks.float()  # Already binary, just convert to float
                filtered_scores = merged_scores
                filtered_boxes = merged_boxes
            else:
                filtered_masks = torch.tensor([])
                filtered_scores = torch.tensor([])
                filtered_boxes = torch.tensor([])
        else:
            # Apply SAM3's NMS pipeline (standard suppression)
            filtered_masks, filtered_scores, filtered_boxes = apply_sam3_nms(
                pred_logits=logits,
                pred_masks=masks,
                pred_boxes=boxes,
                prob_threshold=prob_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections
            )


        if debug and img_id == image_ids[0]:  # 仅对首张图进行调试打印
            print(
                f"  图像 {img_id}: {num_before} 个查询 -> 过滤后剩余 {len(filtered_scores)} 个 (阈值={prob_threshold})"
            )

        if len(filtered_masks) > 0:
            # Convert filtered masks to binary for RLE encoding
            binary_masks = (filtered_masks > 0.5).cpu()
            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, filtered_scores.cpu().tolist(), filtered_boxes.cpu().tolist())):
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """Create COCO ground truth dictionary from dataset.

    OPTIMIZATION: Downsample GT masks to match prediction resolution (288×288)
    instead of upsampling predictions to 1008×1008. Much faster!

    Args:
        dataset: SimpleSAM3Dataset instance
        image_ids: Optional list of specific image IDs to include
        mask_resolution: Resolution to downsample masks to (default: 288 to match model output)

    Returns:
        Dictionary in COCO format
    """
    
    print(f"\n[INFO] Creating COCO ground truth (downsampling to {mask_resolution}×{mask_resolution})...")
    
    coco_gt = {
        "info": {
            "description": "SAM3 LoRA Validation Dataset",
            "version": "1.0",
            "year": 2024,
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    # Scale factor for boxes (masks will be at mask_resolution, boxes scaled accordingly)
    # scale_factor = mask_resolution / dataset.resolution

    for idx in tqdm(list(indices), desc="Creating GT"):
        # Check if idx is within valid range
        if idx >= len(dataset):
            print(f"[WARNING] Skipping index {idx} - out of range (dataset length: {len(dataset)})")
            continue

        # Add image entry at mask resolution
        coco_gt["images"].append(
            {
                "id": int(idx),
                "width": mask_resolution,
                "height": mask_resolution,
                "is_instance_exhaustive": True,  # Required for cgF1 evaluation
            }
        )

        # Get datapoint
        datapoint = dataset[idx]

        # Add annotations
        for obj in datapoint.images[0].objects:
            box = obj.bbox * mask_resolution
            # YOLO 格式，左上角坐标和右下角坐标
            # x1, y1, x2, y2 = box.tolist()
            # Convert normalized CxCyWH box to COCO [x, y, width, height] at mask_resolution
            cx, cy, bw, bh = box.tolist()
            # x, y, w, h = x1, y1, x2-x1, y2-y1   # 转化成 x, y, width, height
            x, y, w, h = cx - bw / 2, cy - bh / 2, bw, bh  # 中心点坐标和宽高，且通常是归一化的

            ann = {
                "id": ann_id,
                "image_id": int(idx),
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "ignore": 0,
            }

            # Add segmentation if available - downsample to mask_resolution
            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = (
                    torch.nn.functional.interpolate(
                        mask_tensor,
                        size=(mask_resolution, mask_resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                    > 0.5
                )

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle["counts"] = rle["counts"].decode("utf-8")
                ann["segmentation"] = rle

            coco_gt["annotations"].append(ann)
            ann_id += 1

    print(f"[INFO] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")

    return coco_gt


def convert_predictions_to_coco_format_original_res(
    predictions_list,
    image_ids,
    dataset,
    model_resolution=288,
    score_threshold=0.0,
    merge_overlaps=True,
    iou_threshold=0.3,
    debug=False,
):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(
            f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)..."
        )
        if merge_overlaps:
            print(
                f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})"
            )

    # for img_id, preds in tqdm(zip(image_ids, predictions_list), \
    #                           total=len(predictions_list), desc="Converting predictions"):
    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get("pred_logits", [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds["pred_logits"]
        boxes = preds["pred_boxes"]
        masks = preds["pred_masks"]  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(
                f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})"
            )
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(
                    f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}"
                )

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        # Process on GPU then immediately move to CPU to save memory
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Free GPU memory immediately after upsampling
        del masks_sigmoid, masks_upsampled
        torch.cuda.empty_cache()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(
                    f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})"
                )

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(
                    f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}"
                )

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(
                zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())
            ):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm / 2) * orig_w
                y = (cy - h_norm / 2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    "image_id": int(img_id),
                    "category_id": 1,
                    "segmentation": rle,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(score),
                    "id": pred_id,
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(
                        f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}"
                    )

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print("\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        "info": {
            "description": "SAM3 LoRA Validation Dataset",
            "version": "1.0",
            "year": 2024,
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt["images"].append(
            {
                "id": int(idx),
                "width": orig_w,
                "height": orig_h,
                "is_instance_exhaustive": True,
            }
        ) 

        for obj in datapoint.images[0].objects:
            # YOLO 格式，左上角坐标和右下角坐标
            # x1, y1, x2, y2 = obj.bbox.tolist()
            cx, cy, bw, bh = obj.bbox.tolist()

            # Convert normalized CxCyWH box to COCO [x, y, w, h] at original size
            w = bw * orig_w
            h = bh * orig_h
            x = cx * orig_w - w / 2
            y = cy * orig_h - h / 2

            # # Convert to COCO format [x, y, w, h]
            # x = x1 * orig_w
            # y = y1 * orig_h
            # w = x2 * orig_w - x
            # h = y2 * orig_h - y

            ann = {
                "id": ann_id,
                "image_id": int(idx),
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "ignore": 0,
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = (
                    torch.nn.functional.interpolate(
                        mask_tensor,
                        size=(orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    > 0.5
                )

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle["counts"] = rle["counts"].decode("utf-8")
                ann["segmentation"] = rle

            coco_gt["annotations"].append(ann)
            ann_id += 1

    if debug:
        print(
            f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations"
        )
        if len(coco_gt["annotations"]) > 0:
            sample_gt = coco_gt["annotations"][0]
            sample_img = coco_gt["images"][0]
            print(
                f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}"
            )

    return coco_gt



def compute_miou(coco_gt_dict, coco_predictions):
    """
    Compute Mean IoU (mIoU) and Overall IoU (oIoU) for Referring Expression Segmentation.

    Args:
        coco_gt_dict: COCO ground truth dict with 'images' and 'annotations'
        coco_predictions: List of COCO prediction dicts

    Returns:
        dict with keys: miou, oiou, iou_object, iou_background, total_tp, total_fp, total_fn, total_tn
    """

    # Group GT annotations by image_id
    gt_by_image = {}
    for ann in coco_gt_dict['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)

    # Group predictions by image_id
    pred_by_image = {}
    for pred in coco_predictions:
        img_id = pred['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(pred)

    # Image dimensions lookup
    img_sizes = {img['id']: (img['height'], img['width']) for img in coco_gt_dict['images']}

    # 累加混淆矩阵
    total_tp = 0  # object pixels correctly predicted
    total_fp = 0  # background pixels predicted as object
    total_fn = 0  # object pixels predicted as background
    total_tn = 0  # background pixels correctly predicted

    # --- 关键修改 1: 增加 RES 专属的全局交集和并集计数器 ---
    total_res_intersection = 0
    total_res_union = 0

    images_processed = 0
    image_ious = []

    for img_id, (h, w) in tqdm(img_sizes.items(), desc="Computing MIoU/OIoU"):
        # Build GT binary mask (union of all object annotations)
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in gt_by_image.get(img_id, []):
            if 'segmentation' in ann:
                try:
                    decoded = mask_utils.decode(ann['segmentation'])
                    gt_mask = np.logical_or(gt_mask, decoded).astype(np.uint8)
                except Exception:
                    pass

        # Build prediction binary mask (union of all predicted objects)
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        for pred in pred_by_image.get(img_id, []):
            if 'segmentation' in pred:
                try:
                    decoded = mask_utils.decode(pred['segmentation'])
                    pred_mask = np.logical_or(pred_mask, decoded).astype(np.uint8)
                except Exception:
                    pass

        # Per-pixel metrics
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
        fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
        tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()

        total_tp += intersection
        total_fp += fp
        total_fn += fn
        total_tn += tn
        images_processed += 1

        # 循环内部 (计算单张图的 IoU 并累加用于 oIoU)
        denom = (intersection + fp + fn)
        img_iou = intersection / denom if denom > 0 else 0.0
        image_ious.append(img_iou)

        # --- 关键修改 2: 累加每张图的交集和并集，用于计算标准的 oIoU ---
        total_res_intersection += intersection
        total_res_union += denom

    # 循环结束后
    # RES 标准的 mIoU: 单张图 IoU 的均值 (也就是你之前的 mean_iou)
    res_mIoU = np.mean(image_ious)

    # --- 关键修改 3: 计算 RES 标准的 oIoU ---
    res_oIoU = total_res_intersection / total_res_union if total_res_union > 0 else 0.0

    # 你原本代码保留的语义分割指标（可用于监控或消融实验）
    iou_object = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    iou_bg = total_tn / (total_tn + total_fp + total_fn) if (total_tn + total_fp + total_fn) > 0 else 0.0

    print(f"\n[RES Evaluation] Processed {images_processed} images")
    print(f"[RES Evaluation] standard mIoU (Instance-average): {res_mIoU:.4f}")
    print(f"[RES Evaluation] standard oIoU (Dataset-cumulative): {res_oIoU:.4f}")
    print(f"[Semantic Meter] Semantic Object IoU: {iou_object:.4f}, Background IoU: {iou_bg:.4f}")

    return {
        'miou': res_mIoU,
        'oiou': res_oIoU,            # 新增返回学术标准的 oIoU
        'iou_object': iou_object,    # 对应你原本的指标
        'iou_background': iou_bg,
        'total_tp': int(total_tp),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
        'total_tn': int(total_tn),
    }

# def compute_miou(coco_gt_dict, coco_predictions):
#     """
#     Compute Mean IoU (mIoU) for binary segmentation.

#     Builds per-image binary masks from RLE annotations, then computes
#     pixel-level confusion matrix accumulated across all images.

#     For binary (crack + background):
#         crack_IoU = TP / (TP + FP + FN)
#         bg_IoU    = TN / (TN + FP + FN)
#         mIoU      = (crack_IoU + bg_IoU) / 2

#     Args:
#         coco_gt_dict: COCO ground truth dict with 'images' and 'annotations'
#         coco_predictions: List of COCO prediction dicts

#     Returns:
#         dict with keys: miou, iou_object, iou_background, total_tp, total_fp, total_fn, total_tn
#     """

#     # Group GT annotations by image_id
#     gt_by_image = {}
#     for ann in coco_gt_dict['annotations']:
#         img_id = ann['image_id']
#         if img_id not in gt_by_image:
#             gt_by_image[img_id] = []
#         gt_by_image[img_id].append(ann)

#     # Group predictions by image_id
#     pred_by_image = {}
#     for pred in coco_predictions:
#         img_id = pred['image_id']
#         if img_id not in pred_by_image:
#             pred_by_image[img_id] = []
#         pred_by_image[img_id].append(pred)

#     # Image dimensions lookup
#     img_sizes = {img['id']: (img['height'], img['width']) for img in coco_gt_dict['images']}

#     # Accumulate confusion matrix
#     total_tp = 0  # object pixels correctly predicted
#     total_fp = 0  # background pixels predicted as object
#     total_fn = 0  # object pixels predicted as background
#     total_tn = 0  # background pixels correctly predicted

#     images_processed = 0
#     image_ious = []

#     for img_id, (h, w) in tqdm(img_sizes.items(), desc="Computing MIoU"):
#         # Build GT binary mask (union of all object annotations)
#         gt_mask = np.zeros((h, w), dtype=np.uint8)
#         for ann in gt_by_image.get(img_id, []):
#             if 'segmentation' in ann:
#                 try:
#                     decoded = mask_utils.decode(ann['segmentation'])
#                     gt_mask = np.logical_or(gt_mask, decoded).astype(np.uint8)
#                 except Exception:
#                     pass

#         # Build prediction binary mask (union of all predicted objects)
#         pred_mask = np.zeros((h, w), dtype=np.uint8)
#         for pred in pred_by_image.get(img_id, []):
#             if 'segmentation' in pred:
#                 try:
#                     decoded = mask_utils.decode(pred['segmentation'])
#                     pred_mask = np.logical_or(pred_mask, decoded).astype(np.uint8)
#                 except Exception:
#                     pass

#         # Per-pixel metrics
#         intersection = np.logical_and(pred_mask, gt_mask).sum()
#         fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
#         fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
#         tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()

#         total_tp += intersection
#         total_fp += fp
#         total_fn += fn
#         total_tn += tn
#         images_processed += 1

#         # 循环内部 (Per-pixel metrics 之后)
#         denom = (intersection + fp + fn)
#         img_iou = intersection / denom if denom > 0 else 0.0
#         image_ious.append(img_iou)

#     # 循环结束后
#     mean_iou = np.mean(image_ious)

#     # Per-class IoU
#     iou_object = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
#     iou_bg = total_tn / (total_tn + total_fp + total_fn) if (total_tn + total_fp + total_fn) > 0 else 0.0
#     # miou = (iou_object + iou_bg) / 2.0

#     print(f"\n[MIoU] Processed {images_processed} images")
#     print(f"[MIoU] Confusion matrix: TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")

#     return {
#         'miou': mean_iou,
#         'iou_object': iou_object,
#         'iou_background': iou_bg,
#         'total_tp': int(total_tp),
#         'total_fp': int(total_fp),
#         'total_fn': int(total_fn),
#         'total_tn': int(total_tn),
#     }