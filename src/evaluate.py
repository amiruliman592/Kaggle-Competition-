"""
evaluate.py — Local evaluation: compute PQ, Dice, and IoU on the validation set.

This mirrors the competition's scoring methodology so you can iterate
locally without burning Kaggle submissions.

Usage:
  python src/evaluate.py
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from config import CFG


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection-over-Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Dice score between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    if total == 0:
        return 0.0
    return 2.0 * intersection / total


def compute_panoptic_quality(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) for a single image.
    
    PQ = SQ × RQ
    where:
      SQ = avg IoU over true positives
      RQ = TP / (TP + 0.5*FP + 0.5*FN)
    
    A prediction matches a ground truth if IoU > iou_threshold.
    
    Args:
        gt_masks: List of ground-truth binary masks
        pred_masks: List of predicted binary masks
        iou_threshold: IoU threshold for matching (default 0.5)
    
    Returns:
        Dict with keys: PQ, SQ, RQ, TP, FP, FN
    """
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return {"PQ": 1.0, "SQ": 1.0, "RQ": 1.0, "TP": 0, "FP": 0, "FN": 0}
    
    if len(gt_masks) == 0:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "TP": 0, "FP": len(pred_masks), "FN": 0}
    
    if len(pred_masks) == 0:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "TP": 0, "FP": 0, "FN": len(gt_masks)}
    
    # Compute IoU matrix (gt × pred)
    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            iou_matrix[i, j] = compute_iou(gt, pred)
    
    # Greedy matching: match GT to pred with highest IoU, above threshold
    matched_gt = set()
    matched_pred = set()
    tp_ious = []
    
    while True:
        # Find the best unmatched pair
        best_iou = -1
        best_i, best_j = -1, -1
        
        for i in range(len(gt_masks)):
            if i in matched_gt:
                continue
            for j in range(len(pred_masks)):
                if j in matched_pred:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_i, best_j = i, j
        
        if best_iou < iou_threshold:
            break
        
        matched_gt.add(best_i)
        matched_pred.add(best_j)
        tp_ious.append(best_iou)
    
    tp = len(tp_ious)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    
    sq = np.mean(tp_ious) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq
    
    return {"PQ": pq, "SQ": sq, "RQ": rq, "TP": tp, "FP": fp, "FN": fn}


def evaluate_val_set(
    predictions: Dict[str, List[np.ndarray]],
    coco_gt: COCO,
    val_img_ids: List,
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth on the validation set.
    
    Args:
        predictions: Dict mapping image_id -> list of predicted masks
        coco_gt: COCO API object with ground truth
        val_img_ids: List of image IDs in the validation set
    
    Returns:
        Dict with mean PQ, SQ, RQ, Dice scores
    """
    all_pq = []
    all_sq = []
    all_rq = []
    all_dice = []
    all_iou = []
    
    for img_id in val_img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        file_stem = img_info["file_name"].replace(".jpeg", "")
        
        # Get ground truth masks
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_masks = [coco_gt.annToMask(ann) for ann in anns]
        
        # Get predictions
        pred_masks = predictions.get(file_stem, [])
        
        # Compute PQ
        pq_result = compute_panoptic_quality(gt_masks, pred_masks)
        all_pq.append(pq_result["PQ"])
        all_sq.append(pq_result["SQ"])
        all_rq.append(pq_result["RQ"])
        
        # Compute per-pair Dice/IoU for matched pairs
        for gt_m in gt_masks:
            best_dice = 0
            best_iou = 0
            for pred_m in pred_masks:
                d = compute_dice(gt_m, pred_m)
                u = compute_iou(gt_m, pred_m)
                best_dice = max(best_dice, d)
                best_iou = max(best_iou, u)
            all_dice.append(best_dice)
            all_iou.append(best_iou)
    
    results = {
        "mean_PQ": np.mean(all_pq) if all_pq else 0.0,
        "mean_SQ": np.mean(all_sq) if all_sq else 0.0,
        "mean_RQ": np.mean(all_rq) if all_rq else 0.0,
        "mean_Dice": np.mean(all_dice) if all_dice else 0.0,
        "mean_IoU": np.mean(all_iou) if all_iou else 0.0,
        "num_images": len(val_img_ids),
    }
    
    print("\n" + "=" * 50)
    print("  Validation Results")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:>15s}: {v:.4f}")
        else:
            print(f"  {k:>15s}: {v}")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    print("Run this module through the training pipeline for val evaluation.")
    print("See train.py for the full workflow.")
