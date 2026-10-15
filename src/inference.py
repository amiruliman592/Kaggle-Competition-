"""
inference.py — Run Mask R-CNN inference on test images and generate submission CSV.

Usage:
  python src/inference.py

This script:
  1. Loads the trained model checkpoint
  2. Runs inference on all 180 test images
  3. Converts predicted masks to RLE format (pycocotools)
  4. Outputs a submission.csv ready for Kaggle upload
"""

import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pycocotools import mask as coco_mask

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from config import CFG


def mask_to_rle(binary_mask: np.ndarray) -> str:
    """
    Convert a binary mask (H×W, uint8) to COCO RLE counts string.
    
    The competition expects only the RLE Counts string, not the Size.
    Size is fixed at 2048×2048 for all images.
    """
    # Ensure correct dtype and order
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = coco_mask.encode(mask)
    return rle["counts"].decode("utf-8")


def resize_mask_to_original(mask: np.ndarray, original_size: tuple = (2048, 2048)) -> np.ndarray:
    """
    Resize a predicted mask back to the original image resolution.
    
    Detectron2 may output masks at a different resolution than the
    native 2048×2048. We need to resize before RLE encoding.
    """
    if mask.shape == original_size:
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized


def postprocess_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """
    Post-process a single predicted mask:
      - Morphological closing to fill small holes
      - Remove masks that are too small
    
    Returns the cleaned mask, or an empty mask if below min_area.
    """
    # Morphological closing (fill small gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Check area
    if mask.sum() < min_area:
        return np.zeros_like(mask)
    
    return mask


def build_predictor(model_weights: str = None) -> DefaultPredictor:
    """
    Build a Detectron2 predictor for inference.
    
    Args:
        model_weights: Path to model weights file. 
                       Defaults to output/detectron2/model_final.pth
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG.MODEL_CONFIG))
    
    # Model settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = CFG.NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CFG.SCORE_THRESHOLD
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = CFG.NMS_THRESHOLD
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = CFG.ANCHOR_SIZES
    
    # Use full resolution for inference (better accuracy)
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    
    # Load weights
    if model_weights is None:
        model_weights = str(CFG.OUTPUT_DIR / "detectron2" / "model_final.pth")
    cfg.MODEL.WEIGHTS = model_weights
    
    print(f"[Inference] Loading model from: {model_weights}")
    predictor = DefaultPredictor(cfg)
    return predictor


def run_inference(predictor, test_img_dir: Path) -> Dict[str, List[np.ndarray]]:
    """
    Run inference on all test images.
    
    Returns:
        Dict mapping image_id -> list of binary masks (each 2048×2048)
    """
    test_images = sorted(test_img_dir.glob("*.jpeg"))
    print(f"[Inference] Processing {len(test_images)} test images...")
    
    predictions = {}
    
    for img_path in tqdm(test_images, desc="Inference"):
        img = cv2.imread(str(img_path))
        
        # Extract image ID from filename (e.g., "20150125172714Mh" from "20150125172714Mh.jpeg")
        img_id = img_path.stem
        
        # Run prediction
        with torch.no_grad():
            outputs = predictor(img)
        
        instances = outputs["instances"].to("cpu")
        
        masks = []
        if len(instances) > 0:
            pred_masks = instances.pred_masks.numpy()  # (N, H, W) boolean
            pred_scores = instances.scores.numpy()
            
            # Sort by confidence (highest first)
            sorted_idx = np.argsort(-pred_scores)
            
            for idx in sorted_idx:
                mask = pred_masks[idx].astype(np.uint8)
                
                # Resize to original 2048×2048
                mask = resize_mask_to_original(mask, CFG.IMAGE_SIZE)
                
                # Post-process
                mask = postprocess_mask(mask, min_area=CFG.MIN_AREA)
                
                if mask.sum() > 0:
                    masks.append(mask)
        
        predictions[img_id] = masks
    
    # Summary
    total_filaments = sum(len(m) for m in predictions.values())
    avg_per_image = total_filaments / len(predictions) if predictions else 0
    print(f"\n[Inference] Predictions summary:")
    print(f"  Total images:    {len(predictions)}")
    print(f"  Total filaments: {total_filaments}")
    print(f"  Avg per image:   {avg_per_image:.1f}")
    
    return predictions


def build_submission_csv(predictions: Dict[str, List[np.ndarray]], output_path: str = None):
    """
    Convert predictions to a submission CSV file.
    
    Format:
      filament_id         | segmentation_rle
      20150125172714Mh_1  | f8uSDds...VQNC
      20150125172714Mh_2  | KHT%$HD...9>km
    """
    if output_path is None:
        output_path = str(CFG.OUTPUT_DIR / CFG.SUBMISSION_FILE)
    
    rows = []
    for img_id, masks in sorted(predictions.items()):
        for i, mask in enumerate(masks, start=1):
            rle_str = mask_to_rle(mask)
            rows.append({
                "filament_id": f"{img_id}_{i}",
                "segmentation_rle": rle_str,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n[Submission] Saved {len(df)} rows to {output_path}")
    print(f"[Submission] Images with predictions: {len(predictions)}")
    
    # Sanity checks
    if len(df) == 0:
        print("[Submission] WARNING: No predictions generated! Check model/threshold.")
    
    return df


def main():
    """Run the full inference pipeline."""
    print("=" * 60)
    print("  Solar Filament Segmentation — Inference & Submission")
    print("=" * 60)
    
    # Build predictor
    predictor = build_predictor()
    
    # Run inference
    predictions = run_inference(predictor, CFG.TEST_IMG_DIR)
    
    # Generate submission CSV
    df = build_submission_csv(predictions)
    
    print("\n[Done] Submission file ready for Kaggle upload!")


if __name__ == "__main__":
    main()
