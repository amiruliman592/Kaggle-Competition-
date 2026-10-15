"""
dataset.py — Dataset preparation & registration for Detectron2.

Handles:
  1. Loading COCO JSON and registering train/val splits with Detectron2
  2. Solar disk masking (optional preprocessing)
  3. Train/val splitting

Usage:
  from dataset import register_datasets
  register_datasets()
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO

from config import CFG


def get_train_val_split(coco: COCO, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List, List]:
    """
    Split image IDs into train and val sets.
    
    Uses a simple random split. For better generalization, consider
    stratifying by year or filament count.
    
    Returns:
        (train_img_ids, val_img_ids)
    """
    random.seed(seed)
    img_ids = list(coco.getImgIds())
    random.shuffle(img_ids)
    
    split_idx = int(len(img_ids) * (1 - val_ratio))
    train_ids = img_ids[:split_idx]
    val_ids = img_ids[split_idx:]
    
    print(f"[Dataset] Split: {len(train_ids)} train, {len(val_ids)} val")
    return train_ids, val_ids


def coco_to_detectron2_dicts(
    coco: COCO,
    img_ids: List,
    img_dir: Path,
    single_class: bool = True,
) -> List[Dict]:
    """
    Convert COCO annotations to Detectron2's standard dataset dict format.
    
    Args:
        coco: COCO API object
        img_ids: List of image IDs to include
        img_dir: Path to image directory
        single_class: If True, map all categories to class 0 (binary filament/not)
    
    Returns:
        List of dataset dicts compatible with Detectron2
    """
    dataset_dicts = []
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = str(img_dir / img_info["file_name"])
        
        if not os.path.exists(img_path):
            print(f"[Dataset] Warning: Image not found: {img_path}")
            continue
        
        record = {
            "file_name": img_path,
            "image_id": img_id,
            "height": img_info["height"],
            "width": img_info["width"],
        }
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        objs = []
        for ann in anns:
            # Convert COCO polygon to Detectron2 format
            segmentation = ann["segmentation"]
            
            # Skip degenerate annotations
            if ann["area"] < 1:
                continue
            
            # COCO bbox format: [x, y, width, height]
            bbox = ann["bbox"]
            # Detectron2 expects [x1, y1, x2, y2]
            bbox_d2 = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            category_id = 0 if single_class else (ann["category_id"] - 1)
            
            obj = {
                "bbox": bbox_d2,
                "bbox_mode": 0,  # BoxMode.XYXY_ABS
                "segmentation": segmentation,
                "category_id": category_id,
                "iscrowd": ann.get("iscrowd", 0),
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


def register_datasets():
    """
    Register train and val datasets with Detectron2's DatasetCatalog.
    
    Call this once before training.
    """
    # Lazy import to avoid requiring detectron2 for EDA
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    coco = COCO(str(CFG.ANNOTATION_FILE))
    train_ids, val_ids = get_train_val_split(coco, val_ratio=CFG.VAL_SPLIT, seed=CFG.SEED)
    
    single_class = (CFG.NUM_CLASSES == 1)
    
    # Register train
    DatasetCatalog.register(
        "filament_train",
        lambda: coco_to_detectron2_dicts(coco, train_ids, CFG.TRAIN_IMG_DIR, single_class),
    )
    MetadataCatalog.get("filament_train").set(
        thing_classes=["filament"] if single_class else ["Left", "Right", "Unidentifiable", "Ambiguous"],
    )
    
    # Register val
    DatasetCatalog.register(
        "filament_val",
        lambda: coco_to_detectron2_dicts(coco, val_ids, CFG.TRAIN_IMG_DIR, single_class),
    )
    MetadataCatalog.get("filament_val").set(
        thing_classes=["filament"] if single_class else ["Left", "Right", "Unidentifiable", "Ambiguous"],
    )
    
    print(f"[Dataset] Registered 'filament_train' ({len(train_ids)} images)")
    print(f"[Dataset] Registered 'filament_val' ({len(val_ids)} images)")


def detect_solar_disk(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Detect the solar disk center and radius using Hough Circle Transform.
    
    Returns:
        (center_x, center_y, radius)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=gray.shape[0] // 2,
        param1=50,
        param2=30,
        minRadius=gray.shape[0] // 3,
        maxRadius=gray.shape[0] // 2,
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        # Take the largest circle (most likely the solar disk)
        idx = np.argmax(circles[:, 2])
        cx, cy, r = circles[idx]
        return cx, cy, r
    
    # Fallback: assume centered disk
    h, w = gray.shape[:2]
    return w // 2, h // 2, min(h, w) // 2 - 20


def apply_disk_mask(image: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    """Zero out pixels outside the solar disk."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    
    if len(image.shape) == 3:
        masked = cv2.bitwise_and(image, image, mask=mask)
    else:
        masked = cv2.bitwise_and(image, image, mask=mask)
    
    return masked


if __name__ == "__main__":
    # Quick test: verify dataset loading works
    coco = COCO(str(CFG.ANNOTATION_FILE))
    train_ids, val_ids = get_train_val_split(coco)
    
    dicts = coco_to_detectron2_dicts(coco, train_ids[:5], CFG.TRAIN_IMG_DIR)
    print(f"\n[Test] Loaded {len(dicts)} sample records")
    for d in dicts:
        print(f"  {d['file_name']}: {len(d['annotations'])} annotations")
