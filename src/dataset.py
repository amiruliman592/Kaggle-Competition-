"""
dataset.py — Dataset preparation & registration for Detectron2.

Handles:
  1. Building Detectron2 dicts per physical image file (one annotator set each)
  2. Registering the frozen train/val split (splits/val_ids.json)
  3. Solar disk masking (optional preprocessing)

Usage:
  from dataset import register_datasets
  register_datasets()
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from annotations import load_annotation_sets, select_annotations
from config import CFG
from preprocess import image_path
from split import load_split


def build_detectron2_dicts(
    stems: List[str],
    single_class: bool = True,
    label_policy: str = "first",
) -> List[Dict]:
    """
    Convert annotations of the given file stems to Detectron2 dataset dicts.

    Args:
        stems: Image file stems (e.g. "20140609195854Bh")
        single_class: If True, map all categories to class 0 (binary filament/not)
        label_policy: "first" — one annotator set per file (CFG.data.ANNOTATOR_POLICY);
            "all" — one dict per annotator set (same image, different labels)
    Images come from preprocess.image_path, so CFG.data.PREPROC is respected.
    """
    sets = load_annotation_sets()
    if label_policy == "all":
        samples = [(stem, k, anns) for stem in stems for k, anns in enumerate(sets[stem])]
    elif label_policy == "first":
        selected = select_annotations(sets)
        samples = [(stem, 0, selected[stem]) for stem in stems]
    else:
        raise ValueError(f"Unknown label policy {label_policy!r}")
    h, w = CFG.data.IMAGE_SIZE
    dataset_dicts = []

    for stem, k, anns in samples:
        img_path = str(image_path(stem, "train"))
        if not os.path.exists(img_path):
            print(f"[Dataset] Warning: Image not found: {img_path}")
            continue

        objs = []
        for ann in anns:
            # Skip degenerate annotations
            if ann["area"] < 1:
                continue
            x, y, bw, bh = ann["bbox"]  # COCO [x, y, width, height]
            objs.append({
                "bbox": [x, y, x + bw, y + bh],  # Detectron2 expects [x1, y1, x2, y2]
                "bbox_mode": 0,  # BoxMode.XYXY_ABS
                "segmentation": ann["segmentation"],
                "category_id": 0 if single_class else (ann["category_id"] - 1),
                "iscrowd": ann.get("iscrowd", 0),
            })

        dataset_dicts.append({
            "file_name": img_path,
            "image_id": stem if label_policy == "first" else f"{stem}#{k}",
            "height": h,
            "width": w,
            "annotations": objs,
        })

    return dataset_dicts


def register_datasets(limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Register 'filament_train' and 'filament_val' with Detectron2's DatasetCatalog.

    Args:
        limit: Keep only the first N images of each split (dry runs).
    Returns:
        (train_stems, val_stems) actually registered
    """
    # Lazy import to avoid requiring detectron2 for EDA
    from detectron2.data import DatasetCatalog, MetadataCatalog

    train_ids, val_ids = load_split()
    if limit:
        train_ids, val_ids = train_ids[:limit], val_ids[:limit]

    single_class = CFG.model.NUM_CLASSES == 1
    classes = ["filament"] if single_class else ["Left", "Right", "Unidentifiable", "Ambiguous"]

    for name, stems, policy in (("filament_train", train_ids, CFG.data.TRAIN_LABEL_POLICY),
                                ("filament_val", val_ids, "first")):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        DatasetCatalog.register(
            name, lambda s=stems, p=policy: build_detectron2_dicts(s, single_class, p))
        MetadataCatalog.get(name).set(thing_classes=classes)
        print(f"[Dataset] Registered '{name}' ({len(stems)} images)")

    return train_ids, val_ids


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
    return cv2.bitwise_and(image, image, mask=mask)


if __name__ == "__main__":
    # Quick test: verify dataset loading works
    train_ids, val_ids = load_split()
    dicts = build_detectron2_dicts(train_ids[:5])
    print(f"\n[Test] Loaded {len(dicts)} sample records")
    for d in dicts:
        print(f"  {d['file_name']}: {len(d['annotations'])} annotations")
