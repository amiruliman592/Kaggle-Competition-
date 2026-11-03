"""
pseudo.py — Mirul's pseudo-labelled GONG images as extra training samples (Phase 6 S5, run_pipeline.py --pseudo).

pseudo_labels_v1.json (M3: m28 at 2048, score >= 0.85, usual post-processing) stores RLE masks. They are turned
into polygons like the real labels, so the same Detectron2 mapper and augmentations apply. The real training
samples are repeated CFG.train.REAL_REPEAT times (2 x ~921 real vs 2,098 pseudo = about half of each batch real).

Safety check: no pseudo image may come from a day that external images must avoid (val 27-day blocks, test day
+-1; src/external/download_gong.py), so val stays an honest check.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from config import CFG
from external.download_gong import day_of, excluded_days, load_exclusion_ids


def mask_to_polygons(mask: np.ndarray) -> List[List[float]]:
    """Outer outline(s) of a binary mask as flat [x1, y1, x2, y2, ...] lists (at least 3 points each)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]


def build_pseudo_dicts(json_path, img_dir, val_ids: Optional[List[str]] = None,
                       test_stems: Optional[List[str]] = None) -> List[Dict]:
    """Detectron2 dicts of the pseudo-labelled images found in img_dir (images without a usable mask are left out)."""
    coco = json.loads(Path(json_path).read_text())
    by_image: Dict[int, List[Dict]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)

    stems = [Path(im["file_name"]).stem for im in coco["images"]]
    if val_ids is None:
        val_ids, test_stems = load_exclusion_ids()
    days = [day_of(s) for s in stems]
    bad = excluded_days(val_ids, test_stems, min(days), max(days))
    clash = [s for s, d in zip(stems, days) if d in bad]
    if clash:
        raise ValueError(f"{len(clash)} pseudo-label images fall on val/test days, e.g. {clash[:3]}")

    dicts, missing = [], 0
    for im, stem in zip(coco["images"], stems):
        path = Path(img_dir) / im["file_name"]
        if not path.exists():
            missing += 1
            continue
        objs = []
        for a in by_image.get(im["id"], []):
            polys = mask_to_polygons(coco_mask.decode(a["segmentation"]))
            if not polys:
                continue
            q = np.concatenate([np.reshape(p, (-1, 2)) for p in polys])
            objs.append({"bbox": [*q.min(0).tolist(), *q.max(0).tolist()], "bbox_mode": 0,  # XYXY_ABS
                         "segmentation": polys, "category_id": 0, "iscrowd": 0})
        if objs:
            dicts.append({"file_name": str(path), "image_id": f"pseudo:{stem}",
                          "height": im["height"], "width": im["width"], "annotations": objs})
    n_masks = sum(len(d["annotations"]) for d in dicts)
    print(f"[Pseudo] {len(dicts)} images, {n_masks} masks from {json_path}"
          + (f"; {missing} image files missing" if missing else ""))
    if not dicts:
        raise FileNotFoundError(f"No pseudo-labelled image found in {img_dir}")
    return dicts


def with_pseudo(real: List[Dict]) -> List[Dict]:
    """Training dicts for --pseudo: real samples repeated REAL_REPEAT times + the pseudo-labelled images."""
    if not CFG.train.PSEUDO_JSON:
        return real
    img_dir = CFG.train.PSEUDO_IMG_DIR or str(Path(CFG.train.PSEUDO_JSON).parent / "images")
    return real * CFG.train.REAL_REPEAT + build_pseudo_dicts(CFG.train.PSEUDO_JSON, img_dir)
