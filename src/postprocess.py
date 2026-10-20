"""
postprocess.py — Mask post-processing shared by Colab inference and local
(CPU-only) Phase 3 ablations. No Detectron2 needed.

Raw predictions (run_pipeline.py --step dump) are stored before any
post-processing, so every setting here can be re-applied offline.

Usage:
  from postprocess import postprocess_mask
"""

from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from config import CFG
from utils.rle import encode_mask, remove_overlaps


def postprocess_mask(mask: np.ndarray, close_kernel: Optional[int] = None,
                     min_area: Optional[int] = None) -> np.ndarray:
    """
    Post-process a single predicted mask:
      - Morphological closing to fill small gaps (kernel 0 = off;
        default CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE)
      - Remove masks smaller than min_area (default CFG.infer.MIN_AREA)

    Returns the cleaned mask, or an empty mask if below min area.
    """
    if close_kernel is None:
        close_kernel = CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0
    min_area = CFG.infer.MIN_AREA if min_area is None else min_area
    mask = mask.astype(np.uint8)
    if close_kernel:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if mask.sum() < min_area:
        return np.zeros_like(mask)
    return mask


def close_all(raw_preds: Dict[str, List[dict]], close_kernel: int) -> Dict[str, List[dict]]:
    """
    Apply closing (no area filter) to every raw mask once; the result is
    reused for every score threshold / min area. Keeps "score" and adds the
    mask "centroid" (x, y) for geometric filters.
    """
    out = {}
    for stem, rles in raw_preds.items():
        done = []
        for r in rles:
            m = coco_mask.decode(r)                      # Fortran-ordered, full size
            x, y, w, h = (int(v) for v in coco_mask.toBbox(r))
            if w == 0 or h == 0:
                continue
            # Closing only reaches k px beyond the mask: work on a padded crop
            # (identical result, ~10x faster than the full 2048x2048 image)
            pad = close_kernel + 1
            y0, y1 = max(y - pad, 0), min(y + h + pad, m.shape[0])
            x0, x1 = max(x - pad, 0), min(x + w + pad, m.shape[1])
            crop = postprocess_mask(m[y0:y1, x0:x1], close_kernel=close_kernel, min_area=0)
            if not crop.any():
                continue
            m[y0:y1, x0:x1] = crop                       # stays Fortran-ordered -> cheap encode
            mo = cv2.moments(crop, binaryImage=True)
            new = encode_mask(m)
            new["score"] = r.get("score", 1.0)
            new["centroid"] = (x0 + mo["m10"] / mo["m00"], y0 + mo["m01"] / mo["m00"])
            done.append(new)
        out[stem] = done
    return out


def select(closed: Dict[str, List[dict]], score_threshold: float, min_area: int,
           keep: Optional[Callable[[str, dict], bool]] = None,
           drop_frac: Optional[float] = None) -> Dict[str, List[dict]]:
    """
    Same order as predict_images: score cut -> (optional filter) -> min area
    -> overlap removal (higher score keeps shared pixels; see remove_overlaps
    for drop_frac).
    """
    out = {}
    for stem, rles in closed.items():
        kept = [r for r in rles if r["score"] >= score_threshold and (keep is None or keep(stem, r))]
        kept = [r for r in kept if int(coco_mask.area(r)) >= min_area]
        out[stem] = remove_overlaps(kept, min_area, drop_frac)
    return out
