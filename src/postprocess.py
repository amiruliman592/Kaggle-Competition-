"""
postprocess.py — Mask post-processing shared by Colab inference and local
(CPU-only) Phase 3 ablations. No Detectron2 needed.

Raw predictions (run_pipeline.py --step dump) are stored before any
post-processing, so every setting here can be re-applied offline.

Usage:
  from postprocess import postprocess_mask
"""

from typing import Optional

import cv2
import numpy as np

from config import CFG


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
