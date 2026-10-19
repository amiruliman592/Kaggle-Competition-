"""
rle.py — COCO RLE helpers matching the competition submission format.

The submission stores only the RLE *counts* string; size is fixed at 2048×2048.
"""

from typing import Dict, List, Sequence

import numpy as np
from pycocotools import mask as coco_mask

IMAGE_HW = (2048, 2048)


def mask_to_rle(binary_mask: np.ndarray) -> str:
    """Binary mask (H×W) -> COCO RLE counts string (submission format)."""
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")


def encode_mask(binary_mask: np.ndarray) -> Dict:
    """Binary mask (H×W) -> compressed RLE dict {"size", "counts"}."""
    return coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))


def counts_to_rle(counts: str, hw: Sequence[int] = IMAGE_HW) -> Dict:
    """Submission counts string -> compressed RLE dict."""
    return {"size": [int(hw[0]), int(hw[1])], "counts": counts.encode("utf-8")}


def rle_to_mask(counts: str, hw: Sequence[int] = IMAGE_HW) -> np.ndarray:
    """Submission counts string -> binary uint8 mask (H×W)."""
    return coco_mask.decode(counts_to_rle(counts, hw))


def polygon_to_rle(segmentation: List[List[float]], hw: Sequence[int] = IMAGE_HW) -> Dict:
    """COCO polygon list -> single merged compressed RLE dict."""
    rles = coco_mask.frPyObjects(segmentation, int(hw[0]), int(hw[1]))
    return coco_mask.merge(rles)
