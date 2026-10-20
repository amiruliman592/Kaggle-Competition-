"""
rle.py — COCO RLE helpers matching the competition submission format.

The submission stores only the RLE *counts* string; size is fixed at 2048×2048.
"""

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


def has_overlap(rles: List[Dict]) -> bool:
    """True if any two masks of one image share a pixel (Kaggle rejects this)."""
    if len(rles) < 2:
        return False
    return int(coco_mask.area(coco_mask.merge(rles))) < int(coco_mask.area(rles).sum())


def remove_overlaps(rles: List[Dict], min_area: int = 0, drop_frac: Optional[float] = None) -> List[Dict]:
    """
    Make one image's masks disjoint, as Kaggle requires. `rles` must be ordered
    highest score first: each mask keeps only pixels not claimed by an earlier
    one. Masks left smaller than `min_area` (or empty) are dropped; extra keys
    such as "score" are kept. With `drop_frac`, a mask that would lose more
    than that fraction of its area is treated as a duplicate and dropped whole.
    """
    if not has_overlap(rles):
        return [r for r in rles if int(coco_mask.area(r)) >= max(min_area, 1)]
    taken = None
    out = []
    for r in rles:
        m = coco_mask.decode(r).astype(bool)
        if taken is None:
            taken = np.zeros_like(m)
        area = m.sum()
        m &= ~taken
        if drop_frac is not None and area and 1 - m.sum() / area > drop_frac:
            continue
        if m.sum() < max(min_area, 1):
            continue
        taken |= m
        new = encode_mask(m)
        new.update({k: v for k, v in r.items() if k not in ("size", "counts")})
        out.append(new)
    return out


def save_predictions(predictions: Dict[str, List[Dict]], path: Path) -> None:
    """stem -> RLE dicts (with "score") -> gzipped JSON; counts and score only."""
    data = {stem: [[r["counts"].decode("utf-8") if isinstance(r["counts"], bytes) else r["counts"],
                    float(r.get("score", 1.0))] for r in rles]
            for stem, rles in predictions.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def load_predictions(path: Path, hw: Sequence[int] = IMAGE_HW) -> Dict[str, List[Dict]]:
    """Inverse of save_predictions: stem -> RLE dicts with "score", highest score first."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for stem, items in data.items():
        rles = []
        for counts, score in sorted(items, key=lambda cs: -cs[1]):
            rle = counts_to_rle(counts, hw)
            rle["score"] = score
            rles.append(rle)
        out[stem] = rles
    return out


def write_submission_csv(predictions: Dict[str, List[Dict]], output_path: Path) -> int:
    """
    Write predictions in the competition format (no quotes, counts only):
      filament_id,segmentation_rle
      20150125172714Mh_1,f8uSDds...VQNC
    Rows of an image keep the list order (highest score first).
    Returns the number of rows written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_path, "w", newline="") as f:
        f.write("filament_id,segmentation_rle" + chr(10))
        for stem in sorted(predictions):
            for i, rle in enumerate(predictions[stem], start=1):
                counts = rle["counts"]
                counts = counts.decode("utf-8") if isinstance(counts, bytes) else counts
                f.write(f"{stem}_{i},{counts}" + chr(10))
                n += 1
    print(f"[Submission] Saved {n} rows ({len(predictions)} images) to {output_path}")
    if n == 0:
        print("[Submission] WARNING: No predictions generated! Check model/threshold.")
    return n
