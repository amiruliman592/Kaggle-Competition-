"""
copy_paste.py — Copy-paste augmentation (Phase 6 S4, run_pipeline.py --aug copypaste).

Real filaments cut out of another training image are pasted at random free spots of this image:
  - inside the solar disk (every outline point within DISK_FRAC of the radius)
  - at least GAP px away from existing (and other pasted) filaments
  - soft edges (blurred cut-out), and the donor's background brightness shifted to the new spot
Each pasted outline is added to the labels. Runs on the full-size image, before resize and flips.
No Detectron2 needed (tested on the laptop: tests/test_copy_paste.py).
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from preprocess import detect_disk

GAP = 10        # px kept free around existing filaments
EDGE_BLUR = 5   # soft-edge width in px
DISK_FRAC = 0.95
TRIES = 20      # random spots tried per filament before giving up


def _fill(polys: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polys], 1)
    return mask


def _polys(anno: Dict) -> List[np.ndarray]:
    return [np.asarray(p, float).reshape(-1, 2) for p in anno["segmentation"]]


def paste_filaments(image: np.ndarray, annos: List[Dict], donor: np.ndarray, donor_annos: List[Dict],
                    n_max: int = 3, rng=np.random) -> Tuple[np.ndarray, List[Dict]]:
    """Paste 1..n_max of the donor's filaments into a copy of `image`; returns (image, annos + pasted)."""
    h, w = image.shape[:2]
    cx, cy, r = detect_disk(image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    grow = np.ones((2 * GAP + 1, 2 * GAP + 1), np.uint8)
    taken = cv2.dilate(_fill([p for a in annos for p in _polys(a)], (h, w)), grow)
    out, new = image.copy(), []
    cands = [a for a in donor_annos if not a.get("iscrowd", 0)]
    for i in rng.permutation(len(cands))[:rng.randint(1, n_max + 1)]:
        a = cands[i]
        pts = _polys(a)
        lo = np.floor(np.concatenate(pts).min(0)).astype(int) - EDGE_BLUR
        hi = np.ceil(np.concatenate(pts).max(0)).astype(int) + EDGE_BLUR + 1
        x0, y0 = np.maximum(lo, 0)
        x1, y1 = np.minimum(hi, (w, h))
        bw, bh = x1 - x0, y1 - y0
        cut = _fill([p - (x0, y0) for p in pts], (bh, bw))
        if not cut.any() or bw >= w or bh >= h:
            continue
        for _ in range(TRIES):
            tx, ty = rng.randint(0, w - bw + 1), rng.randint(0, h - bh + 1)
            moved = [p + (tx - x0, ty - y0) for p in pts]
            q = np.concatenate(moved)
            if np.hypot(q[:, 0] - cx, q[:, 1] - cy).max() <= DISK_FRAC * r \
                    and not (taken[ty:ty + bh, tx:tx + bw] & cut).any():
                break
        else:
            continue
        src = donor[y0:y1, x0:x1].astype(np.float32)
        dst = out[ty:ty + bh, tx:tx + bw].astype(np.float32)
        bg = cut == 0
        if bg.any():  # match the donor's surroundings to the new spot (limb darkening, other station)
            src += np.median(dst[bg], axis=0) - np.median(src[bg], axis=0)
        alpha = cv2.GaussianBlur(cut.astype(np.float32), (2 * EDGE_BLUR + 1,) * 2, 0)
        if image.ndim == 3:
            alpha = alpha[..., None]
        out[ty:ty + bh, tx:tx + bw] = np.clip(alpha * src + (1 - alpha) * dst + 0.5, 0, 255).astype(image.dtype)
        taken[ty:ty + bh, tx:tx + bw] |= cv2.dilate(cut, grow)
        new.append({"bbox": [*q.min(0).tolist(), *q.max(0).tolist()], "bbox_mode": 0,  # XYXY_ABS
                    "segmentation": [p.ravel().tolist() for p in moved],
                    "category_id": a["category_id"], "iscrowd": 0})
    return out, annos + new
