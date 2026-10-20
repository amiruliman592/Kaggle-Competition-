"""
preprocess.py — Solar image preprocessing (Phase 3 step 3, retraining ablations).

Modes (CFG.data.PREPROC):
  "none"        original JPEGs (Phase 2 baseline)
  "clahe_disk"  solar-disk masking (Hough circle on a 512 px copy, everything
                outside radius + margin set to 0) + CLAHE on the grayscale image

Preprocessed images are written ONCE to CFG.paths.PREPROC_DIR/<mode>/{train,test}
(run_pipeline.py --step preprocess); training and inference then just read
them, so the same pixels are used in both and Detectron2 needs no changes.

Usage:
  from preprocess import image_path
  image_path("20140609195854Bh", "train")   # respects CFG.data.PREPROC
"""

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from config import CFG

DISK_MARGIN = 4  # px kept beyond the detected radius (limb filaments)


def source_dir(split: str) -> Path:
    return CFG.paths.TRAIN_IMG_DIR if split == "train" else CFG.paths.TEST_IMG_DIR


def image_path(stem: str, split: str) -> Path:
    """Image the model should see for this file ("train" covers val too)."""
    if CFG.data.PREPROC == "none":
        return source_dir(split) / f"{stem}.jpeg"
    return CFG.paths.PREPROC_DIR / CFG.data.PREPROC / split / f"{stem}.png"


def detect_disk(gray: np.ndarray) -> Tuple[int, int, int]:
    """(cx, cy, r) at full resolution, via Hough on a 512 px copy (fast, stable: r ≈ 892-916 px)."""
    from dataset import detect_solar_disk

    scale = gray.shape[0] / 512
    cx, cy, r = detect_solar_disk(cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA))
    return int(cx * scale), int(cy * scale), int(r * scale)


def clahe_disk(img_bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Disk mask + CLAHE; returns a 3-channel BGR uint8 image (model input format)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    cx, cy, r = detect_disk(gray)
    out = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(gray)
    disk = np.zeros_like(gray)
    cv2.circle(disk, (cx, cy), r + DISK_MARGIN, 255, -1)
    out = cv2.bitwise_and(out, out, mask=disk)
    return cv2.merge([out, out, out])


def preprocess_files(stems: Iterable[str], split: str) -> int:
    """Write preprocessed copies for CFG.data.PREPROC; skips files that already exist."""
    if CFG.data.PREPROC == "none":
        return 0
    if CFG.data.PREPROC != "clahe_disk":
        raise ValueError(f"Unknown PREPROC {CFG.data.PREPROC!r}")
    n = 0
    for stem in stems:
        dst = image_path(stem, split)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(source_dir(split) / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        cv2.imwrite(str(dst), clahe_disk(img))
        n += 1
    return n
