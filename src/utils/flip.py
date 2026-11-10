"""flip.py — Image/mask flips for test-time augmentation (TTA). Each flip is its own inverse."""

import numpy as np

FLIPS = ("none", "h", "v", "hv")


def flip(a: np.ndarray, mode: str) -> np.ndarray:
    """Flip an image (H, W[, C]) or mask (H, W): h = left-right, v = up-down, hv = both, none = unchanged."""
    if mode not in FLIPS:
        raise ValueError(f"Unknown flip {mode!r}, expected one of {FLIPS}")
    if "h" in mode:
        a = a[:, ::-1]
    if "v" in mode:
        a = a[::-1]
    return np.ascontiguousarray(a)
