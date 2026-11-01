"""
augment.py — Training augmentations (run_pipeline.py --aug).

  flip      flips + brightness/contrast at a fixed size (recipe since Phase 2)
  rotscale  flip + 90/180/270 degree rotations + random image size +-20% (Phase 6 S3)

The image is resized whole, never cropped, so no filament is cut in two.
"""

from typing import List

from detectron2.data import transforms as T

from config import CFG


def train_augmentations(size: int) -> List[T.Augmentation]:
    """Augmentations for images trained at `size` px (cfg.INPUT.MIN_SIZE_TRAIN)."""
    if CFG.train.AUG == "rotscale":
        low, high = round(size * (1 - CFG.train.SCALE_JITTER)), round(size * (1 + CFG.train.SCALE_JITTER))
        resize = T.ResizeShortestEdge(short_edge_length=(low, high), max_size=high, sample_style="range")
        rotate = [T.RandomRotation(angle=[0, 90, 180, 270], sample_style="choice")]
    else:
        # Same scale as inference (cfg.INPUT.*_TEST) to avoid a train/test mismatch
        resize = T.ResizeShortestEdge(short_edge_length=(size,), max_size=size, sample_style="choice")
        rotate = []
    return [
        resize,
        T.RandomFlip(prob=0.5, horizontal=True),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        *rotate,
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
    ]
