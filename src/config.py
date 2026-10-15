"""
config.py — Centralized configuration for Solar Filament Segmentation pipeline.

All paths, hyperparameters, and constants live here so every other script
can simply `from config import CFG`.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """Single source of truth for all pipeline settings."""

    # ── Paths ──────────────────────────────────────────────────────────
    # Adjust ROOT if running on Colab (e.g., "/content/drive/MyDrive/Solar_Filament_Kaggle")
    ROOT: Path = Path(".")
    DATA_DIR: Path = field(init=False)
    TRAIN_IMG_DIR: Path = field(init=False)
    TEST_IMG_DIR: Path = field(init=False)
    ANNOTATION_FILE: Path = field(init=False)
    OUTPUT_DIR: Path = field(init=False)

    def __post_init__(self):
        self.ROOT = Path(self.ROOT)
        self.DATA_DIR = self.ROOT / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
        self.TRAIN_IMG_DIR = self.DATA_DIR / "train" / "train_images"
        self.TEST_IMG_DIR = self.DATA_DIR / "test" / "test_images"
        self.ANNOTATION_FILE = self.DATA_DIR / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        self.OUTPUT_DIR = self.ROOT / "output"
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Image ──────────────────────────────────────────────────────────
    IMAGE_SIZE: Tuple[int, int] = (2048, 2048)  # Native resolution
    TRAIN_RESIZE: Tuple[int, int] = (1024, 1024)  # Resize for training (saves VRAM)

    # ── Model (Detectron2 Mask R-CNN) ──────────────────────────────────
    MODEL_CONFIG: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    NUM_CLASSES: int = 1  # Treat all filament categories as a single "filament" class
    # Set to 4 if you want per-category predictions (Left/Right/Unidentifiable/Ambiguous)

    # ── Training ───────────────────────────────────────────────────────
    BASE_LR: float = 0.0025
    MAX_ITER: int = 3000  # ~4 epochs over 707 images at batch_size=2
    BATCH_SIZE: int = 2
    NUM_WORKERS: int = 2
    EVAL_PERIOD: int = 500  # Evaluate every N iterations
    CHECKPOINT_PERIOD: int = 500
    VAL_SPLIT: float = 0.2  # 80/20 train/val split

    # ── Anchors (tuned for small filaments) ────────────────────────────
    ANCHOR_SIZES: List[List[int]] = field(
        default_factory=lambda: [[8], [16], [32], [64], [128]]
    )

    # ── Inference ──────────────────────────────────────────────────────
    SCORE_THRESHOLD: float = 0.5  # Minimum confidence for predictions
    NMS_THRESHOLD: float = 0.3  # NMS IoU threshold
    MIN_AREA: int = 50  # Filter out predicted masks smaller than this (in pixels²)

    # ── Submission ─────────────────────────────────────────────────────
    SUBMISSION_FILE: str = "submission.csv"

    # ── Seed ───────────────────────────────────────────────────────────
    SEED: int = 42


# Global config instance
CFG = Config()
