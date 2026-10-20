"""
config.py — Centralized configuration for Solar Filament Segmentation pipeline.

All paths, hyperparameters, and constants live here so every other script
can simply `from config import CFG`.

Paths resolve portably (local / Colab / Kaggle) in this order:
  1. Environment variables: SOLAR_ROOT, SOLAR_DATA_DIR, SOLAR_OUTPUT_DIR
  2. The repository root (parent of this `src/` folder)

Examples:
  Colab:  os.environ["SOLAR_ROOT"] = "/content/Kaggle-Competition-"
  Kaggle: os.environ["SOLAR_DATA_DIR"] = "/kaggle/input/<dataset>/MAGFiLO_1.0_Kaggle_2026"
          os.environ["SOLAR_OUTPUT_DIR"] = "/kaggle/working/output"
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass
class PathsConfig:
    """Filesystem layout. Nothing is created at import time."""

    ROOT: Path = field(default_factory=lambda: _env_path(
        "SOLAR_ROOT", Path(__file__).resolve().parent.parent))
    DATA_DIR: Path = field(init=False)
    TRAIN_IMG_DIR: Path = field(init=False)
    TEST_IMG_DIR: Path = field(init=False)
    ANNOTATION_FILE: Path = field(init=False)
    OUTPUT_DIR: Path = field(init=False)
    SUBMISSION_DIR: Path = field(init=False)
    SPLIT_FILE: Path = field(init=False)
    EXPERIMENT_LOG: Path = field(init=False)

    def __post_init__(self):
        self.ROOT = Path(self.ROOT)
        self.DATA_DIR = _env_path(
            "SOLAR_DATA_DIR", self.ROOT / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026")
        self.TRAIN_IMG_DIR = self.DATA_DIR / "train" / "train_images"
        self.TEST_IMG_DIR = self.DATA_DIR / "test" / "test_images"
        self.ANNOTATION_FILE = self.DATA_DIR / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        self.OUTPUT_DIR = _env_path("SOLAR_OUTPUT_DIR", self.ROOT / "output")
        self.SUBMISSION_DIR = self.ROOT / "submissions"
        self.SPLIT_FILE = self.ROOT / "splits" / "val_ids.json"
        self.EXPERIMENT_LOG = self.ROOT / "experiments" / "log.csv"

    def ensure_output_dirs(self):
        for d in (self.OUTPUT_DIR, self.SUBMISSION_DIR):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    IMAGE_SIZE: Tuple[int, int] = (2048, 2048)  # Native resolution (H, W)
    # MAGFiLO re-annotates ~42% of files by 2-3 independent annotators; each
    # set is a separate COCO "image" entry. Policy picks ONE set per file:
    #   "first" — lowest entry id (deterministic, default)
    #   "most"  — the set with the most instances
    ANNOTATOR_POLICY: str = "first"
    # Leak-free split: group by ~1 solar rotation, stratify by filament count
    SPLIT_BLOCK_DAYS: int = 27
    SPLIT_N_FOLDS: int = 5  # val fold = 1/5 ≈ 20%
    SPLIT_VAL_FOLD: int = 0


@dataclass
class ModelConfig:
    """Detectron2 Mask R-CNN."""
    MODEL_CONFIG: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    NUM_CLASSES: int = 1  # Single "filament" class (set 4 for Left/Right/Unidentifiable/Ambiguous)
    ANCHOR_SIZES: List[List[int]] = field(
        default_factory=lambda: [[8], [16], [32], [64], [128]]
    )
    # Elongated filaments; 5 ratios -> the COCO RPN head is re-initialised (shape mismatch)
    ASPECT_RATIOS: List[List[float]] = field(
        default_factory=lambda: [[0.2, 0.5, 1.0, 2.0, 5.0]]
    )


@dataclass
class TrainConfig:
    TRAIN_RESIZE: int = 1024  # Images resized to this for training (saves VRAM)
    BASE_LR: float = 0.0025
    MAX_ITER: int = 6000  # ~21 epochs over 566 train images at batch_size=2
    BATCH_SIZE: int = 2
    NUM_WORKERS: int = 2
    # Detectron2's COCOEvaluator is untested with string image_ids and does not
    # measure the official PQ; validate with run_pipeline.py --step val instead.
    EVAL_PERIOD: int = 0
    CHECKPOINT_PERIOD: int = 1000  # Resume point if a Colab session dies


@dataclass
class TilingConfig:
    """Sliding-window inference (Phase 4)."""
    ENABLED: bool = False
    TILE_SIZE: int = 1024
    OVERLAP: int = 256
    STITCH_NMS_IOU: float = 0.35


@dataclass
class InferConfig:
    """Inference & post-processing flags."""
    INFER_RESIZE: int = 1024
    SCORE_THRESHOLD: float = 0.5  # Minimum confidence for predictions
    NMS_THRESHOLD: float = 0.3  # NMS IoU threshold
    MORPH_CLOSE: bool = True
    MORPH_KERNEL: int = 5
    MIN_AREA: int = 50  # Drop predicted masks smaller than this (px²)
    # Val sweep (run_pipeline.py --step sweep): predict once at the lowest
    # threshold, then filter offline. Smallest selected GT is 60 px².
    SWEEP_SCORE_THRESHOLDS: List[float] = field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    SWEEP_MIN_AREAS: List[int] = field(default_factory=lambda: [0, 50, 100, 200, 400])


@dataclass
class Config:
    """Single source of truth for all pipeline settings."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    SEED: int = 42


# Global config instance
CFG = Config()
