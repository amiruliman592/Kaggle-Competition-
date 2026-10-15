"""
train.py — Mask R-CNN training with Detectron2.

Usage:
  python src/train.py

This script:
  1. Registers train/val datasets
  2. Configures Mask R-CNN (ResNet-50 FPN) from COCO pretrained weights
  3. Trains with a learning rate schedule and periodic evaluation
  4. Saves the best model checkpoint

Designed for Google Colab Pro (A100, ~40GB VRAM).
"""

import os
import sys
from pathlib import Path

import torch

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.data import transforms as T

from config import CFG
from dataset import register_datasets


class FilamentTrainer(DefaultTrainer):
    """Custom trainer with COCO evaluation on the validation set."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """Add data augmentation to the training loader."""
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    short_edge_length=[640, 672, 704, 736, 768, 800],
                    max_size=1333,
                    sample_style="choice",
                ),
                T.RandomFlip(prob=0.5, horizontal=True),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
            ],
        )
        return build_detection_train_loader(cfg, mapper=mapper)


def build_detectron2_config() -> "CfgNode":
    """
    Build Detectron2 config for Mask R-CNN.
    
    Returns a fully configured CfgNode ready for training.
    """
    cfg = get_cfg()

    # ── Base model ─────────────────────────────────────────────────────
    cfg.merge_from_file(model_zoo.get_config_file(CFG.MODEL_CONFIG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG.MODEL_CONFIG)

    # ── Datasets ───────────────────────────────────────────────────────
    cfg.DATASETS.TRAIN = ("filament_train",)
    cfg.DATASETS.TEST = ("filament_val",)

    # ── Data loader ────────────────────────────────────────────────────
    cfg.DATALOADER.NUM_WORKERS = CFG.NUM_WORKERS

    # ── Model head ─────────────────────────────────────────────────────
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = CFG.NUM_CLASSES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Larger for dense predictions
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CFG.SCORE_THRESHOLD

    # ── Anchors (tuned for small filaments) ────────────────────────────
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = CFG.ANCHOR_SIZES

    # ── Training schedule ──────────────────────────────────────────────
    cfg.SOLVER.IMS_PER_BATCH = CFG.BATCH_SIZE
    cfg.SOLVER.BASE_LR = CFG.BASE_LR
    cfg.SOLVER.MAX_ITER = CFG.MAX_ITER
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.STEPS = (int(CFG.MAX_ITER * 0.6), int(CFG.MAX_ITER * 0.85))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = CFG.CHECKPOINT_PERIOD

    # ── Image resizing ─────────────────────────────────────────────────
    # Resize to manageable size for training
    cfg.INPUT.MIN_SIZE_TRAIN = (CFG.TRAIN_RESIZE[0],)
    cfg.INPUT.MAX_SIZE_TRAIN = CFG.TRAIN_RESIZE[0]
    cfg.INPUT.MIN_SIZE_TEST = CFG.TRAIN_RESIZE[0]
    cfg.INPUT.MAX_SIZE_TEST = CFG.TRAIN_RESIZE[0]

    # ── NMS ────────────────────────────────────────────────────────────
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = CFG.NMS_THRESHOLD

    # ── Output ─────────────────────────────────────────────────────────
    cfg.OUTPUT_DIR = str(CFG.OUTPUT_DIR / "detectron2")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ── Evaluation ─────────────────────────────────────────────────────
    cfg.TEST.EVAL_PERIOD = CFG.EVAL_PERIOD

    return cfg


def train():
    """Run the full training pipeline."""
    print("=" * 60)
    print("  Solar Filament Segmentation — Mask R-CNN Training")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("[Train] WARNING: No GPU detected! Training will be very slow.")

    # Register datasets
    register_datasets()

    # Build config
    cfg = build_detectron2_config()
    print(f"\n[Train] Config:")
    print(f"  Model:      {CFG.MODEL_CONFIG}")
    print(f"  Classes:    {CFG.NUM_CLASSES}")
    print(f"  Batch size: {CFG.BATCH_SIZE}")
    print(f"  LR:         {CFG.BASE_LR}")
    print(f"  Max iter:   {CFG.MAX_ITER}")
    print(f"  Output:     {cfg.OUTPUT_DIR}")

    # Train
    trainer = FilamentTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print(f"\n[Train] Done! Model saved to {cfg.OUTPUT_DIR}")
    return cfg


if __name__ == "__main__":
    train()
