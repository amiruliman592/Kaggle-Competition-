"""
train.py — Mask R-CNN training with Detectron2.

Usage:
  python src/train.py

This script:
  1. Registers the frozen train/val split
  2. Configures Mask R-CNN (ResNet-50 FPN) from COCO pretrained weights
  3. Trains with a learning rate schedule and periodic evaluation
  4. Saves the final model checkpoint

Designed for Google Colab (T4 / L4 / A100).
"""

import os
from typing import Optional

import torch

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import transforms as T

from config import CFG
from dataset import register_datasets
from utils.seed import set_seed


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
                # Same scale as inference (cfg.INPUT.*_TEST) to avoid a train/test mismatch
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),
                T.RandomFlip(prob=0.5, horizontal=True),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
            ],
        )
        return build_detection_train_loader(cfg, mapper=mapper)


def build_detectron2_config(max_iter: Optional[int] = None, eval_period: Optional[int] = None,
                            name: str = "baseline_r50") -> "CfgNode":
    """
    Build Detectron2 config for Mask R-CNN.

    Args:
        max_iter: Override CFG.train.MAX_ITER (e.g. 5 for a dry run)
        eval_period: Override CFG.train.EVAL_PERIOD (0 disables periodic eval)
        name: Experiment name; checkpoints go to OUTPUT_DIR/detectron2/<name>
    Returns a fully configured CfgNode ready for training.
    """
    max_iter = max_iter or CFG.train.MAX_ITER
    eval_period = CFG.train.EVAL_PERIOD if eval_period is None else eval_period
    cfg = get_cfg()

    # ── Base model ─────────────────────────────────────────────────────
    cfg.merge_from_file(model_zoo.get_config_file(CFG.model.MODEL_CONFIG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG.model.MODEL_CONFIG)
    cfg.SEED = CFG.SEED

    # ── Datasets ───────────────────────────────────────────────────────
    cfg.DATASETS.TRAIN = ("filament_train",)
    cfg.DATASETS.TEST = ("filament_val",)

    # ── Data loader ────────────────────────────────────────────────────
    cfg.DATALOADER.NUM_WORKERS = CFG.train.NUM_WORKERS

    # ── Model head ─────────────────────────────────────────────────────
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = CFG.model.NUM_CLASSES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Larger for dense predictions
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CFG.infer.SCORE_THRESHOLD

    # ── Anchors (tuned for small filaments) ────────────────────────────
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = CFG.model.ANCHOR_SIZES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = CFG.model.ASPECT_RATIOS

    # ── Training schedule ──────────────────────────────────────────────
    cfg.SOLVER.IMS_PER_BATCH = CFG.train.BATCH_SIZE
    cfg.SOLVER.BASE_LR = CFG.train.BASE_LR
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.WARMUP_ITERS = min(200, max_iter)
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.STEPS = (int(max_iter * 0.6), int(max_iter * 0.85))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = CFG.train.CHECKPOINT_PERIOD

    # ── Image resizing ─────────────────────────────────────────────────
    cfg.INPUT.MIN_SIZE_TRAIN = (CFG.train.TRAIN_RESIZE,)
    cfg.INPUT.MAX_SIZE_TRAIN = CFG.train.TRAIN_RESIZE
    cfg.INPUT.MIN_SIZE_TEST = CFG.infer.INFER_RESIZE
    cfg.INPUT.MAX_SIZE_TEST = CFG.infer.INFER_RESIZE

    # ── NMS ────────────────────────────────────────────────────────────
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = CFG.infer.NMS_THRESHOLD

    # ── Output ─────────────────────────────────────────────────────────
    cfg.OUTPUT_DIR = str(checkpoint_dir(name))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ── Evaluation ─────────────────────────────────────────────────────
    cfg.TEST.EVAL_PERIOD = eval_period

    return cfg


def checkpoint_dir(name: str):
    return CFG.paths.OUTPUT_DIR / "detectron2" / name


def train(max_iter: Optional[int] = None, limit: Optional[int] = None, eval_period: Optional[int] = None,
          name: str = "baseline_r50", resume: bool = True):
    """
    Run the full training pipeline.

    Args:
        max_iter: Override number of iterations (dry run: 5)
        limit: Use only the first N images of each split (dry run)
        eval_period: Override periodic evaluation (0 = off)
        name: Experiment name (checkpoint folder)
        resume: Continue from the last checkpoint in the folder if one exists
                (Colab sessions die); otherwise start from COCO weights
    """
    print("=" * 60)
    print("  Solar Filament Segmentation — Mask R-CNN Training")
    print("=" * 60)
    set_seed(CFG.SEED)

    # Check GPU
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[Train] WARNING: No GPU detected! Training will be very slow.")

    # Register datasets
    register_datasets(limit=limit)

    # Build config
    cfg = build_detectron2_config(max_iter=max_iter, eval_period=eval_period, name=name)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    print(f"\n[Train] Config:")
    print(f"  Model:      {CFG.model.MODEL_CONFIG}")
    print(f"  Classes:    {CFG.model.NUM_CLASSES}")
    print(f"  Batch size: {CFG.train.BATCH_SIZE}")
    print(f"  LR:         {CFG.train.BASE_LR}")
    print(f"  Max iter:   {cfg.SOLVER.MAX_ITER}")
    print(f"  Anchors:    {CFG.model.ANCHOR_SIZES} x {CFG.model.ASPECT_RATIOS}")
    print(f"  Output:     {cfg.OUTPUT_DIR}")

    # Train
    trainer = FilamentTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    print(f"\n[Train] Done! Model saved to {cfg.OUTPUT_DIR}")
    return cfg


if __name__ == "__main__":
    train()
