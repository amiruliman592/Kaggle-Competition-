"""
inference.py — Run Mask R-CNN inference and generate a submission CSV.

Usage:
  python src/inference.py [--weights path] [--name baseline_r50]

This script:
  1. Loads the trained model checkpoint
  2. Runs inference on all 180 test images
  3. Converts predicted masks to RLE immediately (no dense masks kept in RAM)
  4. Writes submissions/submission_<name>.csv and validates it
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from config import CFG
from postprocess import postprocess_mask
from utils.rle import encode_mask, remove_overlaps, write_submission_csv


def resize_mask_to_original(mask: np.ndarray, original_size: tuple = (2048, 2048)) -> np.ndarray:
    """
    Resize a predicted mask back to the original image resolution.

    DefaultPredictor already returns masks at the input image size; this is a
    guard for models/tiling paths that do not.
    """
    if mask.shape == tuple(original_size):
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def build_predictor(model_weights: Optional[str] = None,
                    score_threshold: Optional[float] = None) -> DefaultPredictor:
    """
    Build a Detectron2 predictor for inference.

    Args:
        model_weights: Path to model weights file.
                       Defaults to output/detectron2/baseline_r50/model_final.pth
        score_threshold: Override CFG.infer.SCORE_THRESHOLD (the val sweep
                         predicts at its lowest threshold and filters later)
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG.model.MODEL_CONFIG))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = CFG.model.NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        CFG.infer.SCORE_THRESHOLD if score_threshold is None else score_threshold)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = CFG.infer.NMS_THRESHOLD
    # Must match training, or the RPN head weights do not fit
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = CFG.model.ANCHOR_SIZES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = CFG.model.ASPECT_RATIOS
    cfg.INPUT.MIN_SIZE_TEST = CFG.infer.INFER_RESIZE
    cfg.INPUT.MAX_SIZE_TEST = CFG.infer.INFER_RESIZE
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    if model_weights is None:
        model_weights = str(CFG.paths.OUTPUT_DIR / "detectron2" / "baseline_r50" / "model_final.pth")
    cfg.MODEL.WEIGHTS = model_weights

    print(f"[Inference] Loading model from: {model_weights}")
    return DefaultPredictor(cfg)


def predict_images(predictor, img_paths: List[Path], resolve_overlaps: bool = True,
                   raw: bool = False) -> Tuple[Dict[str, List[dict]], float]:
    """
    Run inference on the given images.

    Args:
        resolve_overlaps: Make each image's masks disjoint (Kaggle rejects
            overlapping masks); higher-score masks keep shared pixels. The val
            sweep turns this off and resolves after its own score filtering.
        raw: Skip post-processing and overlap removal entirely (for
            --step dump; post-processing is then applied offline).
    Returns:
        (Dict mapping image stem -> list of RLE dicts highest score first, seconds per image)
        Each RLE dict also carries its confidence under "score" (used by the val sweep).
    """
    print(f"[Inference] Processing {len(img_paths)} images...")
    predictions = {}
    start = time.time()

    for img_path in tqdm(img_paths, desc="Inference"):
        img = cv2.imread(str(img_path))
        with torch.no_grad():
            instances = predictor(img)["instances"].to("cpu")

        rles = []
        if len(instances) > 0:
            pred_masks = instances.pred_masks.numpy()  # (N, H, W) boolean
            scores = instances.scores.numpy()
            for idx in np.argsort(-scores):
                mask = resize_mask_to_original(pred_masks[idx], CFG.data.IMAGE_SIZE)
                if not raw:
                    mask = postprocess_mask(mask)
                if mask.any():
                    rle = encode_mask(mask)
                    rle["score"] = float(scores[idx])
                    rles.append(rle)
        if resolve_overlaps and not raw:
            rles = remove_overlaps(rles, CFG.infer.MIN_AREA, CFG.infer.OVERLAP_DROP_FRAC)
        predictions[img_path.stem] = rles

    sec_per_img = (time.time() - start) / max(len(img_paths), 1)
    total = sum(len(m) for m in predictions.values())
    print(f"\n[Inference] {len(predictions)} images, {total} filaments "
          f"({total / max(len(predictions), 1):.1f}/img), {sec_per_img:.2f} s/img")
    return predictions, sec_per_img


# Lives in utils.rle (no Detectron2 needed); kept under this name for callers
build_submission_csv = write_submission_csv


def main():
    from validate_submission import validate_submission

    parser = argparse.ArgumentParser(description="Inference + submission CSV")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--name", type=str, default="baseline_r50")
    args = parser.parse_args()

    print("=" * 60)
    print("  Solar Filament Segmentation — Inference & Submission")
    print("=" * 60)
    predictor = build_predictor(args.weights)
    test_images = sorted(CFG.paths.TEST_IMG_DIR.glob("*.jpeg"))
    predictions, _ = predict_images(predictor, test_images)
    out = CFG.paths.SUBMISSION_DIR / f"submission_{args.name}.csv"
    build_submission_csv(predictions, out)
    validate_submission(out)


if __name__ == "__main__":
    main()
