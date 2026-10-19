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
from utils.rle import encode_mask


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


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Post-process a single predicted mask:
      - Morphological closing to fill small gaps (CFG.infer.MORPH_CLOSE)
      - Remove masks smaller than CFG.infer.MIN_AREA

    Returns the cleaned mask, or an empty mask if below min area.
    """
    mask = mask.astype(np.uint8)
    if CFG.infer.MORPH_CLOSE:
        k = CFG.infer.MORPH_KERNEL
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if mask.sum() < CFG.infer.MIN_AREA:
        return np.zeros_like(mask)
    return mask


def build_predictor(model_weights: Optional[str] = None) -> DefaultPredictor:
    """
    Build a Detectron2 predictor for inference.

    Args:
        model_weights: Path to model weights file.
                       Defaults to output/detectron2/model_final.pth
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG.model.MODEL_CONFIG))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = CFG.model.NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CFG.infer.SCORE_THRESHOLD
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = CFG.infer.NMS_THRESHOLD
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = CFG.model.ANCHOR_SIZES
    cfg.INPUT.MIN_SIZE_TEST = CFG.infer.INFER_RESIZE
    cfg.INPUT.MAX_SIZE_TEST = CFG.infer.INFER_RESIZE
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    if model_weights is None:
        model_weights = str(CFG.paths.OUTPUT_DIR / "detectron2" / "model_final.pth")
    cfg.MODEL.WEIGHTS = model_weights

    print(f"[Inference] Loading model from: {model_weights}")
    return DefaultPredictor(cfg)


def predict_images(predictor, img_paths: List[Path]) -> Tuple[Dict[str, List[dict]], float]:
    """
    Run inference on the given images.

    Returns:
        (Dict mapping image stem -> list of RLE dicts highest score first, seconds per image)
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
            for idx in np.argsort(-instances.scores.numpy()):
                mask = resize_mask_to_original(pred_masks[idx], CFG.data.IMAGE_SIZE)
                mask = postprocess_mask(mask)
                if mask.any():
                    rles.append(encode_mask(mask))
        predictions[img_path.stem] = rles

    sec_per_img = (time.time() - start) / max(len(img_paths), 1)
    total = sum(len(m) for m in predictions.values())
    print(f"\n[Inference] {len(predictions)} images, {total} filaments "
          f"({total / max(len(predictions), 1):.1f}/img), {sec_per_img:.2f} s/img")
    return predictions, sec_per_img


def build_submission_csv(predictions: Dict[str, List[dict]], output_path: Path) -> int:
    """
    Write predictions in the competition format (no quotes, counts only):
      filament_id,segmentation_rle
      20150125172714Mh_1,f8uSDds...VQNC
    Returns the number of rows written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_path, "w", newline="") as f:
        f.write("filament_id,segmentation_rle\n")
        for stem in sorted(predictions):
            for i, rle in enumerate(predictions[stem], start=1):
                counts = rle["counts"]
                f.write(f"{stem}_{i},{counts.decode('utf-8') if isinstance(counts, bytes) else counts}\n")
                n += 1
    print(f"[Submission] Saved {n} rows ({len(predictions)} images) to {output_path}")
    if n == 0:
        print("[Submission] WARNING: No predictions generated! Check model/threshold.")
    return n


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
