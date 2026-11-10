"""
final_pipeline.py — The final submission pipeline (exp101: val official PQ 0.4480, public LB 0.39) in one command.

Per test image (2048 x 2048, no tiling, no pre-processing):
  1. 2 Mask R-CNN models: X101-FPN with a 56x56 mask head ("x101_2048_m28") and R50-FPN ("r50_2048"),
     both trained at native 2048 on every annotator set of the 566 train files.
  2. Each model predicts 4 views: original + horizontal / vertical / both flips (masks flipped back).
  3. Per model, the 4 views are fused: masks with IoU >= 0.5 are one filament, mask = union,
     score = mean over the 4 views (a view that missed it adds 0).
  4. The 2 models are fused: IoU >= 0.3, mask = intersection, score = mean over the 2 models.
  5. Post-processing: closing (5 px ellipse), score >= 0.6, area >= 50 px², overlap removal
     (higher score keeps shared pixels; a mask more than 20% covered is dropped).
  6. Submission CSV (filament_id, COCO RLE counts), validated.

The model and view order below reproduce exp101's saved predictions bit for bit (checked on all 180 test images).

Usage (GPU + Detectron2, weights from the repo: python src/weights_io.py unpack):
  python src/final_pipeline.py predict --split test --out submissions/submission_final.csv
  python src/final_pipeline.py predict --split val            # official PQ + Dice on the frozen val split
  python src/final_pipeline.py compare --a <csv> --b <csv>     # how close two submissions are (CPU)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG  # noqa: E402
from ensemble import fuse_image  # noqa: E402
from postprocess import close_all, select  # noqa: E402

MODELS = [  # order matters for exact reproduction
    {"name": "x101_2048_m28", "backbone": "x101", "mask_res": 28},
    {"name": "r50_2048", "backbone": "r50", "mask_res": 14},
]
RESOLUTION = 2048
VIEWS = ["none", "h", "v", "hv"]
VIEW_IOU, VIEW_MODE = 0.5, "union"
MODEL_IOU, MODEL_MODE = 0.3, "inter"
SCORE_THR, CLOSE_KERNEL, MIN_AREA, DROP_FRAC = 0.6, 5, 50, 0.2


def fuse_one_image(per_model_views: List[List[List[dict]]]) -> List[dict]:
    """[model][view] -> raw RLE lists of one image  ->  fused raw RLE list (before post-processing)."""
    per_model = [fuse_image(views, VIEW_IOU, VIEW_MODE) for views in per_model_views]
    return fuse_image(per_model, MODEL_IOU, MODEL_MODE)


def postprocess(fused: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Fused raw predictions -> final masks (closing, score + area cut, overlap removal)."""
    return select(close_all(fused, CLOSE_KERNEL), SCORE_THR, MIN_AREA, drop_frac=DROP_FRAC)


def load_predictors(weights_dir: Path):
    """Build both Detectron2 predictors (needs Detectron2 + the unpacked weights)."""
    from config import MODEL_CONFIGS
    from inference import build_predictor

    predictors = []
    for m in MODELS:
        weights = weights_dir / f"{m['name']}.pth"
        if not weights.exists():
            raise FileNotFoundError(f"{weights} missing: run  python src/weights_io.py unpack --dir {weights_dir}")
        CFG.model.MODEL_CONFIG = MODEL_CONFIGS[m["backbone"]]
        CFG.model.MASK_POOLER_RES = m["mask_res"]
        CFG.train.TRAIN_RESIZE = CFG.infer.INFER_RESIZE = RESOLUTION
        predictors.append(build_predictor(str(weights), score_threshold=CFG.infer.DUMP_SCORE_THRESHOLD))
    return predictors


def predict(image_paths: List[Path], predictors):
    """Full pipeline per image. Returns (fused raw preds, final preds, seconds per image)."""
    import cv2
    from tqdm import tqdm

    from inference import predict_one

    fused, final, seconds = {}, {}, []
    for path in tqdm(image_paths, desc="Final pipeline"):
        t0 = time.perf_counter()
        img = cv2.imread(str(path))
        views = [[predict_one(p, img, raw=True, flip_mode=v) for v in VIEWS] for p in predictors]
        fused[path.stem] = fuse_one_image(views)
        final.update(postprocess({path.stem: fused[path.stem]}))
        seconds.append(time.perf_counter() - t0)
    return fused, final, seconds


def latency_report(seconds: List[float], warmup: int = 2) -> Dict:
    s = np.asarray(seconds[warmup:] if len(seconds) > warmup else seconds)
    return {"images": len(seconds), "median_s": float(np.median(s)), "p95_s": float(np.percentile(s, 95)),
            "mean_s": float(s.mean()), "total_s": float(np.sum(seconds))}


def run_predict(split: str, weights_dir: Path, out: Path, raw_out: Path = None, limit: int = None) -> Dict:
    import torch

    from preprocess import image_path
    from split import load_split
    from utils.rle import save_predictions, write_submission_csv
    from validate_submission import validate_submission

    if split == "test":
        paths = [image_path(p.stem, "test") for p in sorted(CFG.paths.TEST_IMG_DIR.glob("*.jpeg"))]
    else:
        paths = [image_path(s, "train") for s in load_split()[1]]
    paths = paths[:limit] if limit else paths

    predictors = load_predictors(weights_dir)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fused, final, seconds = predict(paths, predictors)
    report = {"split": split, **latency_report(seconds),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
              "n_masks": sum(map(len, final.values()))}
    if raw_out:
        save_predictions(fused, raw_out)
    if split == "test":
        write_submission_csv(final, out)
        report["valid_csv"] = bool(validate_submission(out))
        report["csv"] = str(out)
    else:
        from ablate_postprocess import score
        from annotations import load_gt_entries
        report.update(score(load_gt_entries([p.stem for p in paths]), final))
    print(json.dumps(report, indent=1))
    return report


def compare(csv_a: Path, csv_b: Path) -> Dict:
    """Match masks of two submissions per image (IoU > 0.5): share of masks found in the other file."""
    import pandas as pd
    from pycocotools import mask as coco_mask

    from utils.rle import counts_to_rle

    def load(p):
        df = pd.read_csv(p)
        out = {}
        for fid, counts in zip(df["filament_id"], df["segmentation_rle"]):
            out.setdefault(fid.rsplit("_", 1)[0], []).append(counts_to_rle(counts))
        return out

    a, b = load(csv_a), load(csv_b)
    matched, ious = 0, []
    for stem, ra in a.items():
        rb = b.get(stem, [])
        if not rb:
            continue
        iou = np.asarray(coco_mask.iou(ra, rb, [0] * len(rb)))
        best = iou.max(axis=1)
        matched += int((best > 0.5).sum())
        ious += best[best > 0.5].tolist()
    n_a, n_b = sum(map(len, a.values())), sum(map(len, b.values()))
    res = {"masks_a": n_a, "masks_b": n_b, "a_matched_in_b": matched / max(n_a, 1),
           "median_iou": float(np.median(ious)) if ious else 0.0,
           "identical": Path(csv_a).read_bytes() == Path(csv_b).read_bytes()}
    print(json.dumps(res, indent=1))
    return res


def main():
    parser = argparse.ArgumentParser(description="Final submission pipeline (exp101)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--weights", type=Path, default=CFG.paths.ROOT / "weights")
    p.add_argument("--out", type=Path, default=CFG.paths.SUBMISSION_DIR / "submission_final.csv")
    p.add_argument("--raw-out", type=Path, default=None, help="also save the fused raw predictions (.json.gz)")
    p.add_argument("--limit", type=int, default=None, help="only the first N images (quick check)")
    p.add_argument("--report", type=Path, default=None, help="save the latency / score report as JSON")
    c = sub.add_parser("compare")
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "predict":
        rep = run_predict(args.split, args.weights, args.out, args.raw_out, args.limit)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(rep, indent=1))
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
