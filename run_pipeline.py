"""
run_pipeline.py — End-to-end pipeline: EDA → Split → Train → Validate → Inference → Submission.

Main entry point, designed for Google Colab (see notebooks/colab_foundation.ipynb):

  python run_pipeline.py --step all
  python run_pipeline.py --dry-run          # 5 iterations, few images: smoke test
  python run_pipeline.py --step train --name baseline_r50      # resumes if a checkpoint exists
  python run_pipeline.py --step val   --name baseline_r50
  python run_pipeline.py --step sweep --name baseline_r50      # SCORE_THRESHOLD x MIN_AREA on val
  python run_pipeline.py --step infer --name baseline_r50 --score-thr 0.6 --min-area 100
  python run_pipeline.py --step dump  --name baseline_r50      # raw val+test predictions for offline post-processing
  python run_pipeline.py --step train --name x101_2048 --backbone x101 --resolution 2048   # Phase 4
  python run_pipeline.py --step train --name x101_2048_m28 --backbone x101 --resolution 2048 --mask-res 28  # Phase 5
  python run_pipeline.py --step train --name syaz_r50_seed2 --label-policy all --seed 7   # Phase 6: re-train, other seed
  python run_pipeline.py --step dump  --name r50_2048 --resolution 2048 --weights <ckpt> --flip h
      # Phase 6 TTA: predicts on flipped images -> raw_preds_<name>_fliph_<split>.json.gz
  (pass the same --backbone/--resolution/--mask-res to train, val, dump and infer of one model)

Steps:
  1. EDA                       (output/eda/)
  2. Frozen train/val split    (splits/val_ids.json, created once)
  3. Train Mask R-CNN          (output/detectron2/<name>/, resumable)
  4. Val PQ                    (official Kaggle metric + diagnostics, logged to experiments/log.csv)
     Sweep (optional)          (output/sweep_<name>.json)
  5. Test inference + CSV      (submissions/submission_<name>.csv, validated)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import CFG, MODEL_CONFIGS  # noqa: E402
from preprocess import image_path  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# STEP 0: Setup
# ═══════════════════════════════════════════════════════════════════════

def check_environment() -> bool:
    """Print versions; return True if Detectron2 is importable."""
    import torch
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    print(f"  Data dir:   {CFG.paths.DATA_DIR} (exists={CFG.paths.DATA_DIR.exists()})")
    try:
        import detectron2
        print(f"  Detectron2: {detectron2.__version__}")
    except ImportError:
        print("  Detectron2: NOT INSTALLED — see notebooks/colab_foundation.ipynb")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# STEPS
# ═══════════════════════════════════════════════════════════════════════

def run_eda():
    subprocess.check_call([sys.executable, "src/eda.py"], cwd=CFG.paths.ROOT)


def ensure_split():
    from split import main as split_main
    if not CFG.paths.SPLIT_FILE.exists():
        sys.argv = [sys.argv[0]]
        split_main()


def run_training(dry_run: bool, name: str, max_iter=None, resume: bool = True):
    from train import train
    if dry_run:
        return train(max_iter=5, limit=8, eval_period=0, name=name, resume=False)
    return train(max_iter=max_iter, name=name, resume=resume)


def run_validation(weights: str, dry_run: bool):
    """Predict the val split; official Kaggle PQ + single-annotator diagnostics."""
    from evaluate import evaluate_split
    from inference import build_predictor, predict_images
    from split import load_split

    _, val_ids = load_split()
    if dry_run:
        val_ids = val_ids[:4]
    predictor = build_predictor(weights)
    preds, sec_per_img = predict_images(
        predictor, [image_path(s, "train") for s in val_ids])
    res = evaluate_split(val_ids, preds)
    res["sec_per_img"] = sec_per_img
    return res


def run_sweep(weights: str, name: str, dry_run: bool):
    """Predict val once at the lowest threshold, then grid-search SCORE_THRESHOLD x MIN_AREA."""
    from inference import build_predictor, predict_images
    from split import load_split
    from sweep import print_sweep, sweep

    _, val_ids = load_split()
    if dry_run:
        val_ids = val_ids[:4]
    saved_min_area = CFG.infer.MIN_AREA
    CFG.infer.MIN_AREA = min(CFG.infer.SWEEP_MIN_AREAS)  # filter later, per grid cell
    predictor = build_predictor(weights, score_threshold=min(CFG.infer.SWEEP_SCORE_THRESHOLDS))
    preds, _ = predict_images(predictor, [image_path(s, "train") for s in val_ids],
                              resolve_overlaps=False)  # resolved per grid cell, after score filtering
    CFG.infer.MIN_AREA = saved_min_area
    rows = sweep(preds, val_ids)
    print_sweep(rows)
    out = CFG.paths.OUTPUT_DIR / f"sweep_{name}.json"
    out.write_text(json.dumps(rows, indent=1))
    best = rows[0]
    print(f"\n[Sweep] Best official PQ {best['PQ']:.4f} at --score-thr {best['score_threshold']} "
          f"--min-area {best['min_area']}  (saved {out})")
    return rows


def run_preprocess(dry_run: bool):
    """Write preprocessed copies of all train/val + test images (CFG.data.PREPROC); skips existing."""
    from preprocess import preprocess_files
    from split import load_split

    train_ids, val_ids = load_split()
    test_stems = [p.stem for p in sorted(CFG.paths.TEST_IMG_DIR.glob("*.jpeg"))]
    if dry_run:
        train_ids, val_ids, test_stems = train_ids[:8], val_ids[:4], test_stems[:3]
    n = preprocess_files(train_ids + val_ids, "train") + preprocess_files(test_stems, "test")
    print(f"[Preprocess] {CFG.data.PREPROC}: wrote {n} images -> {CFG.paths.PREPROC_DIR / CFG.data.PREPROC}")


def run_dump(weights: str, name: str, dry_run: bool, flip_mode=None):
    """Raw (unprocessed) val + test predictions above DUMP_SCORE_THRESHOLD -> gzipped JSON.

    flip_mode (h / v / hv): predict on flipped images (masks flipped back) and add
    "_flip<mode>" to the file name, so each flip is one ensemble member (TTA).
    """
    import torch
    from inference import build_predictor, predict_images
    from split import load_split
    from utils.rle import save_predictions

    _, val_ids = load_split()
    test_images = [image_path(p.stem, "test") for p in sorted(CFG.paths.TEST_IMG_DIR.glob("*.jpeg"))]
    if dry_run:
        val_ids, test_images = val_ids[:4], test_images[:3]
    predictor = build_predictor(weights, score_threshold=CFG.infer.DUMP_SCORE_THRESHOLD)
    for split, paths in (("val", [image_path(s, "train") for s in val_ids]),
                         ("test", test_images)):
        preds, sec = predict_images(predictor, paths, raw=True, flip_mode=flip_mode or "none")
        tag = f"_flip{flip_mode}" if flip_mode else ""
        out = CFG.paths.OUTPUT_DIR / f"raw_preds_{name}{tag}_{split}.json.gz"
        save_predictions(preds, out)
        print(f"[Dump] {split}: {sum(map(len, preds.values()))} raw masks, {sec:.2f} s/img -> {out}")
    if torch.cuda.is_available():
        print(f"[Dump] Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def run_inference_and_submit(weights: str, name: str, dry_run: bool) -> Path:
    from inference import build_predictor, build_submission_csv, predict_images
    from validate_submission import validate_submission

    test_images = [image_path(p.stem, "test") for p in sorted(CFG.paths.TEST_IMG_DIR.glob("*.jpeg"))]
    if dry_run:
        test_images = test_images[:3]
    predictor = build_predictor(weights)
    preds, _ = predict_images(predictor, test_images)
    out = CFG.paths.SUBMISSION_DIR / f"submission_{name}.csv"
    n_rows = build_submission_csv(preds, out)
    # A 5-iteration model may legitimately predict nothing; the dry run checks the plumbing
    if n_rows:
        ok = validate_submission(out)
        if not ok:
            sys.exit(1)
    return out


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Solar Filament Segmentation Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["eda", "split", "preprocess", "train", "val", "sweep", "infer", "dump", "all"],
                        help="Which step to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="5 iterations on 8 train / 4 val / 3 test images")
    parser.add_argument("--name", type=str, default=None, help="Submission/experiment name")
    parser.add_argument("--weights", type=str, default=None,
                        help="Model weights for val/sweep/infer (default: output/detectron2/<name>/model_final.pth)")
    parser.add_argument("--max-iter", type=int, default=None, help="Override CFG.train.MAX_ITER")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints, start from COCO")
    parser.add_argument("--score-thr", type=float, default=None, help="Override CFG.infer.SCORE_THRESHOLD")
    parser.add_argument("--min-area", type=int, default=None, help="Override CFG.infer.MIN_AREA")
    parser.add_argument("--preproc", choices=["none", "clahe_disk"], default=None,
                        help="Override CFG.data.PREPROC (use the same value for preprocess/train/val/dump)")
    parser.add_argument("--label-policy", choices=["first", "all"], default=None,
                        help="Override CFG.data.TRAIN_LABEL_POLICY (training labels only)")
    parser.add_argument("--backbone", choices=sorted(MODEL_CONFIGS), default=None,
                        help="Override CFG.model.MODEL_CONFIG (r50 = Phase 2/3 default)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Override TRAIN_RESIZE and INFER_RESIZE together (2048 = native, Phase 4)")
    parser.add_argument("--mask-res", type=int, default=None,
                        help="Override CFG.model.MASK_POOLER_RES (14 = default 28x28 masks, 28 = 56x56, Phase 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override CFG.SEED (42); a re-train with another seed measures training noise (Phase 6)")
    parser.add_argument("--flip", choices=["h", "v", "hv"], default=None,
                        help="--step dump only: predict on flipped images for TTA (Phase 6)")
    args = parser.parse_args(argv)
    if args.flip and args.step != "dump":
        parser.error("--flip only works with --step dump")
    return args


def apply_overrides(args):
    """Copy command-line overrides into the global CFG."""
    if args.backbone is not None:
        CFG.model.MODEL_CONFIG = MODEL_CONFIGS[args.backbone]
    if args.resolution is not None:
        CFG.train.TRAIN_RESIZE = CFG.infer.INFER_RESIZE = args.resolution
    if args.mask_res is not None:
        CFG.model.MASK_POOLER_RES = args.mask_res
    if args.preproc is not None:
        CFG.data.PREPROC = args.preproc
    if args.label_policy is not None:
        CFG.data.TRAIN_LABEL_POLICY = args.label_policy
    if args.score_thr is not None:
        CFG.infer.SCORE_THRESHOLD = args.score_thr
    if args.min_area is not None:
        CFG.infer.MIN_AREA = args.min_area
    if args.seed is not None:
        CFG.SEED = args.seed


if __name__ == "__main__":
    args = parse_args()
    apply_overrides(args)

    from utils.exp_log import log_experiment
    from utils.seed import set_seed

    set_seed(CFG.SEED)
    CFG.paths.ensure_output_dirs()
    name = args.name or ("dryrun" if args.dry_run else "baseline_r50")
    weights = args.weights or str(CFG.paths.OUTPUT_DIR / "detectron2" / name / "model_final.pth")

    print("\n[Pipeline] Environment check:")
    ready = check_environment()
    needs_d2 = args.step in ("train", "val", "sweep", "infer", "dump", "all")
    if needs_d2 and not ready:
        print("[Pipeline] Detectron2 is required for this step.")
        sys.exit(1)

    if args.step in ("eda",) or (args.step == "all" and not args.dry_run):
        run_eda()
    if args.step in ("split", "all"):
        ensure_split()
    if args.step in ("preprocess", "all"):
        run_preprocess(args.dry_run)
    if args.step in ("train", "all"):
        run_training(args.dry_run, name, args.max_iter, resume=not args.no_resume)

    res = None
    if args.step in ("val", "all"):
        res = run_validation(weights, args.dry_run)
    if args.step == "sweep":
        run_sweep(weights, name, args.dry_run)
    if args.step == "dump":
        run_dump(weights, name, args.dry_run, args.flip)
    if args.step in ("infer", "all"):
        out = run_inference_and_submit(weights, name, args.dry_run)
        print(f"\n[Pipeline] Submission file: {out}")

    if res is not None:
        log_experiment(
            model=f"mask_rcnn_{args.backbone or 'r50'}_{name}", resolution=CFG.infer.INFER_RESIZE,
            preproc=CFG.data.PREPROC + ("" if CFG.data.TRAIN_LABEL_POLICY == "first" else "+labels_all"),
            postproc=f"close{CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0}_minarea{CFG.infer.MIN_AREA}",
            val_pq=res["official_PQ"], val_sq=res["official_SQ"], val_rq=res["official_RQ"],
            val_dice=res["official_Dice_samplewise"],
            sec_per_img=round(res["sec_per_img"], 3),
            notes=("DRY RUN 5 iters; " if args.dry_run else "")
                  + f"val_pq=official(all annotators); first-annotator PQ={res['PQ']:.4f}; "
                    f"thr={CFG.infer.SCORE_THRESHOLD} frag={res['fragmented_gt_1toN']} "
                    f"merge={res['merged_pred_Mto1']} imgs={res['num_images']}",
        )
        (CFG.paths.OUTPUT_DIR / f"val_results_{name}.json").write_text(json.dumps(res, indent=1))

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)
