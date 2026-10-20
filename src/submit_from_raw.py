"""
submit_from_raw.py — Build a submission CSV from saved raw test predictions
(run_pipeline.py --step dump) with any post-processing setting. CPU only.

Defaults are the current pipeline settings in CFG.infer.

Usage:
  python src/submit_from_raw.py "Result Phase 3/Result Step 1/raw_preds_baseline_r50_test.json.gz" \
      submissions/submission_baseline_r50_pp.csv [--score-thr 0.65] [--drop-frac 0.2]
"""

import argparse
from pathlib import Path

from config import CFG
from postprocess import close_all, select
from utils.rle import load_predictions, write_submission_csv
from validate_submission import validate_submission


def main():
    parser = argparse.ArgumentParser(description="Submission CSV from raw test predictions")
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--score-thr", type=float, default=CFG.infer.SCORE_THRESHOLD)
    parser.add_argument("--close-kernel", type=int,
                        default=CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0)
    parser.add_argument("--min-area", type=int, default=CFG.infer.MIN_AREA)
    parser.add_argument("--drop-frac", type=float, default=CFG.infer.OVERLAP_DROP_FRAC)
    args = parser.parse_args()

    raw = load_predictions(args.raw)
    preds = select(close_all(raw, args.close_kernel), args.score_thr, args.min_area, drop_frac=args.drop_frac)
    print(f"[SubmitFromRaw] thr={args.score_thr} close={args.close_kernel} min_area={args.min_area} "
          f"drop_frac={args.drop_frac}")
    write_submission_csv(preds, args.out)
    validate_submission(args.out)


if __name__ == "__main__":
    main()
