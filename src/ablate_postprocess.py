"""
ablate_postprocess.py — Phase 3 step 2: post-processing ablations on saved raw
val predictions (run_pipeline.py --step dump). CPU only, no model needed.

Every row is scored with the official PQ and the official Dice variants
(evaluate.evaluate_official / official_dice) on the frozen val split.

Usage:
  python src/ablate_postprocess.py --raw "Result Phase 3/raw_preds_baseline_r50_val.json.gz" \
      --out "Result Phase 3/ablation_postproc.json"
"""

import argparse
import json
from pathlib import Path

from annotations import load_gt_entries
from evaluate import evaluate_official, official_dice
from postprocess import close_all, select
from split import load_split
from utils.rle import load_predictions

SCORE_THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
CLOSE_KERNELS = [0, 3, 5, 7]
# Round 2 (closing has no effect in round 1 -> fixed at the pipeline default 5)
R2_SCORE_THRESHOLDS = [0.7, 0.75, 0.8]
R2_DROP_FRACS = [None, 0.3, 0.5, 0.7]
R2_DISK_FACTORS = [None, 0.95, 0.98, 1.0]  # drop masks whose centroid is beyond factor x radius


def score(entries, preds):
    pq = evaluate_official(entries, preds)
    dice = official_dice(entries, preds)
    return {"PQ": pq["PQ"], "SQ": pq["SQ"], "RQ": pq["RQ"], "TP": pq["TP"], "FP": pq["FP"], "FN": pq["FN"],
            "Dice": dice["Dice_samplewise"], "pair_Dice": dice["pair_Dice_mean"],
            "n_pred": sum(map(len, preds.values()))}


def round2(raw, entries, disk):
    closed = close_all(raw, 5)
    rows = []
    for factor in R2_DISK_FACTORS:
        def inside(stem, r, f=factor):
            if f is None:
                return True
            cx, cy, rad = disk[stem]
            x, y = r["centroid"]
            return (x - cx) ** 2 + (y - cy) ** 2 <= (f * rad) ** 2
        for drop in R2_DROP_FRACS:
            for thr in R2_SCORE_THRESHOLDS:
                r = {"close_kernel": 5, "score_threshold": thr, "drop_frac": drop, "disk_factor": factor,
                     **score(entries, select(closed, thr, 0, keep=inside, drop_frac=drop))}
                rows.append(r)
                print(f"disk={factor} drop={drop} thr={thr:.2f}  PQ={r['PQ']:.4f} RQ={r['RQ']:.4f} "
                      f"Dice={r['Dice']:.4f} FP={r['FP']} FN={r['FN']} n={r['n_pred']}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Post-processing ablations on raw val predictions")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1, choices=[1, 2])
    parser.add_argument("--disk", type=Path, help="JSON stem -> [cx, cy, r] (round 2)")
    args = parser.parse_args()

    _, val_ids = load_split()
    raw = load_predictions(args.raw)
    raw = {s: raw.get(s, []) for s in val_ids}
    entries = load_gt_entries(val_ids)

    rows = round2(raw, entries, json.loads(args.disk.read_text())) if args.round == 2 else []
    for k in (CLOSE_KERNELS if args.round == 1 else []):
        closed = close_all(raw, k)
        for thr in SCORE_THRESHOLDS:
            r = {"close_kernel": k, "score_threshold": thr, "min_area": 0, **score(entries, select(closed, thr, 0))}
            rows.append(r)
            print(f"close={k} thr={thr:.2f}  PQ={r['PQ']:.4f} SQ={r['SQ']:.4f} RQ={r['RQ']:.4f} "
                  f"Dice={r['Dice']:.4f} n={r['n_pred']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    best = max(rows, key=lambda r: r["PQ"])
    print(f"\nBest PQ {best['PQ']:.4f} (Dice {best['Dice']:.4f}) at close={best['close_kernel']} "
          f"thr={best['score_threshold']} -> {args.out}")


if __name__ == "__main__":
    main()
