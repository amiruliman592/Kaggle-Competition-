"""
sweep.py — Tune SCORE_THRESHOLD × MIN_AREA on the val split with the official PQ.

Val images are predicted ONCE at the lowest threshold (each RLE dict carries
its "score"); every grid cell is then a cheap offline filter. Pick the best
cell here, never on the leaderboard.

Usage (from run_pipeline.py --step sweep):
  table = sweep(preds, val_ids)   # list of dicts, best official PQ first
"""

from typing import Dict, List, Optional

from annotations import load_gt_entries
from config import CFG
from evaluate import evaluate_official
from utils.rle import remove_overlaps


def filter_predictions(preds: Dict[str, List[dict]], score_threshold: float,
                       min_area: int) -> Dict[str, List[dict]]:
    """
    Keep masks with score >= score_threshold, make them disjoint (higher score
    wins, as at submission time), then drop masks with area < min_area.
    """
    return {stem: remove_overlaps([r for r in rles if r.get("score", 1.0) >= score_threshold], min_area)
            for stem, rles in preds.items()}


def sweep(preds: Dict[str, List[dict]], stems: List[str],
          score_thresholds: Optional[List[float]] = None,
          min_areas: Optional[List[int]] = None) -> List[Dict]:
    """Official PQ for every (score threshold, min area) pair, best first."""
    score_thresholds = score_thresholds or CFG.infer.SWEEP_SCORE_THRESHOLDS
    min_areas = min_areas if min_areas is not None else CFG.infer.SWEEP_MIN_AREAS
    gt_entries = load_gt_entries(stems)
    rows = []
    for thr in score_thresholds:
        for area in min_areas:
            filtered = filter_predictions(preds, thr, area)
            res = evaluate_official(gt_entries, filtered)
            rows.append({"score_threshold": thr, "min_area": area,
                         "n_pred": sum(len(v) for v in filtered.values()),
                         **{k: res[k] for k in ("PQ", "SQ", "RQ", "TP", "FP", "FN")}})
    rows.sort(key=lambda r: -r["PQ"])
    return rows


def print_sweep(rows: List[Dict], top: int = 10):
    print(f"\n{'thr':>5} {'min_area':>8} {'n_pred':>7} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    for r in rows[:top]:
        print(f"{r['score_threshold']:>5.2f} {r['min_area']:>8d} {r['n_pred']:>7d} {r['PQ']:>7.4f} "
              f"{r['SQ']:>7.4f} {r['RQ']:>7.4f} {r['TP']:>5d} {r['FP']:>5d} {r['FN']:>5d}")
