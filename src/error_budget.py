"""
error_budget.py — Phase 6 M1: where is the PQ lost? CPU only.

Same matching as the official metric (evaluate_official): every annotator set is
its own entry, a hit is IoU > 0.5. Every mistake of one run on val goes in one box:

  missed filament (FN), first box that applies:
    fn_low_score   a raw prediction hits it (IoU > 0.5) but its score < threshold  -> confidence problem
    fn_postproc    a raw hit with score >= threshold was removed/carved by min area
                   or overlap removal                                            -> post-processing problem
    fn_near_miss   a kept mask overlaps it with IoU 0.3-0.5                       -> outline problem
    fn_never       nothing close                                                 -> recall problem
  false alarm (FP), per entry:
    fp_other_ann   another annotator's set of the same image has it (IoU > 0.5)  -> label noise, not fixable
    fp_nobody      no annotator has it                                           -> real false alarm
  plus SQ loss: hits whose outline is not perfect (IoU < 1).

"PQ if fixed": FN boxes turned into hits at the current mean IoU, FP boxes removed;
for SQ every hit at IoU 1 (upper bound). Tables per GT size bin (FP: predicted
area, as evaluate.py) and per GONG station (filename char 15).

Usage:
  python src/error_budget.py --out "Result Phase 6/Mirul/error_budget_exp010.json"
  python src/error_budget.py --raw <val raw file> --thr 0.7 --out ...
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from pycocotools import mask as coco_mask

from annotations import entry_stem, load_gt_entries
from config import CFG
from eda import STATIONS, station_of
from evaluate import MATCH_IOU, SIZE_BINS, size_bin
from postprocess import close_all, select
from split import load_split
from utils.rle import load_predictions

NEAR_IOU = 0.3
DEFAULT_RAW = "Result Step 5/Step 1b/raw_preds_ens_x101_r50_2048_union_val.json.gz"  # exp010, best LB
DEFAULT_THR = 0.6
FN_BOXES = ["fn_low_score", "fn_postproc", "fn_near_miss", "fn_never"]
FP_BOXES = ["fp_other_ann", "fp_nobody"]


def _iou(pr: List[dict], gt: List[dict]) -> np.ndarray:
    if not pr or not gt:
        return np.zeros((len(pr), len(gt)))
    return np.asarray(coco_mask.iou(pr, gt, [0] * len(gt)), dtype=float)  # (P, G)


def classify_entry(gt: List[dict], final: List[dict], raw: List[dict], others: List[List[dict]],
                   thr: float) -> List[Dict]:
    """One annotator entry -> one record per hit / FN / FP: {"box", "area", "iou"}."""
    out = []
    iou = _iou(final, gt)
    raw_iou = _iou(raw, gt)
    raw_score = np.array([r["score"] for r in raw])
    gt_area = coco_mask.area(gt).astype(float) if gt else []
    for p, q in np.argwhere(iou > MATCH_IOU).tolist():  # non-unique matching, like the official metric
        out.append({"box": "tp", "area": float(gt_area[q]), "iou": float(iou[p, q])})
    for g in range(len(gt)):
        if (iou[:, g] > MATCH_IOU).any():
            continue
        raw_hit = raw_iou[:, g] > MATCH_IOU
        if raw_hit.any() and (raw_score[raw_hit] < thr).all():
            box = "fn_low_score"
        elif raw_hit.any():
            box = "fn_postproc"
        elif (iou[:, g] >= NEAR_IOU).any():
            box = "fn_near_miss"
        else:
            box = "fn_never"
        out.append({"box": box, "area": float(gt_area[g]), "iou": 0.0})
    pr_area = coco_mask.area(final).astype(float) if final else []
    for p in range(len(final)):
        if (iou[p] > MATCH_IOU).any():
            continue
        other_hit = any((_iou([final[p]], o) > MATCH_IOU).any() for o in others)
        out.append({"box": "fp_other_ann" if other_hit else "fp_nobody", "area": float(pr_area[p]), "iou": 0.0})
    return out


def budget(records: List[Dict]) -> Dict:
    """Counts per box, PQ, and PQ if a box were fixed."""
    n = Counter(r["box"] for r in records)
    tp_iou = [r["iou"] for r in records if r["box"] == "tp"]
    s, tp = sum(tp_iou), len(tp_iou)
    fp, fn = sum(n[b] for b in FP_BOXES), sum(n[b] for b in FN_BOXES)
    pq = lambda s_, tp_, fp_, fn_: s_ / (tp_ + 0.5 * (fp_ + fn_)) if tp_ + fp_ + fn_ else 0.0
    sq = s / tp if tp else 0.0
    res = {"PQ": pq(s, tp, fp, fn), "SQ": sq, "TP": tp, "FP": fp, "FN": fn, "boxes": {}}
    for b in FN_BOXES:
        res["boxes"][b] = {"n": n[b], "PQ_if_fixed": pq(s + n[b] * sq, tp + n[b], fp, fn - n[b])}
    for b in FP_BOXES:
        res["boxes"][b] = {"n": n[b], "PQ_if_fixed": pq(s, tp, fp - n[b], fn)}
    res["boxes"]["sq_loss"] = {"n": tp, "PQ_if_fixed": pq(tp, tp, fp, fn)}
    return res


def run(entries: Dict[str, List[dict]], final: Dict[str, List[dict]], raw: Dict[str, List[dict]],
        thr: float) -> Dict:
    by_stem = defaultdict(list)
    for e, g in entries.items():
        if g:  # the official GT has no rows for empty entries
            by_stem[entry_stem(e)].append(g)
    total, by_size, by_station = [], defaultdict(list), defaultdict(list)
    for stem, gts in by_stem.items():
        for k, gt in enumerate(gts):
            recs = classify_entry(gt, final.get(stem, []), raw.get(stem, []), gts[:k] + gts[k + 1:], thr)
            total += recs
            by_station[station_of(stem)] += recs
            for r in recs:
                by_size[size_bin(r["area"])].append(r)
    return {"all": budget(total),
            "by_size": {b: budget(by_size[b]) for b in SIZE_BINS},
            "by_station": {s: budget(by_station[s]) for s in sorted(by_station)}}


def _print(res: Dict):
    a = res["all"]
    print(f"\nPQ {a['PQ']:.4f}  SQ {a['SQ']:.4f}  TP {a['TP']}  FP {a['FP']}  FN {a['FN']}")
    print(f"{'box':14s} {'n':>5s} {'PQ if fixed':>12s} {'gain':>7s}")
    for b, v in a["boxes"].items():
        print(f"{b:14s} {v['n']:5d} {v['PQ_if_fixed']:12.4f} {v['PQ_if_fixed'] - a['PQ']:+7.4f}")
    cols = ["TP"] + FN_BOXES + FP_BOXES
    for title, group in (("size", res["by_size"]), ("station", res["by_station"])):
        print(f"\n{title:22s} {'PQ':>6s} {'SQ':>6s} " + " ".join(f"{c.replace('fn_', '').replace('fp_', ''):>10s}"
                                                       for c in cols))
        for k, v in group.items():
            name = f"{k} {STATIONS.get(k, '')}"[:22] if title == "station" else k
            vals = [v["TP"]] + [v["boxes"][c]["n"] for c in FN_BOXES + FP_BOXES]
            print(f"{name:22s} {v['PQ']:6.4f} {v['SQ']:6.4f} " + " ".join(f"{x:10d}" for x in vals))


def main():
    parser = argparse.ArgumentParser(description="Error budget of one run on val (Phase 6 M1)")
    parser.add_argument("--raw", type=Path, default=CFG.paths.ROOT / DEFAULT_RAW)
    parser.add_argument("--thr", type=float, default=DEFAULT_THR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _, stems = load_split()
    entries = load_gt_entries(stems)
    loaded = load_predictions(args.raw)
    raw = close_all({s: loaded.get(s, []) for s in stems},
                    CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0)
    final = select(raw, args.thr, CFG.infer.MIN_AREA, drop_frac=CFG.infer.OVERLAP_DROP_FRAC)
    res = run(entries, final, raw, args.thr)
    _print(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"raw": str(args.raw), "thr": args.thr, **res}, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
