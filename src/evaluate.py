"""
evaluate.py — Local evaluation engine: PQ / SQ / RQ, Dice, IoU distribution,
fragmentation & over-merging counters, size-stratified PQ.

Definitions (Kirillov et al., CVPR 2019 — the official competition metric):
  - A prediction p and ground truth g match iff IoU(p, g) > 0.5.
    Matching is one-to-one, greedy by descending IoU (equivalent to the
    unique matching whenever predictions do not overlap).
  - PQ is POOLED over the whole dataset (all TP/FP/FN from all images):
        SQ = sum_TP IoU / |TP|,  RQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
    A per-image mean PQ is also reported, as a secondary number.
  - Fragmentation (1->N): a GT that has >= 2 predictions each lying mostly
    (>= 50% of the prediction's area) inside it.
  - Over-merging (M->1): a prediction that covers >= 50% of the area of
    >= 2 different GTs.
  - Size bins (px²): small < 200 <= medium <= 2000 < large. TP/FN are binned
    by GT area, FP by predicted area.
  - Semantic Dice: per image, Dice of union(pred) vs union(GT), averaged over
    images (binary torchmetrics DiceScore equivalent).

All masks are compressed COCO RLE dicts, so evaluation never holds dense
2048×2048 arrays.

Usage:
  python src/evaluate.py --pred submissions/submission_x.csv        # val split
  python src/evaluate.py --gt-vs-gt                                 # sanity: PQ == 1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from pycocotools import mask as coco_mask

from config import CFG
from utils.rle import counts_to_rle

SIZE_BINS = {"small": (0, 200), "medium": (200, 2000), "large": (2000, float("inf"))}
MATCH_IOU = 0.5
COVER_FRAC = 0.5


def size_bin(area: float) -> str:
    for name, (lo, hi) in SIZE_BINS.items():
        if lo <= area < hi or (name == "medium" and area == hi):
            return name
    return "large"


def match_image(gt: List[dict], pred: List[dict]) -> Dict:
    """Match one image's predictions to its ground truth."""
    gt_area = coco_mask.area(gt).astype(float) if gt else np.zeros(0)
    pr_area = coco_mask.area(pred).astype(float) if pred else np.zeros(0)
    if gt and pred:
        iou = np.asarray(coco_mask.iou(pred, gt, [0] * len(gt)), dtype=float)  # (P, G)
        inter = iou * (pr_area[:, None] + gt_area[None, :]) / (1.0 + iou)
    else:
        iou = inter = np.zeros((len(pred), len(gt)))

    # One-to-one greedy matching above threshold
    tp_pairs = []
    used_p, used_g = set(), set()
    cand = np.argwhere(iou > MATCH_IOU)
    for p, g in sorted(cand.tolist(), key=lambda pg: -iou[pg[0], pg[1]]):
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            tp_pairs.append((p, g))

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_of_pred = np.where(pr_area[:, None] > 0, inter / pr_area[:, None], 0)
        frac_of_gt = np.where(gt_area[None, :] > 0, inter / gt_area[None, :], 0)
    preds_inside_gt = ((frac_of_pred >= COVER_FRAC) & (inter > 0)).sum(axis=0)  # per GT
    gts_covered_by_pred = ((frac_of_gt >= COVER_FRAC) & (inter > 0)).sum(axis=1)  # per pred
    frag = int((preds_inside_gt >= 2).sum())
    merge = int((gts_covered_by_pred >= 2).sum())

    # Semantic (union) Dice
    if gt and pred:
        u_gt = coco_mask.merge(gt)
        u_pr = coco_mask.merge(pred)
        a_gt, a_pr = float(coco_mask.area(u_gt)), float(coco_mask.area(u_pr))
        u_iou = float(coco_mask.iou([u_pr], [u_gt], [0])[0][0])
        u_inter = u_iou * (a_gt + a_pr) / (1.0 + u_iou)
        dice = 2 * u_inter / (a_gt + a_pr)
    else:
        dice = 1.0 if not gt and not pred else 0.0

    return {
        "tp_iou": [float(iou[p, g]) for p, g in tp_pairs],
        "tp_gt_area": [float(gt_area[g]) for _, g in tp_pairs],
        "fp_area": [float(pr_area[p]) for p in range(len(pred)) if p not in used_p],
        "fn_area": [float(gt_area[g]) for g in range(len(gt)) if g not in used_g],
        "best_iou_per_gt": iou.max(axis=0).tolist() if pred else [0.0] * len(gt),
        "n_fragmented_gt": frag,
        "n_merged_pred": merge,
        "semantic_dice": dice,
    }


def _pq(tp_iou, n_fp, n_fn) -> Dict[str, float]:
    tp = len(tp_iou)
    sq = float(np.mean(tp_iou)) if tp else 0.0
    denom = tp + 0.5 * n_fp + 0.5 * n_fn
    rq = tp / denom if denom else 0.0
    return {"PQ": sq * rq, "SQ": sq, "RQ": rq, "TP": tp, "FP": int(n_fp), "FN": int(n_fn)}


def evaluate(gt: Dict[str, List[dict]], pred: Dict[str, List[dict]], verbose: bool = True) -> Dict:
    """
    Evaluate predictions on the images in `gt` (stem -> list of RLE dicts).
    Images missing from `pred` count as having no predictions; prediction
    images not in `gt` are ignored.
    """
    per_image = {s: match_image(gt[s], pred.get(s, [])) for s in gt}

    tp_iou = [x for r in per_image.values() for x in r["tp_iou"]]
    n_fp = sum(len(r["fp_area"]) for r in per_image.values())
    n_fn = sum(len(r["fn_area"]) for r in per_image.values())
    res = _pq(tp_iou, n_fp, n_fn)

    img_pq = [_pq(r["tp_iou"], len(r["fp_area"]), len(r["fn_area"]))["PQ"] for r in per_image.values()]
    best = np.array([x for r in per_image.values() for x in r["best_iou_per_gt"]])

    by_size = defaultdict(lambda: {"iou": [], "fp": 0, "fn": 0})
    for r in per_image.values():
        for i, a in zip(r["tp_iou"], r["tp_gt_area"]):
            by_size[size_bin(a)]["iou"].append(i)
        for a in r["fp_area"]:
            by_size[size_bin(a)]["fp"] += 1
        for a in r["fn_area"]:
            by_size[size_bin(a)]["fn"] += 1

    res.update({
        "mean_image_PQ": float(np.mean(img_pq)) if img_pq else 0.0,
        "mean_semantic_Dice": float(np.mean([r["semantic_dice"] for r in per_image.values()])),
        "mean_matched_Dice": float(np.mean([2 * i / (1 + i) for i in tp_iou])) if tp_iou else 0.0,
        "IoU_p25_p50_p75": [round(float(v), 4) for v in np.percentile(best, [25, 50, 75])] if len(best) else [],
        "fragmented_gt_1toN": sum(r["n_fragmented_gt"] for r in per_image.values()),
        "merged_pred_Mto1": sum(r["n_merged_pred"] for r in per_image.values()),
        "num_images": len(per_image),
        "num_gt": sum(len(v) for v in gt.values()),
        "num_pred": sum(len(pred.get(s, [])) for s in gt),
    })
    for name in SIZE_BINS:
        b = by_size[name]
        p = _pq(b["iou"], b["fp"], b["fn"])
        res[f"PQ_{name}"] = p["PQ"]
        res[f"TP/FP/FN_{name}"] = f"{p['TP']}/{p['FP']}/{p['FN']}"

    if verbose:
        print_report(res)
    return res


def print_report(res: Dict):
    print("\n" + "=" * 50)
    print("  Validation Results (pooled PQ, IoU > 0.5)")
    print("=" * 50)
    for k, v in res.items():
        print(f"  {k:>22s}: {v:.4f}" if isinstance(v, float) else f"  {k:>22s}: {v}")
    print("=" * 50)


def load_submission_rles(csv_path: Path) -> Dict[str, List[dict]]:
    """Submission CSV -> stem -> list of RLE dicts."""
    preds = defaultdict(list)
    with open(csv_path) as f:
        header = f.readline().strip()
        assert header == "filament_id,segmentation_rle", f"bad header: {header}"
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            fid, counts = line.split(",", 1)
            preds[fid.rsplit("_", 1)[0]].append(counts_to_rle(counts, CFG.data.IMAGE_SIZE))
    return dict(preds)


def main():
    from annotations import load_gt_rles
    from split import load_split

    parser = argparse.ArgumentParser(description="Evaluate a submission-format CSV locally")
    parser.add_argument("--pred", type=Path, help="Submission-format CSV of predictions")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--gt-vs-gt", action="store_true", help="Sanity check: GT scored against itself")
    parser.add_argument("--out", type=Path, help="Optional JSON file to save results")
    args = parser.parse_args()

    train_ids, val_ids = load_split()
    stems = {"val": val_ids, "train": train_ids, "all": train_ids + val_ids}[args.split]
    gt = load_gt_rles(stems)
    if args.gt_vs_gt:
        pred = gt
    elif args.pred:
        pred = load_submission_rles(args.pred)
    else:
        parser.error("pass --pred or --gt-vs-gt")
    res = evaluate(gt, pred)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
