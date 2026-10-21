"""
ensemble.py — Fuse raw predictions of several models (run_pipeline.py --step dump). CPU only.

Per image, masks of different models that overlap (IoU >= match_iou) are one
filament. Its score is the mean over ALL models (a model that missed it adds
0, so a filament found by one of two models gets half its score and survives
only low thresholds). Its mask is, by mask_mode:
  best   the mask of the highest-scoring member
  union  OR of the members
  inter  AND of the members
The fused file is an ordinary raw prediction file, so the usual post-processing
(close, score threshold, min area, overlap removal) follows.

Usage:
  python src/ensemble.py ablate --raw "Result Phase 4/raw_preds_x101_2048_val.json.gz" \
      "Result Phase 4/raw_preds_r50_2048_val.json.gz" --out "Result Step 5/ensemble_ablation.json"
  python src/ensemble.py export --raw <test raw files...> --match-iou 0.4 --mask-mode best --out <fused raw>
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from pycocotools import mask as coco_mask

from config import CFG
from utils.rle import load_predictions, save_predictions

MATCH_IOUS = [0.3, 0.5]
MASK_MODES = ["best", "union", "inter"]
SCORE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def fuse_image(model_rles: List[List[Dict]], match_iou: float, mask_mode: str) -> List[Dict]:
    """One image: list (per model) of RLE dicts with "score" -> fused RLE dicts, highest score first."""
    items = sorted(((r["score"], m, i) for m, rles in enumerate(model_rles) for i, r in enumerate(rles)),
                   key=lambda t: -t[0])
    used = set()
    fused = []
    for score, m, i in items:
        if (m, i) in used:
            continue
        used.add((m, i))
        anchor = model_rles[m][i]
        members = [anchor]
        for m2, rles in enumerate(model_rles):
            free = [j for j in range(len(rles)) if m2 != m and (m2, j) not in used]
            if not free:
                continue
            iou = np.asarray(coco_mask.iou([rles[j] for j in free], [anchor], [0])).ravel()
            k = int(np.argmax(iou))
            if iou[k] >= match_iou:
                used.add((m2, free[k]))
                members.append(rles[free[k]])
        if mask_mode == "best" or len(members) == 1:
            out = max(members, key=lambda r: r["score"])
        elif mask_mode in ("union", "inter"):
            out = coco_mask.merge(members, intersect=mask_mode == "inter")
            if not coco_mask.area(out):
                out = anchor
        else:
            raise ValueError(f"Unknown mask mode {mask_mode!r}")
        fused.append({"size": out["size"], "counts": out["counts"],
                      "score": sum(r["score"] for r in members) / len(model_rles)})
    return sorted(fused, key=lambda r: -r["score"])


def fuse(preds: List[Dict[str, List[Dict]]], match_iou: float, mask_mode: str) -> Dict[str, List[Dict]]:
    stems = sorted(set().union(*preds))
    return {s: fuse_image([p.get(s, []) for p in preds], match_iou, mask_mode) for s in stems}


def ablate(raw_paths: List[Path], out: Path):
    from ablate_postprocess import score
    from annotations import load_gt_entries
    from postprocess import close_all, select
    from split import load_split

    _, val_ids = load_split()
    entries = load_gt_entries(val_ids)
    close_k = CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0
    # Closing per model once (same result as closing after fusion for "best"; cheap for the rest)
    closed = [close_all({s: p.get(s, []) for s in val_ids}, close_k) for p in map(load_predictions, raw_paths)]
    rows = []
    for match_iou in MATCH_IOUS:
        for mode in MASK_MODES:
            fused = fuse(closed, match_iou, mode)
            for thr in SCORE_THRESHOLDS:
                r = {"match_iou": match_iou, "mask_mode": mode, "score_threshold": thr,
                     **score(entries, select(fused, thr, CFG.infer.MIN_AREA, drop_frac=CFG.infer.OVERLAP_DROP_FRAC))}
                rows.append(r)
                print(f"iou={match_iou} {mode:5s} thr={thr:.1f}  PQ={r['PQ']:.4f} SQ={r['SQ']:.4f} "
                      f"RQ={r['RQ']:.4f} Dice={r['Dice']:.4f} n={r['n_pred']}", flush=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"models": [str(p) for p in raw_paths], "rows": rows}, indent=1))
    best = max(rows, key=lambda r: r["PQ"])
    print(f"\nBest PQ {best['PQ']:.4f} (Dice {best['Dice']:.4f}): iou={best['match_iou']} {best['mask_mode']} "
          f"thr={best['score_threshold']} -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Fuse raw predictions of several models")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ablate", help="score match IoU x mask mode x threshold on val")
    a.add_argument("--raw", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("export", help="write one fusion as a normal raw prediction file")
    e.add_argument("--raw", type=Path, nargs="+", required=True)
    e.add_argument("--match-iou", type=float, required=True)
    e.add_argument("--mask-mode", choices=MASK_MODES, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "ablate":
        ablate(args.raw, args.out)
    else:
        fused = fuse([load_predictions(p) for p in args.raw], args.match_iou, args.mask_mode)
        save_predictions(fused, args.out)
        print(f"[Ensemble] {sum(map(len, fused.values()))} masks -> {args.out}")


if __name__ == "__main__":
    main()
