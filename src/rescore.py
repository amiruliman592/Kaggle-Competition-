"""
rescore.py — Phase 7 prep D (Kimee): give every fused filament of exp101 a better confidence score. CPU only.

Why: the M1/M4 error budget shows the biggest loss is confidence — ~280 true filaments score < 0.6 and are
thrown away, while real false alarms survive. The fused score only averages the views. This script adds
what the 8 single views (m28 + r50, each original + h/v/hv flips) say about each fused filament
(how many views found it, how well they agree on its shape, the spread of their scores) plus its size and
position, and lets a small gradient-boosting model learn on val which filaments are real.

Honest scoring: val images are split into 5 folds by 27-day time block (like the frozen split). Each fold is
rescored by a model fitted on the other 4, so no image is scored by a model that saw its labels. The test set is
rescored by a model fitted on all of val.

Target per fused filament: the share of the image's annotator sets in which it matches a GT filament
(IoU > 0.5), i.e. how often the official metric counts it as a TP.

Usage:
  python src/rescore.py --out-dir "Result Phase 7/rescore"
  -> raw_preds_rescore_{val,test}.json.gz (normal raw files, new "score"), rescore.json (CV sweep)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from pycocotools import mask as coco_mask
from sklearn.ensemble import HistGradientBoostingRegressor

from config import CFG
from utils.rle import load_predictions, save_predictions

ROOT = CFG.paths.ROOT
FUSED = "Result Phase 6/Mirul/raw_preds_tta2_m28_r50_{s}.json.gz"  # exp101 before post-processing
VIEWS = {  # model -> 4 view files (original, h, v, hv)
    "m28": ["Result Step 5/Step 2/raw_preds_x101_2048_m28_{s}.json.gz"]
           + [f"Result Phase 6/Mirul/raw_preds_x101_2048_m28_flip{f}_{{s}}.json.gz" for f in ("h", "v", "hv")],
    "r50": ["Result Phase 4/raw_preds_r50_2048_{s}.json.gz"]
           + [f"Result Phase 6/Mirul/raw_preds_r50_2048_flip{f}_{{s}}.json.gz" for f in ("h", "v", "hv")],
}
FEATURES = ["score", "log_area", "elong", "fill", "r_center", "n_m28", "n_r50", "s_m28", "s_r50",
            "iou_mean", "s_std", "n_crowd"]
N_FOLDS = 5
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def image_features(cands: List[dict], views: Dict[str, List[List[dict]]]) -> np.ndarray:
    """(n_cands, len(FEATURES)) for one image. views: model -> list of 4 lists of raw RLE dicts."""
    n = len(cands)
    X = np.zeros((n, len(FEATURES)))
    if n == 0:
        return X
    area = coco_mask.area(cands).astype(float)
    box = coco_mask.toBbox(cands)  # x, y, w, h
    w, h = np.maximum(box[:, 2], 1), np.maximum(box[:, 3], 1)
    cx, cy = box[:, 0] + w / 2, box[:, 1] + h / 2
    size = cands[0]["size"][0]
    X[:, 0] = [c["score"] for c in cands]
    X[:, 1] = np.log1p(area)
    X[:, 2] = np.maximum(w, h) / np.minimum(w, h)
    X[:, 3] = area / (w * h)
    X[:, 4] = np.hypot(cx - size / 2, cy - size / 2) / (size / 2)
    view_scores, view_ious = [], []
    for k, model in enumerate(("m28", "r50")):
        hits, scores = np.zeros(n), np.zeros(n)
        for v in views[model]:
            if not v:
                view_scores.append(np.zeros(n)); view_ious.append(np.zeros(n))
                continue
            iou = np.asarray(coco_mask.iou(v, cands, [0] * n), dtype=float).reshape(len(v), n)
            j = iou.argmax(0)
            best = iou[j, np.arange(n)]
            s = np.array([v[i]["score"] for i in j]) * (best >= 0.5)
            hits += best >= 0.5
            scores += s
            view_scores.append(s); view_ious.append(best)
        X[:, 5 + k] = hits
        X[:, 7 + k] = scores / 4
    X[:, 9] = np.mean(view_ious, axis=0)
    X[:, 10] = np.std(view_scores, axis=0)
    X[:, 11] = (X[:, 0] >= 0.5).sum()
    return X


def match_share(cands: List[dict], gt_sets: List[List[dict]]) -> np.ndarray:
    """Share of the image's (non-empty) annotator sets in which each candidate matches a GT (IoU > 0.5)."""
    sets = [g for g in gt_sets if g]
    if not cands or not sets:
        return np.zeros(len(cands))
    hit = np.zeros(len(cands))
    for g in sets:
        iou = np.asarray(coco_mask.iou(cands, g, [0] * len(g)), dtype=float).reshape(len(cands), len(g))
        hit += iou.max(1) > 0.5
    return hit / len(sets)


def build(split: str, stems: List[str]):
    fused = load_predictions(ROOT / FUSED.format(s=split))
    views = {m: [load_predictions(ROOT / p.format(s=split)) for p in ps] for m, ps in VIEWS.items()}
    X, owner = [], []
    for i, stem in enumerate(stems):
        c = fused.get(stem, [])
        X.append(image_features(c, {m: [v.get(stem, []) for v in vs] for m, vs in views.items()}))
        owner += [i] * len(c)
    return fused, np.vstack(X), np.array(owner)


def make_model():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=CFG.SEED)


def with_scores(fused: Dict[str, List[dict]], stems: List[str], new: np.ndarray) -> Dict[str, List[dict]]:
    out, k = {}, 0
    for stem in stems:
        rles = []
        for r in fused.get(stem, []):
            rles.append({"size": r["size"], "counts": r["counts"], "score": float(np.clip(new[k], 0, 1))})
            k += 1
        out[stem] = sorted(rles, key=lambda r: -r["score"])
    return out


def main():
    from ablate_postprocess import score
    from annotations import entry_stem, load_gt_entries
    from postprocess import close_all, select
    from split import load_split, time_block

    parser = argparse.ArgumentParser(description="Rescore exp101's fused filaments from view agreement")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(CFG.SEED)

    _, val = load_split()
    entries = load_gt_entries(val)
    fused, X, owner = build("val", val)
    y = np.zeros(len(X))
    for i, stem in enumerate(val):
        idx = np.where(owner == i)[0]
        y[idx] = match_share(fused.get(stem, []), [g for e, g in entries.items() if entry_stem(e) == stem])
    print(f"[Rescore] val: {len(X)} fused filaments, mean target {y.mean():.3f}")

    # 5 folds of whole 27-day blocks (round-robin over sorted blocks -> similar fold sizes)
    blocks = np.array([time_block(s, CFG.data.SPLIT_BLOCK_DAYS) for s in val])
    fold_of_block = {b: i % N_FOLDS for i, b in enumerate(sorted(set(blocks)))}
    img_fold = np.array([fold_of_block[b] for b in blocks])
    fold = img_fold[owner]
    oof = np.zeros(len(X))
    for f in range(N_FOLDS):
        m = make_model().fit(X[fold != f], y[fold != f])
        oof[fold == f] = m.predict(X[fold == f])
    print(f"[Rescore] CV corr(new, target) {np.corrcoef(oof, y)[0, 1]:.3f} "
          f"vs old score {np.corrcoef(X[:, 0], y)[0, 1]:.3f}")

    close_k = CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0
    rows = []
    for name, s in (("exp101 score", X[:, 0]), ("rescored (CV)", oof)):
        closed = close_all(with_scores(fused, val, s), close_k)
        for thr in THRESHOLDS:
            r = {"run": name, "thr": thr,
                 **score(entries, select(closed, thr, CFG.infer.MIN_AREA, drop_frac=CFG.infer.OVERLAP_DROP_FRAC))}
            rows.append(r)
            print(f"{name:14s} thr={thr:.2f} PQ={r['PQ']:.4f} SQ={r['SQ']:.4f} RQ={r['RQ']:.4f} "
                  f"TP={r['TP']} FP={r['FP']} FN={r['FN']} Dice={r['Dice']:.4f} n={r['n_pred']}", flush=True)
    save_predictions(with_scores(fused, val, oof), args.out_dir / "raw_preds_rescore_val.json.gz")

    # Test: one model fitted on all of val
    final = make_model().fit(X, y)
    test = sorted(set(load_predictions(ROOT / FUSED.format(s="test"))))
    fused_t, X_t, _ = build("test", test)
    save_predictions(with_scores(fused_t, test, final.predict(X_t)), args.out_dir / "raw_preds_rescore_test.json.gz")

    best = max((r for r in rows if r["run"] != "exp101 score"), key=lambda r: r["PQ"])
    (args.out_dir / "rescore.json").write_text(json.dumps({"features": FEATURES, "rows": rows}, indent=1))
    print(f"\n[Rescore] best rescored: thr {best['thr']} PQ {best['PQ']:.4f} Dice {best['Dice']:.4f} -> {args.out_dir}")


if __name__ == "__main__":
    main()
