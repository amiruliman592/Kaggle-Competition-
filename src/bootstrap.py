"""
bootstrap.py — Phase 6 M1: how big must a gain be to be real? CPU only.

Official PQ is pooled over all annotator entries:
    PQ = SQ x RQ = sum(TP IoU) / (TP + 0.5 FP + 0.5 FN)
so each val image is scored once into 4 numbers (sum IoU, TP, FP, FN, summed over
its annotator entries). A bootstrap sample re-draws images with replacement and
re-pools those numbers — no mask is touched again.

  val: 141 of 141 images with replacement   -> how much val PQ wobbles
  LB:   90 of 141 images with replacement, rounded to 2 dp
        (public LB ~ 90 test images)        -> how often the LB ranking flips

Pairs of runs use the SAME draws (paired bootstrap). "Min real gain" is
1.96 x SD of the paired difference: a smaller gain is within the noise.

Default runs = the val raw files in git, post-processed as logged in
experiments/log.csv (close 5, min area 50, overlap drop 0.2, thr per run).

Usage:
  python src/bootstrap.py --out "Result Phase 6/Mirul/bootstrap.json"
  python src/bootstrap.py --run exp101 "Result Phase 6/Mirul/raw_preds_x_val.json.gz" 0.6 --out ...
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from annotations import entry_stem, load_gt_entries
from config import CFG
from evaluate import evaluate_official
from postprocess import close_all, select
from split import load_split
from utils.rle import load_predictions

RUNS = {  # exp_id -> (val raw file, score threshold)
    "exp006": ("Result Phase 3/Result Step 3/raw_preds_r50_allann_val.json.gz", 0.6),
    "exp010": ("Result Step 5/Step 1b/raw_preds_ens_x101_r50_2048_union_val.json.gz", 0.6),
    "exp012": ("Result Step 5/Step 2/raw_preds_x101_2048_m28_val.json.gz", 0.7),
    "exp013": ("Result Step 5/Step 2/raw_preds_ens_m28_r50_union_val.json.gz", 0.6),
}
N_BOOT = 2000
LB_IMAGES = 90


def image_components(entries: Dict[str, List[dict]], preds: Dict[str, List[dict]],
                     stems: List[str]) -> np.ndarray:
    """(n_images, 4) array: sum of TP IoU, TP, FP, FN per image (all its annotator entries)."""
    comp = np.zeros((len(stems), 4))
    for i, stem in enumerate(stems):
        r = evaluate_official({e: g for e, g in entries.items() if entry_stem(e) == stem}, preds)
        comp[i] = (r["SQ"] * r["TP"], r["TP"], r["FP"], r["FN"])
    return comp


def pooled_pq(comp: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """counts (n_samples, n_images): how often each image is drawn -> PQ per sample."""
    s = counts @ comp
    denom = s[:, 1] + 0.5 * (s[:, 2] + s[:, 3])
    return np.divide(s[:, 0], denom, out=np.zeros(len(s)), where=denom > 0)


def draw_counts(n_images: int, n_draw: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n_draw, np.full(n_images, 1.0 / n_images), size=n_boot)


def compare(pq_a: np.ndarray, pq_b: np.ndarray) -> Dict[str, float]:
    """Paired bootstrap of B - A."""
    d = pq_b - pq_a
    return {"mean_delta": float(d.mean()), "ci95": [float(v) for v in np.percentile(d, [2.5, 97.5])],
            "sd_delta": float(d.std()), "min_real_gain": float(1.96 * d.std()),
            "P_B_better": float((d > 0).mean())}


def compare_lb(pq_a: np.ndarray, pq_b: np.ndarray) -> Dict[str, float]:
    """Simulated public LB: both runs on the same 90 images, rounded to 2 dp like Kaggle."""
    a, b = np.round(pq_a, 2), np.round(pq_b, 2)
    return {**compare(pq_a, pq_b), "P_lb_B_higher": float((b > a).mean()),
            "P_lb_tie": float((b == a).mean()), "P_lb_B_lower": float((b < a).mean())}


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap of official val PQ (Phase 6 M1)")
    parser.add_argument("--run", nargs=3, action="append", metavar=("ID", "RAW", "THR"),
                        help="extra run: id, val raw file, score threshold (repeatable)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--lb-images", type=int, default=LB_IMAGES)
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = {k: (CFG.paths.ROOT / p, t) for k, (p, t) in RUNS.items()}
    runs.update({k: (Path(p), float(t)) for k, p, t in args.run or []})

    _, stems = load_split()
    entries = load_gt_entries(stems)
    close_k = CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0
    comps, full = {}, {}
    for name, (path, thr) in runs.items():
        raw = load_predictions(path)
        preds = select(close_all({s: raw.get(s, []) for s in stems}, close_k), thr, CFG.infer.MIN_AREA,
                       drop_frac=CFG.infer.OVERLAP_DROP_FRAC)
        comps[name] = image_components(entries, preds, stems)
        full[name] = float(pooled_pq(comps[name], np.ones((1, len(stems))))[0])
        print(f"[Bootstrap] {name}: val official PQ {full[name]:.4f} (thr {thr})", flush=True)

    rng = np.random.default_rng(args.seed)
    val_counts = draw_counts(len(stems), len(stems), args.n_boot, rng)
    lb_counts = draw_counts(len(stems), args.lb_images, args.n_boot, rng)
    val = {k: pooled_pq(c, val_counts) for k, c in comps.items()}
    lb = {k: pooled_pq(c, lb_counts) for k, c in comps.items()}

    singles = {k: {"val_PQ": full[k], "val_sd": float(val[k].std()), "lb_sd": float(lb[k].std())} for k in runs}
    pairs = []
    for a, b in itertools.combinations(runs, 2):
        pairs.append({"A": a, "B": b, "val": compare(val[a], val[b]), "lb": compare_lb(lb[a], lb[b])})

    print(f"\n{'run':8s} {'val PQ':>7s} {'sd val':>7s} {'sd LB':>7s}")
    for k, s in singles.items():
        print(f"{k:8s} {s['val_PQ']:7.4f} {s['val_sd']:7.4f} {s['lb_sd']:7.4f}")
    print(f"\n{'B vs A':16s} {'dPQ':>7s} {'95% CI':>17s} {'P(B>A)':>7s} {'min val':>8s} {'min LB':>7s} "
          f"{'LB B>A/tie/B<A':>16s}")
    for p in pairs:
        v, l = p["val"], p["lb"]
        print(f"{p['B']} vs {p['A']} {v['mean_delta']:+7.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
              f"{v['P_B_better']:7.2f} {v['min_real_gain']:8.4f} {l['min_real_gain']:7.4f} "
              f"{l['P_lb_B_higher']:5.2f}/{l['P_lb_tie']:.2f}/{l['P_lb_B_lower']:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_boot": args.n_boot, "lb_images": args.lb_images, "seed": args.seed,
                                    "runs": {k: [str(p), t] for k, (p, t) in runs.items()},
                                    "singles": singles, "pairs": pairs}, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
