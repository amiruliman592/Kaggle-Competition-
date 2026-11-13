"""
make_fig_examples.py — Figure 2 of the report: example segmentations of the final pipeline (exp101) on validation.

Three validation images, chosen by their own official PQ (all annotator sets pooled): a good one (90th percentile),
a typical one (median) and a hard one (10th percentile), among images with at least 5 filaments in the first
annotator set. For each image, a 640 x 640 px crop is shown twice:
  left  = the first annotator's filaments (outlines),
  right = our final masks, judged against that annotator: green = hit (IoU > 0.5), red = false alarm,
          dashed cyan = that annotator's filament we missed.
The crop is the window with the most filaments (the most errors for the hard case).

Inputs: results/phase6/Mirul/raw_preds_tta2_m28_r50_val.json.gz (fused exp101 predictions) + final_pipeline.postprocess;
ground truth via src/annotations.py. Drawing helpers copied from notebooks/pipeline_story.ipynb (chapter 10).
Output: docs/report/figures/fig_examples.pdf (+ .png preview). CPU, about a minute.
Run:  python docs/report/make_fig_examples.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from pycocotools import mask as cm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import final_pipeline as fp  # noqa: E402
from annotations import entry_stem, load_gt_entries  # noqa: E402
from config import CFG  # noqa: E402
from evaluate import evaluate_official  # noqa: E402
from split import load_split  # noqa: E402
from utils.rle import load_predictions  # noqa: E402

RAW = ROOT / "results/phase6/Mirul/raw_preds_tta2_m28_r50_val.json.gz"
OUT = Path(__file__).resolve().parent / "figures"
HIT, FALSE_ALARM, MISSED, GT_LINE = "#4be37a", "#ff5a4d", "#39d5ff", "#ffd21f"   # story-notebook colours
WIN, STEP, MIN_GT = 640, 64, 5
W = 7.0      # inches: full text width (figure*)
FONT = 6.5


def mask(r):
    return cm.decode({"size": r["size"], "counts": r["counts"]}).astype(bool)


def iou(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(cm.iou(a, b, [0] * len(b)), dtype=float)


def centres(rles):
    b = np.array([cm.toBbox(r) for r in rles]).reshape(-1, 4)
    return np.c_[b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]


def best_window(points):
    """Top-left corner of the WIN x WIN window holding the most points."""
    best, corner = -1, (0, 0)
    for y in range(0, 2048 - WIN + 1, STEP):
        for x in range(0, 2048 - WIN + 1, STEP):
            n = int(((points[:, 0] >= x) & (points[:, 0] < x + WIN) & (points[:, 1] >= y) & (points[:, 1] < y + WIN)).sum())
            if n > best:
                best, corner = n, (x, y)
    return corner


def show(ax, img, x, y):
    ax.imshow(img[y:y + WIN, x:x + WIN], cmap="gray", vmin=0, vmax=255, extent=(x, x + WIN, y + WIN, y),
              interpolation="nearest")
    ax.set_xlim(x, x + WIN)
    ax.set_ylim(y + WIN, y)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def outline(ax, r, color, lw=0.8, ls="-"):
    contours, _ = cv2.findContours(mask(r).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        c = np.vstack([c[:, 0, :], c[:1, 0, :]]) + 0.5
        ax.plot(c[:, 0], c[:, 1], color=color, lw=lw, ls=ls)


def fill(ax, r, color, x, y, alpha=0.75):
    m = mask(r)[y:y + WIN, x:x + WIN]
    rgba = np.zeros(m.shape + (4,))
    rgba[m] = (*mcolors.to_rgb(color), alpha)
    ax.imshow(rgba, extent=(x, x + WIN, y + WIN, y), interpolation="nearest")


def main():
    _, val_ids = load_split()
    gt_all = load_gt_entries(val_ids)
    final = fp.postprocess(load_predictions(RAW))

    # per-image official PQ (all annotator sets of that image pooled)
    by_stem = {}
    for e, g in gt_all.items():
        by_stem.setdefault(entry_stem(e), {})[e] = g
    rows = []
    for stem, entries in by_stem.items():
        first = next(iter(entries))
        if len(entries[first]) < MIN_GT:
            continue
        pq = evaluate_official(entries, {stem: final.get(stem, [])})["PQ"]
        rows.append((pq, stem, first))
    rows.sort()
    pick = [rows[int(q * (len(rows) - 1))] for q in (0.9, 0.5, 0.1)]
    labels = ["good", "typical", "hard"]

    fig, axes = plt.subplots(1, 6, figsize=(W, W / 6 + 0.42))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.12, wspace=0.03)
    for k, ((pq, stem, first), label) in enumerate(zip(pick, labels)):
        img = cv2.imread(str(CFG.paths.TRAIN_IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        gt, pred = gt_all[first], final.get(stem, [])
        match = iou(pred, gt) > 0.5
        hit = match.any(axis=1)
        miss = ~match.any(axis=0) if gt else np.zeros(0, bool)
        if label == "hard":
            pts = [r for r, h in zip(pred, hit) if not h] + [r for r, m in zip(gt, miss) if m]
        else:
            pts = gt + pred
        x, y = best_window(centres(pts))

        a, b = axes[2 * k], axes[2 * k + 1]
        show(a, img, x, y)
        for r in gt:
            outline(a, r, GT_LINE)
        show(b, img, x, y)
        for r, h in zip(pred, hit):
            fill(b, r, HIT if h else FALSE_ALARM, x, y)
        for r, m in zip(gt, miss):
            if m:
                outline(b, r, MISSED, lw=1.0, ls="--")
        b.set_xlim(x, x + WIN)
        b.set_ylim(y + WIN, y)
        a.set_xlim(x, x + WIN)
        a.set_ylim(y + WIN, y)
        fig.text((a.get_position().x0 + b.get_position().x1) / 2, 0.93,
                 f"({'abc'[k]}) {label}: image PQ {pq:.2f}", ha="center", va="center", fontsize=FONT + 0.5,
                 fontweight="bold")
        a.set_title("annotator", fontsize=FONT, pad=1.5)
        b.set_title("ours", fontsize=FONT, pad=1.5)
        print(f"({'abc'[k]}) {label}: {stem}  image PQ {pq:.3f}  crop x {x} y {y}  "
              f"GT {len(gt)}  ours {len(pred)}  hits {int(hit.sum())}  false alarms {int((~hit).sum())}  "
              f"missed {int(miss.sum())}")

    handles = [Line2D([], [], color=GT_LINE, lw=1.2, label="annotator's filament"),
               mp.Patch(fc=HIT, ec="#555555", lw=0.4, label="our mask: hit (IoU > 0.5)"),
               mp.Patch(fc=FALSE_ALARM, ec="#555555", lw=0.4, label="our mask: false alarm"),
               Line2D([], [], color=MISSED, lw=1.2, ls="--", label="annotator's filament we missed")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=FONT, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.6, columnspacing=1.4)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_examples.pdf", dpi=400)
    fig.savefig(OUT / "fig_examples.png", dpi=250)
    print(f"{len(rows)} validation images with >= {MIN_GT} filaments; ->", OUT / "fig_examples.pdf")


if __name__ == "__main__":
    main()
