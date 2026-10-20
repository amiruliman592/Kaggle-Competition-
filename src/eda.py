"""
eda.py — Exploratory Data Analysis for Solar Filament Segmentation.

Run this first to understand the dataset:
  python src/eda.py [--no-images]

Outputs plots and eda_summary.json to output/eda/. Key analyses:
  - Annotation structure: COCO entries vs physical files (multi-annotator sets)
  - GONG station & year distribution (filename suffix: B/C/L/M/T/U + 'h')
  - Instance area distribution with the PQ size bins
  - Inter-annotator agreement: PQ of annotator k vs annotator 0 on files with
    several sets — a practical human ceiling for our model's PQ
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from annotations import anns_to_rles, load_annotation_sets, select_annotations
from config import CFG
from evaluate import evaluate, size_bin

STATIONS = {
    "B": "Big Bear (USA)", "C": "Cerro Tololo (Chile)", "L": "Learmonth (Australia)",
    "M": "Mauna Loa (USA)", "T": "El Teide (Spain)", "U": "Udaipur (India)",
}
CAT_COLORS = {
    1: (1.0, 0.2, 0.2, 0.5),   # Left - red
    2: (0.2, 0.5, 1.0, 0.5),   # Right - blue
    3: (0.2, 1.0, 0.2, 0.5),   # Unidentifiable - green
    4: (1.0, 1.0, 0.2, 0.5),   # Ambiguous - yellow
}
CAT_NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


def station_of(stem: str) -> str:
    return stem[14] if len(stem) > 14 else "?"


def plot_sample_images(selected, img_dir: Path, save_dir: Path, n: int = 6):
    """Plot sample training images with their (selected) filament masks overlaid."""
    from pycocotools import mask as coco_mask

    sample = random.sample(sorted(selected), min(n, len(selected)))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, stem in zip(axes.flatten(), sample):
        img = cv2.cvtColor(cv2.imread(str(img_dir / f"{stem}.jpeg")), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        overlay = np.zeros((*img.shape[:2], 4))
        for ann, rle in zip(selected[stem], anns_to_rles(selected[stem])):
            overlay[coco_mask.decode(rle) == 1] = CAT_COLORS.get(ann["category_id"], (1, 1, 1, 0.5))
        ax.imshow(overlay)
        ax.set_title(f"{stem} [{STATIONS.get(station_of(stem), '?')}]\n{len(selected[stem])} filaments", fontsize=9)
        ax.axis("off")
    patches = [mpatches.Patch(color=c[:3], label=CAT_NAMES[k]) for k, c in CAT_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=12)
    plt.suptitle("Sample Training Images with Filament Masks", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / "sample_images.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved {save_dir / 'sample_images.png'}")


def plot_stats(selected, areas: np.ndarray, save_dir: Path):
    """Area distribution, filaments/image, station and year distributions."""
    stems = sorted(selected)
    counts = [len(selected[s]) for s in stems]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.hist(np.log10(areas), bins=60, color="darkorange", edgecolor="black", alpha=0.8)
    for edge in (200, 2000):
        ax.axvline(np.log10(edge), color="red", linestyle="--")
    ax.set_xlabel("log10(area px²)  [dashed: PQ size-bin edges 200 / 2000]")
    ax.set_ylabel("Instances")
    ax.set_title("Filament Area Distribution")

    ax = axes[0, 1]
    ax.hist(counts, bins=range(0, max(counts) + 2), color="seagreen", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Filaments per image (selected annotator set)")
    ax.set_ylabel("Images")
    ax.set_title("Filaments per Image")

    ax = axes[1, 0]
    st = Counter(station_of(s) for s in stems)
    keys = sorted(st)
    ax.bar([STATIONS.get(k, k) for k in keys], [st[k] for k in keys], color="steelblue")
    ax.set_title("Images per GONG Station")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    yr = Counter(s[:4] for s in stems)
    ax.bar(sorted(yr), [yr[y] for y in sorted(yr)], color="slateblue")
    ax.set_title("Images per Year")

    plt.tight_layout()
    plt.savefig(save_dir / "filament_stats.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved {save_dir / 'filament_stats.png'}")


def inter_annotator_agreement(sets) -> dict:
    """PQ of each extra annotator set vs set 0, pooled over multi-annotated files."""
    multi = {s: v for s, v in sets.items() if len(v) > 1}
    gt, pred = {}, {}
    for stem, v in multi.items():
        ref = anns_to_rles(v[0])
        for k, other in enumerate(v[1:], start=1):
            key = f"{stem}#{k}"
            gt[key], pred[key] = ref, anns_to_rles(other)
    res = evaluate(gt, pred, verbose=False)
    return {"files": len(multi), "comparisons": len(gt),
            **{k: res[k] for k in ("PQ", "SQ", "RQ", "mean_semantic_Dice",
                                   "fragmented_gt_1toN", "merged_pred_Mto1")}}


def main():
    parser = argparse.ArgumentParser(description="Exploratory data analysis")
    parser.add_argument("--no-images", action="store_true", help="Skip the sample-image figure")
    args = parser.parse_args()

    random.seed(CFG.SEED)
    print("=" * 60)
    print("  Solar Filament Segmentation — Exploratory Data Analysis")
    print("=" * 60)
    save_dir = CFG.paths.OUTPUT_DIR / "eda"
    save_dir.mkdir(parents=True, exist_ok=True)

    sets = load_annotation_sets()
    selected = select_annotations(sets)
    all_anns = [a for v in sets.values() for s in v for a in s]
    sel_anns = [a for v in selected.values() for a in v]
    areas = np.array([float(a["area"]) for a in sel_anns])
    counts = np.array([len(v) for v in selected.values()])

    summary = {
        "coco_image_entries": sum(len(v) for v in sets.values()),
        "physical_files": len(sets),
        "files_by_num_annotator_sets": dict(sorted(Counter(len(v) for v in sets.values()).items())),
        "annotations_total_all_sets": len(all_anns),
        "annotations_selected_policy": len(sel_anns),
        "annotator_policy": CFG.data.ANNOTATOR_POLICY,
        "filaments_per_image": {"mean": round(float(counts.mean()), 2), "min": int(counts.min()),
                                "max": int(counts.max())},
        "area_px2": {"min": float(areas.min()), "median": float(np.median(areas)),
                     "mean": round(float(areas.mean()), 1), "max": float(areas.max())},
        "instances_per_size_bin": dict(Counter(size_bin(a) for a in areas)),
        "instances_below_50px2": int((areas < 50).sum()),
        "categories": {CAT_NAMES[k]: v for k, v in sorted(Counter(a["category_id"] for a in sel_anns).items())},
        "stations": {STATIONS.get(k, k): v for k, v in sorted(Counter(station_of(s) for s in sets).items())},
        "years": dict(sorted(Counter(s[:4] for s in sets).items())),
        "unique_dates": len({s[:8] for s in sets}),
    }
    print("[EDA] Computing inter-annotator agreement (human PQ ceiling)...")
    summary["inter_annotator_agreement"] = inter_annotator_agreement(sets)

    plot_stats(selected, areas, save_dir)
    if not args.no_images:
        plot_sample_images(selected, CFG.paths.TRAIN_IMG_DIR, save_dir)

    (save_dir / "eda_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"\n[EDA] Done! See {save_dir}")


if __name__ == "__main__":
    main()
