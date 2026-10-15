"""
eda.py — Exploratory Data Analysis for Solar Filament Segmentation.

Run this first to understand the dataset:
  python src/eda.py

Outputs visualizations to output/eda/
"""

import json
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pycocotools.coco import COCO
from collections import Counter

from config import CFG


def load_coco(annotation_path: Path) -> COCO:
    """Load COCO-format annotations."""
    coco = COCO(str(annotation_path))
    return coco


def plot_sample_images(coco: COCO, img_dir: Path, n: int = 6, save_dir: Path = None):
    """Plot sample images with their filament masks overlaid."""
    save_dir = save_dir or CFG.OUTPUT_DIR / "eda"
    save_dir.mkdir(parents=True, exist_ok=True)

    img_ids = coco.getImgIds()
    sample_ids = random.sample(img_ids, min(n, len(img_ids)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Category colors
    cat_colors = {
        1: (1.0, 0.2, 0.2, 0.5),   # Left - red
        2: (0.2, 0.5, 1.0, 0.5),   # Right - blue
        3: (0.2, 1.0, 0.2, 0.5),   # Unidentifiable - green
        4: (1.0, 1.0, 0.2, 0.5),   # Ambiguous - yellow
    }
    cat_names = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}

    for ax, img_id in zip(axes, sample_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img, cmap="gray")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            mask = coco.annToMask(ann)
            cat_id = ann["category_id"]
            color = cat_colors.get(cat_id, (1, 1, 1, 0.5))

            # Overlay colored mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask == 1] = color
            ax.imshow(colored_mask)

        ax.set_title(f"{img_info['file_name']}\n{len(anns)} filaments", fontsize=9)
        ax.axis("off")

    # Legend
    patches = [mpatches.Patch(color=c[:3], label=cat_names[k]) for k, c in cat_colors.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=12)

    plt.suptitle("Sample Training Images with Filament Masks", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved sample images to {save_dir / 'sample_images.png'}")


def plot_filament_stats(coco: COCO, save_dir: Path = None):
    """Plot filament area distribution and filament-count-per-image distribution."""
    save_dir = save_dir or CFG.OUTPUT_DIR / "eda"
    save_dir.mkdir(parents=True, exist_ok=True)

    anns = coco.loadAnns(coco.getAnnIds())

    # ── Area distribution ──
    areas = [ann["area"] for ann in anns]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].hist(areas, bins=80, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Filament Area (px²)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Filament Area Distribution")
    axes[0].axvline(np.median(areas), color="red", linestyle="--", label=f"Median: {np.median(areas):.0f}")
    axes[0].legend()

    # Log-scale area
    axes[1].hist(np.log10(np.array(areas) + 1), bins=60, color="darkorange", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("log₁₀(Area + 1)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Filament Area (Log Scale)")

    # ── Filaments per image ──
    img_ids = coco.getImgIds()
    counts = [len(coco.getAnnIds(imgIds=img_id)) for img_id in img_ids]
    axes[2].hist(counts, bins=range(0, max(counts) + 2), color="seagreen", edgecolor="black", alpha=0.8)
    axes[2].set_xlabel("Number of Filaments")
    axes[2].set_ylabel("Number of Images")
    axes[2].set_title("Filaments per Image")

    plt.suptitle("Dataset Statistics", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "filament_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved stats to {save_dir / 'filament_stats.png'}")

    # ── Category distribution ──
    cat_counts = Counter(ann["category_id"] for ann in anns)
    cat_names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    print("\n[EDA] Category Distribution:")
    for cat_id, count in sorted(cat_counts.items()):
        print(f"  {cat_names.get(cat_id, cat_id):>20s}: {count:>5d} ({100*count/len(anns):.1f}%)")

    # ── Summary stats ──
    print(f"\n[EDA] Summary:")
    print(f"  Total images:      {len(img_ids)}")
    print(f"  Total annotations: {len(anns)}")
    print(f"  Avg filaments/img: {np.mean(counts):.1f}")
    print(f"  Min/Max count:     {min(counts)} / {max(counts)}")
    print(f"  Area min/max/mean: {min(areas):.0f} / {max(areas):.0f} / {np.mean(areas):.0f} px²")
    print(f"  Area median:       {np.median(areas):.0f} px²")


def plot_category_distribution(coco: COCO, save_dir: Path = None):
    """Pie chart of filament categories."""
    save_dir = save_dir or CFG.OUTPUT_DIR / "eda"
    save_dir.mkdir(parents=True, exist_ok=True)

    anns = coco.loadAnns(coco.getAnnIds())
    cat_counts = Counter(ann["category_id"] for ann in anns)
    cat_names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    labels = [cat_names[k] for k in sorted(cat_counts.keys())]
    sizes = [cat_counts[k] for k in sorted(cat_counts.keys())]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
    ax.set_title("Filament Category Distribution", fontsize=14)
    plt.savefig(save_dir / "category_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved category pie chart to {save_dir / 'category_distribution.png'}")


def main():
    """Run full EDA pipeline."""
    random.seed(CFG.SEED)
    print("=" * 60)
    print("  Solar Filament Segmentation — Exploratory Data Analysis")
    print("=" * 60)

    coco = load_coco(CFG.ANNOTATION_FILE)

    plot_sample_images(coco, CFG.TRAIN_IMG_DIR, n=6)
    plot_filament_stats(coco)
    plot_category_distribution(coco)

    print("\n[EDA] Done! Check output/eda/ for all visualizations.")


if __name__ == "__main__":
    main()
