"""
visualize.py — Visualize model predictions overlaid on images.

Useful for sanity-checking predictions before submitting.

Usage:
  python src/visualize.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import random

from config import CFG


def visualize_predictions(
    img_dir: Path,
    predictions: Dict[str, List[np.ndarray]],
    n: int = 6,
    save_path: str = None,
):
    """
    Visualize predicted masks overlaid on images.
    
    Args:
        img_dir: Path to image directory
        predictions: Dict mapping image_id -> list of binary masks
        n: Number of images to visualize
        save_path: Where to save the figure (optional)
    """
    img_ids = list(predictions.keys())
    sample_ids = random.sample(img_ids, min(n, len(img_ids)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Generate random colors for each filament
    cmap = plt.cm.Set1
    
    for ax, img_id in zip(axes, sample_ids):
        img_path = img_dir / f"{img_id}.jpeg"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img, cmap="gray")
        
        masks = predictions[img_id]
        for i, mask in enumerate(masks):
            color = cmap(i % 9)
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask > 0] = (*color[:3], 0.4)
            ax.imshow(colored_mask)
            
            # Draw contour
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                cnt = cnt.squeeze()
                if cnt.ndim == 2 and len(cnt) > 2:
                    ax.plot(cnt[:, 0], cnt[:, 1], color=color[:3], linewidth=1.5)
        
        ax.set_title(f"{img_id}\n{len(masks)} filaments predicted", fontsize=10)
        ax.axis("off")
    
    plt.suptitle("Model Predictions on Test Images", fontsize=16)
    plt.tight_layout()
    
    if save_path is None:
        save_path = str(CFG.OUTPUT_DIR / "predictions_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved predictions visualization to {save_path}")


def visualize_gt_vs_pred(
    img_path: Path,
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    save_path: str = None,
):
    """
    Side-by-side comparison of ground truth and predictions for one image.
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ground truth
    ax1.imshow(img, cmap="gray")
    for i, mask in enumerate(gt_masks):
        color = plt.cm.Set1(i % 9)
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = (*color[:3], 0.5)
        ax1.imshow(colored_mask)
    ax1.set_title(f"Ground Truth ({len(gt_masks)} filaments)", fontsize=14)
    ax1.axis("off")
    
    # Predictions
    ax2.imshow(img, cmap="gray")
    for i, mask in enumerate(pred_masks):
        color = plt.cm.Set2(i % 8)
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = (*color[:3], 0.5)
        ax2.imshow(colored_mask)
    ax2.set_title(f"Predictions ({len(pred_masks)} filaments)", fontsize=14)
    ax2.axis("off")
    
    plt.suptitle(img_path.stem, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
