"""
viewer.py — One-window viewer for filament segmentation results (no GPU).

Shows each image with the annotators' ground truth (left) and a model's
predictions (right), using the saved raw predictions
(`Result Phase 3/**/raw_preds_<run>_<split>.json.gz`) and the SAME
post-processing as the submission, so what you see is what Kaggle scores.

  Left  — H-alpha image + ground-truth outlines, one colour per annotator;
          a GT filament that no prediction matches (IoU > 0.5) is filled red (FN)
  Right — predictions: green = matches at least one annotator (TP),
          red = matches none (FP); optional confidence labels

Controls: ◀/▶ buttons or Left/Right arrow keys, model selector, check boxes,
"Worst PQ first" sort, jump box (index or image id), "Save PNG" (or key s)
-> output/figures/.

Usage:
  python src/viewer.py                       # val, best model
  python src/viewer.py --run baseline_r50    # start with another model
  python src/viewer.py --split test          # test images (predictions only)
"""

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from config import CFG
from evaluate import evaluate_official
from postprocess import close_all, select
from utils.rle import load_predictions

DISPLAY = 1024                      # displayed size (2048 / 2)
RESULTS_DIR = CFG.paths.ROOT / "Result Phase 3"
# Score threshold per model = its best on val (experiments/ablation_table.md)
RUN_THRESHOLDS = {"baseline_r50": 0.65, "r50_clahe_disk": 0.6, "r50_allann": 0.6}
RUN_ORDER = ["r50_allann", "baseline_r50", "r50_clahe_disk"]
ANNOTATOR_COLORS = [(0, 220, 255), (255, 210, 0), (255, 80, 255)]    # RGB: cyan, yellow, magenta
TP_COLOR, FP_COLOR, FN_COLOR = (40, 220, 60), (255, 50, 50), (255, 50, 50)


def find_runs(split: str) -> Dict[str, Path]:
    """run name -> raw prediction file for this split, best model first."""
    runs = {}
    for p in sorted(RESULTS_DIR.rglob(f"raw_preds_*_{split}.json.gz")):
        runs[p.name[len("raw_preds_"):-len(f"_{split}.json.gz")]] = p
    order = {n: i for i, n in enumerate(RUN_ORDER)}
    return dict(sorted(runs.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])))


@lru_cache(maxsize=8)
def _raw(path: str) -> Dict[str, List[dict]]:
    return load_predictions(Path(path))


@lru_cache(maxsize=2)
def _gt_entries(split: str) -> Dict[str, Dict[str, List[dict]]]:
    """stem -> {annotator entry id -> GT RLEs}; empty for the test split."""
    if split == "test":
        return {}
    from annotations import load_gt_entries
    out: Dict[str, Dict[str, List[dict]]] = {}
    for entry_id, rles in load_gt_entries().items():
        if rles:
            out.setdefault(entry_id.split("-", 1)[1], {})[entry_id] = rles
    return out


def stems_for(split: str) -> List[str]:
    if split == "test":
        return [p.stem for p in sorted(CFG.paths.TEST_IMG_DIR.glob("*.jpeg"))]
    from split import load_split
    return load_split()[1]


@lru_cache(maxsize=1024)
def predictions_for(stem: str, run_path: Path, run: str) -> List[dict]:
    """One image's predictions after the submission post-processing."""
    raw = _raw(str(run_path)).get(stem, [])
    closed = close_all({stem: raw}, CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0)
    thr = RUN_THRESHOLDS.get(run, CFG.infer.SCORE_THRESHOLD)
    return select(closed, thr, CFG.infer.MIN_AREA, drop_frac=CFG.infer.OVERLAP_DROP_FRAC)[stem]


def match(preds: List[dict], entries: Dict[str, List[dict]]) -> Tuple[List[bool], Dict[str, List[bool]]]:
    """(pred is TP vs any annotator, per entry: GT is matched) at IoU > 0.5."""
    pred_tp = [False] * len(preds)
    gt_hit = {}
    for entry_id, gt in entries.items():
        if preds:
            hit = np.asarray(coco_mask.iou(preds, gt, [0] * len(gt)), dtype=float) > 0.5   # (P, G)
            pred_tp = [a or b for a, b in zip(pred_tp, hit.any(axis=1))]
            gt_hit[entry_id] = hit.any(axis=0).tolist()
        else:
            gt_hit[entry_id] = [False] * len(gt)
    return pred_tp, gt_hit


def _small(rle: dict) -> np.ndarray:
    step = CFG.data.IMAGE_SIZE[0] // DISPLAY
    return np.ascontiguousarray(coco_mask.decode(rle)[::step, ::step])


def _fill(img: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    img[mask > 0] = ((1 - alpha) * img[mask > 0] + alpha * np.array(color)).astype(np.uint8)


def _outline(img: np.ndarray, mask: np.ndarray, color, thickness: int = 2) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, tuple(int(c) for c in color), thickness)


def load_image(stem: str, split: str) -> np.ndarray:
    d = CFG.paths.TRAIN_IMG_DIR if split != "test" else CFG.paths.TEST_IMG_DIR
    gray = cv2.imread(str(d / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def render(stem: str, split: str, run: str, run_path: Path, show_gt: bool = True,
           show_pred: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """(left RGB, right RGB, stats) for one image; no GUI involved."""
    base = load_image(stem, split)
    preds = predictions_for(stem, run_path, run)
    entries = _gt_entries(split).get(stem, {})
    pred_tp, gt_hit = match(preds, entries)

    left, right = base.copy(), base.copy()
    if show_gt:
        for k, (entry_id, gt) in enumerate(entries.items()):
            color = ANNOTATOR_COLORS[k % len(ANNOTATOR_COLORS)]
            for rle, hit in zip(gt, gt_hit[entry_id]):
                m = _small(rle)
                if not hit:
                    _fill(left, m, FN_COLOR, 0.35)
                _outline(left, m, color)
    labels = []
    if show_pred:
        for rle, tp in zip(preds, pred_tp):
            m = _small(rle)
            color = TP_COLOR if (tp or not entries) else FP_COLOR
            _fill(right, m, color, 0.40)
            _outline(right, m, color)
            cx, cy = rle["centroid"]
            scale = DISPLAY / CFG.data.IMAGE_SIZE[0]
            labels.append((cx * scale, cy * scale, rle.get("score", 1.0)))

    stats = {"stem": stem, "station": stem[14] if len(stem) > 14 else "?", "n_pred": len(preds),
             "n_annotators": len(entries), "labels": labels}
    if entries:
        r = evaluate_official(entries, {stem: preds})
        stats.update(PQ=r["PQ"], TP=r["TP"], FP=r["FP"], FN=r["FN"])
    return left, right, stats


def image_pq(stem: str, run: str, run_path: Path) -> float:
    entries = _gt_entries("val").get(stem, {})
    return evaluate_official(entries, {stem: predictions_for(stem, run_path, run)})["PQ"] if entries else 1.0


class Viewer:
    """Single matplotlib window; every control redraws the same two panels."""

    def __init__(self, split: str, run: Optional[str], backend: str = "TkAgg"):
        import matplotlib
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

        self.plt, self.split = plt, split
        self.runs = find_runs(split)
        if not self.runs:
            raise SystemExit(f"No raw_preds_*_{split}.json.gz under {RESULTS_DIR}")
        self.run = run if run in self.runs else next(iter(self.runs))
        self.stems = stems_for(split)
        self.order, self.idx = list(self.stems), 0
        self.sorted_by_pq = False
        self.show = {"Ground truth": split != "test", "Predictions": True, "Scores": False}

        self.fig = plt.figure(figsize=(16, 9.2))
        self.fig.canvas.manager.set_window_title("Solar filament viewer")
        self.ax_l = self.fig.add_axes([0.01, 0.17, 0.485, 0.75])
        self.ax_r = self.fig.add_axes([0.505, 0.17, 0.485, 0.75])

        self.b_prev = Button(self.fig.add_axes([0.01, 0.04, 0.07, 0.06]), "◀ Prev")
        self.b_next = Button(self.fig.add_axes([0.09, 0.04, 0.07, 0.06]), "Next ▶")
        self.r_run = RadioButtons(self.fig.add_axes([0.18, 0.01, 0.16, 0.13]), list(self.runs),
                                  active=list(self.runs).index(self.run))
        self.c_show = CheckButtons(self.fig.add_axes([0.36, 0.01, 0.13, 0.13]), list(self.show),
                                   list(self.show.values()))
        self.b_sort = Button(self.fig.add_axes([0.51, 0.04, 0.12, 0.06]), "Worst PQ first")
        self.t_jump = TextBox(self.fig.add_axes([0.70, 0.04, 0.12, 0.06]), "Go to # / id ")
        self.b_save = Button(self.fig.add_axes([0.85, 0.04, 0.12, 0.06]), "Save PNG")
        if split == "test":
            self.b_sort.label.set_text("(no GT on test)")

        self.b_prev.on_clicked(lambda _: self.step(-1))
        self.b_next.on_clicked(lambda _: self.step(+1))
        self.r_run.on_clicked(self.set_run)
        self.c_show.on_clicked(self.toggle)
        self.b_sort.on_clicked(self.toggle_sort)
        self.t_jump.on_submit(self.jump)
        self.b_save.on_clicked(lambda _: self.save())
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    # --- controls -------------------------------------------------------
    def step(self, d: int):
        self.idx = (self.idx + d) % len(self.order)
        self.draw()

    def set_run(self, label: str):
        self.run = label
        if self.sorted_by_pq:
            self.sort_by_pq()
        self.draw()

    def toggle(self, label: str):
        self.show[label] = not self.show[label]
        self.draw()

    def toggle_sort(self, _):
        if self.split == "test":
            return
        self.sorted_by_pq = not self.sorted_by_pq
        if self.sorted_by_pq:
            self.sort_by_pq()
            self.b_sort.label.set_text("File order")
        else:
            self.order = list(self.stems)
            self.b_sort.label.set_text("Worst PQ first")
        self.idx = 0
        self.draw()

    def sort_by_pq(self):
        self.fig.suptitle(f"Scoring all {len(self.stems)} images for {self.run} ...", fontsize=13)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        path = self.runs[self.run]
        self.order = sorted(self.stems, key=lambda s: image_pq(s, self.run, path))

    def jump(self, text: str):
        text = text.strip()
        if text.isdigit() and 1 <= int(text) <= len(self.order):
            self.idx = int(text) - 1
        elif text in self.order:
            self.idx = self.order.index(text)
        else:
            return
        self.draw()

    def on_key(self, event):
        if event.key == "right":
            self.step(+1)
        elif event.key == "left":
            self.step(-1)
        elif event.key == "s":
            self.save()

    def save(self):
        out = CFG.paths.OUTPUT_DIR / "figures"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"viewer_{self.split}_{self.run}_{self.order[self.idx]}.png"
        self.fig.savefig(path, dpi=110)
        print(f"[Viewer] saved {path}")

    # --- drawing --------------------------------------------------------
    def draw(self):
        stem = self.order[self.idx]
        left, right, st = render(stem, self.split, self.run, self.runs[self.run],
                                 self.show["Ground truth"], self.show["Predictions"])
        for ax in (self.ax_l, self.ax_r):
            ax.clear()
            ax.set_axis_off()
        self.ax_l.imshow(left)
        self.ax_r.imshow(right)
        n_ann = st["n_annotators"]
        self.ax_l.set_title("Ground truth" + (f" — {n_ann} annotator(s): cyan / yellow / magenta; red fill = missed (FN)"
                                              if n_ann else " — none (test image)"), fontsize=10)
        thr = RUN_THRESHOLDS.get(self.run, CFG.infer.SCORE_THRESHOLD)
        self.ax_r.set_title(f"{self.run} predictions (thr {thr}) — green = correct (TP), red = extra (FP)"
                            if n_ann else f"{self.run} predictions (thr {thr})", fontsize=10)
        if self.show["Scores"]:
            for x, y, s in st["labels"]:
                self.ax_r.text(x, y, f"{s:.2f}", color="white", fontsize=8, ha="center", va="center",
                               bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
        head = f"{stem}   ({self.split} {self.idx + 1}/{len(self.order)})   station {st['station']}   "
        head += f"{st['n_pred']} predicted"
        if "PQ" in st:
            head += f"   ·   image PQ {st['PQ']:.2f}   TP {st['TP']}  FP {st['FP']}  FN {st['FN']}"
        if self.sorted_by_pq:
            head += "   [worst PQ first]"
        self.fig.suptitle(head, fontsize=13)
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="One-window filament segmentation viewer")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--run", default=None, help="Model to show first (default: best available)")
    args = parser.parse_args()
    viewer = Viewer(args.split, args.run)
    viewer.plt.show()


if __name__ == "__main__":
    main()
