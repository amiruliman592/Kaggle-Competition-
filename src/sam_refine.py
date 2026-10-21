"""
sam_refine.py — Phase 5 step 1: SAM 2 box-prompt refinement of saved raw
Mask R-CNN predictions (run_pipeline.py --step dump). No Detectron2 needed.

SAM 2 redraws every predicted mask from its box. Each mask gets its own square
crop around the box (SAM works at 1024 px: on the whole 2048 disk a ~90 px
filament would be drawn on a ~45 px grid). Two prompts per mask:
  box     the mask's bounding box only
  boxpts  box + 3 positive points along the mask's long axis (a thin filament
          fills little of its box, the points say which part is the object)
SAM masks are saved in the raw file's order, so they pair 1:1 with the
originals; the combinations below are then scored on CPU:
  sam     SAM mask replaces the original
  union   original OR SAM
  inter   original AND SAM (original if they do not overlap)
  gate    SAM only where it agrees with the original (IoU >= 0.5), else original

Usage:
  # Colab GPU (notebooks/colab_phase5_sam2.ipynb): one file per prompt and split
  python src/sam_refine.py dump --raw "Result Phase 4/raw_preds_x101_2048_val.json.gz" --split val
  # local CPU: official PQ / Dice of every prompt x combination x score threshold
  python src/sam_refine.py ablate --raw "Result Phase 4/raw_preds_x101_2048_val.json.gz" \
      --sam-dir "Result Phase 5" --out "Result Phase 5/sam_ablation_x101_2048.json"
  # local CPU: chosen combination as a normal raw file (-> submit_from_raw.py, viewer)
  python src/sam_refine.py export --raw "Result Phase 4/raw_preds_x101_2048_test.json.gz" \
      --sam "Result Phase 5/sam_box_x101_2048_test.json.gz" --mode gate \
      --out "Result Phase 5/raw_preds_x101_2048_sam_box_gate_test.json.gz"
"""

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from config import CFG
from utils.rle import IMAGE_HW, counts_to_rle, encode_mask, load_predictions, save_predictions

PROMPTS = ["box", "boxpts"]
MODES = ["sam", "union", "inter", "gate"]
GATE_IOU = 0.5
SAM_MIN_SCORE = 0.5        # masks below are never kept (best thr 0.6-0.8), so not refined
CROP_MIN = 256             # crop side (px) before SAM's resize to 1024
CROP_SCALE = 1.5           # crop side = 1.5 x longest box side (context around the filament)
SCORE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
SAM_MODEL = "facebook/sam2.1-hiera-large"


def crop_window(bbox: Sequence[float], hw: Sequence[int] = IMAGE_HW, min_side: int = CROP_MIN,
                scale: float = CROP_SCALE):
    """Square crop (x0, y0, x1, y1) centred on a COCO [x, y, w, h] box, shifted to stay inside the image."""
    x, y, w, h = bbox
    side = int(min(max(min_side, np.ceil(scale * max(w, h))), min(hw)))
    x0 = int(np.clip(round(x + w / 2 - side / 2), 0, hw[1] - side))
    y0 = int(np.clip(round(y + h / 2 - side / 2), 0, hw[0] - side))
    return x0, y0, x0 + side, y0 + side


def mask_points(mask: np.ndarray, n: int = 3) -> np.ndarray:
    """
    Up to n positive (x, y) points spread along the mask's long axis: the mask
    is cut into n equal slices along that axis and each slice gives its pixel
    farthest from the mask edge.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)[ys, xs]
    pts = np.stack([xs, ys], 1).astype(np.float32)
    centred = pts - pts.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    proj = centred @ axis
    edges = np.linspace(proj.min(), proj.max() + 1e-6, n + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((proj >= lo) & (proj < hi))
        if len(idx):
            out.append(pts[idx[np.argmax(depth[idx])]])
    return np.array(out, np.float32)


def combine(orig: Dict, sam: Dict, mode: str, gate_iou: float = GATE_IOU) -> Dict:
    """One original mask + its SAM mask -> combined RLE dict carrying the original score."""
    if mode == "sam":
        out = sam
    elif mode == "union":
        out = coco_mask.merge([orig, sam], intersect=False)
    elif mode == "inter":
        out = coco_mask.merge([orig, sam], intersect=True)
        if not coco_mask.area(out):
            out = orig
    elif mode == "gate":
        out = sam if coco_mask.iou([sam], [orig], [0])[0][0] >= gate_iou else orig
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return {"size": out["size"], "counts": out["counts"], "score": orig["score"]}


def save_sam(preds: Dict[str, List[Dict]], path: Path) -> None:
    """stem -> [[counts, score, sam_iou], ...] in list order (sam_iou -1 = SAM gave nothing, original kept)."""
    data = {stem: [[r["counts"].decode("utf-8") if isinstance(r["counts"], bytes) else r["counts"],
                    float(r["score"]), float(r["sam_iou"])] for r in rles]
            for stem, rles in preds.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def load_sam(path: Path, hw: Sequence[int] = IMAGE_HW) -> Dict[str, List[Dict]]:
    """Inverse of save_sam; keeps file order (= raw order) so masks pair with the originals."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for stem, items in data.items():
        out[stem] = []
        for counts, score, sam_iou in items:
            rle = counts_to_rle(counts, hw)
            rle.update(score=score, sam_iou=sam_iou)
            out[stem].append(rle)
    return out


def load_raw(path: Path, min_score: float = SAM_MIN_SCORE) -> Dict[str, List[Dict]]:
    """Raw predictions above min_score, highest score first (the order SAM files follow)."""
    return {s: [r for r in rles if r["score"] >= min_score] for s, rles in load_predictions(path).items()}


def paired(raw: Dict[str, List[Dict]], sam: Dict[str, List[Dict]], mode: str) -> Dict[str, List[Dict]]:
    """Combine every original mask with its SAM mask (same image, same position)."""
    out = {}
    for stem, rles in raw.items():
        s = sam.get(stem, [])
        assert [r["score"] for r in rles] == [r["score"] for r in s], f"{stem}: SAM file does not match raw file"
        out[stem] = [combine(o, m, mode) for o, m in zip(rles, s)]
    return out


def sam_name(prompt: str, raw_path: Path) -> str:
    """raw_preds_x101_2048_val.json.gz -> sam_<prompt>_x101_2048_val.json.gz"""
    return f"sam_{prompt}_" + Path(raw_path).name.removeprefix("raw_preds_")


# ═══════════════════════════════════════════════════════════════════════
# dump (GPU)
# ═══════════════════════════════════════════════════════════════════════

def dump(raw_path: Path, split: str, out_dir: Path, model_id: str = SAM_MODEL, limit: int = 0):
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from tqdm import tqdm

    from preprocess import image_path

    raw = load_raw(raw_path)
    stems = sorted(raw)[:limit] if limit else sorted(raw)
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    # bf16 only where the GPU has it natively (A100: yes, T4: no -> fp32)
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    out = {p: {} for p in PROMPTS}
    n_masks, n_empty, t0 = 0, 0, time.time()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        for stem in tqdm(stems, desc=f"SAM 2 {split}"):
            img = cv2.cvtColor(cv2.imread(str(image_path(stem, "train" if split == "val" else "test"))),
                               cv2.COLOR_BGR2RGB)
            for p in PROMPTS:
                out[p][stem] = []
            for r in raw[stem]:
                x, y, w, h = coco_mask.toBbox(r)
                x0, y0, x1, y1 = crop_window((x, y, w, h), img.shape[:2])
                orig_crop = coco_mask.decode(r)[y0:y1, x0:x1]
                predictor.set_image(img[y0:y1, x0:x1])
                box = np.array([x - x0, y - y0, x + w - x0, y + h - y0], np.float32)
                pts = mask_points(orig_crop)
                for p in PROMPTS:
                    kw = {"box": box, "multimask_output": False}
                    if p == "boxpts" and len(pts):
                        kw.update(point_coords=pts, point_labels=np.ones(len(pts), np.int32))
                    masks, ious, _ = predictor.predict(**kw)
                    m = masks[0] > 0.5
                    if m.any():
                        full = np.zeros(img.shape[:2], np.uint8)
                        full[y0:y1, x0:x1] = m
                        new = encode_mask(full)
                        new.update(score=r["score"], sam_iou=float(ious[0]))
                    else:                                  # keep the pairing: fall back to the original
                        n_empty += 1
                        new = {"size": r["size"], "counts": r["counts"], "score": r["score"], "sam_iou": -1.0}
                    out[p][stem].append(new)
                n_masks += 1

    sec = time.time() - t0
    for p in PROMPTS:
        path = Path(out_dir) / sam_name(p, raw_path)
        save_sam(out[p], path)
        print(f"[SAM] {p}: {n_masks} masks -> {path}")
    print(f"[SAM] {split}: {len(stems)} images, {n_masks} masks, {sec / max(n_masks, 1):.3f} s/mask, "
          f"{n_empty} empty SAM masks (original kept)")
    if torch.cuda.is_available():
        print(f"[SAM] Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════
# ablate / export (CPU)
# ═══════════════════════════════════════════════════════════════════════

def ablate(raw_path: Path, sam_dir: Path, out: Path):
    from ablate_postprocess import score
    from annotations import load_gt_entries
    from postprocess import close_all, select
    from split import load_split

    _, val_ids = load_split()
    entries = load_gt_entries(val_ids)
    raw = load_raw(raw_path)
    raw = {s: raw.get(s, []) for s in val_ids}
    variants = [("orig", "orig", raw)]
    for p in PROMPTS:
        path = Path(sam_dir) / sam_name(p, raw_path)
        if not path.exists():
            print(f"[SAM] missing {path}, skipped")
            continue
        sam = load_sam(path)
        variants += [(p, mode, paired(raw, sam, mode)) for mode in MODES]

    close_k = CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0
    rows = []
    for prompt, mode, preds in variants:
        closed = close_all(preds, close_k)
        for thr in SCORE_THRESHOLDS:
            r = {"prompt": prompt, "mode": mode, "score_threshold": thr,
                 **score(entries, select(closed, thr, CFG.infer.MIN_AREA, drop_frac=CFG.infer.OVERLAP_DROP_FRAC))}
            rows.append(r)
            print(f"{prompt:6s} {mode:5s} thr={thr:.1f}  PQ={r['PQ']:.4f} SQ={r['SQ']:.4f} RQ={r['RQ']:.4f} "
                  f"Dice={r['Dice']:.4f} n={r['n_pred']}", flush=True)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    best = max(rows, key=lambda r: r["PQ"])
    print(f"\nBest PQ {best['PQ']:.4f} (Dice {best['Dice']:.4f}): {best['prompt']} {best['mode']} "
          f"thr={best['score_threshold']} -> {out}")


def export(raw_path: Path, sam_path: Path, mode: str, out: Path):
    preds = paired(load_raw(raw_path), load_sam(sam_path), mode)
    save_predictions(preds, out)
    print(f"[SAM] {mode}: {sum(map(len, preds.values()))} masks -> {out}")


def main():
    parser = argparse.ArgumentParser(description="SAM 2 refinement of raw predictions")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="GPU: SAM 2 masks for every raw mask (score >= 0.5)")
    d.add_argument("--raw", type=Path, required=True)
    d.add_argument("--split", choices=["val", "test"], required=True)
    d.add_argument("--out-dir", type=Path, default=None, help="default: CFG.paths.OUTPUT_DIR")
    d.add_argument("--model", default=SAM_MODEL)
    d.add_argument("--limit", type=int, default=0, help="first N images only (probe)")
    a = sub.add_parser("ablate", help="CPU: score prompt x combination x threshold on val")
    a.add_argument("--raw", type=Path, required=True)
    a.add_argument("--sam-dir", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("export", help="CPU: write one combination as a normal raw prediction file")
    e.add_argument("--raw", type=Path, required=True)
    e.add_argument("--sam", type=Path, required=True)
    e.add_argument("--mode", choices=MODES, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "dump":
        dump(args.raw, args.split, args.out_dir or CFG.paths.OUTPUT_DIR, args.model, args.limit)
    elif args.cmd == "ablate":
        ablate(args.raw, args.sam_dir, args.out)
    else:
        export(args.raw, args.sam, args.mode, args.out)


if __name__ == "__main__":
    main()
