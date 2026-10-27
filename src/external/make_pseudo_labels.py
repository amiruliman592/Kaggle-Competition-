"""
make_pseudo_labels.py — Phase 6 M3: raw predictions on external images -> COCO pseudo-labels.

Labeller: x101_2048_m28 alone, one view. Chosen on val (2026-10-27): with the usual
post-processing, thr 0.85 keeps 84% correct masks (matched by some annotator,
IoU > 0.5) at 4.7 masks/img, recall 0.53. Flip TTA of m28 + r50 is only ~2 points
more precise at the same recall for 8x the T4 time.

Masks go through the usual post-processing (close 5, min area 50, overlap removal
with drop 0.2); images with no kept mask are dropped. Every annotation also keeps its
"score", and the raw file (score >= 0.2) stays next to it, so S5 can pick another
threshold or ignore unsure regions: at thr 0.85 about half the filaments of an image
are NOT labelled, so unlabelled pixels are not reliable background.

Usage:
  python src/external/make_pseudo_labels.py labels --raw raw_preds_x101_2048_m28_part*.json.gz \
      --out pseudo_labels_v1.json [--thr 0.85]
  python src/external/make_pseudo_labels.py score-val --raw raw_preds_x101_2048_m28_valcopies.json.gz
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from pycocotools import mask as coco_mask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG  # noqa: E402
from postprocess import close_all, select  # noqa: E402
from utils.rle import load_predictions  # noqa: E402

DEFAULT_THR = 0.85


def postprocess(raw: Dict[str, List[dict]], thr: float) -> Dict[str, List[dict]]:
    closed = close_all(raw, CFG.infer.MORPH_KERNEL if CFG.infer.MORPH_CLOSE else 0)
    return select(closed, thr, CFG.infer.MIN_AREA, drop_frac=CFG.infer.OVERLAP_DROP_FRAC)


def to_coco(final: Dict[str, List[dict]], thr: float, labeller: str) -> Dict:
    """stem -> RLE dicts -> COCO instance JSON (RLE segmentation); images with no mask are left out."""
    images, anns = [], []
    for stem in sorted(final):
        if not final[stem]:
            continue
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": f"{stem}.jpg", "width": 2048, "height": 2048})
        for r in final[stem]:
            counts = r["counts"].decode("utf-8") if isinstance(r["counts"], bytes) else r["counts"]
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                         "segmentation": {"size": [2048, 2048], "counts": counts},
                         "area": int(coco_mask.area(r)), "bbox": [float(v) for v in coco_mask.toBbox(r)],
                         "score": round(float(r["score"]), 4)})
    return {"info": {"description": "Phase 6 M3 pseudo-labels (GONG hag, not human labels)",
                     "labeller": labeller, "score_threshold": thr,
                     "postprocess": f"close{CFG.infer.MORPH_KERNEL}_minarea{CFG.infer.MIN_AREA}"
                                    f"_overlapdrop{CFG.infer.OVERLAP_DROP_FRAC}"},
            "images": images, "annotations": anns, "categories": [{"id": 1, "name": "filament"}]}


def score_val(raw_path: Path, thr: float = 0.6):
    """Official PQ / Dice of predictions on the archive copies of val (keys = val stems)."""
    from annotations import load_gt_entries
    from evaluate import evaluate_official, official_dice
    from split import load_split

    _, stems = load_split()
    loaded = load_predictions(raw_path)
    missing = [s for s in stems if s not in loaded]
    final = postprocess({s: loaded.get(s, []) for s in stems}, thr)
    entries = load_gt_entries(stems)
    res = evaluate_official(entries, final)
    dice = official_dice(entries, final)["Dice_samplewise"]
    print(f"[Val copies] {len(stems) - len(missing)}/{len(stems)} images, thr {thr}: "
          f"official PQ {res['PQ']:.4f} (SQ {res['SQ']:.4f}, RQ {res['RQ']:.4f}), Dice {dice:.4f}")
    return res


def main():
    p = argparse.ArgumentParser(description="Pseudo-labels from raw predictions (Phase 6 M3)")
    sub = p.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("labels")
    lab.add_argument("--raw", type=Path, nargs="+", required=True, help="one or more raw files (parts)")
    lab.add_argument("--out", type=Path, required=True)
    lab.add_argument("--thr", type=float, default=DEFAULT_THR)
    lab.add_argument("--labeller", default="x101_2048_m28")
    sv = sub.add_parser("score-val")
    sv.add_argument("--raw", type=Path, required=True)
    sv.add_argument("--thr", type=float, default=0.6)
    args = p.parse_args()
    if args.cmd == "score-val":
        score_val(args.raw, args.thr)
        return
    raw = {}
    for path in args.raw:
        raw.update(load_predictions(path))
    final = postprocess(raw, args.thr)
    coco = to_coco(final, args.thr, args.labeller)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(coco))
    n_img, n_ann = len(coco["images"]), len(coco["annotations"])
    print(f"[Pseudo-labels] {len(final)} images in, {n_img} with masks, {n_ann} masks "
          f"({n_ann / max(n_img, 1):.1f}/img), thr {args.thr} -> {args.out}")


if __name__ == "__main__":
    main()
