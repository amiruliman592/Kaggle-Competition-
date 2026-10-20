"""
make_metric_check_csvs.py — Build a GT/prediction CSV pair for the Kaggle
self-evaluation notebook (Phase 2, step 2A: metric alignment).

No model is needed: on validation files annotated by 2+ annotators, set 0 is
written as ground truth and set 1 as "predictions". Both CSVs use the
submission format, and our own evaluator scores the same pair so the numbers
can be compared with the official notebook (target: within ±0.001 PQ).

Usage:
  python src/make_metric_check_csvs.py
  -> output/metric_check/{gt.csv, pred.csv, local_results.json}
"""

import json

from annotations import _load_json, anns_to_rles, load_annotation_sets
from config import CFG
from evaluate import evaluate, load_submission_rles
from split import load_split


def write_csv(path, stem_to_rles, id_prefix=None):
    """id_prefix: stem -> prefix; the official GT ids are "<annotator>-<stem>_<i>"."""
    with open(path, "w", newline="\n") as f:
        f.write("filament_id,segmentation_rle\n")
        for stem, rles in stem_to_rles.items():
            image_id = f"{id_prefix[stem]}-{stem}" if id_prefix else stem
            for i, rle in enumerate(rles, start=1):
                f.write(f"{image_id}_{i},{rle['counts'].decode('utf-8')}\n")


def first_entry_prefixes():
    """stem -> annotator prefix of its first COCO entry (same order as load_annotation_sets)."""
    data = _load_json(str(CFG.paths.ANNOTATION_FILE))
    prefixes = {}
    for entry_id in sorted(im["id"] for im in data["images"]):
        prefix, stem = entry_id.split("-", 1)
        prefixes.setdefault(stem, prefix)
    return prefixes


def main():
    _, val_ids = load_split()
    sets = load_annotation_sets()
    stems = [s for s in val_ids if len(sets[s]) >= 2]

    gt = {s: anns_to_rles(sets[s][0]) for s in stems}
    pred = {s: anns_to_rles(sets[s][1]) for s in stems}

    out_dir = CFG.paths.OUTPUT_DIR / "metric_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "gt.csv", gt, id_prefix=first_entry_prefixes())
    write_csv(out_dir / "pred.csv", pred)

    # Score the CSVs as written (round trip), exactly what Kaggle will see.
    gt_rt = {k.split("-", 1)[1]: v for k, v in load_submission_rles(out_dir / "gt.csv").items()}
    res = evaluate(gt_rt, load_submission_rles(out_dir / "pred.csv"))
    (out_dir / "local_results.json").write_text(json.dumps(res, indent=1))
    print(f"{len(stems)} val files, {sum(map(len, gt.values()))} GT / "
          f"{sum(map(len, pred.values()))} pred instances -> {out_dir}")


if __name__ == "__main__":
    main()
