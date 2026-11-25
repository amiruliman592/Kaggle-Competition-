"""Reproduce the submitted R50 baseline from the archived raw predictions."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ablate_postprocess import score
from annotations import load_gt_entries
from postprocess import close_all, select
from utils.rle import load_predictions

PREDICTIONS = ROOT / "results" / "phase3" / "step1"


def baseline_masks():
    raw = load_predictions(PREDICTIONS / "raw_preds_baseline_r50_val.json.gz")
    # Closing is independent for each mask. Filter first to skip masks below 0.80.
    retained = {stem: [mask for mask in masks if mask["score"] >= 0.8] for stem, masks in raw.items()}
    return select(close_all(retained, 5), score_threshold=0.8, min_area=0, drop_frac=None)


def main():
    split = json.loads((ROOT / "splits" / "val_ids.json").read_text(encoding="utf-8"))
    validation_ids = split["val_ids"]
    masks = baseline_masks()
    predictions = {stem: masks.get(stem, []) for stem in validation_ids}
    entries = load_gt_entries(validation_ids)
    results = {"images": len(validation_ids), "annotator_sets": len(entries), **score(entries, predictions)}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
