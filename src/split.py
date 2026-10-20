"""
split.py — Leak-free, frozen train/val split.

Filaments persist for days and several GONG stations image the same disk
within hours (276 of 656 training dates have another image <=1 day away), so
splitting by image or even by date leaks the same filament into both sets.
We group images into SPLIT_BLOCK_DAYS-day blocks (~1 solar rotation) and run
StratifiedGroupKFold stratified on filament-count bins. Fold SPLIT_VAL_FOLD
becomes validation.

The result is written once to splits/val_ids.json and committed. It is never
regenerated unless --force is passed.

Usage:
  python src/split.py            # create (refuses to overwrite)
  python src/split.py --force    # regenerate
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from annotations import load_annotation_sets, select_annotations
from config import CFG


def time_block(stem: str, block_days: int) -> int:
    """'20140609195854Bh' -> index of its block_days-long time block."""
    day = dt.date(int(stem[:4]), int(stem[4:6]), int(stem[6:8])).toordinal()
    return day // block_days


def count_bins(counts: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Quantile bins of per-image filament counts (stratification label)."""
    edges = np.unique(np.quantile(counts, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.digitize(counts, edges)


def make_split(seed: Optional[int] = None) -> Dict:
    seed = CFG.SEED if seed is None else seed
    selected = select_annotations(load_annotation_sets())
    stems = sorted(selected)
    counts = np.array([len(selected[s]) for s in stems])
    groups = np.array([time_block(s, CFG.data.SPLIT_BLOCK_DAYS) for s in stems])

    sgkf = StratifiedGroupKFold(n_splits=CFG.data.SPLIT_N_FOLDS, shuffle=True, random_state=seed)
    folds = list(sgkf.split(stems, count_bins(counts), groups))
    train_idx, val_idx = folds[CFG.data.SPLIT_VAL_FOLD]

    train_ids = [stems[i] for i in train_idx]
    val_ids = [stems[i] for i in val_idx]
    assert not set(groups[train_idx]) & set(groups[val_idx]), "time-block leakage"

    return {
        "method": f"StratifiedGroupKFold(n={CFG.data.SPLIT_N_FOLDS}, fold={CFG.data.SPLIT_VAL_FOLD}), "
                  f"groups={CFG.data.SPLIT_BLOCK_DAYS}-day blocks, strata=filament-count quantile bins",
        "seed": seed,
        "annotator_policy": CFG.data.ANNOTATOR_POLICY,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_mean_filaments": round(float(counts[train_idx].mean()), 3),
        "val_mean_filaments": round(float(counts[val_idx].mean()), 3),
        "train_ids": train_ids,
        "val_ids": val_ids,
    }


def load_split(path: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    """Returns (train_stems, val_stems) from the frozen split file."""
    path = Path(path or CFG.paths.SPLIT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python src/split.py` once and commit it.")
    with open(path) as f:
        split = json.load(f)
    return split["train_ids"], split["val_ids"]


def main():
    parser = argparse.ArgumentParser(description="Create the frozen train/val split")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing split")
    args = parser.parse_args()

    path = CFG.paths.SPLIT_FILE
    if path.exists() and not args.force:
        print(f"[Split] {path} already exists and is frozen. Use --force to regenerate.")
        return
    split = make_split()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=1)
    print(f"[Split] {split['method']}")
    print(f"[Split] train={split['n_train']} (mean {split['train_mean_filaments']} filaments/img), "
          f"val={split['n_val']} (mean {split['val_mean_filaments']}) -> {path}")


if __name__ == "__main__":
    main()
