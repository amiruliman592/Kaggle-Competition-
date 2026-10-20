"""
fix_overlaps.py — Make an existing submission CSV Kaggle-valid by removing
mask overlaps (no GPU / model needed).

Rows of one image must be in score order (highest first), as written by
utils.rle.write_submission_csv: each mask keeps only pixels not claimed by an
earlier row. Masks left smaller than --min-area are dropped.

Usage:
  python src/fix_overlaps.py in.csv out.csv [--min-area 50]
"""

import argparse
from pathlib import Path

from config import CFG
from evaluate import load_submission_rles
from utils.rle import has_overlap, remove_overlaps, write_submission_csv
from validate_submission import validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--min-area", type=int, default=CFG.infer.MIN_AREA)
    args = parser.parse_args()

    preds = load_submission_rles(args.src)
    n_fixed = sum(has_overlap(r) for r in preds.values())
    fixed = {stem: remove_overlaps(rles, args.min_area) for stem, rles in preds.items()}
    n_before = sum(map(len, preds.values()))
    n_after = sum(map(len, fixed.values()))
    print(f"[FixOverlaps] {n_fixed} images had overlaps; masks {n_before} -> {n_after}")
    write_submission_csv(fixed, args.dst)
    validate_submission(args.dst)


if __name__ == "__main__":
    main()
