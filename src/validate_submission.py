"""
validate_submission.py — Check a submission CSV before uploading to Kaggle.

Checks:
  1. Header is exactly `filament_id,segmentation_rle`
  2. Every filament_id is `<image_id>_<suffix>`, unique, and <image_id> is a test image
  3. RLE counts are bare strings (no quotes, brackets or size array)
  4. Every RLE parses with a non-zero area; a random sample decodes to a 2048×2048 binary mask

Usage:
  python src/validate_submission.py submissions/submission_baseline_r50.csv
"""

import argparse
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from pycocotools import mask as coco_mask

from config import CFG
from utils.rle import counts_to_rle

HEADER = "filament_id,segmentation_rle"
ID_RE = re.compile(r"^(?P<image>.+)_(?P<idx>[A-Za-z0-9]+)$")
# COCO counts use ASCII 48-111 (includes '[' and ']'), so only quotes/whitespace are illegal;
# a size array would add a comma and fail the 2-column check.
FORBIDDEN = re.compile(r"[\"'\s]")


def validate_submission(
    csv_path: Path,
    test_dir: Optional[Path] = None,
    sample: int = 25,
    allow_missing_images: bool = True,
    seed: int = 42,
) -> bool:
    """
    Returns True if the file passes. Prints every problem found.
    Test images without rows are only a warning by default: the official
    metric counts their GT as FN (no format error), and a high score
    threshold can legitimately leave an image empty.
    """
    test_dir = Path(test_dir or CFG.paths.TEST_IMG_DIR)
    test_ids = {p.stem for p in test_dir.glob("*.jpeg")}
    errors: List[str] = []
    warnings: List[str] = []

    with open(csv_path, newline="") as f:
        lines = f.read().splitlines()
    if not lines or lines[0] != HEADER:
        errors.append(f"Header must be '{HEADER}', got '{lines[0] if lines else ''}'")
    rows = [ln for ln in lines[1:] if ln]

    seen, images, rles = set(), set(), []
    for n, line in enumerate(rows, start=2):
        if line.count(",") != 1:
            errors.append(f"line {n}: expected exactly 2 columns")
            continue
        fid, counts = line.split(",")
        m = ID_RE.match(fid)
        if not m:
            errors.append(f"line {n}: bad filament_id '{fid}'")
            continue
        if fid in seen:
            errors.append(f"line {n}: duplicate filament_id '{fid}'")
        seen.add(fid)
        if test_ids and m["image"] not in test_ids:
            errors.append(f"line {n}: image id '{m['image']}' is not a test image")
        images.add(m["image"])
        if not counts or FORBIDDEN.search(counts):
            errors.append(f"line {n}: RLE must be a bare counts string")
            continue
        rles.append((n, counts))

    for n, counts in rles:
        try:
            if coco_mask.area(counts_to_rle(counts, CFG.data.IMAGE_SIZE)) == 0:
                errors.append(f"line {n}: empty mask")
        except Exception as e:  # malformed counts
            errors.append(f"line {n}: RLE does not parse ({e})")

    for n, counts in random.Random(seed).sample(rles, min(sample, len(rles))):
        try:
            mask = coco_mask.decode(counts_to_rle(counts, CFG.data.IMAGE_SIZE))
            if mask.shape != tuple(CFG.data.IMAGE_SIZE) or not set(np.unique(mask)) <= {0, 1}:
                errors.append(f"line {n}: decodes to {mask.shape}, values {np.unique(mask)[:5]}")
        except Exception as e:
            errors.append(f"line {n}: RLE does not decode ({e})")

    missing = test_ids - images
    if missing:
        msg = f"{len(missing)}/{len(test_ids)} test images have no predictions"
        (warnings if allow_missing_images else errors).append(msg)
    if not rows:
        errors.append("no prediction rows")

    print(f"[Validate] {csv_path}: {len(rows)} rows, {len(images)} images")
    for w in warnings:
        print(f"  WARNING: {w}")
    for e in errors[:30]:
        print(f"  ERROR: {e}")
    if len(errors) > 30:
        print(f"  ... and {len(errors) - 30} more errors")
    print("[Validate] PASS" if not errors else f"[Validate] FAIL ({len(errors)} errors)")
    return not errors


def main():
    parser = argparse.ArgumentParser(description="Validate a Kaggle submission CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=25, help="RLEs to fully decode")
    parser.add_argument("--strict-images", action="store_true",
                        help="Fail (instead of warn) when some test images have no rows")
    args = parser.parse_args()
    ok = validate_submission(args.csv, args.test_dir, args.sample, not args.strict_images)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
