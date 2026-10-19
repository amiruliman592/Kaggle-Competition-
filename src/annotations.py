"""
annotations.py — Load MAGFiLO COCO annotations grouped by physical image file.

Important dataset quirk:
  The JSON has 1,154 "image" entries for only 707 JPEG files. Files with 2-3
  entries were annotated independently by different annotators (the same
  filament appears in each set with median IoU ~0.56). Merging the sets would
  create duplicate overlapping ground truth, and treating each entry as its
  own image leaks the same file into train AND val. Everything downstream is
  therefore keyed by the file stem (e.g. "20140609195854Bh").

Usage:
  from annotations import load_annotation_sets, select_annotations
  sets = load_annotation_sets()              # stem -> [set_0, set_1, ...]
  anns = select_annotations(sets, "first")   # stem -> [ann, ...]
"""

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from config import CFG
from utils.rle import polygon_to_rle

AnnSets = Dict[str, List[List[dict]]]


def file_stem(file_name: str) -> str:
    return Path(file_name).stem


@lru_cache(maxsize=2)
def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_annotation_sets(path: Optional[Path] = None) -> AnnSets:
    """
    Returns stem -> list of annotator sets, ordered by COCO entry id.
    Each set is the list of COCO annotation dicts of one entry.
    """
    data = _load_json(str(path or CFG.paths.ANNOTATION_FILE))
    entry_to_stem = {im["id"]: file_stem(im["file_name"]) for im in data["images"]}

    by_entry = defaultdict(list)
    for ann in data["annotations"]:
        by_entry[ann["image_id"]].append(ann)

    stem_entries = defaultdict(list)
    for entry_id in sorted(entry_to_stem):
        stem_entries[entry_to_stem[entry_id]].append(by_entry.get(entry_id, []))
    return dict(sorted(stem_entries.items()))


def select_annotations(sets: AnnSets, policy: Optional[str] = None) -> Dict[str, List[dict]]:
    """Pick ONE annotator set per file according to `policy` ("first" | "most")."""
    policy = policy or CFG.data.ANNOTATOR_POLICY
    if policy == "first":
        return {stem: s[0] for stem, s in sets.items()}
    if policy == "most":
        return {stem: max(s, key=len) for stem, s in sets.items()}
    raise ValueError(f"Unknown annotator policy: {policy!r}")


def anns_to_rles(anns: List[dict]) -> List[dict]:
    """COCO polygon annotations -> compressed RLE dicts at native resolution."""
    return [polygon_to_rle(a["segmentation"], CFG.data.IMAGE_SIZE) for a in anns]


def load_gt_rles(stems: Optional[List[str]] = None, policy: Optional[str] = None) -> Dict[str, List[dict]]:
    """Ground truth for evaluation: stem -> list of RLE dicts."""
    selected = select_annotations(load_annotation_sets(), policy)
    stems = stems if stems is not None else list(selected)
    return {s: anns_to_rles(selected[s]) for s in stems}
