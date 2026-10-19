"""Frozen split integrity: disjoint, complete, no time-block leakage, reproducible."""

from annotations import load_annotation_sets
from config import CFG
from split import load_split, make_split, time_block


def test_split_disjoint_and_complete():
    train, val = load_split()
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(load_annotation_sets())
    assert 0.15 < len(val) / (len(train) + len(val)) < 0.25


def test_no_time_block_leakage():
    train, val = load_split()
    b = CFG.data.SPLIT_BLOCK_DAYS
    assert not {time_block(s, b) for s in train} & {time_block(s, b) for s in val}


def test_regenerated_split_is_valid():
    """
    The exact folds depend on the scikit-learn version (Colab differs from local),
    which is why splits/val_ids.json is frozen and committed. A regenerated split
    must still be a valid one, but is not required to equal the frozen file.
    """
    fresh = make_split()
    train, val = fresh["train_ids"], fresh["val_ids"]
    b = CFG.data.SPLIT_BLOCK_DAYS
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(load_annotation_sets())
    assert not {time_block(s, b) for s in train} & {time_block(s, b) for s in val}
