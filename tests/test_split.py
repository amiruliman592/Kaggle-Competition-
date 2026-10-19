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


def test_frozen_split_is_reproducible():
    train, val = load_split()
    fresh = make_split()
    assert fresh["val_ids"] == val and fresh["train_ids"] == train
