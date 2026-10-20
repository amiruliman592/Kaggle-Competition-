"""Metric engine: hand-computed PQ cases and GT-vs-GT on the real val split."""

import numpy as np
import pytest

from evaluate import evaluate, match_image
from utils.rle import encode_mask

HW = (64, 64)


def box(y0, y1, x0, x1):
    m = np.zeros(HW, np.uint8)
    m[y0:y1, x0:x1] = 1
    return encode_mask(m)


def test_perfect_match():
    g = [box(0, 10, 0, 10), box(20, 30, 20, 40)]
    r = evaluate({"a": g}, {"a": g}, verbose=False)
    assert r["PQ"] == pytest.approx(1.0) and r["TP"] == 2 and r["FP"] == r["FN"] == 0


def test_iou_exactly_half_is_not_a_match():
    g = [box(0, 10, 0, 10)]  # area 100
    p = [box(0, 10, 0, 5) ]  # IoU = 50/100 = 0.5 -> not > 0.5
    r = evaluate({"a": g}, {"a": p}, verbose=False)
    assert r["TP"] == 0 and r["FP"] == 1 and r["FN"] == 1 and r["PQ"] == 0


def test_fragmentation_penalty():
    # GT 10x20; prediction split in two halves at IoU 0.6 and 0.4
    g = [box(0, 10, 0, 20)]
    p = [box(0, 10, 0, 12), box(0, 10, 12, 20)]
    r = evaluate({"a": g}, {"a": p}, verbose=False)
    assert r["TP"] == 1 and r["FP"] == 1 and r["FN"] == 0
    assert r["RQ"] == pytest.approx(1 / 1.5)
    assert r["PQ"] == pytest.approx(0.6 / 1.5)
    assert r["fragmented_gt_1toN"] == 1


def test_over_merging_counter():
    g = [box(0, 10, 0, 10), box(0, 10, 12, 22)]
    p = [box(0, 10, 0, 22)]
    r = match_image(g, p)
    assert r["n_merged_pred"] == 1


def test_pq_is_pooled_not_per_image_mean():
    g1 = [box(0, 10, 0, 10)]
    g2 = [box(0, 10, 0, 10), box(20, 30, 0, 10), box(40, 50, 0, 10)]
    r = evaluate({"a": g1, "b": g2}, {"a": g1, "b": []}, verbose=False)
    # pooled: TP=1, FN=3 -> RQ = 1/2.5; per-image mean = (1 + 0)/2
    assert r["PQ"] == pytest.approx(1 / 2.5)
    assert r["mean_image_PQ"] == pytest.approx(0.5)


def test_empty_cases():
    g = [box(0, 10, 0, 10)]
    assert evaluate({"a": g}, {}, verbose=False)["FN"] == 1
    r = evaluate({"a": []}, {"a": g}, verbose=False)
    assert r["FP"] == 1 and r["PQ"] == 0


def test_gt_vs_gt_on_val_split_is_one():
    from annotations import load_gt_rles
    from split import load_split

    _, val_ids = load_split()
    gt = load_gt_rles(val_ids)
    r = evaluate(gt, gt, verbose=False)
    assert r["PQ"] == pytest.approx(1.0) and r["FP"] == 0 and r["FN"] == 0
    assert r["mean_semantic_Dice"] == pytest.approx(1.0)
