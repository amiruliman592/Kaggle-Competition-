import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np

from bootstrap import compare, compare_lb, draw_counts, image_components, pooled_pq
from evaluate import evaluate_official
from utils.rle import encode_mask


def _rle(y0, y1, x0, x1, score=1.0):
    m = np.zeros((64, 64), np.uint8)
    m[y0:y1, x0:x1] = 1
    r = encode_mask(m)
    r["score"] = score
    return r


ENTRIES = {  # s1 annotated twice, s2 once
    "a-s1": [_rle(0, 20, 0, 20), _rle(40, 50, 40, 50)],
    "b-s1": [_rle(0, 18, 0, 20)],
    "a-s2": [_rle(10, 30, 10, 30)],
}
PREDS = {"s1": [_rle(0, 20, 0, 20), _rle(30, 35, 0, 5)], "s2": [_rle(12, 30, 10, 30)]}


def test_components_repool_to_official_pq():
    comp = image_components(ENTRIES, PREDS, ["s1", "s2"])
    assert comp[:, 1:].tolist() == [[2, 2, 1], [1, 0, 0]]   # TP/FP/FN per image, entries summed
    pq = pooled_pq(comp, np.ones((1, 2)))[0]
    assert np.isclose(pq, evaluate_official(ENTRIES, PREDS)["PQ"])


def test_counts_weight_images():
    comp = image_components(ENTRIES, PREDS, ["s1", "s2"])
    only_s2 = pooled_pq(comp, np.array([[0, 3]]))[0]
    assert np.isclose(only_s2, evaluate_official({"a-s2": ENTRIES["a-s2"]}, PREDS)["PQ"])
    assert pooled_pq(np.zeros((2, 4)), np.ones((1, 2)))[0] == 0.0   # no TP/FP/FN -> 0, no crash


def test_draw_counts_and_identical_runs():
    counts = draw_counts(141, 90, 50, np.random.default_rng(0))
    assert counts.shape == (50, 141) and (counts.sum(axis=1) == 90).all()
    pq = np.random.default_rng(1).uniform(0.3, 0.5, 200)
    same = compare(pq, pq)
    assert same["mean_delta"] == 0 and same["min_real_gain"] == 0 and same["P_B_better"] == 0
    assert compare_lb(pq, pq)["P_lb_tie"] == 1.0
    assert compare(pq, pq + 0.01)["P_B_better"] == 1.0
