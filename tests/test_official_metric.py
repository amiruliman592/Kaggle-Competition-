"""Official-mode PQ: hand cases, agreement with the organisers' code, real-data regressions."""

import sys
from pathlib import Path

import numpy as np
import pytest

from evaluate import evaluate_official
from utils.rle import encode_mask

HW = (64, 64)


def box(y0, y1, x0, x1):
    m = np.zeros(HW, np.uint8)
    m[y0:y1, x0:x1] = 1
    return encode_mask(m)


def test_duplicates_both_count_as_tp():
    g = [box(0, 10, 0, 10)]
    r = evaluate_official({"a1-s": g}, {"s": [g[0], g[0]]})
    assert r["TP"] == 2 and r["FP"] == 0 and r["FN"] == 0


def test_each_annotator_entry_is_scored_and_pooled():
    g1, g2 = [box(0, 10, 0, 10)], [box(30, 40, 30, 40)]
    r = evaluate_official({"a1-s": g1, "a2-s": g2}, {"s": g1})
    # entry a1: 1 TP; entry a2: 1 FP + 1 FN -> PQ = 1 / (1 + 0.5 + 0.5)
    assert (r["TP"], r["FP"], r["FN"]) == (1, 1, 1)
    assert r["PQ"] == pytest.approx(0.5)


def test_empty_entries_skipped_and_missing_preds_are_fn():
    g = [box(0, 10, 0, 10)]
    r = evaluate_official({"a1-s": [], "a1-t": g}, {"s": g})
    assert r["num_entries"] == 1 and r["FN"] == 1 and r["FP"] == 0


def test_matches_organisers_code_on_random_masks():
    torch = pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SelfEvaluationResult"))
    from official_metric import get_overlap_df, get_pq_score
    import official_metric

    rng = np.random.default_rng(42)

    def rand_masks(n):
        out = []
        for _ in range(n):
            y, x = rng.integers(0, 48, 2)
            h, w = rng.integers(4, 16, 2)
            out.append(box(y, y + h, x, x + w))
        return out

    gt_entries = {f"a{k}-s{i}": rand_masks(rng.integers(1, 5)) for i in range(6) for k in range(1, 3)}
    pred = {f"s{i}": rand_masks(rng.integers(0, 6)) for i in range(6)}

    def to_df(d):
        return pd.DataFrame([(f"{key}_{j}", r["counts"].decode()) for key, rles in d.items()
                             for j, r in enumerate(rles, 1)], columns=["filament_id", "segmentation_rle"])

    orig = official_metric.rles_to_layers.__defaults__
    official_metric.rles_to_layers.__defaults__ = HW  # organisers' code assumes 2048x2048
    try:
        expected = get_pq_score(get_overlap_df(to_df(gt_entries), to_df(pred)))
    finally:
        official_metric.rles_to_layers.__defaults__ = orig
    assert evaluate_official(gt_entries, pred)["PQ"] == pytest.approx(expected, abs=1e-9)


@pytest.fixture(scope="module")
def val_sets():
    from annotations import load_annotation_sets
    from split import load_split

    _, val_ids = load_split()
    return val_ids, load_annotation_sets()


def test_kaggle_notebook_value_annotator1_vs_annotator2(val_sets):
    """Same pair as output/metric_check/*.csv, which scored 0.3296 on Kaggle."""
    from annotations import anns_to_rles, load_gt_entries

    val_ids, sets = val_sets
    stems = [s for s in val_ids if len(sets[s]) >= 2]
    first = {}
    for entry_id, rles in load_gt_entries(stems).items():
        first.setdefault(entry_id.split("-", 1)[1], (entry_id, rles))
    gt = dict(first.values())
    pred = {s: anns_to_rles(sets[s][1]) for s in stems}
    assert evaluate_official(gt, pred)["PQ"] == pytest.approx(0.3296, abs=5e-5)


def test_copy_of_one_annotator_vs_all_sets(val_sets):
    from annotations import anns_to_rles, load_gt_entries

    val_ids, sets = val_sets
    pred = {s: anns_to_rles(sets[s][0]) for s in val_ids}
    r = evaluate_official(load_gt_entries(val_ids), pred)
    assert r["num_entries"] == 233
    assert r["PQ"] == pytest.approx(0.7394, abs=5e-5)


def test_sweep_filters_by_score_and_area():
    from sweep import filter_predictions

    small, big = box(0, 5, 0, 5), box(10, 30, 10, 30)  # areas 25, 400
    small["score"], big["score"] = 0.9, 0.4
    out = filter_predictions({"s": [small, big]}, score_threshold=0.5, min_area=0)
    assert out["s"] == [small]
    out = filter_predictions({"s": [small, big]}, score_threshold=0.3, min_area=100)
    assert out["s"] == [big]
