import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np

from error_budget import budget, classify_entry, run
from evaluate import evaluate_official
from utils.rle import encode_mask


def _rle(y0, y1, x0, x1, score=1.0):
    m = np.zeros((64, 64), np.uint8)
    m[y0:y1, x0:x1] = 1
    r = encode_mask(m)
    r["score"] = score
    return r


def boxes(recs):
    return sorted(r["box"] for r in recs)


def test_fn_boxes():
    gt = [_rle(0, 10, 0, 10), _rle(20, 30, 0, 10), _rle(40, 50, 0, 10), _rle(0, 10, 40, 50)]
    final = [_rle(0, 10, 0, 10, 0.9),        # hit
             _rle(40, 50, 4, 14, 0.9)]       # IoU 60/140 = 0.43 -> near miss
    raw = final + [_rle(20, 30, 0, 10, 0.3)]  # low-score raw hit on GT 2
    recs = classify_entry(gt, final, raw, [], thr=0.6)
    assert boxes(recs) == ["fn_low_score", "fn_near_miss", "fn_never", "fp_nobody", "tp"]  # near miss is also an FP


def test_postproc_box_and_fp_boxes():
    gt = [_rle(0, 10, 0, 10)]
    raw = [_rle(0, 10, 0, 10, 0.9)]           # good raw hit, but post-processing removed it
    final = [_rle(30, 40, 30, 40, 0.9), _rle(50, 60, 50, 60, 0.9)]
    other = [[_rle(30, 40, 30, 40)]]          # another annotator drew the first false alarm
    recs = classify_entry(gt, final, raw, other, thr=0.6)
    assert boxes(recs) == ["fn_postproc", "fp_nobody", "fp_other_ann"]


def test_run_matches_official_pq():
    entries = {"a-s1": [_rle(0, 20, 0, 20), _rle(40, 50, 40, 50)], "b-s1": [_rle(0, 18, 0, 20)],
               "a-s2": [_rle(10, 30, 10, 30)], "a-s3": []}
    final = {"s1": [_rle(0, 20, 0, 20, 0.9), _rle(30, 35, 0, 5, 0.9)], "s2": [_rle(12, 30, 10, 30, 0.9)]}
    res = run(entries, final, final, thr=0.6)
    off = evaluate_official(entries, final)
    assert np.isclose(res["all"]["PQ"], off["PQ"])
    assert (res["all"]["TP"], res["all"]["FP"], res["all"]["FN"]) == (off["TP"], off["FP"], off["FN"])
    # the FP at s1 is counted once per entry; nobody drew it
    assert res["all"]["boxes"]["fp_nobody"]["n"] == 2


def test_pq_if_fixed():
    recs = [{"box": "tp", "area": 1, "iou": 0.6}, {"box": "fn_never", "area": 1, "iou": 0},
            {"box": "fp_nobody", "area": 1, "iou": 0}]
    b = budget(recs)
    assert np.isclose(b["PQ"], 0.6 / 2)
    assert np.isclose(b["boxes"]["fn_never"]["PQ_if_fixed"], 1.2 / 2.5)
    assert np.isclose(b["boxes"]["fp_nobody"]["PQ_if_fixed"], 0.6 / 1.5)
    assert np.isclose(b["boxes"]["sq_loss"]["PQ_if_fixed"], 1 / 2)
