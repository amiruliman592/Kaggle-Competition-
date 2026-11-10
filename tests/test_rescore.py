import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _rle(x0, y0, x1, y1, score=1.0, size=64):
    m = np.zeros((size, size), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    r = coco_mask.encode(np.asfortranarray(m))
    r["score"] = score
    return r


def test_features_count_agreeing_views():
    from rescore import FEATURES, image_features
    a, b = _rle(5, 5, 25, 15, 0.9), _rle(40, 40, 50, 50, 0.4)
    near_a = _rle(6, 5, 25, 15, 0.8)
    views = {"m28": [[near_a], [near_a], [], [b]], "r50": [[near_a], [], [], []]}
    X = image_features([a, b], views)
    f = {k: X[:, i] for i, k in enumerate(FEATURES)}
    assert list(f["n_m28"]) == [2, 1] and list(f["n_r50"]) == [1, 0]
    assert np.isclose(f["s_m28"][0], 0.8 * 2 / 4) and np.isclose(f["s_m28"][1], 0.4 / 4)
    assert np.isclose(f["elong"][0], 2.0) and np.isclose(f["fill"][0], 1.0)
    assert list(f["n_crowd"]) == [1, 1]  # one candidate with score >= 0.5 in the image


def test_match_share_over_annotator_sets():
    from rescore import match_share
    a, b = _rle(5, 5, 25, 15), _rle(40, 40, 50, 50)
    sets = [[_rle(5, 5, 25, 14)], [_rle(40, 40, 50, 50), _rle(5, 5, 24, 15)], []]  # empty set is ignored
    assert list(match_share([a, b], sets)) == [1.0, 0.5]
    assert list(match_share([], sets)) == []
