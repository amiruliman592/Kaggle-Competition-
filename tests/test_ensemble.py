import numpy as np
from pycocotools import mask as coco_mask

from ensemble import fuse_image
from utils.rle import encode_mask


def _rle(y0, y1, x0, x1, score):
    m = np.zeros((64, 64), np.uint8)
    m[y0:y1, x0:x1] = 1
    r = encode_mask(m)
    r["score"] = score
    return r


def test_fuse_pairs_matching_masks_and_halves_singletons():
    a = [_rle(0, 20, 0, 20, 0.9), _rle(40, 50, 40, 50, 0.8)]   # model A: shared + A-only
    b = [_rle(2, 20, 0, 20, 0.7)]                               # model B: shared (IoU 0.9)
    out = fuse_image([a, b], match_iou=0.4, mask_mode="best")
    assert [round(r["score"], 3) for r in out] == [0.8, 0.4]   # (0.9+0.7)/2, 0.8/2
    assert int(coco_mask.area(out[0])) == 400                  # best = A's mask (higher score)


def test_fuse_mask_modes():
    a = [_rle(0, 20, 0, 20, 0.6)]
    b = [_rle(2, 20, 0, 20, 0.9)]
    area = lambda mode: int(coco_mask.area(fuse_image([a, b], 0.4, mode)[0]))
    assert area("best") == 360       # B scores higher
    assert area("union") == 400
    assert area("inter") == 360


def test_fuse_does_not_pair_below_match_iou():
    a = [_rle(0, 20, 0, 20, 0.9)]
    b = [_rle(0, 20, 15, 35, 0.9)]   # IoU 100/700
    out = fuse_image([a, b], match_iou=0.4, mask_mode="best")
    assert sorted(round(r["score"], 3) for r in out) == [0.45, 0.45]
