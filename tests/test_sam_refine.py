import numpy as np
from pycocotools import mask as coco_mask

from sam_refine import combine, crop_window, load_sam, mask_points, save_sam
from utils.rle import encode_mask


def _rle(mask, score=0.9):
    r = encode_mask(mask)
    r["score"] = score
    return r


def test_crop_window_is_square_inside_image_and_holds_box():
    for bbox in ([0, 0, 40, 10], [2000, 2030, 48, 18], [900, 1000, 700, 120], [1000, 5, 30, 30]):
        x0, y0, x1, y1 = crop_window(bbox)
        assert x1 - x0 == y1 - y0 >= 256
        assert 0 <= x0 and 0 <= y0 and x1 <= 2048 and y1 <= 2048
        x, y, w, h = bbox
        assert x0 <= x and y0 <= y and x + w <= x1 and y + h <= y1


def test_mask_points_lie_inside_a_thin_mask():
    m = np.zeros((200, 200), np.uint8)
    m[95:100, 20:180] = 1                      # long thin horizontal "filament"
    pts = mask_points(m, 3)
    assert pts.shape == (3, 2)
    assert all(m[int(y), int(x)] for x, y in pts)
    assert pts[:, 0].max() - pts[:, 0].min() > 80   # spread along the long axis
    assert mask_points(np.zeros((10, 10), np.uint8)).shape == (0, 2)


def test_combine_modes():
    a = np.zeros((64, 64), np.uint8); a[10:30, 10:30] = 1    # original, area 400
    b = np.zeros((64, 64), np.uint8); b[12:30, 10:30] = 1    # SAM, IoU 0.9 with a
    c = np.zeros((64, 64), np.uint8); c[40:60, 40:60] = 1    # SAM, disjoint from a
    o = _rle(a, 0.7)
    agree, off = _rle(b), _rle(c)
    area = lambda r: int(coco_mask.area(r))
    assert area(combine(o, agree, "sam")) == 360
    assert area(combine(o, agree, "union")) == 400
    assert area(combine(o, agree, "inter")) == 360
    assert area(combine(o, agree, "gate")) == 360          # agrees -> SAM
    assert area(combine(o, off, "gate")) == 400            # disagrees -> original
    assert area(combine(o, off, "inter")) == 400           # empty intersection -> original
    assert combine(o, agree, "union")["score"] == 0.7      # keeps the detector score


def test_save_load_sam_keeps_order_and_fields(tmp_path):
    m = np.zeros((8, 8), np.uint8); m[1:3, 1:3] = 1
    preds = {"img": [dict(_rle(m, 0.5), sam_iou=0.8), dict(_rle(m, 0.9), sam_iou=-1.0)]}
    save_sam(preds, tmp_path / "s.json.gz")
    back = load_sam(tmp_path / "s.json.gz", hw=(8, 8))
    assert [r["score"] for r in back["img"]] == [0.5, 0.9]   # file order, not re-sorted
    assert [r["sam_iou"] for r in back["img"]] == [0.8, -1.0]
    assert coco_mask.area(back["img"][0]) == 4
