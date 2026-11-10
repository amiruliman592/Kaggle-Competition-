"""Phase 6 S5: pseudo-labels load as polygons that match the masks, and never come from val/test days."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import CFG  # noqa: E402
from pseudo import build_pseudo_dicts, mask_to_polygons, with_pseudo  # noqa: E402

N = 2048
VAL = ["20120301120000Bh"]          # its 27-day block is off limits
TEST = ["20120601120000Ch"]         # this day +-1 is off limits


def _shapes():
    a = np.zeros((N, N), np.uint8)
    cv2.rectangle(a, (300, 400), (700, 460), 1, -1)
    b = np.zeros((N, N), np.uint8)
    cv2.ellipse(b, (1000, 1000), (250, 40), 30, 0, 360, 1, -1)
    c = np.zeros((N, N), np.uint8)                        # two pieces in one mask
    cv2.circle(c, (1400, 600), 30, 1, -1)
    cv2.circle(c, (1500, 650), 20, 1, -1)
    d = np.zeros((N, N), np.uint8)                        # thin curved filament
    cv2.ellipse(d, (900, 1500), (300, 120), 0, 180, 330, 1, 4)
    return [a, b, c, d]


def _rle(m):
    r = coco_mask.encode(np.asfortranarray(m))
    return {"size": [N, N], "counts": r["counts"].decode("utf-8")}


def _write(tmp_path, stems, masks_per_image):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    images, anns = [], []
    for i, (stem, masks) in enumerate(zip(stems, masks_per_image), start=1):
        images.append({"id": i, "file_name": f"{stem}.jpg", "width": N, "height": N})
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), np.zeros((8, 8, 3), np.uint8))
        for m in masks:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "iscrowd": 0,
                         "segmentation": _rle(m), "score": 0.9})
    path = tmp_path / "pseudo_labels_v1.json"
    path.write_text(json.dumps({"images": images, "annotations": anns}))
    return path, img_dir


def test_polygons_match_masks():
    for m in _shapes():
        polys = mask_to_polygons(m)
        back = np.zeros_like(m)
        cv2.fillPoly(back, [np.round(np.reshape(p, (-1, 2))).astype(np.int32) for p in polys], 1)
        inter, union = (m & back).sum(), (m | back).sum()
        assert inter / union > 0.95
    assert len(mask_to_polygons(_shapes()[2])) == 2


def test_build_dicts(tmp_path):
    stems = ["20150110120000Lh", "20150210120000Mh", "20150310120000Th"]
    shapes = _shapes()
    path, img_dir = _write(tmp_path, stems, [shapes[:2], shapes[2:], shapes[:1]])
    (img_dir / f"{stems[2]}.jpg").unlink()                # missing image file is skipped
    dicts = build_pseudo_dicts(path, img_dir, VAL, TEST)
    assert [d["image_id"] for d in dicts] == [f"pseudo:{s}" for s in stems[:2]]
    assert [len(d["annotations"]) for d in dicts] == [2, 2]
    for d in dicts:
        assert Path(d["file_name"]).exists() and (d["height"], d["width"]) == (N, N)
        for a in d["annotations"]:
            q = np.concatenate([np.reshape(p, (-1, 2)) for p in a["segmentation"]])
            assert a["bbox"] == [*q.min(0).tolist(), *q.max(0).tolist()]
            assert a["category_id"] == 0 and a["iscrowd"] == 0 and a["bbox_mode"] == 0
    x0, y0, x1, y1 = dicts[0]["annotations"][0]["bbox"]
    assert (x0, y0, x1, y1) == (300, 400, 700, 460)


@pytest.mark.parametrize("bad", ["20120305120000Bh", "20120602120000Uh", "20120531120000Uh"])
def test_val_or_test_day_is_refused(tmp_path, bad):
    path, img_dir = _write(tmp_path, ["20150110120000Lh", bad], [_shapes()[:1]] * 2)
    with pytest.raises(ValueError, match="val/test days"):
        build_pseudo_dicts(path, img_dir, VAL, TEST)


def test_with_pseudo_mixes_real_twice(tmp_path, monkeypatch):
    path, img_dir = _write(tmp_path, ["20150110120000Lh"], [_shapes()[:1]])
    import pseudo
    monkeypatch.setattr(pseudo, "load_exclusion_ids", lambda: (VAL, TEST))
    real = [{"image_id": "a"}, {"image_id": "b"}]
    monkeypatch.setattr(CFG.train, "PSEUDO_JSON", "")
    assert with_pseudo(real) == real                      # off by default
    monkeypatch.setattr(CFG.train, "PSEUDO_JSON", str(path))   # images default to <json folder>/images
    out = with_pseudo(real)
    assert [d["image_id"] for d in out] == ["a", "b", "a", "b", "pseudo:20150110120000Lh"]
