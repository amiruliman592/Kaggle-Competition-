"""Phase 6 S4: pasted filaments land inside the disk, on free sky, with matching outlines. No Detectron2 needed."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import copy_paste  # noqa: E402
from copy_paste import EDGE_BLUR, paste_filaments  # noqa: E402

N, R = 2048, 900  # synthetic disk: centred, radius 900


def _anno(x, y, bw, bh):
    poly = [x, y, x + bw, y, x + bw, y + bh, x + bw // 3, y + bh, x, y + bh // 2]
    return {"bbox": [x, y, x + bw, y + bh], "bbox_mode": 0, "segmentation": [poly], "category_id": 0, "iscrowd": 0}


def _image(sky, annos, value):
    img = np.zeros((N, N), np.uint8)
    cv2.circle(img, (N // 2, N // 2), R, sky, -1)
    for a in annos:
        img[copy_paste._fill(copy_paste._polys(a), (N, N)) > 0] = value
    return cv2.merge([img] * 3)


def _mask(anno):
    return copy_paste._fill(copy_paste._polys(anno), (N, N)).astype(bool)


TARGET = [_anno(900, 900, 300, 60)]
DONOR = [_anno(600, 700, 200, 40), _anno(1100, 1300, 80, 160), _anno(1300, 600, 120, 30)]


def test_paste():
    image, donor = _image(180, TARGET, 60), _image(140, DONOR, 30)
    before = image.copy()
    pasted = 0
    for seed in range(5):
        out, annos = paste_filaments(image, TARGET, donor, DONOR, n_max=3, rng=np.random.RandomState(seed))
        assert np.array_equal(image, before)            # input not changed in place
        assert annos[:1] == TARGET                      # own labels kept
        new = annos[1:]
        assert 0 <= len(new) <= 3
        pasted += len(new)
        changed = np.zeros((N, N), bool)
        masks = [_mask(a) for a in annos]
        for i, a in enumerate(new, start=1):
            q = np.asarray(a["segmentation"][0], float).reshape(-1, 2)
            assert np.hypot(q[:, 0] - N / 2, q[:, 1] - N / 2).max() <= 0.95 * R + 1   # inside the disk
            assert a["bbox"] == [*q.min(0).tolist(), *q.max(0).tolist()]
            assert not any((masks[i] & m).any() for j, m in enumerate(masks) if j != i)  # no overlap
            core = cv2.erode(masks[i].astype(np.uint8), np.ones((2 * EDGE_BLUR + 1,) * 2, np.uint8)) > 0
            # filament pixels (30 on a 140 sky) shifted to this sky (180): 30 + 40 = 70
            assert abs(out[core][:, 0].astype(float).mean() - 70) < 2
            changed |= cv2.dilate(masks[i].astype(np.uint8), np.ones((4 * EDGE_BLUR + 1,) * 2, np.uint8)) > 0
        assert np.array_equal(out[~changed], before[~changed])   # nothing else touched
    assert pasted >= 5


def test_no_free_spot_pastes_nothing():
    full = [{**_anno(0, 0, 1, 1), "segmentation": [[100, 100, 1948, 100, 1948, 1948, 100, 1948]]}]  # covers the disk
    image, donor = _image(180, full, 60), _image(140, DONOR, 30)
    out, annos = paste_filaments(image, full, donor, DONOR, rng=np.random.RandomState(0))
    assert annos == full and np.array_equal(out, image)
