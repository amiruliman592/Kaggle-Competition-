"""Phase 6 S3: outlines must stay on the filament after rotation + zoom. Needs Detectron2 (runs on Colab)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

T = pytest.importorskip("detectron2.data.transforms")

from augment import train_augmentations  # noqa: E402
from config import CFG  # noqa: E402

# L-shape: a wrong rotation or flip cannot land on itself
POLY = np.array([[200, 150], [760, 150], [760, 300], [340, 300], [340, 800], [200, 800]], dtype=float)


@pytest.fixture
def aug():
    saved = CFG.train.AUG
    yield CFG
    CFG.train.AUG = saved


def _draw(h, w, poly):
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask.astype(bool)


def test_default_keeps_fixed_size_and_no_rotation(aug):
    aug.train.AUG = "flip"
    augs = train_augmentations(1024)
    assert not any(isinstance(a, T.RandomRotation) for a in augs)
    img = np.zeros((2048, 2048, 3), np.uint8)
    for _ in range(5):
        out = T.AugInput(img)
        T.AugmentationList(augs)(out)
        assert out.image.shape[:2] == (1024, 1024)


def test_rotscale_outline_stays_on_the_filament(aug):
    aug.train.AUG = "rotscale"
    img = np.zeros((1024, 1024, 3), np.uint8)
    img[_draw(1024, 1024, POLY)] = 255
    augs = T.AugmentationList(train_augmentations(1024))
    seen, sizes = set(), set()
    np.random.seed(0)
    for _ in range(40):
        inp = T.AugInput(img)
        tf = augs(inp)
        h, w = inp.image.shape[:2]
        assert h == w and 819 <= h <= 1228
        sizes.add(h)
        box = _draw(h, w, tf.apply_coords(POLY.copy()))
        seen.add(tuple(box[[h // 4, h // 4, 3 * h // 4, 3 * h // 4], [w // 4, 3 * w // 4, w // 4, 3 * w // 4]]))
        pix = inp.image[:, :, 0] > 128
        assert (pix & box).sum() / (pix | box).sum() > 0.95
    assert len(sizes) > 5   # zoom varies
    assert len(seen) > 2    # the L points different ways
