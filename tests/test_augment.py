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


def test_copypaste_mapper_adds_filaments(aug, tmp_path):
    from detectron2.config import get_cfg
    from augment import CopyPasteMapper

    aug.train.AUG = "copypaste"
    saved = aug.train.COPY_PASTE_PROB
    aug.train.COPY_PASTE_PROB = 1.0
    try:
        cfg = get_cfg()
        cfg.MODEL.MASK_ON = True
        dicts = []
        for k, (x, y) in enumerate([(900, 900), (700, 1200)]):
            img = np.zeros((2048, 2048, 3), np.uint8)
            cv2.circle(img, (1024, 1024), 900, (180, 180, 180), -1)
            poly = [x, y, x + 200, y, x + 200, y + 50, x, y + 50]
            cv2.fillPoly(img, [np.array(poly).reshape(-1, 2)], (60, 60, 60))
            cv2.imwrite(str(tmp_path / f"{k}.png"), img)
            dicts.append({"file_name": str(tmp_path / f"{k}.png"), "image_id": k, "height": 2048, "width": 2048,
                          "annotations": [{"bbox": [x, y, x + 200, y + 50], "bbox_mode": 0, "segmentation": [poly],
                                           "category_id": 0, "iscrowd": 0}]})
        mapper = CopyPasteMapper(cfg, train_augmentations(1024), dicts)
        np.random.seed(0)
        counts = []
        for _ in range(10):
            out = mapper(dicts[0])
            assert tuple(out["image"].shape) == (3, 1024, 1024)
            inst = out["instances"]
            assert len(inst.gt_masks) == len(inst) and inst.gt_boxes.tensor.max() <= 1024
            counts.append(len(inst))
        assert min(counts) >= 1 and max(counts) >= 2      # own filament kept, pasted ones added
        assert len(dicts[0]["annotations"]) == 1          # dataset dict not changed in place
    finally:
        aug.train.COPY_PASTE_PROB = saved
