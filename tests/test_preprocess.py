"""Phase 3 retraining toggles: CLAHE + disk masking, training label policy."""

import cv2
import numpy as np


def test_clahe_disk_blanks_outside_and_keeps_shape():
    from preprocess import clahe_disk

    img = np.full((2048, 2048), 20, np.uint8)
    cv2.circle(img, (1024, 1024), 900, 140, -1)            # synthetic solar disk
    cv2.circle(img, (1024, 1024), 60, 60, -1)              # dark "filament"
    out = clahe_disk(cv2.merge([img, img, img]))
    assert out.shape == (2048, 2048, 3) and out.dtype == np.uint8
    assert out[0, 0].tolist() == [0, 0, 0] and out[2047, 2047].tolist() == [0, 0, 0]
    assert out[1024, 1024, 0] < out[1024, 1024 + 500, 0]   # contrast order preserved


def test_image_path_follows_preproc_mode():
    from config import CFG
    from preprocess import image_path

    saved = CFG.data.PREPROC
    try:
        CFG.data.PREPROC = "none"
        assert image_path("x", "test") == CFG.paths.TEST_IMG_DIR / "x.jpeg"
        CFG.data.PREPROC = "clahe_disk"
        assert image_path("x", "train") == CFG.paths.PREPROC_DIR / "clahe_disk" / "train" / "x.png"
    finally:
        CFG.data.PREPROC = saved


def test_label_policy_all_uses_every_annotator_set():
    from annotations import load_annotation_sets
    from dataset import build_detectron2_dicts
    from split import load_split

    train_ids, _ = load_split()
    sets = load_annotation_sets()
    first = build_detectron2_dicts(train_ids, label_policy="first")
    every = build_detectron2_dicts(train_ids, label_policy="all")
    assert len(first) == len(train_ids)
    assert len(every) == sum(len(sets[s]) for s in train_ids)
    assert len({d["image_id"] for d in every}) == len(every)
