"""RLE round-trip and the submission CSV path: write -> validate -> evaluate."""

import numpy as np
from pycocotools import mask as coco_mask

from annotations import anns_to_rles, load_annotation_sets, load_gt_rles
from evaluate import evaluate, load_submission_rles
from utils.rle import mask_to_rle, polygon_to_rle, rle_to_mask


def test_polygon_rle_roundtrip_is_lossless_for_all_annotations():
    """polygon -> mask -> counts string -> mask is identical for all 8,199 instances."""
    sets = load_annotation_sets()
    n = 0
    for v in sets.values():
        for s in v:
            for ann in s:
                rle = polygon_to_rle(ann["segmentation"])
                counts = rle["counts"].decode()
                # string form is what goes into the CSV; re-encoding a decoded mask must give it back
                assert mask_to_rle(rle_to_mask(counts)) == counts
                n += 1
    assert n == 8199


def test_dense_roundtrip_spot_check():
    rng = np.random.default_rng(42)
    m = (rng.random((2048, 2048)) > 0.999).astype(np.uint8)
    assert np.array_equal(rle_to_mask(mask_to_rle(m)), m)


def test_csv_gt_submission_scores_one(tmp_path):
    """Write val GT as a submission CSV, read it back: PQ must be exactly 1."""
    from split import load_split

    _, val_ids = load_split()
    gt = load_gt_rles(val_ids[:20])
    csv = tmp_path / "sub.csv"
    with open(csv, "w", newline="") as f:
        f.write("filament_id,segmentation_rle\n")
        for stem, rles in gt.items():
            for i, r in enumerate(rles, 1):
                f.write(f"{stem}_{i},{r['counts'].decode()}\n")
    assert '"' not in csv.read_text()
    pred = load_submission_rles(csv)
    r = evaluate(gt, pred, verbose=False)
    assert r["PQ"] == 1.0


def test_validator_accepts_good_and_rejects_bad(tmp_path):
    from config import CFG
    from validate_submission import validate_submission

    stem = sorted(p.stem for p in CFG.paths.TEST_IMG_DIR.glob("*.jpeg"))[0]
    m = np.zeros((2048, 2048), np.uint8)
    m[100:200, 100:300] = 1
    good = tmp_path / "good.csv"
    good.write_text(f"filament_id,segmentation_rle\n{stem}_1,{mask_to_rle(m)}\n")
    assert validate_submission(good)  # 179 test images missing: warning only
    assert not validate_submission(good, allow_missing_images=False)

    bad = tmp_path / "bad.csv"
    bad.write_text(f'filament_id,segmentation_rle\nnot_a_test_image_1,"{mask_to_rle(m)}"\n')
    assert not validate_submission(bad, allow_missing_images=True)


def test_remove_overlaps_higher_score_wins():
    from utils.rle import encode_mask, has_overlap, remove_overlaps

    a = np.zeros((32, 32), np.uint8); a[0:10, 0:10] = 1    # area 100, first = higher score
    b = np.zeros((32, 32), np.uint8); b[5:15, 5:15] = 1    # overlaps a by 25 px
    c = np.zeros((32, 32), np.uint8); c[2:4, 2:4] = 1      # fully inside a -> dropped
    ra, rb, rc = encode_mask(a), encode_mask(b), encode_mask(c)
    rb["score"] = 0.7
    out = remove_overlaps([ra, rb, rc])
    assert len(out) == 2 and not has_overlap(out)
    assert coco_mask.area(out[0]) == 100 and coco_mask.area(out[1]) == 75
    assert out[1]["score"] == 0.7
    assert len(remove_overlaps([ra, rb], min_area=80)) == 1


def test_validator_rejects_overlapping_masks(tmp_path):
    from config import CFG
    from validate_submission import validate_submission

    stem = sorted(p.stem for p in CFG.paths.TEST_IMG_DIR.glob("*.jpeg"))[0]
    m1 = np.zeros((2048, 2048), np.uint8); m1[100:200, 100:300] = 1
    m2 = np.zeros((2048, 2048), np.uint8); m2[150:250, 100:300] = 1
    csv = tmp_path / "overlap.csv"
    csv.write_text(f"filament_id,segmentation_rle\n{stem}_1,{mask_to_rle(m1)}\n{stem}_2,{mask_to_rle(m2)}\n")
    assert not validate_submission(csv)
