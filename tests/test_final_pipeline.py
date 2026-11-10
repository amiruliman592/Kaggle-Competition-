"""Final pipeline (exp101) on CPU: fusion order + post-processing reproduce the saved exp101 predictions,
and the weights survive pack -> unpack. No Detectron2 needed."""

import json
from pathlib import Path

import pytest

from final_pipeline import compare, fuse_one_image, postprocess
from utils.rle import load_predictions, write_submission_csv

ROOT = Path(__file__).resolve().parents[1]
MIRUL = ROOT / "Result Phase 6" / "Mirul"
VIEW_FILES = {
    "x101_2048_m28": [ROOT / "Result Step 5" / "Step 2" / "raw_preds_x101_2048_m28_test.json.gz"]
    + [MIRUL / f"raw_preds_x101_2048_m28_flip{f}_test.json.gz" for f in ("h", "v", "hv")],
    "r50_2048": [ROOT / "Result Phase 4" / "raw_preds_r50_2048_test.json.gz"]
    + [MIRUL / f"raw_preds_r50_2048_flip{f}_test.json.gz" for f in ("h", "v", "hv")],
}
FUSED = MIRUL / "raw_preds_tta2_m28_r50_test.json.gz"


@pytest.mark.skipif(not FUSED.exists(), reason="exp101 raw files not in this clone")
def test_reproduces_exp101_on_test_images(tmp_path):
    views = {m: [load_predictions(p) for p in ps] for m, ps in VIEW_FILES.items()}
    ref = load_predictions(FUSED)
    stems = sorted(ref)[:4]
    fused = {s: fuse_one_image([[v.get(s, []) for v in views[m]] for m in VIEW_FILES]) for s in stems}
    for s in stems:
        assert [(r["counts"], round(r["score"], 6)) for r in fused[s]] == \
               [(r["counts"], round(r["score"], 6)) for r in ref[s]]
    # post-processing per image (as in the GPU loop) == on the whole dict (as submit_from_raw.py)
    per_image = {}
    for s in stems:
        per_image.update(postprocess({s: fused[s]}))
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_submission_csv(per_image, a)
    write_submission_csv(postprocess({s: ref[s] for s in stems}), b)
    res = compare(a, b)
    assert res["identical"] and res["a_matched_in_b"] == 1.0


def test_weights_pack_unpack_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from weights_io import pack, unpack

    ckpt = tmp_path / "model_final.pth"
    model = {"w": torch.arange(5000, dtype=torch.float32), "b": torch.ones(3)}
    torch.save({"model": model, "optimizer": {"m": torch.zeros(5000)}, "iteration": 7}, ckpt)
    info = pack(ckpt, "toy", tmp_path / "weights", part_bytes=4096)
    assert len(info["parts"]) > 1                                   # really split
    out = unpack(tmp_path / "weights")["toy"]
    loaded = torch.load(out, map_location="cpu")
    assert set(loaded) == {"model"}                                 # optimizer stripped
    assert torch.equal(loaded["model"]["w"], model["w"])

    out.unlink()                                                    # a damaged part must be caught
    part = tmp_path / "weights" / info["parts"][0]
    part.write_bytes(b"x" + part.read_bytes()[1:])
    with pytest.raises(ValueError):
        unpack(tmp_path / "weights")
    assert json.loads((tmp_path / "weights" / "manifest.json").read_text())["toy"]["sha256"] == info["sha256"]
