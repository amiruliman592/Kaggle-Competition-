import csv
import datetime as dt
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "external"))

from download_gong import (STATIONS, allowed_days, block_of, day_of, excluded_days,  # noqa: E402
                           load_exclusion_ids, parse_listing, pick_frame, qc_ok)
from make_pseudo_labels import to_coco  # noqa: E402
from utils.rle import encode_mask  # noqa: E402

MANIFEST = ROOT / "external" / "manifest.csv"


def test_exclusion_val_blocks_and_test_days():
    val, test = ["20140609195854Bh"], ["20150125172714Mh"]
    bad = excluded_days(val, test, dt.date(2014, 1, 1), dt.date(2015, 12, 31))
    vday = dt.date(2014, 6, 9)
    block = [d for d in (vday + dt.timedelta(k) for k in range(-30, 31)) if block_of(d) == block_of(vday)]
    assert len(block) == 27 and set(block) <= bad                       # whole val block
    assert {dt.date(2015, 1, 24), dt.date(2015, 1, 25), dt.date(2015, 1, 26)} <= bad   # test day +-1
    assert dt.date(2015, 1, 27) not in bad and dt.date(2015, 1, 23) not in bad
    assert len(bad) == 27 + 3


def test_allowed_days_never_touch_real_val_or_test():
    val, test = load_exclusion_ids()
    ok = set(allowed_days(val, test))
    val_blocks = {block_of(day_of(s)) for s in val}
    assert not any(block_of(d) in val_blocks for d in ok)
    assert not {day_of(s) + dt.timedelta(k) for s in test for k in (-1, 0, 1)} & ok
    assert len(ok) > 2500


def test_parse_listing_keeps_competition_stations_only():
    html = ('<a href="20150125000214Mh.jpg">x</a><a href="20150125000014Mh.jpg">x</a>'
            '<a href="20150125172714Sh.jpg">x</a><a href="20150125000054Bh.jpg">x</a>'
            '<a href="20150125000054Bh.fits.fz">x</a><a href="/secret0x2112">x</a>')
    assert parse_listing(html) == {"M": ["20150125000014Mh.jpg", "20150125000214Mh.jpg"],
                                   "B": ["20150125000054Bh.jpg"]}
    assert parse_listing("") == {}


def test_pick_frame_balances_stations_and_skips_day_edges():
    names = {s: [f"20150125{h:02d}0000{s}h.jpg" for h in range(10)] for s in "BM"}
    counts = {s: 0 for s in STATIONS}
    counts["B"] = 3
    rng = random.Random(0)
    for _ in range(20):
        name = pick_frame(names, counts, rng)
        assert name[14] in "BM" and name not in (names[name[14]][0], names[name[14]][-1])
    assert abs(counts["B"] - counts["M"]) <= 1
    assert pick_frame({}, counts, rng) is None


def test_qc_ok():
    bounds = {"disk_frac": (0.7, 0.8), "sharpness": (10, 100)}
    assert qc_ok({"disk_frac": 0.75, "sharpness": 50}, bounds)
    assert not qc_ok({"disk_frac": 0.5, "sharpness": 50}, bounds)


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not built yet (Colab, M3)")
def test_manifest_has_no_excluded_day():
    val, test = load_exclusion_ids()
    ok = set(allowed_days(val, test))
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    days = [day_of(r["file"]) for r in rows]
    assert rows and all(d in ok for d in days)
    assert len(set(days)) == len(days)                                   # one frame per day
    assert all(r["station"] in STATIONS and r["file"][14] == r["station"] for r in rows)


def test_to_coco_drops_empty_images():
    m = np.zeros((2048, 2048), np.uint8)
    m[10:20, 30:60] = 1
    r = encode_mask(m)
    r["score"] = 0.9
    coco = to_coco({"20150125000014Mh": [r], "20150125000054Bh": []}, 0.85, "test")
    assert [i["file_name"] for i in coco["images"]] == ["20150125000014Mh.jpg"]
    (a,) = coco["annotations"]
    assert a["area"] == 300 and a["bbox"] == [30.0, 10.0, 30.0, 10.0] and a["image_id"] == 1


def test_m3_notebook_compiles_and_saves_to_own_drive():
    import ast
    import json
    nb = json.loads((ROOT / "notebooks" / "colab_phase6_mirul_m3.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in cells:
        ast.parse("\n".join(l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l
                            for l in src.splitlines()))
    text = "\n".join(cells)
    assert "MyDrive/m3_mirul" in text and "--image-dir" in text and "--mask-res 28" in text


def test_image_dir_flag_only_with_dump():
    from run_pipeline import parse_args
    assert parse_args(["--step", "dump", "--image-dir", "x"]).image_dir == "x"
    with pytest.raises(SystemExit):
        parse_args(["--step", "val", "--image-dir", "x"])
