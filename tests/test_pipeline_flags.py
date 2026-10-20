import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def restore_cfg():
    from config import CFG
    saved = (CFG.model.MODEL_CONFIG, CFG.train.TRAIN_RESIZE, CFG.infer.INFER_RESIZE)
    yield CFG
    CFG.model.MODEL_CONFIG, CFG.train.TRAIN_RESIZE, CFG.infer.INFER_RESIZE = saved


def test_backbone_and_resolution_flags(restore_cfg):
    from run_pipeline import apply_overrides, parse_args
    cfg = restore_cfg
    apply_overrides(parse_args(["--backbone", "x101", "--resolution", "2048"]))
    assert cfg.model.MODEL_CONFIG == "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    assert cfg.train.TRAIN_RESIZE == cfg.infer.INFER_RESIZE == 2048


def test_no_flags_keep_phase3_defaults(restore_cfg):
    from run_pipeline import apply_overrides, parse_args
    cfg = restore_cfg
    apply_overrides(parse_args([]))
    assert cfg.model.MODEL_CONFIG == "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    assert cfg.train.TRAIN_RESIZE == cfg.infer.INFER_RESIZE == 1024


def _code_cells(name):
    nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_phase4_notebook_compiles_and_uses_one_resolution():
    cells = _code_cells("colab_phase4_native.ipynb")
    for src in cells:
        # Shell (!) and magic (%) lines are IPython syntax, not Python
        py = "\n".join(l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l
                       for l in src.splitlines())
        ast.parse(py)
    text = "\n".join(cells)
    flags = re.findall(r'FLAGS_\w+ = "([^"]+)"', text)
    assert flags and all("--resolution 2048" in f for f in flags)
    # Every pipeline command of a model passes its FLAGS (same backbone + resolution for train and dump)
    runs = re.findall(r'run\(f"([^"]+)"\)', text)
    assert runs and all("{FLAGS_" in r for r in runs)
