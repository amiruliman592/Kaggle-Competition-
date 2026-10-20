"""Viewer: rendering/statistics without a window, and the GUI wiring on the Agg backend."""

import pytest


@pytest.fixture(scope="module")
def runs():
    from viewer import find_runs
    r = find_runs("val")
    if not r:
        pytest.skip("no raw predictions under Result Phase 3/")
    return r


def test_render_matches_official_score(runs):
    from evaluate import evaluate_official
    from viewer import _gt_entries, predictions_for, render, stems_for

    run, path = next(iter(runs.items()))
    stem = stems_for("val")[0]
    left, right, st = render(stem, "val", run, path)
    assert left.shape == right.shape == (1024, 1024, 3)
    r = evaluate_official(_gt_entries("val")[stem], {stem: predictions_for(stem, path, run)})
    assert (st["TP"], st["FP"], st["FN"]) == (r["TP"], r["FP"], r["FN"])
    assert st["n_pred"] == len(st["labels"])


def test_gui_controls_on_agg(runs, tmp_path, monkeypatch):
    from config import CFG
    from viewer import Viewer

    monkeypatch.setattr(CFG.paths, "OUTPUT_DIR", tmp_path)
    v = Viewer("val", None, backend="Agg")
    first = v.order[0]
    v.step(+1); v.step(-1)
    assert v.order[v.idx] == first
    v.jump("3")
    assert v.idx == 2
    v.toggle("Scores")
    v.set_run(list(v.runs)[-1])
    v.save()
    assert list((tmp_path / "figures").glob("viewer_val_*.png"))
    v.plt.close("all")
