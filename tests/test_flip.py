import numpy as np
import pytest

from utils.flip import FLIPS, flip


@pytest.mark.parametrize("mode", FLIPS)
def test_flip_is_its_own_inverse(mode):
    img = np.random.default_rng(0).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    assert np.array_equal(flip(flip(img, mode), mode), img)


def test_flip_directions():
    mask = np.zeros((4, 6), bool)
    mask[0, 1] = True  # top row, near the left edge
    assert flip(mask, "h")[0, 4]   # left-right: column 1 -> 6 - 1 - 1
    assert flip(mask, "v")[3, 1]   # up-down: row 0 -> 4 - 0 - 1
    assert flip(mask, "hv")[3, 4]
    assert np.array_equal(flip(mask, "none"), mask)


def test_flipped_prediction_maps_back():
    """A fake model that finds a blob in the flipped image gives the blob back at its original place."""
    img = np.zeros((8, 10), np.uint8)
    img[1:3, 2:5] = 255
    for mode in ("h", "v", "hv"):
        pred_on_flipped = flip(img, mode) > 0   # what the model "sees" and paints
        assert np.array_equal(flip(pred_on_flipped, mode), img > 0)


def test_unknown_flip_rejected():
    with pytest.raises(ValueError):
        flip(np.zeros((2, 2)), "x")
