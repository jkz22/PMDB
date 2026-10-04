"""Pure helpers of scripts/eval_harmonise_ext.py on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_harmonise_ext as E  # noqa: E402

import pandas as pd  # noqa: E402
import pmdb.clean as C  # noqa: E402


def test_fill_invalid_ignores_masked_pixels():
    img = np.full((10, 10, 3), 50, dtype=np.uint8)
    img[:, :3, 1] = 255
    msk = np.zeros(img.shape, dtype=np.uint8)
    msk[:, :3, 1] = C.BIT_CLIP_HIGH
    out = E._fill_invalid(img, msk)
    assert (out[..., 1] == 50).all() and (img[:, :3, 1] == 255).all()
    assert (out[..., 0] == img[..., 0]).all()


def test_recall_is_not_accuracy():
    y = np.array([True] * 17 + [False] * 14)
    pred = np.ones(31, dtype=bool)
    assert E._recall(pred, y) == 1.0
    assert (pred == y).mean() == pytest.approx(17 / 31)


def test_gallery_includes_clipping_only_fields():
    t = pd.DataFrame({"batch": ["B", "B"], "site": ["a", "b"], "detector": ["Inlens", "BSE"],
                      "frac_invalid_stats_inner": [0.07, 0.0], "frac_crack": [0.0, 0.0]})
    assert E._gallery_keys(t) == [("B", "a")]
