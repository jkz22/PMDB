"""Unit checks for scripts/eval_cut.py (CUT evaluation helpers)."""
import importlib.util
from pathlib import Path

import numpy as np

from pmdb.segment import segment_bse

_spec = importlib.util.spec_from_file_location("eval_cut", Path(__file__).resolve().parents[1] / "scripts" / "eval_cut.py")
EC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EC)


def _phantom():
    rng = np.random.default_rng(1)
    img = np.full((300, 300), 120.0)
    yy, xx = np.mgrid[:300, :300]
    for cy, cx, r, v in ((60, 60, 25, 10.0), (200, 90, 18, 10.0), (150, 220, 30, 230.0), (250, 250, 12, 230.0)):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = v
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def test_fixed_threshold_phases_reproduce_segment_bse_on_the_input():
    img = _phantom()
    valid = np.ones(img.shape, bool)
    m = segment_bse(img.astype(np.float64), EC.E.NM)
    pore, si = EC.phases_fixed(img, valid, m.params["T_pore"], m.params["T_si"])
    assert (pore == m.pore).all() and (si == m.si).all()
    assert pore.any() and si.any()


def test_identity_translation_scores_zero_label_change():
    img = _phantom()
    valid = np.ones(img.shape, bool)
    prm = segment_bse(img.astype(np.float64), EC.E.NM).params
    a = EC.phases_fixed(img, valid, prm["T_pore"], prm["T_si"])
    b = EC.phases_fixed(img.copy(), valid, prm["T_pore"], prm["T_si"])
    assert all((x == y).all() for x, y in zip(a, b))
