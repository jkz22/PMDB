"""Tests for scripts/fem_docs.py (local only; not part of the Modal image)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pmdb.fem.config import load_params

_SPEC = importlib.util.spec_from_file_location(
    "fem_docs", Path(__file__).resolve().parents[1] / "scripts" / "fem_docs.py")
fem_docs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fem_docs)


def test_fill_blocks(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("a\n<!-- AUTO:x -->\nold\n<!-- /AUTO:x -->\nb\n<!-- AUTO:y -->\n<!-- /AUTO:y -->\n")
    fem_docs.fill_blocks(f, {"x": "NEW_X", "y": "NEW_Y"})
    t = f.read_text()
    assert "NEW_X" in t and "NEW_Y" in t and "old" not in t
    assert t.count("<!-- AUTO:") == 2 and t.count("<!-- /AUTO:") == 2
    before = f.read_text()
    with pytest.raises(ValueError):
        fem_docs.fill_blocks(f, {"x": "1", "y": "2", "z": "3"})
    with pytest.raises(ValueError):
        fem_docs.fill_blocks(f, {"x": "1"})
    assert f.read_text() == before


def test_lit_param_rows_cover_maps():
    rows = fem_docs.lit_param_rows()
    for name in list(fem_docs.PARAM_MAP) + fem_docs.PHYSICS_ROWS:
        assert name in rows, name
        assert rows[name]["source"] and rows[name]["evidence"]


def test_method_blocks():
    b = fem_docs.method_blocks(load_params(), None)
    assert set(b) == {"params", "physics", "soc", "gates", "cost"}
    assert b["cost"] == fem_docs.PENDING
    assert "96000" in b["params"]


def test_results_blocks_synthetic(tmp_path):
    import json

    import numpy as np
    import pandas as pd

    from pmdb.fem.features import METRIC_NAMES

    rng = np.random.default_rng(0)
    srows, vrows, cases = [], [], []
    for b in ("Batch_1", "Batch_2", "Batch_3"):
        for k in range(2):
            site = f"{b[-1]}s{k}"
            sw = 0.1 + 0.02 * k + 0.01 * int(b[-1])
            for fr in range(11):
                row = {"batch": b, "site": site, "heldout": False, "orientation": "sym", "frame": fr, "s": fr / 10,
                       "converged": True, "failed_at_s": np.nan, "first_pore_closure_s": np.nan}
                row.update({m: float(rng.normal()) for m in METRIC_NAMES})
                row["swelling"] = sw * fr / 10
                srows.append(row)
            vrows.append({"batch": b, "site": site, "heldout": False, "swelling_bottom": sw, "swelling_top": sw,
                          "swelling_sym": sw, "gate_ok": True, "lit_band_ok": True, "sxx_ratio_vt2": 250.0,
                          "porosity_change": -0.01, "porosity_rel_change": -0.1, "J_si_mean": 1.7,
                          "si_yield_frac": 1.0, "failed_at_s": np.nan, "first_pore_closure_s": 0.1})
            for o in ("bottom", "top"):
                cases.append({"tag": "full", "orientation": o, "n_substeps": 11, "newton_its_total": 11,
                              "wall_s": 100.0, "cost_usd": 0.02, "failed_at_s": None})
    pd.DataFrame(srows).to_csv(tmp_path / "site_curves.csv", index=False)
    pd.DataFrame(vrows).to_csv(tmp_path / "validation.csv", index=False)
    (tmp_path / "run_log.json").write_text(json.dumps({
        "orientations_run": ["bottom", "top"], "missing_cases": [], "rerun_cases": [], "budget": {},
        "remediation": [], "cases": cases, "ledger_total_usd": 1.0, "ledger_per_mode_usd": {"full": 1.0},
        "cost_model": {"cap_usd": 180.0}}))
    b = fem_docs.results_blocks(tmp_path, pytest_txt=None, figures=False)
    assert set(b) == {"flags", "validation", "convergence", "cost", "figures", "batch_diff"}
    assert all(fem_docs.PENDING not in v for v in b.values())
