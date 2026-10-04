import hashlib
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def test_null_guard(tmp_path):
    ev = importlib.import_module("xgb_kpi_eval")
    f = tmp_path / "t.csv"
    f.write_text("a\n1\n")
    good = {"fem_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    ev.check_null_matches_table(good, f, "outputs/fem/site_curves.csv")
    with pytest.raises(SystemExit):
        ev.check_null_matches_table({"fem_sha256": "0" * 64}, f, "outputs/fem/site_curves.csv")
    with pytest.raises(SystemExit):
        ev.check_null_matches_table({}, f, "outputs/fem/site_curves.csv")


def test_ledger_rows_cover_failed_chunks():
    pytest.importorskip("modal")
    me = importlib.import_module("modal_eval")
    calls = [("X1", None, None, 3, 0, [0, 1]), ("X1", None, None, 3, 0, [2, 3])]
    outs = [{"arm": "X1", "ks": [0, 1], "acc": [0.1, 0.2], "wall_s": 5.0}, RuntimeError("boom")]
    rows, res, bad = me.ledger_rows(calls, outs, "main", "t")
    assert len(rows) == 2 and len(res) == 1 and len(bad) == 1
    assert rows[1]["status"] == "lost" and rows[1]["wall_s"] == me.TIMEOUT_S and rows[1]["cost_usd"] > rows[0]["cost_usd"]


def test_legacy_null_needs_explicit_opt_in():
    ev = importlib.import_module("xgb_kpi_eval")
    default = "outputs/fem/site_curves.csv"
    with pytest.raises(SystemExit, match="no FEM table hash"):
        ev.check_null_matches_table({}, ROOT / default, default)
    ev.check_null_matches_table({}, ROOT / default, default, accept_legacy=True)


def test_inputs_hash_detects_kpi_change():
    import numpy as np
    ev = importlib.import_module("xgb_kpi_eval")
    from pmdb.xgb_kpi import training_fingerprint
    tr = {"index": np.array([["B1", "a"], ["B2", "b"]], dtype=object), "kpi": np.array([[1.0, 2.0], [3.0, 4.0]])}
    y = ["B1", "B2"]
    raw = {"inputs_sha256": training_fingerprint(tr, y)}
    ev.check_null_matches_inputs(raw, tr, y)
    tr2 = {**tr, "kpi": tr["kpi"] + np.array([[0.0, 0.5], [0.0, 0.0]])}
    with pytest.raises(SystemExit, match="different training inputs"):
        ev.check_null_matches_inputs(raw, tr2, y)
    with pytest.raises(SystemExit, match="training-inputs hash"):
        ev.check_null_matches_inputs({}, tr, y)
    ev.check_null_matches_inputs({}, tr, y, accept_legacy=True)
