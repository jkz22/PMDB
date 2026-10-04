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
