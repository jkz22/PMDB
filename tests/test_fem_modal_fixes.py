"""Unit tests for the PR #32 review fixes (no Modal calls, no data)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

modal_fem = pytest.importorskip("modal_fem")
import fem_collect  # noqa: E402


def test_failed_cases():
    cases = [{"batch": "B", "site": s, "orientation": "bottom"} for s in ("a", "b", "c")]
    res = [{"meta": {"batch": "B", "site": "a", "orientation": "bottom"}},
           {"meta": {"batch": "B", "site": "b", "orientation": "bottom", "error": "x"}}]
    assert modal_fem._failed_cases(cases, res) == ["B/b", "B/c"]


def test_worst_with_retry_exceeds_single():
    one = modal_fem._worst(10, 4.0, 16384, 7200)
    assert modal_fem._worst_with_retry(10, 4.0, 16384, 7200) > 2 * one


def test_budget_refuses_worst_case(monkeypatch):
    monkeypatch.setattr(modal_fem, "_load_budget", lambda: {"drop_stage2": False, "drop_top": False, "log": []})
    monkeypatch.setattr(modal_fem, "ledger_total", lambda: 0.0)
    items = [{"orientation": "bottom"}] * 34
    with pytest.raises(SystemExit) as e:
        modal_fem.budget_check("full", "full", items, 4.0, 16384, 28800)
    assert e.value.code == 3


def _write(d: Path, site: str, err=None):
    p = d / "bottom" / f"B__{site}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"batch": "B", "site": site, "orientation": "bottom", "error": err},
                             "site_rows": []}))


def test_g2_fails_on_missing_case(tmp_path, capsys):
    _write(tmp_path / "r100", "a")
    _write(tmp_path / "r100", "b")
    _write(tmp_path / "r50", "a")
    assert fem_collect.g2(tmp_path / "r100", tmp_path / "r50") == 1
    assert "B/b/bottom" in capsys.readouterr().out
