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
