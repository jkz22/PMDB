"""Left-to-right chain ordering helper of scripts/plot_parents.py (no data needed)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "plot_parents", Path(__file__).resolve().parents[1] / "scripts" / "plot_parents.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)


def row(chain, left="[]", right="[]"):
    return {"chain": chain, "art_left_of_first": left, "art_right_of_last": right}


def names(rows):
    return [r["chain"] for r in rows]


def test_verdict_reverses_csv_order():
    rows = [row("p"), row("x")]
    assert names(pp.order_chains(rows, [("x", "p")])) == ["x", "p"]


def test_multi_site_chain_and_artefact_fallback():
    rows = [row("avn", right="[0]"), row("fn | 38", left="[0]")]
    assert names(pp.order_chains(rows, [("fn", "avn")])) == ["fn | 38", "avn"]
    assert names(pp.order_chains(rows, [])) == ["fn | 38", "avn"]


def test_no_evidence_keeps_input_order():
    rows = [row("a"), row("b")]
    assert names(pp.order_chains(rows, [])) == ["a", "b"]
