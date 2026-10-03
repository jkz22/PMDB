"""Unit tests for the materials-KPI converter (synthetic inputs) plus a real-data staleness guard."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from scripts import materials_kpis_to_csv as conv

ROOT = Path(__file__).resolve().parents[1]


def _long() -> pd.DataFrame:
    rows = []
    for img, batch, d50 in (("img_aaa1_BSE", "Batch_1", 2.5), ("img_bbb2_BSE", "Batch_3", 5.0)):
        rows.append({"image": img, "batch": batch, "kpi": "mat_graphite_d50_um", "value": d50, "tier": "t"})
        rows.append({"image": img, "batch": batch, "kpi": "mat_porosity", "value": 0.1, "tier": "t"})
    return pd.DataFrame(rows)


def _ref(sites=(("Batch_1", "aaa1"), ("Batch_3", "bbb2"))) -> pd.DataFrame:
    return pd.DataFrame(list(sites), columns=["batch", "site"])


def test_to_wide_strips_keys_and_orders_canonically():
    w = conv.to_wide(_long())
    assert list(w.columns) == ["batch", "site", "mat_porosity", "mat_graphite_d50_um"]
    assert list(w["site"]) == ["aaa1", "bbb2"]
    assert list(w["mat_graphite_d50_um"]) == pytest.approx([2.5, 5.0])
    assert list(w["mat_porosity"]) == [0.1, 0.1]


def test_to_wide_appends_unknown_kpis_sorted():
    extra = [
        {"image": img, "batch": b, "kpi": k, "value": 1.0, "tier": "t"}
        for k in ("geo_zeta", "geo_alpha")
        for img, b in (("img_aaa1_BSE", "Batch_1"), ("img_bbb2_BSE", "Batch_3"))
    ]
    w = conv.to_wide(pd.concat([_long(), pd.DataFrame(extra)], ignore_index=True))
    assert list(w.columns[2:]) == ["mat_porosity", "mat_graphite_d50_um", "geo_alpha", "geo_zeta"]


def test_to_wide_rejects_legacy_pixel_kpis():
    long = _long().replace({"kpi": {"mat_graphite_d50_um": "mat_graphite_d50"}})
    with pytest.raises(conv.ConversionError, match="pixel-unit"):
        conv.to_wide(long)


def test_to_wide_rejects_duplicate_rows():
    long = pd.concat([_long(), _long().iloc[[0]]], ignore_index=True)
    with pytest.raises(conv.ConversionError, match="duplicate"):
        conv.to_wide(long)


def test_to_wide_rejects_unknown_image_pattern():
    long = _long()
    long.loc[0, "image"] = "img_aaa1_SE"
    with pytest.raises(conv.ConversionError, match=re.escape("img_<site>_BSE")):
        conv.to_wide(long)


def test_align_rejects_site_mismatch():
    with pytest.raises(conv.ConversionError, match="site"):
        conv.align_to_reference(conv.to_wide(_long()), _ref((("Batch_1", "aaa1"),)))


def test_align_orders_rows_like_reference():
    ref = _ref((("Batch_3", "bbb2"), ("Batch_1", "aaa1")))
    out = conv.align_to_reference(conv.to_wide(_long()), ref)
    assert list(out["site"]) == ["bbb2", "aaa1"]


def test_main_roundtrip_and_error_exit(tmp_path, capsys):
    pq, ref, out = tmp_path / "k.parquet", tmp_path / "r.csv", tmp_path / "o.csv"
    _long().to_parquet(pq)
    _ref().to_csv(ref, index=False)
    args = ["--parquet", str(pq), "--reference", str(ref), "--out", str(out)]
    assert conv.main(args) == 0
    expected = conv.align_to_reference(conv.to_wide(_long()), _ref())
    got = pd.read_csv(out, dtype={"batch": str, "site": str})
    pd.testing.assert_frame_equal(got, expected)
    _ref((("Batch_1", "aaa1"),)).to_csv(ref, index=False)
    capsys.readouterr()
    assert conv.main(args) == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.data
def test_converter_on_real_results():
    pq = ROOT / "results/kpis.parquet"
    refp = ROOT / "outputs/kpis/site_kpis.csv"
    if not pq.exists() or not refp.exists():
        pytest.skip("results/kpis.parquet or outputs/kpis/site_kpis.csv absent")
    ref = pd.read_csv(refp, dtype={"batch": str, "site": str})
    wide = conv.align_to_reference(conv.to_wide(pd.read_parquet(pq)), ref)
    assert wide.shape == (31, 10)
    assert list(wide.columns[2:]) == [
        "mat_porosity", "mat_bright_fraction", "mat_active_fraction",
        "mat_graphite_d10_um", "mat_graphite_d50_um", "mat_graphite_d90_um",
        "mat_crack_fraction", "mat_rim_coverage"]
    assert list(zip(wide.batch, wide.site)) == list(zip(ref.batch, ref.site))
    committed = pd.read_csv(ROOT / "outputs/kpis/materials_site_kpis.csv", dtype={"batch": str, "site": str})
    pd.testing.assert_frame_equal(committed, wide, check_exact=False, rtol=1e-12)
