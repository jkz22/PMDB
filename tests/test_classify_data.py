"""Batch classifier tests against the real data and the committed classifier outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmdb import segment as segment_mod
from pmdb.classify.features import fem_grid_slices
from pmdb.classify.kpi_tiles import compute_tile_kpis_on_grid
from pmdb.io import load_site
from pmdb.kpis import compute_tile_kpis, tile_slices

pytestmark = pytest.mark.data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "classifier"


def test_grid_copy_matches_original():
    key = "Batch_1/4ih2ggld"
    raw = load_site("Batch_1", "4ih2ggld", resolution="half", normalise="none")
    norm = load_site("Batch_1", "4ih2ggld", resolution="half")
    masks = segment_mod.segment(raw)
    bse = norm.image[..., 0]
    a = compute_tile_kpis_on_grid(masks, bse, 50.0, key, tile_slices(masks.shape[1]))
    b = compute_tile_kpis(masks, bse, 50.0, key)
    assert len(a) == len(b) == 4
    for ra, rb in zip(a, b):
        assert ra.keys() == rb.keys()
        for k in ra:
            if k == "nan_reason":
                assert ra[k] == rb[k]
            else:
                np.testing.assert_allclose(ra[k], rb[k], rtol=0, atol=0, equal_nan=True)


def test_kpi_tiles6_table():
    df = pd.read_csv(OUT / "kpi_tiles6.csv")
    assert df.site.nunique() == 34 and len(df) == 204
    assert (df.groupby("site").size() == 6).all()
    held = df[df.heldout.astype(str) == "True"]
    assert held.site.nunique() == 3 and (held.batch == "Batch_heldout").all()
    widths = {}
    for man in (ROOT / "cache/half/manifest.csv", ROOT / "cache_heldout/half/manifest.csv"):
        m = pd.read_csv(man)
        widths.update(dict(zip(m.site, m.width)))
    for site, g in df.groupby("site"):
        t0 = g[g.tile == 0].iloc[0]
        assert t0.tile_x0_um == pytest.approx(fem_grid_slices(int(widths[site]))[0].start * 0.05)


def test_outputs_consistent():
    ls = pd.read_csv(OUT / "loso_site_predictions.csv")
    hp = pd.read_csv(OUT / "heldout_predictions.csv")
    mt = pd.read_csv(OUT / "metrics.csv")
    arms = sorted(mt.arm.unique())
    for arm in arms:
        a = ls[ls.arm == arm]
        assert a.site.nunique() == 31 and len(a) == 31 and "Batch_heldout" not in set(a.batch)
        h = hp[hp.arm == arm]
        assert sorted(h.site) == ["3e122cbj", "fn0mhxef", "xrv9xvzb"]
        assert ((h.confidence > 0) & (h.confidence <= 1)).all()
        assert sorted(mt[mt.arm == arm].level) == ["end_to_end", "stage1", "stage2"]
    fin = hp[hp.final.astype(str) == "True"]
    assert fin.arm.nunique() == 1 and len(fin) == 3
    assert fin.explanation.fillna("").str.len().gt(0).all()
