"""Batch classifier tests on synthetic FEM / KPI tile tables (no data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmdb.classify.features import (
    FEM_FEATURES, FEM_REQUIRED_METRICS, TILE_ID, build_arm_table, fem_grid_slices, fem_tile_features,
    kpi_tile_columns, load_fem_tile_curves)

WIDTH = 3494


def synthetic_tile_curves(n_sites=(4, 4, 6), n_heldout=2, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sl = fem_grid_slices(WIDTH)
    sites = []
    for bi, n in enumerate(n_sites):
        sites += [(f"Batch_{bi + 1}", f"s{bi + 1}{i}") for i in range(n)]
    sites += [("Batch_heldout", f"h{i}") for i in range(n_heldout)]
    rows = []
    for batch, site in sites:
        is_b3 = batch in ("Batch_3", "Batch_heldout")
        is_b1 = batch == "Batch_1"
        for orient in ("bottom", "top", "sym"):
            for t in range(6):
                failed = (site == "s10" and t == 0 and orient == "sym")
                for f in range(11):
                    s = f / 10
                    r = {"batch": batch, "site": site, "heldout": batch == "Batch_heldout",
                         "config": "si", "orientation": orient, "region": t, "frame": f, "s": s,
                         "converged": not (failed and f >= 7),
                         "tile_x0_um": sl[t].start * 0.05, "tile_x1_um": sl[t].stop * 0.05,
                         "first_pore_closure_s": 0.6 if is_b3 else np.nan, "failed_at_s": np.nan}
                    shift = 2.0 * is_b3 + 1.0 * is_b1
                    for m in FEM_REQUIRED_METRICS:
                        r[m] = s * (1.0 + shift) + rng.normal(0, 0.1)
                    r["swelling"] = s * (0.05 + 0.02 * is_b3 + 0.01 * is_b1) + rng.normal(0, 0.001)
                    r["pore_closed_frac"] = 0.1 if (is_b3 and f >= 6) else 0.0
                    r["extra_metric"] = rng.normal()
                    if failed and f >= 7:
                        for m in FEM_REQUIRED_METRICS:
                            r[m] = np.nan
                    rows.append(r)
    return pd.DataFrame(rows)


def synthetic_kpi_tiles(curves: pd.DataFrame, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sym = curves[(curves.orientation == "sym") & (curves.frame == 0)]
    cols = kpi_tile_columns()
    rows = []
    for r in sym.itertuples():
        row = {"batch": r.batch, "site": r.site, "heldout": r.heldout, "tile": r.region,
               "tile_x0_um": r.tile_x0_um, "tile_x1_um": r.tile_x1_um}
        for c in cols:
            row[c] = rng.normal() + (1.5 if r.batch in ("Batch_3", "Batch_heldout") else 0.0)
        row["K16_si_graphite_dist_median_um"] = 0.0
        row["nan_reason"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def curves() -> pd.DataFrame:
    return synthetic_tile_curves()


def test_fem_grid_slices():
    sl = fem_grid_slices(WIDTH)
    assert len(sl) == 6
    assert sl[0].start == 400 and sl[-1].stop == 3094
    assert all(a.stop == b.start for a, b in zip(sl[:-1], sl[1:]))


def test_fem_loader_filters_and_validates(curves, tmp_path):
    p = tmp_path / "t.csv"
    curves.to_csv(p, index=False)
    out = load_fem_tile_curves(p)
    assert sorted(out.tile.unique()) == list(range(6))
    assert len(out) == curves.site.nunique() * 66
    bad = curves[~((curves.site == "s20") & (curves.region == 5))]
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_fem_tile_curves(p)
    curves.rename(columns={"region": "tile"}).to_csv(p, index=False)
    assert len(load_fem_tile_curves(p)) == len(out)


def test_fem_features(tmp_path):
    cv = synthetic_tile_curves()
    cv["swelling"] = (0.05 * cv["s"]).where(cv["swelling"].notna())
    p = tmp_path / "t.csv"
    cv.to_csv(p, index=False)
    feats = fem_tile_features(load_fem_tile_curves(p))
    assert [c for c in feats.columns if c.startswith("fem_")] == list(FEM_FEATURES)
    ok = feats[(feats.site == "s11") & (feats.tile == 1)].iloc[0]
    assert ok.fem_swell_slope_early == pytest.approx(0.05, rel=1e-9)
    assert ok.fem_swell_slope_late == pytest.approx(0.05, rel=1e-9)
    failed = feats[(feats.site == "s10") & (feats.tile == 0)].iloc[0]
    assert np.isnan(failed.fem_swell_100) and np.isnan(failed.fem_first_closure_s)
    assert feats[(feats.site == "s20") & (feats.tile == 2)].iloc[0].fem_first_closure_s == pytest.approx(1.1)
    assert feats[(feats.site == "s30") & (feats.tile == 2)].iloc[0].fem_first_closure_s == pytest.approx(0.6)


def test_arm_join_checks_alignment(curves, tmp_path):
    p = tmp_path / "t.csv"
    curves.to_csv(p, index=False)
    fem = fem_tile_features(load_fem_tile_curves(p))
    kpi = synthetic_kpi_tiles(curves)
    table, feats = build_arm_table("KPI+FEM", kpi, fem)
    assert len(feats) == 15 + 16
    assert len(table) == curves.site.nunique() * 6
    kpi2 = kpi.copy()
    kpi2.loc[0, "tile_x0_um"] += 0.5
    with pytest.raises(ValueError):
        build_arm_table("KPI+FEM", kpi2, fem)
