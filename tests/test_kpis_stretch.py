"""Known-answer tests for the v2 stretch KPIs (pmdb.kpis.stretch) on synthetic masks."""

import numpy as np
import pytest

from pmdb.kpis import STRETCH_REGISTRY, catalogue_columns
from pmdb.kpis import stretch
from pmdb.kpis.common import KpiContext
from tests.synthetic_patterns import discs_mask, make_masks

NM = 50.0  # 50 nm/px -> 20 px per um


def _ctx(si: np.ndarray) -> KpiContext:
    return KpiContext(masks=make_masks(si), nm_per_px=NM, seed_key="t")


def _lattice(spacing: int, radius: int, size: int = 600) -> np.ndarray:
    g = np.arange(spacing // 2, size, spacing)
    pts = np.array([(r, c) for r in g for c in g])
    return discs_mask(pts, radius=radius, size=size)


def _bar_c2_length(bar_px: int, field_px: int, r_max: int) -> float:
    """Pair-fraction C2 of one bar of length ``bar_px`` in a field of ``field_px``, integrated to r_max."""
    r = np.arange(0, min(bar_px, r_max) + 1)
    return float(np.trapz((bar_px - r) / bar_px * field_px / (field_px - r), dx=1.0))


def test_c2_bar_lengths_follow_the_pair_fraction_formula():
    si = np.zeros((300, 300), dtype=bool)
    si[100:110, 50:250] = True  # 200 px along x, 10 px along z
    lab = si.astype(int)
    cx = stretch.two_point_cluster(lab, np.ones_like(si), axis=1, r_max_px=250)
    cz = stretch.two_point_cluster(lab, np.ones_like(si), axis=0, r_max_px=250)
    assert cx[0] == pytest.approx(si.mean())
    assert stretch.c2_length_px(cx) == pytest.approx(_bar_c2_length(200, 300, 250), rel=0.01)
    assert stretch.c2_length_px(cz) == pytest.approx(_bar_c2_length(10, 300, 250), rel=0.01)
    assert stretch.c2_length_px(cx) > 10 * stretch.c2_length_px(cz)


def test_s01_anisotropy_follows_streak_direction():
    si = np.zeros((600, 600), dtype=bool)
    for r in range(20, 600, 40):
        si[r:r + 6, 50:250] = True  # 200 px = 10 um bars, within S01_R_MAX_UM
    out = stretch.s01_two_point_cluster(_ctx(si)).values
    assert out["S01_c2_anisotropy"] > 20
    assert out["S01_c2_length_x_um"] == pytest.approx(_bar_c2_length(200, 600, 200) * 0.05, rel=0.03)


def test_s02_merge_radius_is_half_the_gap():
    si = _lattice(spacing=30, radius=5)  # gap 20 px -> merge at eps = 10 px = 0.5 um
    out = stretch.s02_euler_merge_radius(_ctx(si)).values
    assert out["S02_euler_merge_radius_um"] == pytest.approx(0.5, abs=0.05)
    far = stretch.s02_euler_merge_radius(_ctx(_lattice(spacing=50, radius=5))).values
    assert far["S02_euler_merge_radius_um"] == pytest.approx(1.0, abs=0.05)


def test_half_crossing_interpolates():
    assert stretch.half_crossing(np.array([0, 1, 2.0]), np.array([10, 10, 0.0])) == pytest.approx(1.5)
    assert np.isnan(stretch.half_crossing(np.array([0, 1.0]), np.array([10, 9.0])))


def test_h0_lifetimes_lattice():
    si = _lattice(spacing=30, radius=5)
    births, deaths, n_cens = stretch.h0_lifetimes(si, t_max_px=60)
    n = si.astype(int).sum() // int(si[si].size / len(births)) if False else len(births)
    assert n == 400
    assert np.allclose(births, -5.0, atol=0.5)
    assert n_cens == 1  # one component survives
    merged = deaths[deaths < 60]
    assert np.allclose(merged, 10.0, atol=1.0)


def test_s03_wider_lattice_has_longer_lifetimes():
    near = stretch.s03_persistence(_ctx(_lattice(30, 5))).values
    far = stretch.s03_persistence(_ctx(_lattice(50, 5))).values
    assert near["S03_h0_life_p50_um"] == pytest.approx(15 * 0.05, abs=0.06)
    assert far["S03_h0_life_p50_um"] > near["S03_h0_life_p50_um"] + 0.4
    assert near["S03_h0_inradius_p50_um"] == pytest.approx(0.25, abs=0.03)
    assert near["S03_h0_life_iqr_um"] < 0.06


def test_s04_discs_isotropic_stripes_aligned():
    beta, _ = stretch.tensor_anisotropy(stretch.minkowski_w102(_lattice(30, 5)))
    assert beta > 0.9
    si = np.zeros((600, 600), dtype=bool)
    for r in range(20, 600, 40):
        si[r:r + 6, 50:550] = True
    beta_x, ang_x = stretch.tensor_anisotropy(stretch.minkowski_w102(si))
    beta_z, ang_z = stretch.tensor_anisotropy(stretch.minkowski_w102(si.T))
    assert beta_x < 0.1 and beta_x == pytest.approx(beta_z, rel=0.05)
    assert min(ang_x, 180 - ang_x) < 3
    assert abs(ang_z - 90) < 3


def test_registry_and_catalogue_v2_columns_agree():
    site_cols, _ = catalogue_columns("v2")
    si = _lattice(30, 5)
    for kpi_id in ("S01", "S02", "S03", "S04"):
        out = STRETCH_REGISTRY[kpi_id].site_fn(_ctx(si)).values
        assert set(site_cols[kpi_id]) <= set(out), kpi_id
        assert all(np.isfinite(out[c]) for c in site_cols[kpi_id]), kpi_id
