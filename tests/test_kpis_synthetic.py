"""Spec 002 Acceptance 1: synthetic known-answer tests (no real data needed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pmdb.kpis import REGISTRY, catalogue_columns, check_registry, compute_site_kpis, compute_tile_kpis
from pmdb.kpis import crossphase, fields, objects, pointpattern
from pmdb.kpis.common import KpiContext, stable_seed
from pmdb.segment import segment
from synthetic_patterns import (NM_PER_PX, SIZE, discs_mask, hex_points, make_masks, poisson_points,
                                stripe_points, synthetic_bse, thomas_points)

SEED = 20240601


def _ctx(si, graphite=None, pore=None, key="synthetic"):
    return KpiContext(make_masks(si, graphite, pore), NM_PER_PX, key)


@pytest.fixture(scope="module")
def poisson_ctx():
    pts = poisson_points(np.random.default_rng(SEED), 400)
    return _ctx(discs_mask(pts), key="poisson")


@pytest.fixture(scope="module")
def thomas_ctx():
    pts = thomas_points(np.random.default_rng(SEED))
    return _ctx(discs_mask(pts), key="thomas")


# ------------------------------------------------------------------ point patterns

def test_poisson(poisson_ctx):
    assert poisson_ctx.n_objects == 400
    k05 = pointpattern.k05_voronoi_tile(poisson_ctx).values
    k07 = pointpattern.k07_nearest_neighbour(poisson_ctx).values
    k08 = pointpattern.k08_pair_correlation(poisson_ctx).values
    assert k05["K05_voronoi_sigma"] == pytest.approx(0.53, abs=0.07)
    assert k07["K07_R_csr"] == pytest.approx(1.0, abs=0.1)
    assert k07["K07_R_rl"] == pytest.approx(1.0, abs=0.1)
    assert k08["K08_pcf_excess_max_x"] <= 0.2
    assert k08["K08_pcf_excess_max_z"] <= 0.2


def test_hexagonal_lattice():
    ctx = _ctx(discs_mask(hex_points()), key="hex")
    k05 = pointpattern.k05_voronoi_tile(ctx).values
    k07 = pointpattern.k07_nearest_neighbour(ctx).values
    assert k07["K07_R_csr"] > 1.8
    assert k05["K05_voronoi_sigma"] < 0.15


def test_thomas_cluster(thomas_ctx):
    k05 = pointpattern.k05_voronoi_tile(thomas_ctx).values
    k07 = pointpattern.k07_nearest_neighbour(thomas_ctx).values
    k08 = pointpattern.k08_pair_correlation(thomas_ctx).values
    assert k07["K07_R_rl"] < 0.7
    assert k05["K05_voronoi_sigma"] > 0.8
    for d in ("x", "z"):
        assert k08[f"K08_pcf_excess_max_{d}"] > 0.5
        assert 0 < k08[f"K08_pcf_rpeak_{d}_um"] < 3.0


def test_k04_parameters(thomas_ctx, poisson_ctx):
    thomas = objects.k04_agglomerates(thomas_ctx, d_um=0.5, d_star_um=1.0).values
    poisson = objects.k04_agglomerates(poisson_ctx, d_um=0.5, d_star_um=1.0).values
    assert thomas["K04_agglom_frac"] > 0.5
    assert poisson["K04_agglom_frac"] < 0.1


def test_admissible_stripe_null():
    """Si only inside horizontal admissible stripes (~30% of area), graphite elsewhere."""
    stripes = [(a, a + 100) for a in range(100, SIZE, 333)][:6]
    adm = np.zeros((SIZE, SIZE), dtype=bool)
    for a, b in stripes:
        adm[a:b] = True
    assert adm.mean() == pytest.approx(0.30, abs=0.01)
    si = discs_mask(stripe_points(np.random.default_rng(SEED), 400, stripes))
    ctx = _ctx(si, graphite=~adm, key="stripes")
    k07 = pointpattern.k07_nearest_neighbour(ctx).values
    assert k07["K07_R_rl"] == pytest.approx(1.0, abs=0.15)
    assert k07["K07_R_csr"] < 0.9


# ------------------------------------------------------------------ localisation

def test_localised_top_rows():
    pts = poisson_points(np.random.default_rng(SEED), 400, rows=(0, int(0.4 * SIZE)))
    k12 = fields.k12_depth_profile(_ctx(discs_mask(pts))).values
    assert k12["K12_depth_maxdev"] > 0.8


def test_uniform_random_map(poisson_ctx):
    k12 = fields.k12_depth_profile(poisson_ctx).values
    k10 = fields.k10_scale_of_segregation(poisson_ctx).values
    k11 = fields.k11_lacey(poisson_ctx).values
    assert k12["K12_depth_maxdev"] < 0.2
    assert k10["K10_cv_slope"] < -0.6
    assert k11["K11_lacey_w10"] > 0.7


# ------------------------------------------------------------------ cross-phase

def _disc(centre, r, size=400):
    return discs_mask(np.array([centre], dtype=float), r, size)


def test_si_embedded_in_graphite():
    si = _disc((200, 200), 4)
    graphite = _disc((200, 200), 60)
    assert crossphase.k15_contact(_ctx(si, graphite)).values["K15_si_graphite_contact_frac"] == 1.0


def test_si_inside_pore():
    si = _disc((200, 120), 4)
    pore = _disc((200, 120), 60)
    graphite = np.zeros_like(si)
    graphite[:, 300:] = True
    ctx = _ctx(si, graphite, pore)
    assert crossphase.k15_contact(ctx).values["K15_si_graphite_contact_frac"] == 0.0
    assert crossphase.si_graphite_distance(*ctx.si_labels, ctx.masks.graphite, ctx.px_um) > 0


# ------------------------------------------------------------------ segmentation

def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_segmentation_recovers_synthetic_bse():
    """Mid-grey matrix, bright grainy discs, dark holes, Gaussian noise (contrast/noise ~3.5, as in real BSE)."""
    img, si_true, pore_true = synthetic_bse(np.random.default_rng(SEED), noise=0.1)
    masks = segment(SimpleNamespace(image=img[..., None], nm_per_px=NM_PER_PX))
    assert _iou(masks.si, si_true) > 0.85
    assert _iou(masks.pore, pore_true) > 0.85


# ------------------------------------------------------------------ plumbing

def test_registry_covers_catalogue():
    check_registry()
    site_cols, tile_cols = catalogue_columns()
    assert set(site_cols) == set(REGISTRY)
    assert not any(k.startswith("S") for k in site_cols)  # no v2 KPIs


def test_stable_seed():
    assert stable_seed("Batch_1/abc/tile0") == stable_seed("Batch_1/abc/tile0")
    assert stable_seed("Batch_1/abc/tile0") != stable_seed("Batch_1/abc/tile1")


def test_all_site_columns_finite_on_synthetic_slice():
    rng = np.random.default_rng(SEED)
    h, w = 600, 2400
    pts = poisson_points(rng, 300, rows=(0, h), size=w)
    pts = pts[pts[:, 0] < h - 6]
    si = discs_mask(pts, size=(h, w))
    graphite = np.zeros((h, w), dtype=bool)
    for c in range(100, w, 400):
        graphite[100:180, c:c + 200] = True
    pore = discs_mask(poisson_points(rng, 40, rows=(0, h), size=w, min_dist=40)[:, :], 8, (h, w))
    pore[300:420, 1000:1120] = True  # one large void -> artefact
    masks = make_masks(si, graphite & ~pore, pore)
    masks.artefact[300:420, 1000:1120] = True
    masks.pore[300:420, 1000:1120] = False
    masks.admissible &= ~masks.artefact
    bse = rng.uniform(0, 1, (h, w))
    values, curves, _ = compute_site_kpis(masks, bse, NM_PER_PX, "synthetic/slice")
    site_cols, _ = catalogue_columns()
    expected = {c for cols in site_cols.values() for c in cols}
    assert set(values) == expected
    assert all(np.isfinite(v) for v in values.values())
    rows = compute_tile_kpis(masks, bse, NM_PER_PX, "synthetic/slice")
    assert len(rows) == 4
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and not np.isfinite(v):
                assert r["nan_reason"]
