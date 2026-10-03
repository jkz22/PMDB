"""Known-answer tests for pmdb.functional (no real data needed)."""

from __future__ import annotations

import numpy as np
import pytest

from pmdb import functional as F
from pmdb.segment import Masks


def _masks(si=None, graphite=None, pore=None, shape=(64, 64)) -> Masks:
    z = np.zeros(shape, dtype=bool)
    si = z if si is None else si
    graphite = z if graphite is None else graphite
    pore = z if pore is None else pore
    return Masks(si=si, graphite=graphite, pore=pore, artefact=z.copy(), admissible=~graphite)


# ------------------------------------------------------------------ percolation / tortuosity

def test_open_domain_has_unit_tortuosity():
    t = F.tortuosity_factor(np.ones((40, 30), dtype=bool), axis=0)
    assert t.spans and t.phase_frac == 1.0
    assert t.d_eff == pytest.approx(1.0, abs=1e-6)
    assert t.tau == pytest.approx(1.0, abs=1e-6)


def test_straight_channels_tau_one_in_both_directions():
    m = np.zeros((50, 40), dtype=bool)
    m[:, ::2] = True  # vertical open columns, half the area
    tz = F.tortuosity_factor(m, axis=0)
    assert tz.phase_frac == pytest.approx(0.5)
    assert tz.d_eff == pytest.approx(0.5, abs=1e-6)
    assert tz.tau == pytest.approx(1.0, abs=1e-6)
    tx = F.tortuosity_factor(m, axis=1)
    assert not tx.spans and tx.d_eff == 0.0 and np.isinf(tx.tau)
    assert F.percolation(m, 0).spans and not F.percolation(m, 1).spans


def test_blocked_domain_does_not_percolate():
    m = np.ones((30, 30), dtype=bool)
    m[15, :] = False
    p = F.percolation(m, 0)
    assert not p.spans and p.n_clusters == 2 and p.spanning_frac == 0.0
    assert F.percolation(m, 1).spans


def test_dead_end_cluster_carries_no_flux():
    m = np.zeros((40, 20), dtype=bool)
    m[:, 2] = True  # spanning channel
    m[5:15, 10] = True  # isolated dead-end pocket
    t = F.tortuosity_factor(m, axis=0)
    assert t.spans
    assert t.d_eff == pytest.approx(1.0 / 20, abs=1e-6)  # one open column out of 20
    assert t.tau == pytest.approx(t.phase_frac / t.d_eff)


def test_tilted_channel_increases_tau():
    # a staircase channel of unit width has a longer path than the straight one
    n = 60
    m = np.zeros((n, n), dtype=bool)
    for r in range(n):
        m[r, r] = True
        if r + 1 < n:
            m[r, r + 1] = True
    t = F.tortuosity_factor(m, axis=0)
    assert t.spans and t.tau > 1.3


def test_domain_argument_excludes_pixels_from_fraction():
    m = np.ones((20, 20), dtype=bool)
    dom = np.ones((20, 20), dtype=bool)
    dom[:, :10] = False
    t = F.tortuosity_factor(m, axis=0, domain=dom)
    assert t.phase_frac == pytest.approx(2.0)  # fraction over the restricted domain, by design


# ------------------------------------------------------------------ chords and access

def test_chord_lengths_directional():
    m = np.zeros((10, 30), dtype=bool)
    m[2, 5:15] = True  # one horizontal run of 10
    m[4:7, 20] = True  # one vertical run of 3
    assert sorted(F.chord_lengths(m, 1).tolist()) == [1.0, 1.0, 1.0, 10.0]
    assert sorted(F.chord_lengths(m, 0).tolist()) == [1.0] * 10 + [3.0]


def test_pore_access_distances_and_isolation():
    shape = (100, 100)
    pore = np.zeros(shape, dtype=bool)
    pore[:, 0:5] = True
    si = np.zeros(shape, dtype=bool)
    si[10:20, 25:35] = True  # 20-30 px from the pore wall
    gr = np.zeros(shape, dtype=bool)
    gr[50:90, 50:90] = True
    out = F.pore_access(_masks(si, gr, pore, shape), nm_per_px=100.0)  # 0.1 um / px
    assert out["F01_pore_frac"] == pytest.approx(0.05)
    assert out["F01_pore_spans_z"] == 1.0 and out["F01_pore_spans_x"] == 0.0
    assert 2.0 <= out["F01_si_pore_dist_p50_um"] <= 3.0
    assert out["F01_si_pore_access_frac"] == 0.0  # all Si further than 1 um from a pore
    assert out["F01_si_isolated_frac"] == 0.0  # the solid is one connected block
    assert out["F01_pore_chord_z_p50_um"] == pytest.approx(10.0)  # full-height columns
    assert out["F01_pore_chord_x_p50_um"] == pytest.approx(0.5)


def test_si_islanded_in_pore_counts_as_isolated():
    shape = (60, 60)
    pore = np.zeros(shape, dtype=bool)
    pore[20:40, 20:40] = True
    si = np.zeros(shape, dtype=bool)
    si[28:32, 28:32] = True
    pore &= ~si
    out = F.pore_access(_masks(si, None, pore, shape), nm_per_px=50.0)
    assert out["F01_si_isolated_frac"] == pytest.approx(1.0)


# ------------------------------------------------------------------ swelling

def test_area_factor_limits():
    assert F.area_factor(0.0) == 1.0
    assert F.area_factor(1.0) == pytest.approx(3.8 ** (2 / 3))


def test_disc_swells_to_target_area():
    yy, xx = np.mgrid[:200, :200]
    si = (yy - 100) ** 2 + (xx - 100) ** 2 <= 20 ** 2
    for soc in (0.25, 0.5, 1.0):
        sw = F.swell_si(si, soc)
        assert sw.sum() / si.sum() == pytest.approx(F.area_factor(soc), rel=0.08)
        assert (sw & si).sum() == si.sum()  # growth only


def test_objects_grow_by_their_own_radius():
    yy, xx = np.mgrid[:200, :400]
    big = (yy - 100) ** 2 + (xx - 100) ** 2 <= 30 ** 2
    small = (yy - 100) ** 2 + (xx - 300) ** 2 <= 8 ** 2
    sw = F.swell_si(big | small, 1.0)
    f = F.area_factor(1.0)
    assert sw[:, :200].sum() / big.sum() == pytest.approx(f, rel=0.08)
    assert sw[:, 200:].sum() / small.sum() == pytest.approx(f, rel=0.2)


def test_swelling_accounting_sums_to_one():
    shape = (120, 120)
    yy, xx = np.mgrid[:120, :120]
    si = (yy - 60) ** 2 + (xx - 60) ** 2 <= 15 ** 2
    gr = (xx >= 75) & ~si
    pore = (xx <= 45) & ~si
    out = F.swelling_test(_masks(si, gr, pore, shape), soc=1.0)
    assert out["into_graphite"] + out["into_pore"] + out["into_binder"] == pytest.approx(1.0)
    assert out["into_graphite"] > 0 and out["into_pore"] > 0 and out["into_binder"] > 0
    assert 0 < out["pore_loss"] < 1
    assert out["si_objects_ratio"] == 1.0


def test_swelling_merges_neighbours():
    shape = (100, 200)
    yy, xx = np.mgrid[:100, :200]
    si = ((yy - 50) ** 2 + (xx - 80) ** 2 <= 12 ** 2) | ((yy - 50) ** 2 + (xx - 115) ** 2 <= 12 ** 2)
    out = F.swelling_test(_masks(si, None, None, shape), soc=1.0)
    assert out["si_objects_ratio"] == pytest.approx(0.5)


def test_compute_functional_columns():
    shape = (80, 80)
    yy, xx = np.mgrid[:80, :80]
    si = (yy - 40) ** 2 + (xx - 40) ** 2 <= 6 ** 2
    pore = (xx < 8) & ~si
    out = F.compute_functional(_masks(si, None, pore, shape), nm_per_px=50.0)
    assert {"F01_pore_frac", "F01_si_pore_dist_p50_um", "F02_soc100_into_pore", "F02_soc025_pore_loss"} <= out.keys()
    assert all(isinstance(v, float) for v in out.values())
