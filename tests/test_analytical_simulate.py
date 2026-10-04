import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analytical_benchmarks"))
S = pytest.importorskip("simulate")


def disc_lab(n_disc, size=120, r=12):
    lab = np.ones((size, size), np.uint8)
    yy, xx = np.mgrid[:size, :size]
    for k in range(n_disc):
        cy, cx = 30 + 60 * (k // 2), 30 + 60 * (k % 2)
        lab[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 2
    lab[55:60, :] = 0
    return lab


def test_no_eigenstrain_no_deformation():
    m = S.Mesh(20, 30, 0.2, 0.25); m.factor(np.full(600, 10.0))
    r = m.solve(np.zeros(600))
    assert abs(r["thick"]) < 1e-12 and np.abs(r["s1"]).max() < 1e-9


def test_uniform_swelling_matches_uniaxial_strain():
    nu, e = 0.25, 0.01
    m = S.Mesh(20, 30, 0.2, nu); m.factor(np.full(600, 10.0))
    assert m.solve(np.full(600, e))["thick"] == pytest.approx(e / (1 - nu), rel=1e-6)


def test_more_si_swells_more_and_trajectory_is_sane():
    t1 = S.simulate(disc_lab(1), cycles=4, seed=0, snap=False)["traj"]
    t4 = S.simulate(disc_lab(4), cycles=4, seed=0, snap=False)["traj"]
    assert t4.thickness_charged_pct.iloc[1] > t1.thickness_charged_pct.iloc[1] > 0
    for t in (t1, t4):
        c = t[t.cycle > 0]
        assert np.isfinite(c[["retention_pct", "thickness_charged_pct", "porosity", "si_frac", "sei_frac"]].values).all()
        assert (np.diff(c.capacity_mAh_cm3) <= 1e-9).all()
        assert (np.diff(c.sei_frac) >= 0).all() and (np.diff(c.n_cracks) >= 0).all()
        assert t.cycle.tolist() == [0, 1, 2, 3, 4]


def test_deterministic_per_seed():
    a = S.simulate(disc_lab(2), cycles=3, seed=1, snap=False)["traj"]
    b = S.simulate(disc_lab(2), cycles=3, seed=1, snap=False)["traj"]
    assert a.drop(columns="cycle").fillna(-1).equals(b.drop(columns="cycle").fillna(-1))


def test_constant_current_li_is_gradual_and_reversible():
    act = disc_lab(4) == 2
    li = S.LiFD(act, 3600.0)
    ch = li.run(np.zeros(act.shape), True, {12, 60, 120})
    assert ch[12][act].mean() == pytest.approx(0.1, abs=0.01)       # 1C: 10% of the Si filled after 6 min
    assert ch[60][act].mean() == pytest.approx(0.5, abs=0.03)
    dis = li.run(ch[120], False, {12, 120})
    assert ch[120][act].mean() - dis[12][act].mean() == pytest.approx(0.1, abs=0.01)   # no jump when the current reverses
    assert dis[120][act].mean() < 0.05 and (dis[120] >= 0).all()
