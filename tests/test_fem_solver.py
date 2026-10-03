"""Analytic tests of the dolfinx solver (run inside the Modal image; skipped without dolfinx)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.optimize import brentq

pytest.importorskip("dolfinx")

from pmdb.fem.config import load_params  # noqa: E402
from pmdb.fem.materials import (  # noqa: E402
    BINDER, GRAPHITE, PORE, SI, PhaseProps, graphite_strains, lame, phase_properties,
)
from pmdb.fem.solver import BCSpec, simulate  # noqa: E402

LABELS = (BINDER, SI, GRAPHITE, PORE, 4)


def _opts(**kw):
    o = dict(load_params()["solver"])
    o.update(snes_rtol=1e-12, snes_atol=1e-9)
    o.update(kw)
    return o


def _uniform(fn):
    """props_fn that gives every label the same PhaseProps fn(s)."""
    return lambda s: {lab: fn(s) for lab in LABELS}


def _closed_form_f(mu, lam_l, lam_y):
    return brentq(lambda f: mu * (f * f - 1) + lam_l * np.log(f * f / lam_y), 0.5, 2.0, xtol=1e-14)


def test_t1a_free_eigenstretch():
    lab = np.full((4, 6), SI, dtype=np.uint8)
    fn = _uniform(lambda s: PhaseProps(1000.0, 0.3, (1 + 0.2 * s, 1 + 0.1 * s, 1.0)))
    r = simulate(lab, 1.0, fn, BCSpec("bottom", "left"), _opts(), np.array([0.0, 1.0]))
    assert r.converged.all()
    assert np.allclose(r.fields["J"][1], 1.32, rtol=1e-8, atol=0)
    smax = max(np.abs(r.fields[k][1]).max() for k in ("sxx", "szz", "sxz", "syy"))
    assert smax < 1e-6


def _si_props_at(s):
    p = load_params()
    u = 0.8 * s
    lam = (1 + p["si"]["beta"] * u) ** (1 / 3)
    return PhaseProps(p["si"]["E_MPa"][0] + p["si"]["E_MPa"][1] * u,
                      p["si"]["nu"][0] + p["si"]["nu"][1] * u, (lam, lam, lam))


def test_t1b_si_plane_strain():
    lab = np.full((4, 6), SI, dtype=np.uint8)
    r = simulate(lab, 1.0, _uniform(_si_props_at), BCSpec("bottom", "left"), _opts(), np.array([0.0, 1.0]))
    assert r.converged.all()
    pp = _si_props_at(1.0)
    lam = pp.stretch[0]
    mu, lam_l = lame(pp.E, pp.nu)
    f = _closed_form_f(mu, lam_l, lam)
    assert np.allclose(r.fields["J"][1], lam**2 * f**2, rtol=1e-6)
    assert np.abs(r.fields["sxx"][1]).max() < 1e-6 * pp.E
    assert np.abs(r.fields["szz"][1]).max() < 1e-6 * pp.E
    je = f * f / lam
    syy = (mu * (1 / lam**2 - 1) + lam_l * np.log(je)) / je
    assert np.allclose(r.fields["syy"][1], syy, rtol=1e-6)


def _gr_props(y):
    p = load_params()
    g = p["graphite"]
    ea, ec = graphite_strains(y, p)
    return PhaseProps(g["E_MPa"][0] + g["E_MPa"][1] * y, g["nu"][0] + g["nu"][1] * y,
                      (1 + ea, 1 + ec, 1 + ea))


def test_t1c_graphite_plane_strain():
    lab = np.full((4, 6), GRAPHITE, dtype=np.uint8)
    r = simulate(lab, 1.0, _uniform(lambda s: _gr_props(0.91 * s)), BCSpec("bottom", "left"), _opts(),
                 np.array([0.0, 1.0]))
    assert r.converged.all()
    pp = _gr_props(0.91)
    mu, lam_l = lame(pp.E, pp.nu)
    lx, lz, ly = pp.stretch
    fe = _closed_form_f(mu, lam_l, ly)
    assert np.allclose(r.fields["J"][1], lx * lz * fe**2, rtol=1e-6)
    assert np.abs(r.fields["sxx"][1]).max() < 1e-6 * pp.E
    assert np.abs(r.fields["szz"][1]).max() < 1e-6 * pp.E


def _bilayer(eps, orientation):
    h, E, nu = 1.0, 1000.0, 0.3
    lab = np.zeros((10, 8), dtype=np.uint8)
    top = np.zeros((10, 8), dtype=bool)
    top[:5] = True
    lab[top] = SI  # layer B
    lab[~top] = GRAPHITE  # layer A

    def fn(s):
        e = eps * s
        return {SI: PhaseProps(E, nu, (1 + e, 1 + e, 1 + e)), GRAPHITE: PhaseProps(E, nu, (1.0, 1.0, 1.0)),
                BINDER: PhaseProps(E, nu, (1.0,) * 3), PORE: PhaseProps(E, nu, (1.0,) * 3),
                4: PhaseProps(E, nu, (1.0,) * 3)}

    r = simulate(lab, h, fn, BCSpec(orientation, "both"), _opts(), np.array([0.0, 1.0]))
    assert r.converged.all()
    return r, h, E, nu


def _check_bilayer(eps, orientation, rel):
    r, h, E, nu = _bilayer(eps, orientation)
    s_b = -E * eps / (1 - nu)
    F = r.fields
    for k in ("sxx", "syy"):
        assert np.allclose(F[k][1][:5], s_b, rtol=rel)
    assert np.abs(F["szz"][1][:5]).max() < max(1e-3 * abs(s_b), 1e-3 * abs(s_b) * rel * 100)
    if eps < 1e-3:
        for k in ("sxx", "szz", "syy", "sxz"):
            assert np.abs(F[k][1][5:]).max() < 1e-6 * abs(s_b)
    uz_free = 5 * h * eps * (1 + nu) / (1 - nu)
    if orientation == "bottom":
        assert r.u_nodes[1, 0, :, 1].mean() == pytest.approx(uz_free, rel=rel)
        assert abs(r.u_nodes[1, 5, :, 1].mean()) < 1e-3 * uz_free
    else:
        assert r.u_nodes[1, 10, :, 1].mean() == pytest.approx(-uz_free, rel=rel)
        assert r.u_nodes[1, 5, :, 1].mean() == pytest.approx(-uz_free, rel=rel)


def test_t2_bilayer_small():
    _check_bilayer(1e-4, "bottom", 1e-3)


def test_t2_bilayer_finite():
    r, h, E, nu = _bilayer(1e-2, "bottom")
    s_b = -E * 1e-2 / (1 - nu)
    assert (r.fields["sxx"][1][:5] < 0).all() and (r.fields["syy"][1][:5] < 0).all()
    assert np.allclose(r.fields["sxx"][1][:5], s_b, rtol=3e-2)
    assert np.allclose(r.fields["syy"][1][:5], s_b, rtol=3e-2)
    uz = 5 * h * 1e-2 * (1 + nu) / (1 - nu)
    assert r.u_nodes[1, 0, :, 1].mean() == pytest.approx(uz, rel=3e-2)


def test_t3_orientation_top():
    _check_bilayer(1e-4, "top", 1e-3)


def test_t4_pixel_mapping():
    lab = np.full((7, 9), BINDER, dtype=np.uint8)
    lab[1, 6] = SI
    p = load_params()
    r = simulate(lab, 0.1, lambda s: phase_properties(s, p), BCSpec("bottom", "both"), _opts(),
                 np.array([0.0, 0.2]))
    assert r.converged.all()
    vm = r.fields["vm"][1]
    assert np.unravel_index(np.argmax(vm), vm.shape) == (1, 6)
    assert np.array_equal(r.labels, lab)


def test_t5_failure_path():
    lab = np.full((4, 6), SI, dtype=np.uint8)
    opts = _opts(snes_max_it=1, ds_min=0.05)
    r = simulate(lab, 1.0, _uniform(_si_props_at), BCSpec("bottom", "left"), opts, np.linspace(0, 1, 11))
    assert np.isfinite(r.failed_at_s) and r.failed_at_s <= 0.1
    assert r.converged[0] and not r.converged[1:].any()
    assert np.isnan(r.fields["vm"][1:]).all() and np.isnan(r.u_nodes[1:]).all()


def test_t6_soft_pores():
    lab = np.full((12, 12), BINDER, dtype=np.uint8)
    lab[3:9, 3:9] = PORE
    lab[4:8, 4:8] = SI
    p = load_params()
    frames = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    r = simulate(lab, 0.1, lambda s: phase_properties(s, p), BCSpec("bottom", "both"), load_params()["solver"],
                 frames, extra_targets=(p["soc"]["s_star"],))
    pore = lab == PORE
    minj = [float(np.nanmin(r.fields["J"][i][pore])) if r.converged[i] else float("nan")
            for i in range(len(frames))]
    print("T6_REPORT " + json.dumps({"failed_at_s": None if np.isnan(r.failed_at_s) else r.failed_at_s,
                                     "min_pore_J": minj, "n_substeps": len(r.substeps)}))
    assert r.converged[1]
    assert minj[1] < 0.9
