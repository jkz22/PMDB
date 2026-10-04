"""Unit tests for pmdb.fem (pure numpy; also run inside the Modal image)."""

from __future__ import annotations

import numpy as np
import pytest

from pmdb.fem.config import case_params_hash, load_params, params_hash
from pmdb.fem.geometry import central_cols, coarsen_image, coarsen_labels
from pmdb.fem.materials import (
    ARTEFACT, BINDER, GRAPHITE, PORE, SI, graphite_strains, lithiation_state,
    phase_properties, si_yield_MPa, soc_fractions,
)


# ---------------------------------------------------------------- config
def test_config_gate():
    assert load_params()["gates"]["swelling_stop"] == [0.03, 0.39]


def test_config_keys():
    assert set(load_params()) == {"version", "mesh", "soc", "si", "graphite", "binder", "pore",
                                  "solver", "remediation", "features", "gif", "gates", "mechanics"}


def test_config_hash():
    h = params_hash(load_params())
    assert len(h) == 12 and int(h, 16) >= 0
    assert h == params_hash(load_params())


def test_hash_includes_lateral_bc():
    p = load_params()
    assert case_params_hash(p, "both") == params_hash(p)  # default runs keep their committed hash
    assert case_params_hash(p, "left") != case_params_hash(p, "both")


# ------------------------------------------------------------- materials
# s, f_Si, f_Gr, u, y, J_Si, eps_c  (docs/fem/literature-review.md, "Overall SOC")
SOC_TABLE = [
    (0.0, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000),
    (0.1, 0.142, 0.012, 0.114, 0.011, 1.319, 0.002),
    (0.2, 0.284, 0.025, 0.228, 0.022, 1.637, 0.005),
    (0.3, 0.399, 0.095, 0.319, 0.087, 1.893, 0.019),
    (0.4, 0.484, 0.225, 0.388, 0.204, 2.085, 0.045),
    (0.5, 0.570, 0.354, 0.456, 0.322, 2.278, 0.055),
    (0.6, 0.656, 0.483, 0.525, 0.440, 2.470, 0.055),
    (0.7, 0.742, 0.612, 0.594, 0.557, 2.662, 0.061),
    (0.8, 0.828, 0.742, 0.662, 0.675, 2.855, 0.072),
    (0.9, 0.914, 0.871, 0.731, 0.792, 3.047, 0.084),
    (1.0, 1.000, 1.000, 0.800, 0.910, 3.240, 0.095),
]


@pytest.mark.parametrize("row", SOC_TABLE)
def test_materials_soc_table(row):
    p = load_params()
    s, f_si, f_gr, u, y, j_si, eps_c = row
    a, b = soc_fractions(s, p)
    uu, yy = lithiation_state(s, p)
    assert a == pytest.approx(f_si, abs=1.5e-3)
    assert b == pytest.approx(f_gr, abs=1.5e-3)
    assert uu == pytest.approx(u, abs=1.5e-3)
    assert yy == pytest.approx(y, abs=1.5e-3)
    assert 1 + p["si"]["beta"] * uu == pytest.approx(j_si, abs=1.5e-3)
    assert graphite_strains(yy, p)[1] == pytest.approx(eps_c, abs=1.5e-3)


def test_materials_yield():
    p = load_params()
    assert si_yield_MPa(0.0, p) == pytest.approx(3000.0)
    assert si_yield_MPa(1.0, p) == pytest.approx(3000 - 3150 * 3.75 / 4.75, abs=0.01)


def test_materials_si_stretch_cubed():
    p = load_params()
    lam = phase_properties(1.0, p)[SI].stretch
    assert lam[0] ** 3 == pytest.approx(3.24, abs=2e-3)
    assert lam[0] == lam[1] == lam[2]


def test_materials_pore_modulus():
    pp = phase_properties(0.5, load_params())
    assert pp[PORE].E == pytest.approx(0.05)
    assert pp[ARTEFACT].E == pytest.approx(0.05)
    assert pp[BINDER].stretch == (1.0, 1.0, 1.0)
    assert set(pp) == {BINDER, SI, GRAPHITE, PORE, ARTEFACT}


# -------------------------------------------------------------- geometry
_S, _G, _B, _P, _A = SI, GRAPHITE, BINDER, PORE, ARTEFACT


@pytest.mark.parametrize("block,expected", [
    ([_S, _S, _G, _G], _S),
    ([_S, _G, _G, _B], _G),
    ([_S, _P, _G, _B], _S),
    ([_P, _P, _G, _G], _P),
    ([_G, _G, _B, _B], _G),
    ([_A, _A, _B, _B], _B),
])
def test_geometry_block_votes(block, expected):
    lab = np.array(block, dtype=np.uint8).reshape(2, 2)
    out = coarsen_labels(lab, 2)
    assert out.shape == (1, 1) and out[0, 0] == expected


def test_geometry_odd_shape_dropped():
    lab = np.zeros((5, 7), dtype=np.uint8)
    assert coarsen_labels(lab, 2).shape == (2, 3)
    assert coarsen_image(np.zeros((5, 7)), 2).shape == (2, 3)
    assert coarsen_labels(lab, 1) is lab


def test_geometry_central_cols():
    sl = central_cols(1747, 40.0, 50.0)
    assert (sl.stop - sl.start) == 800 and sl.start % 2 == 0


# -------------------------------------------------------------- features
import pandas as pd  # noqa: E402

from pmdb.fem.features import (  # noqa: E402
    METRIC_NAMES, fem_tile_slices, region_metrics, run_curves, swelling_gate, symmetrise,
)
from pmdb.fem.result import FIELD_KEYS, SimResult  # noqa: E402


def _synthetic(labels, orientation="bottom", n=11, px=0.1, J=1.1, sxx=-5.0, slope=0.1):
    H, W = labels.shape
    s = np.linspace(0, 1, n)
    zc = (H - np.arange(H + 1)) * px  # node-row z
    if orientation == "bottom":
        uz = slope * zc
    else:
        uz = -slope * (H * px - zc)
    un = np.zeros((n, H + 1, W + 1, 2), dtype=np.float32)
    un[..., 1] = uz[None, :, None]
    fields = {k: np.zeros((n, H, W), dtype=np.float32) for k in FIELD_KEYS}
    fields["J"][:] = J
    fields["sxx"][:] = sxx
    fields["vm"][:] = 1.0
    fields["J"][0] = 1.0
    return SimResult(labels=labels, px_um=px, s=s, converged=np.ones(n, dtype=bool), u_nodes=un,
                     fields=fields, failed_at_s=float("nan"))


def test_features_tile_slices():
    sl = fem_tile_slices(1747, 0.1)
    assert len(sl) == 6 and sl[0].start == 200 and sl[-1].stop == 1547
    assert all(a.stop == b.start for a, b in zip(sl[:-1], sl[1:]))


@pytest.mark.parametrize("orientation", ["bottom", "top"])
def test_features_swelling_uniform(orientation):
    lab = np.full((20, 600), GRAPHITE, dtype=np.uint8)
    r = _synthetic(lab, orientation)
    m = region_metrics(r, 5, slice(0, 600), orientation, load_params())
    assert m["swelling"] == pytest.approx(0.1, rel=1e-6)
    assert m["surface_rough"] == pytest.approx(0.0, abs=1e-9)
    assert m["sxx_mean_MPa"] == pytest.approx(-5.0)
    assert m["J_gr_mean"] == pytest.approx(1.1, rel=1e-6)
    assert np.isnan(m["J_si_mean"])


def test_features_pore_closure():
    lab = np.full((10, 10), GRAPHITE, dtype=np.uint8)
    lab[4, 4] = PORE
    r = _synthetic(lab)
    r.fields["J"][3, 4, 4] = 0.05
    m = region_metrics(r, 3, slice(0, 10), "bottom", load_params())
    assert m["pore_closed_frac"] > 0
    site, _ = run_curves(r, "bottom", load_params(), window=True)
    assert site[0]["first_pore_closure_s"] == pytest.approx(r.s[3])


def test_features_symmetrise():
    keys = ["batch", "site", "frame"]
    base = {"batch": "B", "site": "x", "heldout": False, "frame": 1, "s": 0.1, "converged": True,
            "failed_at_s": np.nan, "first_pore_closure_s": np.nan}
    df = pd.DataFrame([{**base, "orientation": "bottom", "swelling": 0.1},
                       {**base, "orientation": "top", "swelling": 0.2}])
    out = symmetrise(df, keys)
    assert out[out.orientation == "sym"]["swelling"].iloc[0] == pytest.approx(0.15)
    df.loc[1, "swelling"] = np.nan
    out = symmetrise(df, keys)
    assert np.isnan(out[out.orientation == "sym"]["swelling"].iloc[0])


def test_features_metric_names():
    expected = "swelling surface_rough sxx_mean_MPa syy_mean_MPa porosity porosity_change " \
               "porosity_rel_change J_si_mean J_gr_mean J_binder_mean vm_si_p50_MPa vm_si_p95_MPa " \
               "vm_gr_p95_MPa vm_binder_p95_MPa p_si_mean_MPa si_yield_frac pore_closed_frac".split()
    assert len(expected) == 17
    for f in ("vm", "p", "J"):
        for ph in ("si", "gr", "binder"):
            expected += [f"q{q}_{f}_{ph}" for q in (5, 25, 50, 75, 95, 99)]
    expected += [f"q{q}_J_pore" for q in (5, 25, 50, 75, 95, 99)]
    expected += ["band_vm_maxdev", "band_vm_absslope", "band_J_maxdev", "band_J_absslope"]
    assert len(expected) == 81
    assert list(METRIC_NAMES) == expected
    lab = np.full((10, 10), SI, dtype=np.uint8)
    m = region_metrics(_synthetic(lab), 2, slice(0, 10), "bottom", load_params())
    assert list(m) == expected


def test_features_unconverged_nan():
    lab = np.full((10, 10), SI, dtype=np.uint8)
    r = _synthetic(lab)
    r.converged[4:] = False
    m = region_metrics(r, 5, slice(0, 10), "bottom", load_params())
    assert len(m) == 81 and all(np.isnan(v) for v in m.values())


@pytest.mark.parametrize("val,ok,lit", [(0.05, True, False), (0.02, False, False),
                                        (0.20, True, True), (0.40, False, False)])
def test_features_swelling_gate(val, ok, lit):
    g = swelling_gate(val, load_params())
    assert g["gate_ok"] is ok and g["lit_band_ok"] is lit
    assert swelling_gate(float("nan"), load_params())["gate_ok"] is False


# ------------------------------------------------------------------- gif
def test_gif_synthetic_site():
    import io

    from PIL import Image
    from scipy import ndimage

    from pmdb.fem.gif import render_site_gif

    rng = np.random.default_rng(0)
    H, W, n = 580, 1748, 11
    labels = rng.choice([BINDER, SI, GRAPHITE, PORE], size=(H, W), p=[0.2, 0.2, 0.5, 0.1]).astype(np.uint8)
    bse = ndimage.gaussian_filter(rng.normal(size=(H, W)), 1.0).astype(np.float32)
    s = np.linspace(0, 1, n)
    zc = (H - np.arange(H + 1)) * 0.1
    un = np.zeros((n, H + 1, W + 1, 2), dtype=np.float32)
    un[..., 1] = 0.1 * s[:, None, None] * zc[None, :, None]
    fields = {k: np.zeros((n, H, W), dtype=np.float32) for k in ("J", "sxx", "szz", "sxz", "syy")}
    fields["vm"] = (10.0 ** rng.uniform(0, 4, size=(n, H, W))).astype(np.float32)
    fields["J"][:] = 1.0
    conv = np.ones(n, dtype=bool)
    conv[-2:] = False
    un[~conv] = np.nan
    for k in fields:
        fields[k][~conv] = np.nan
    r = SimResult(labels=labels, px_um=0.1, s=s, converged=conv, u_nodes=un, fields=fields, failed_at_s=0.85)
    data, info = render_site_gif(bse, r, "Batch_X/site | collector assumed at bottom", 0.1 * s,
                                 load_params()["gif"])
    im = Image.open(io.BytesIO(data))
    assert im.n_frames == 11
    assert len(data) <= 2_000_000
    assert im.size[0] in (900, 720, 600)
    assert info["bytes"] == len(data)


def test_features_z_edge_default_identical():
    lab = np.full((20, 40), GRAPHITE, dtype=np.uint8)
    lab[5, 5] = PORE
    r = _synthetic(lab)
    r.fields["J"][4, 5, 5] = 0.05
    p = load_params()
    for o in ("bottom", "top"):
        a = run_curves(r, o, p, window=True)
        b = run_curves(r, o, p, window=True, z_edge_um=0.0)
        assert str(a) == str(b)
        assert str(region_metrics(r, 5, slice(0, 40), o, p)) == str(region_metrics(r, 5, slice(0, 40), o, p, 0.0))


@pytest.mark.parametrize("orientation", ["bottom", "top"])
def test_features_z_edge_cropped(orientation):
    lab = np.full((100, 40), GRAPHITE, dtype=np.uint8)
    lab[2, 5] = PORE  # inside the 3 um (30 row) edge band
    r = _synthetic(lab, orientation)
    r.fields["J"][:, 2, 5] = 0.05
    r.fields["vm"][5, :30, :] = 99.0
    p = load_params()
    m = region_metrics(r, 5, slice(0, 40), orientation, p, z_edge_um=3.0)
    assert m["swelling"] == pytest.approx(0.1, rel=1e-5)  # linear uz: interior strain = slope
    assert m["surface_rough"] == pytest.approx(0.0, abs=1e-9)
    assert m["q99_vm_gr"] == pytest.approx(1.0)  # edge rows excluded
    assert np.isnan(m["pore_closed_frac"])  # the only pore sits in the dropped band
    site, _ = run_curves(r, orientation, p, window=True, z_edge_um=3.0)
    assert np.isnan(site[0]["first_pore_closure_s"])
    assert region_metrics(r, 5, slice(0, 40), orientation, p)["q99_vm_gr"] == pytest.approx(99.0)


def test_features_oversized_z_edge_raises():
    lab = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="no interior"):
        region_metrics(_synthetic(lab, px=0.1), 2, slice(0, 10), "bottom", load_params(), z_edge_um=1.0)


def test_features_oversized_z_edge_raises_even_if_unconverged():
    lab = np.zeros((20, 20), dtype=np.uint8)
    r = _synthetic(lab, px=0.1)
    r.converged[:] = False
    with pytest.raises(ValueError, match="no interior"):
        region_metrics(r, 2, slice(0, 10), "bottom", load_params(), z_edge_um=1.0)
