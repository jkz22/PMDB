"""Phantom tests for pmdb.clean (acceptance check 7.1) — all synthetic, < 60 s."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from clean_phantom import make_phantom  # noqa: E402

import pmdb.clean as C  # noqa: E402

SHAPE = (768, 1536)


@pytest.fixture(scope="module")
def phantom():
    return make_phantom(SHAPE, seed=1, bands=[(200, 210, 0.95), (450, 470, 0.97)], charging=2, sigma_blur=1.2)


@pytest.fixture(scope="module")
def result(phantom):
    return C.clean_site(phantom.raw)


def test_dark_level_within_1dn(phantom, result):
    for d in ("BSE", "SE_type"):
        a = result.params["anchors"][d]
        assert a["D_method"] == "gauss"
        assert abs(a["D"] - phantom.d[d]) <= 1.0, (d, a["D"], phantom.d[d])
    # Inlens uses the lower edge (1 % quantile) of the pore pixels: on a Gaussian phantom that sits
    # ~2.3 noise SDs below the true level, never above it
    a = result.params["anchors"]["Inlens"]
    sd = phantom.gain["Inlens"] * np.sqrt(phantom.beta)
    assert a["D_method"] == "quantile"
    assert phantom.d["Inlens"] - 3.5 * sd < a["D"] < phantom.d["Inlens"]


def test_dark_level_tobit_on_clipped_phantom():
    p = make_phantom(SHAPE, seed=3, clip_low=True)  # true D = -2: most pore pixels clip at 0
    res = C.clean_site(p.raw, detect_fov=False, charging=False, correct_bands=False)
    for d in ("BSE", "SE_type"):
        a = res.params["anchors"][d]
        assert a["D_method"] == "tobit"
        assert abs(a["D"] - p.d[d]) <= 1.0, (d, a["D"])
        assert ((res.mask[d] & C.BIT_CLIP_LOW) != 0).any()
    a = res.params["anchors"]["Inlens"]
    assert a["D_method"] == "quantile-clipped" and a["D"] == 0.0


def test_graphite_map_within_1pct(phantom, result):
    for d in C.DETECTORS:
        a = result.params["anchors"][d]
        gm = C.GraphiteMap(np.array(a["G_coef"]), None, 64, SHAPE).evaluate()
        truth = phantom.gain[d] * phantom.gmap
        shape_rel = np.abs((gm / gm.mean()) / (truth / truth.mean()) - 1.0)
        assert float(np.percentile(shape_rel, 99)) < 0.01, (d, shape_rel.max())  # shading shape
        if d != "Inlens":  # Inlens D is a lower-edge anchor, so its absolute G level is not the phantom gain
            rel = np.abs(gm / truth - 1.0)
            assert float(np.percentile(rel, 99)) < 0.01, (d, rel.max())


def test_row_gains_within_0p3pct(phantom, result):
    corrected = result.params["bands"]["corrected"]
    assert len(corrected) >= 2 * len(phantom.bands)  # found in every detector
    for ev in corrected:
        true_g = float(np.mean(phantom.row_gain[ev["start"]: ev["end"] + 1]))
        tol = 0.01 if ev["detector"] == "Inlens" else 0.003  # Inlens gain inherits its D anchor bias
        assert abs(ev["gain"] / true_g - 1.0) < tol, ev
    assert not result.params["bands"]["masked"]
    for d in C.DETECTORS:
        assert ((result.mask[d] & C.BIT_BAND_CORRECTED) != 0)[205].all()


def test_band_rows_within_5(phantom, result):
    for s, e in phantom.bands:
        hits = [ev for ev in result.params["bands"]["corrected"] if abs(ev["start"] - s) <= 5 and abs(ev["end"] - e) <= 5]
        assert len(hits) == 3, (s, e, result.params["bands"]["corrected"])


def test_corrected_band_profile_flat(phantom, result):
    ph = result.phases
    valid = C.valid_for_kpis(result.mask["BSE"])
    _, zr = C._row_z(result.norm["BSE"], ph.graphite, valid)
    for s, e in phantom.bands:
        assert float(np.nanmax(np.abs(zr[s: e + 1]))) < 3.0


def test_edge_sigma_within_0p05px(phantom, result):
    s = result.params["fingerprint"]["BSE"]["sigma_e_px"]
    assert abs(s - phantom.sigma_blur) < 0.05, s


def test_noise_model_recovered(phantom, result):
    fp = result.params["fingerprint"]["BSE"]
    assert abs(fp["noise_alpha"] - phantom.alpha) < 0.25 * phantom.alpha
    assert abs(fp["noise_beta"] - phantom.beta) < 0.25 * phantom.beta


def test_charging_blobs_found(phantom, result):
    found = result.params["charging"]["local"]
    assert len(found) == len(phantom.charging_bboxes)
    for (y0, x0, y1, x1) in phantom.charging_bboxes:
        best = 0.0
        for f in found:
            iy = max(0, min(y1, f["y1"]) - max(y0, f["y0"]))
            ix = max(0, min(x1, f["x1"]) - max(x0, f["x0"]))
            best = max(best, iy * ix / ((y1 - y0) * (x1 - x0)))
        assert best >= 0.5
    assert ((result.mask["Inlens"] & C.BIT_CHARGE_LOCAL) != 0).any()
    assert ((result.mask["SE_type"] & C.BIT_CHARGE_LOCAL) != 0).any()
    assert not ((result.mask["BSE"] & C.BIT_CHARGE_LOCAL) != 0).any()


def test_normalised_anchors(result):
    ph = result.phases
    z = result.norm["BSE"]
    v = C.valid_for_stats(result.mask["BSE"])
    assert abs(float(np.median(z[ph.pore & v]))) < 0.03
    assert abs(float(np.median(z[ph.graphite & v])) - 1.0) < 0.02
    assert 1.7 < result.params["anchors"]["BSE"]["si_graphite_ratio"] < 1.9


def test_clean_phantom_masks_little():
    p = make_phantom(SHAPE, seed=5, bands=None, charging=0)
    res = C.clean_site(p.raw)
    for d in C.DETECTORS:
        assert res.params["masked_beyond_border"][d] < 0.005, d


def test_idempotence(result):
    """Running 4.2–4.4 on a norm output gives D ≈ 0 and G ≈ 1."""
    z = result.norm["BSE"]
    valid = C.valid_for_kpis(result.mask["BSE"])
    ph = C.provisional_phases(z, valid)
    raw_proxy = np.ones_like(z, dtype=np.uint8)  # no clipped pixels -> robust-mode path
    dl = C.dark_level(z, raw_proxy, ph.pore, valid)
    gm = C.graphite_map(z - dl.value, ph.graphite, valid).evaluate()
    assert abs(dl.value) < 0.01
    assert abs(float(gm.mean()) - 1.0) < 0.01 and float(gm.max() - gm.min()) < 0.02


def test_masked_pixel_invariance(result):
    """With the mask fixed, changing pixel values under the invalid mask changes no anchor or KPI."""
    ph = result.phases
    out = {}
    for tag, scale in (("a", 1.0), ("b", 0.0)):
        vals = {}
        for d in C.DETECTORS:
            z = result.norm[d].copy()
            mask = result.mask[d]
            inval = ~C.valid_for_stats(mask)
            assert inval.any()
            if scale == 0.0:
                z[inval] = 50.0  # garbage under the mask
            v_kpi, v_stat = C.valid_for_kpis(mask), C.valid_for_stats(mask)
            raw_proxy = np.ones_like(z, dtype=np.uint8)
            dl = C.dark_level(z, raw_proxy, ph.pore, v_stat)
            gm = C.graphite_map(z - dl.value, ph.graphite, v_stat)
            nm = C.noise_model(z, ph, v_stat)
            ew = C.edge_width(z, ph.pore, v_stat, n_edges=2000)
            vals[d] = (dl.value, gm.level, tuple(gm.coef), nm.alpha, nm.beta, ew.sigma_px,
                       C.eighth_porosity(ph.pore, v_kpi)["all"], len(C.pore_network_areas(ph.pore, v_kpi)))
        out[tag] = vals
    assert out["a"] == out["b"]


def test_harmonise_down(phantom, result):
    sharper = make_phantom(SHAPE, seed=7, sigma_blur=0.8, alpha=0.002, beta=0.0005)
    res_s = C.clean_site(sharper.raw, detect_fov=False, charging=False, correct_bands=False)
    targets = C.build_targets({"a": result.params, "b": res_s.params}, ["a", "b"])
    h = C.harmonise_site(res_s, targets)
    info = h.params["harmonisation"]["BSE"]
    assert info["sigma_k_px"] > 0
    assert abs(info["sigma_e_after_px"] - targets.sigma_t["BSE"]) < 0.1
    assert info["noise_within_10pct"]
    # the reference itself is not blurred or noised
    h_ref = C.harmonise_site(result, targets)
    assert h_ref.params["harmonisation"]["BSE"]["sigma_k_px"] == 0.0


def test_write_read_roundtrip(tmp_path, result):
    C.write_site(tmp_path, result)
    z, m = C.read_site(tmp_path, "BSE", "norm")
    assert z.dtype == np.float32 and m.dtype == np.uint16
    assert np.allclose(z, np.clip(result.norm["BSE"], 0, 6.5535), atol=1.0 / C.Z_SCALE)
    assert np.array_equal(m, result.mask["BSE"])


def test_load_clean_roundtrip_and_half(phantom, result, tmp_path):
    from pmdb import io as pio

    site_dir = tmp_path / "Batch_9" / "site1"
    site_dir.mkdir(parents=True)
    C.write_site(site_dir, result)
    z, m = pio.load_clean("Batch_9", "site1", "BSE", "norm", clean_root=tmp_path)
    assert z.shape == SHAPE and m.dtype == np.uint16
    assert np.allclose(z, np.clip(result.norm["BSE"], 0, 6.5535), atol=1e-4)
    zh, mh = pio.load_clean("Batch_9", "site1", "BSE", "norm", resolution="half", clean_root=tmp_path)
    assert zh.shape == (SHAPE[0] // 2, SHAPE[1] // 2)
    # a block is valid iff one of its parents is; a fully masked block carries the OR of all flags
    m4 = m.reshape(SHAPE[0] // 2, 2, SHAPE[1] // 2, 2)
    any_ok = C.valid_for_kpis(m4).any(axis=(1, 3))
    assert np.array_equal(C.valid_for_kpis(mh), any_ok)
    all_or = m4[:, 0, :, 0] | m4[:, 1, :, 0] | m4[:, 0, :, 1] | m4[:, 1, :, 1]
    assert np.array_equal(mh[~any_ok], all_or[~any_ok])
    # a block with one masked parent averages only the valid ones and stays valid
    z4 = np.array([[1.0, 3.0], [5.0, 100.0]], dtype=np.float32)
    m4 = np.array([[0, 0], [0, C.BIT_CHARGE_LOCAL]], dtype=np.uint16)
    zh2, mh2 = pio.downsample_clean(z4, m4)
    assert abs(zh2[0, 0] - 3.0) < 1e-6 and C.valid_for_kpis(mh2)[0, 0]
    # informational bits of the contributing parents are kept, those of masked parents dropped
    m4i = np.array([[C.BIT_CLIP_LOW, 0], [0, C.BIT_CHARGE_LOCAL | C.BIT_CRACK]], dtype=np.uint16)
    _, mh2i = pio.downsample_clean(z4, m4i)
    assert mh2i[0, 0] == C.BIT_CLIP_LOW
    # a fully masked block stays invalid
    m4f = np.full((2, 2), C.BIT_BAND_BAD, dtype=np.uint16)
    _, mh2f = pio.downsample_clean(z4, m4f)
    assert not C.valid_for_kpis(mh2f)[0, 0]
    # clip bits do not exclude a pixel from the KPI-valid mean
    m4c = np.array([[0, 0], [0, C.BIT_CLIP_HIGH]], dtype=np.uint16)
    zh3, _ = pio.downsample_clean(z4, m4c)
    assert abs(zh3[0, 0] - z4.mean()) < 1e-5
    import pytest

    with pytest.raises(FileNotFoundError):
        pio.load_clean("Batch_9", "nope", clean_root=tmp_path)


def test_collector_touching_edge_is_masked():
    h, w = 400, 600
    img = np.full((h, w), 50, dtype=np.uint8)
    img[:41] = 255
    valid = C.valid_for_kpis(C.sanitise((h, w)))
    coll, info = C.detect_collector(img, valid)
    assert info["found"] and info["edge"] == "top"
    assert coll[:41].all()
    assert not coll[41 + C.DILATE_HALF_UM_PX + 2:].any()


def test_blur_does_not_bleed_from_masked_pixels():
    z = np.ones((120, 120), dtype=np.float32)
    z[:, 60] = 50.0
    valid = np.ones_like(z, dtype=bool)
    valid[:, 60] = False
    out, sk = C.harmonise_resolution(z, 0.5, 1.5, valid=valid)
    assert sk > 0
    assert np.abs(out[valid] - 1.0).max() < 1e-4
    assert out[0, 60] == 50.0
    out_plain, _ = C.harmonise_resolution(z, 0.5, 1.5)
    assert out_plain[:, 59].max() > 1.5
