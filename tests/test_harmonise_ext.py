"""Imported harmonisation pipelines (pmdb.harmonise_ext): synthetic checks, no real data."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

import pmdb.clean as C
import pmdb.harmonise_ext as X

pytest.importorskip("SimpleITK")
pytest.importorskip("intensity_normalization")
pytest.importorskip("basicpy")


@pytest.fixture(scope="module")
def phantoms():
    """Three 'clean' textured images and one affine-distorted copy (offset +22, gain 0.7)."""
    rng = np.random.default_rng(0)
    base = []
    for _ in range(3):
        img = np.full((160, 400), 56.0)
        img[rng.random((160, 400)) < 0.12] = 2.0  # pores
        img[rng.random((160, 400)) < 0.06] = 150.0  # Si
        img = img + rng.normal(0, 3, img.shape)
        base.append(np.clip(img, 0, 255).astype(np.float32))
    distorted = np.clip(0.7 * base[0] + 22.0, 0, 255).astype(np.float32)
    valid = np.ones((160, 400), dtype=bool)
    valid[:8], valid[-8:], valid[:, :8], valid[:, -8:] = False, False, False, False
    return base, distorted, valid


def test_nyul_maps_affine_distortion_back_onto_the_standard_scale(phantoms):
    base, distorted, valid = phantoms
    model = X.nyul_fit(base, [valid] * 3)
    assert model.standard_scale.shape == model.percentiles.shape == (11,)
    out, landmarks = X.nyul_apply(distorted, valid, model)
    ref, _ = X.nyul_apply(base[0], valid, model)
    # same material → same standardised grey (distortion removed) at every landmark
    for p in (1, 10, 50, 90, 99):
        assert abs(np.percentile(out[valid], p) - np.percentile(ref[valid], p)) < 1.5
    assert abs(landmarks[0] - 22.0) < 6  # the distorted image's own p1 landmark sits near 0.7·pore + 22


def test_n4_returns_smooth_bias_field_near_one_for_flat_image(phantoms):
    base, _, valid = phantoms
    corrected, bias = X.n4_correct(base[0], valid)
    assert bias.shape == base[0].shape
    assert 0.9 < bias.min() <= bias.max() < 1.1
    assert abs(np.median(corrected[valid]) - np.median(base[0][valid])) < 3


def test_basic_removes_per_image_offset(phantoms):
    base, distorted, valid = phantoms
    offset_copy = np.clip(base[1] + 30.0, 0, 255).astype(np.float32)
    model = X.basic_fit(base + [offset_copy], [valid] * 4)
    assert model.flatfield.shape == tuple(X.BASIC_PARAMS["working_shape"])
    out_a, b_a = X.basic_apply(base[1], valid, model)
    out_b, b_b = X.basic_apply(offset_copy, valid, model)
    assert abs((b_b - b_a) - 30.0) < 3.0
    assert abs(np.median(out_a[valid]) - np.median(out_b[valid])) < 3.0


def test_store_roundtrip_and_loader(tmp_path):
    rng = np.random.default_rng(1)
    imgs = {d: rng.uniform(-40, 120, (20, 30)).astype(np.float32) for d in X.DETECTORS}
    msks = {d: np.zeros((20, 30), dtype=np.uint16) for d in X.DETECTORS}
    msks["BSE"][0] = C.BIT_BORDER
    for method in X.METHODS:
        p = X.write_site(tmp_path, method, "Batch_9", "abc", imgs, msks)
        assert p.exists()
        img, msk = X.load_ext("Batch_9", "abc", method, ext_root=tmp_path, as_uint8=False)
        assert img.shape == (20, 30, 3) and msk.shape == (20, 30, 3) and msk.dtype == np.uint16
        gain, off = X.store_scale(method)
        lo, hi = (0 - off) / gain, (255 - off) / gain
        inside = (imgs["BSE"] > lo) & (imgs["BSE"] < hi)
        assert np.abs(img[..., 0][inside] - imgs["BSE"][inside]).max() <= 0.5 / gain + 1e-6
        assert (msk[0, :, 0] & C.BIT_BORDER).all()
    with pytest.raises(ValueError):
        X.load_ext("Batch_9", "abc", "nope", ext_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        X.load_ext("Batch_9", "zzz", "nyul", ext_root=tmp_path)


def test_models_save_load(tmp_path, phantoms):
    base, _, valid = phantoms
    ny = {"BSE": X.nyul_fit(base, [valid] * 3)}
    X.save_models(tmp_path, "nyul", ny)
    back = X.load_models(tmp_path, "nyul")["BSE"]
    assert np.allclose(back.standard_scale, ny["BSE"].standard_scale)
    ba = {"BSE": X.basic_fit(base, [valid] * 3)}
    X.save_models(tmp_path, "basic", ba)
    back_b = X.load_models(tmp_path, "basic")["BSE"]
    assert np.allclose(back_b.flatfield, ba["BSE"].flatfield) and back_b.working_shape == ba["BSE"].working_shape


def test_load_half_raw_uses_clean_mask_and_relative_paths(tmp_path, monkeypatch):
    # a tiny fake 'clean' site directory + raw RGB TIFF under a fake repo root
    repo = tmp_path
    (repo / "data" / "B").mkdir(parents=True)
    raw = np.full((12, 16, 3), 100, dtype=np.uint8)
    raw[:, :, :] = 100
    tifffile.imwrite(repo / "data" / "B" / "img_s_BSE.tif", raw)
    site = repo / "clean" / "B" / "s"
    site.mkdir(parents=True)
    mask = C.sanitise((12, 16), border_px=2)
    mask[6, 6] |= C.BIT_CHARGE_LOCAL
    tifffile.imwrite(site / "BSE_mask.tif", mask)
    (site / "params.json").write_text(json.dumps({"paths": {"BSE": "data/B/img_s_BSE.tif"}}))
    monkeypatch.setattr(X, "REPO_ROOT", repo)
    g, m = X.load_half_raw("B", "s", "BSE", clean_root=repo / "clean")
    assert g.shape == (6, 8) and m.dtype == np.uint16
    assert np.allclose(g, 100)
    assert (m[0] & C.BIT_BORDER).all() and not (m[3, 3] & C.BIT_CHARGE_LOCAL)  # a valid sibling → block valid
    assert C.valid_for_kpis(m)[3, 3]


# --- evaluation helpers (scripts/eval_harmonise_ext.py) -------------------------------------------------------


def test_eval_site_metrics_ignore_masked_pixels(phantoms):
    from scripts.eval_harmonise_ext import _site_metrics

    base, _, _ = phantoms
    img = np.repeat(base[0].astype(np.uint8)[..., None], 3, axis=2)
    msk = np.zeros(img.shape, dtype=np.uint16)
    # saturate a block on every detector and mark it excluded (clipped + charging): anchors, thresholds and phase
    # fractions must be the same as for the clean image under the same mask
    bad = img.copy()
    bad[20:60, 50:200] = 255
    msk_bad = msk.copy()
    msk_bad[20:60, 50:200] = C.BIT_CLIP_HIGH | C.BIT_CHARGE_LOCAL
    ref = _site_metrics(img, msk_bad, img)
    got = _site_metrics(bad, msk_bad, bad)
    for k in ("BSE_anchor_black", "BSE_anchor_graphite", "BSE_anchor_si", "BSE_contrast_ratio", "Inlens_anchor_si", "seg_f_si", "seg_f_pore"):
        assert got[k] == pytest.approx(ref[k], abs=2e-3 if k.startswith("seg") else 1.0), k
    # same image, no mask: the saturated block does leak into the anchors
    leaked = _site_metrics(bad, msk, bad)
    assert leaked["BSE_anchor_si"] > ref["BSE_anchor_si"] + 10


def test_eval_shortcut_recall_is_recall_not_accuracy():
    import pandas as pd

    from scripts.eval_harmonise_ext import _shortcut, _shortcut_recall

    rng = np.random.default_rng(1)
    # feature carries no information: recall of a rare positive class must be well below the accuracy
    df = pd.DataFrame({"x": rng.normal(size=40)})
    y = np.zeros(40, dtype=bool)
    y[:4] = True
    acc, rec = _shortcut(df, ["x"], y), _shortcut_recall(df, ["x"], y)
    assert acc > 0.7 and rec <= 0.5
    # perfectly separable: both 1
    df2 = pd.DataFrame({"x": np.where(y, 5.0, -5.0) + rng.normal(0, 0.1, 40)})
    assert _shortcut(df2, ["x"], y) == 1.0 and _shortcut_recall(df2, ["x"], y) == 1.0
