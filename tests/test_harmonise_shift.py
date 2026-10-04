"""Synthetic tests for the shift (non-intensity) harmonisation routes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pmdb import harmonise_shift as S


def _phantom(seed: int, blur: float, noise: float, shape=(1024, 1024)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full(shape, 110.0)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for _ in range(300):  # dark pores and bright Si particles
        cy, cx, r = rng.integers(0, shape[0]), rng.integers(0, shape[1]), rng.integers(4, 20)
        base[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 10.0 if rng.random() < 0.6 else 200.0
    img = gaussian_filter(base, blur) + rng.normal(0, noise, shape)
    return np.clip(img, 0, 255).astype(np.float32)


def _hf(img, valid):
    a = S.radial_amplitude(img, valid)
    f = S.bin_centres()
    return a[(f >= 0.35)].mean() / a[(f >= 0.05) & (f < 0.15)].mean()


def test_spectrum_moves_blurred_site_to_reference_texture_without_changing_mean():
    valid = np.ones((1024, 1024), bool)
    refs = [_phantom(i, 1.0, 3.0) for i in range(3)]
    model = S.spectrum_fit(refs, [valid] * 3)
    blurred = _phantom(7, 2.0, 1.0)
    before = _hf(blurred, valid)
    out, h = S.spectrum_apply(blurred, valid, model)
    after = _hf(out, valid)
    target = np.mean([_hf(r, valid) for r in refs])
    assert abs(after - target) < 0.5 * abs(before - target)
    assert h[0] == 1.0 and h.min() >= S.SPECTRUM_CLAMP[0] and h.max() <= S.SPECTRUM_CLAMP[1]
    assert abs(out.mean() - blurred.mean()) < 0.05
    assert out.shape == blurred.shape and out.dtype == np.float32


def test_spectrum_leaves_a_reference_like_site_nearly_unchanged():
    valid = np.ones((1024, 1024), bool)
    refs = [_phantom(i, 1.0, 3.0) for i in range(3)]
    model = S.spectrum_fit(refs, [valid] * 3)
    same = _phantom(11, 1.0, 3.0)
    out, h = S.spectrum_apply(same, valid, model)
    assert np.abs(h - 1).max() < 0.25
    assert np.abs(out - same).mean() < 1.5


def test_masked_pixels_do_not_leak_and_are_passed_through():
    valid = np.ones((1024, 1024), bool)
    img = _phantom(3, 1.0, 3.0)
    sat = img.copy()
    sat[100:300, 100:300] = 255.0
    v2 = valid.copy()
    v2[100:300, 100:300] = False
    a_clean = S.radial_amplitude(img, valid)
    a_masked = S.radial_amplitude(sat, v2)
    a_leak = S.radial_amplitude(sat, valid)
    assert np.abs(np.log(a_masked / a_clean)).mean() < 0.5 * np.abs(np.log(a_leak / a_clean)).mean()
    model = S.spectrum_fit([img], [valid])
    out, _ = S.spectrum_apply(sat, v2, model)
    assert np.array_equal(out[100:300, 100:300], sat[100:300, 100:300])


def test_fda_swaps_only_low_frequency_amplitude_and_keeps_phase():
    valid = np.ones((512, 512), bool)
    ref = _phantom(1, 1.0, 3.0, (512, 512))
    src = _phantom(2, 1.0, 3.0, (512, 512)) + 30.0  # offset = low-frequency difference
    model = S.fda_fit([ref], [valid], ["ref"], beta=0.01)
    out, b = S.fda_apply(src, valid, model)
    assert b == 5
    F_src, F_out = np.fft.fftshift(np.fft.fft2(src)), np.fft.fftshift(np.fft.fft2(out))
    # phase preserved everywhere (up to numerical noise where amplitude is tiny)
    big = np.abs(F_src) > 1e3
    assert np.abs(np.angle(F_src[big]) - np.angle(F_out[big])).max() < 1e-3
    # amplitude outside the window unchanged, DC moved to the reference
    outside = np.ones((512, 512), bool)
    outside[256 - 5:256 + 6, 256 - 5:256 + 6] = False
    assert np.allclose(np.abs(F_src[outside]), np.abs(F_out[outside]), rtol=1e-4, atol=1e-2)
    assert abs(out.mean() - ref.mean()) < 1.0


def test_load_shift_rejects_unknown_method_and_missing_file(tmp_path):
    with pytest.raises(ValueError):
        S.load_shift("Batch_1", "x", method="nope")
    with pytest.raises(FileNotFoundError):
        S.load_shift("Batch_1", "x", method="spectrum", root=tmp_path)


def test_models_round_trip(tmp_path):
    valid = np.ones((512, 512), bool)
    imgs = [_phantom(i, 1.0, 3.0, (512, 512)) for i in range(2)]
    sm = {d: S.spectrum_fit(imgs, [valid] * 2, ["a", "b"]) for d in S.DETECTORS}
    S.save_models(tmp_path, "spectrum", sm)
    back = S.load_models(tmp_path, "spectrum")
    assert np.allclose(back["BSE"].reference, sm["BSE"].reference)
    fm = {d: S.fda_fit(imgs, [valid] * 2, ["a", "b"]) for d in S.DETECTORS}
    S.save_models(tmp_path, "fda", fm)
    back = S.load_models(tmp_path, "fda")
    assert np.allclose(back["Inlens"].amplitude, fm["Inlens"].amplitude, rtol=1e-5) and back["Inlens"].beta == S.FDA_BETA


def test_fda_reference_mean_is_independent_of_field_size():
    big = np.full((1000, 1000), 100.0, np.float32)
    small = np.full((500, 500), 40.0, np.float32)
    model = S.fda_fit([big], [np.ones(big.shape, bool)], ["ref"])
    out, _ = S.fda_apply(small, np.ones(small.shape, bool), model)
    assert abs(out.mean() - 100.0) < 1e-3


def test_corner_frequencies_do_not_enter_the_nyquist_bin():
    bins = S._radial_bins((512, 512))
    fy = np.fft.fftfreq(512)[:, None]
    fx = np.fft.fftfreq(512)[None, :]
    r = np.hypot(fy, fx)
    assert (bins[r > 0.5] == S.N_BINS).all() and (bins[r <= 0.5] < S.N_BINS).all()
    rng = np.random.default_rng(0)
    F = np.abs(np.fft.fft2(rng.normal(0, 1, (512, 512))))
    F2 = F.copy()
    F2[r > 0.5] *= 10  # change only the corners
    prof = lambda a: np.bincount(bins.ravel(), a.ravel(), minlength=S.N_BINS + 1)[:S.N_BINS]
    assert np.array_equal(prof(F), prof(F2))
    assert bins.max() == S.N_BINS and (bins == S.N_BINS).sum() > 0


def _eval_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_harmonise_ext.py"
    spec = importlib.util.spec_from_file_location("eval_harmonise_ext_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_figures_only_methods_come_from_saved_summary(tmp_path):
    import pandas as pd

    mod = _eval_script()
    pd.DataFrame({"method": ["none", "spectrum", "fda"]}).to_csv(tmp_path / "summary.csv", index=False)
    assert mod.resolve_methods(None, True, tmp_path) == ["none", "spectrum", "fda"]
    assert mod.resolve_methods(["none", "fda"], True, tmp_path) == ["none", "fda"]
    assert mod.resolve_methods(None, False, tmp_path) == ["none", "nyul", "basic", "hybrid"]


def test_figures_only_without_summary_fails_clearly(tmp_path):
    with pytest.raises(SystemExit):
        _eval_script().resolve_methods(None, True, tmp_path)


def test_hybrid_spectrum_is_a_spectrum_model_and_lists_as_a_method(tmp_path):
    assert "hybrid_spectrum" in S.METHODS and "hybrid_spectrum" in S.SPECTRUM_METHODS
    rng = np.random.default_rng(0)
    img = rng.normal(100, 10, (600, 600)).astype(np.float32)
    sm = {"BSE": S.spectrum_fit([img], [np.ones(img.shape, bool)], ["a"])}
    S.save_models(tmp_path, "hybrid_spectrum", sm)
    back = S.load_models(tmp_path, "hybrid_spectrum")
    assert isinstance(back["BSE"], S.SpectrumModel)
    np.testing.assert_allclose(back["BSE"].reference, sm["BSE"].reference, rtol=1e-6)
