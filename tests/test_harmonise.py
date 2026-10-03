"""Tests for pmdb.harmonise: synthetic (no data) and real-data checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmdb import harmonise as H

NM = 50.0
PORE, GRAPHITE, SI = 0, 57, 112


def _synthetic_site(rng: np.random.Generator, size: int = 600) -> np.ndarray:
    """Three-phase BSE-like image (pores, graphite matrix, Si discs) on three channels."""
    img = np.full((size, size), GRAPHITE, dtype=np.float64)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(60):  # Si discs
        cy, cx, r = rng.integers(10, size - 10, 2).tolist() + [rng.integers(4, 9)]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = SI
    for _ in range(25):  # pores
        cy, cx, r = rng.integers(10, size - 10, 2).tolist() + [rng.integers(6, 14)]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = PORE
    img += rng.normal(0, 3.0, img.shape)
    base = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return np.stack([base, base, base], axis=-1)


def _distort(arr: np.ndarray, gain: float, offset: float) -> np.ndarray:
    return np.clip(np.round(arr.astype(np.float64) * gain + offset), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def clean_and_reference():
    rng = np.random.default_rng(0)
    sites = [_synthetic_site(rng) for _ in range(4)]
    rows, hists = [], []
    for i, s in enumerate(sites):
        a, h = H.estimate_anchors(s, NM)
        rows.append(H.anchors_to_row("Batch_1", f"s{i}", a))
        hists.append(h)
    ref = H.build_reference(pd.DataFrame(rows), np.stack(hists))
    return sites, ref


def test_estimate_anchors_recovers_phase_levels(clean_and_reference) -> None:
    sites, ref = clean_and_reference
    a, hist = H.estimate_anchors(sites[0], NM)
    assert hist.shape == (3, 256) and np.allclose(hist.sum(axis=1), 1.0)
    for ch in H.CHANNELS:
        assert abs(a[ch]["black"] - PORE) <= 2
        assert abs(a[ch]["graphite"] - GRAPHITE) <= 2
        assert abs(a[ch]["si"] - SI) <= 4


@pytest.mark.parametrize("method", ["offset", "affine2", "affine3", "histmatch", "hybrid"])
def test_methods_remove_black_level(clean_and_reference, method: str) -> None:
    sites, ref = clean_and_reference
    gain, offset = (1.0, 20.0) if method == "offset" else (0.78, 20.0)
    bad = _distort(sites[1], gain, offset)
    a, hist = H.estimate_anchors(bad, NM)
    assert a["BSE"]["black"] >= 15  # the artefact is there before harmonisation
    lut, params = H.fit_lut(method, a, hist, ref)
    assert lut.shape == (3, 256) and lut.dtype == np.uint8
    assert np.all(np.diff(lut.astype(int), axis=1) >= 0)  # monotone grey-level maps
    fixed = H.apply_lut(bad, lut)
    b, _ = H.estimate_anchors(fixed, NM)
    assert abs(b["BSE"]["black"] - ref.anchors["BSE"]["black"]) <= 2
    if method != "offset":  # gain-correcting methods also put graphite back
        assert abs(b["BSE"]["graphite"] - ref.anchors["BSE"]["graphite"]) <= 3
    if method in ("affine2", "affine3", "hybrid"):
        params = params[params.channel != "Inlens"] if method == "hybrid" else params
        assert params.slope.between(1.2, 1.4).all()  # ~1/0.78
        # the whole image is recovered, not just the anchors
        assert np.mean(np.abs(fixed.astype(int) - sites[1].astype(int))) < 3.0


def test_affine2_preserves_si_graphite_contrast(clean_and_reference) -> None:
    sites, ref = clean_and_reference
    # a site whose Si is genuinely brighter relative to graphite (a material difference)
    brighter = sites[2].astype(np.float64)
    brighter[brighter > 90] += 20
    brighter = np.clip(np.round(brighter), 0, 255).astype(np.uint8)
    bad = _distort(brighter, 0.8, 18.0)
    a, hist = H.estimate_anchors(bad, NM)
    ratio_before = (a["BSE"]["si"] - a["BSE"]["black"]) / (a["BSE"]["graphite"] - a["BSE"]["black"])
    lut, _ = H.fit_lut("affine2", a, hist, ref)
    b, _ = H.estimate_anchors(H.apply_lut(bad, lut), NM)
    ratio_after = (b["BSE"]["si"] - b["BSE"]["black"]) / (b["BSE"]["graphite"] - b["BSE"]["black"])
    assert ratio_after == pytest.approx(ratio_before, rel=0.05)
    assert b["BSE"]["si"] > ref.anchors["BSE"]["si"] + 10  # the extra Si brightness survives


def test_none_is_identity_and_lut_roundtrip(tmp_path: Path, clean_and_reference) -> None:
    sites, ref = clean_and_reference
    a, hist = H.estimate_anchors(sites[3], NM)
    lut, params = H.fit_lut("none", a, hist, ref)
    assert np.array_equal(H.apply_lut(sites[3], lut), sites[3])
    params.insert(0, "site", "s3")
    params.insert(0, "batch", "Batch_1")
    H.save_luts(tmp_path, "affine2", {H.lut_key("Batch_1", "s3"): lut}, params, ref)
    assert np.array_equal(H.load_lut(tmp_path, "affine2", "Batch_1", "s3"), lut)
    ref2 = H.load_reference(tmp_path, "affine2")
    assert ref2.anchors == ref.anchors and np.allclose(ref2.cdf, ref.cdf)
    with pytest.raises(KeyError):
        H.load_lut(tmp_path, "affine2", "Batch_1", "missing")
    with pytest.raises(FileNotFoundError):
        H.load_lut(tmp_path, "offset", "Batch_1", "s3")
    with pytest.raises(ValueError):
        H.fit_lut("bogus", a, hist, ref)


def test_load_site_applies_lut(tmp_path: Path, clean_and_reference) -> None:
    from pmdb.io import load_site

    sites, ref = clean_and_reference
    bad = _distort(sites[0], 0.78, 20.0)
    half = tmp_path / "half"
    half.mkdir()
    np.savez_compressed(half / "Batch_9__abc.npz", image=bad)
    pd.DataFrame([{"batch": "Batch_9", "site": "abc", "se_detector": "ETD"}]).to_csv(half / "manifest.csv", index=False)
    a, hist = H.estimate_anchors(bad, NM)
    lut, params = H.fit_lut("affine2", a, hist, ref)
    H.save_luts(tmp_path, "affine2", {H.lut_key("Batch_9", "abc"): lut}, params, ref)

    raw = load_site("Batch_9", "abc", resolution="half", normalise="none", cache_root=tmp_path)
    assert raw.harmonise == "none" and raw.harmonised_stats is None
    harm = load_site("Batch_9", "abc", resolution="half", normalise="none", cache_root=tmp_path, harmonise="affine2")
    assert harm.harmonise == "affine2"
    assert harm.raw_stats["BSE"]["p0_5"] >= 15  # the raw record keeps the confound
    assert harm.harmonised_stats["BSE"]["p0_5"] <= 2
    assert np.array_equal(harm.image, H.apply_lut(bad, lut))
    fixed = load_site("Batch_9", "abc", resolution="half", normalise="fixed", cache_root=tmp_path, harmonise="affine2")
    assert fixed.image.dtype == np.float32
    assert np.allclose(fixed.image, harm.image.astype(np.float32) / 255.0)
    with pytest.raises(ValueError):
        load_site("Batch_9", "abc", resolution="half", cache_root=tmp_path, harmonise="bogus")
    default = load_site("Batch_9", "abc", resolution="half", cache_root=tmp_path, harmonise="affine2")
    assert default.image.dtype == np.float32 and np.allclose(default.image, fixed.image)  # defaults to 'fixed'
    plain = load_site("Batch_9", "abc", resolution="half", cache_root=tmp_path)
    assert plain.image.max() == pytest.approx(1.0)  # unharmonised default is still percentile
    with pytest.warns(UserWarning, match="percentile"):
        load_site("Batch_9", "abc", resolution="half", cache_root=tmp_path, harmonise="affine2", normalise="percentile")
    with pytest.raises(ValueError):
        load_site("Batch_9", "abc", resolution="half", cache_root=tmp_path, normalise="bogus")


# ----------------------------------------------------------------------------------------
# Real data
# ----------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
STRONG = ("71vgq3fw", "kbdh4tri", "tuy3zymq", "x7u69zsw")


@pytest.mark.data
@pytest.mark.parametrize("method", ["offset", "affine2", "affine3", "histmatch", "hybrid"])
def test_real_data_black_level_removed(method: str) -> None:
    from pmdb.io import load_site

    if not (REPO / "cache" / "harmonised" / method / "luts.npz").exists():
        pytest.skip("harmonisation LUTs not built")
    for site in STRONG:
        s = load_site("Batch_3", site, resolution="half", normalise="none", harmonise=method)
        assert s.raw_stats["BSE"]["p0_5"] >= 15
        # affine3 is known to leave an Inlens black level on these sites (anchors not collinear)
        for ch in (("BSE", "SE_type") if method == "affine3" else H.CHANNELS):
            assert s.harmonised_stats[ch]["p0_5"] <= 2, (site, ch, method)
    # clean sites' BSE is left (almost) alone by the methods that do not anchor Si
    if method in ("offset", "affine2", "hybrid"):
        s = load_site("Batch_1", "4ih2ggld", resolution="half", normalise="none")
        h = load_site("Batch_1", "4ih2ggld", resolution="half", normalise="none", harmonise=method)
        assert np.mean(np.abs(h.image[..., 0].astype(int) - s.image[..., 0].astype(int))) < 4.0


@pytest.mark.data
def test_real_data_lut_coverage() -> None:
    luts = REPO / "cache" / "harmonised" / "affine2" / "luts.npz"
    if not luts.exists():
        pytest.skip("harmonisation LUTs not built")
    manifest = pd.read_csv(REPO / "cache" / "half" / "manifest.csv")
    with np.load(luts) as z:
        keys = set(z.files)
    assert keys == {H.lut_key(b, s) for b, s in zip(manifest.batch, manifest.site)}
    heldout = REPO / "cache_heldout" / "harmonised" / "affine2" / "luts.npz"
    if heldout.exists():
        with np.load(heldout) as z:
            assert len(z.files) == 3
