"""Unit tests with synthetic data (no real data required)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
from skimage.transform import downscale_local_mean

from pmdb.io import (
    downsample_to_half,
    list_sites,
    load_site,
    normalise_image,
    read_detector_image,
)
from pmdb.stats import raw_intensity_stats


def make_tiff(
    path: Path,
    shape: tuple[int, int] = (20, 32),
    base_val: int = 100,
    border_mismatch: bool = False,
    interior_mismatch: bool = False,
    scale_nm: float = 25.0,
) -> None:
    """Create a synthetic LZW-compressed TIFF with specified properties."""
    H, W = shape
    arr = np.full((H, W, 3), base_val, dtype=np.uint8)
    if border_mismatch:
        # Channels differ in border columns (x <= 3 or x >= W - 2)
        arr[:, 0, 1] = 200
        arr[:, 1, 1] = 200
        arr[:, W - 1, 1] = 200
    if interior_mismatch:
        # Channels differ in an interior column (e.g. col 10 where 4 <= 10 < W - 4)
        arr[:, 10, 1] = 250

    xres = 25_400_000.0 / scale_nm
    tifffile.imwrite(path, arr, compression="lzw", resolution=(xres, xres), resolutionunit="INCH")


def test_raw_intensity_stats() -> None:
    """Test raw_intensity_stats on known values."""
    arr = np.array([0, 0, 50, 100, 150, 200, 255], dtype=np.uint8)
    stats = raw_intensity_stats(arr)

    expected_keys = {"mean", "std", "p0_5", "p1", "p50", "p99", "p99_5", "frac_zero", "frac_255"}
    assert set(stats.keys()) == expected_keys
    assert np.isclose(stats["mean"], np.mean(arr))
    assert np.isclose(stats["std"], np.std(arr))
    assert stats["p50"] == 100.0
    assert np.isclose(stats["frac_zero"], 2 / 7)
    assert np.isclose(stats["frac_255"], 1 / 7)

    with pytest.raises(ValueError, match="Cannot compute intensity statistics on an empty array"):
        raw_intensity_stats(np.array([], dtype=np.uint8))


def test_read_detector_image_margin_crop_and_assertion(tmp_path: Path) -> None:
    """Test 4-px margin crop on both sides and asserting R == G == B.

    Per D-003 (rev 2) & Scope:
      - Border column differences (x <= 3 or x >= W - 2) are cropped away and pass.
      - Interior column differences (4 <= x < W - 4) raise an error naming file and offending columns.
    """
    # Border mismatch should pass because 4-px margin is cropped from both sides
    border_file = tmp_path / "border_mismatch.tif"
    make_tiff(border_file, shape=(20, 32), border_mismatch=True)
    img = read_detector_image(border_file)
    assert img.shape == (20, 24)  # 32 - 8 = 24
    assert img.dtype == np.uint8

    # Interior mismatch (e.g. col 10) must raise an error naming the file and offending columns
    interior_file = tmp_path / "interior_mismatch.tif"
    make_tiff(interior_file, shape=(20, 32), interior_mismatch=True)
    with pytest.raises(ValueError) as excinfo:
        read_detector_image(interior_file)
    err_msg = str(excinfo.value)
    assert "interior_mismatch.tif" in err_msg
    assert "offending original columns [10]" in err_msg


def test_list_sites_synthetic(tmp_path: Path) -> None:
    """Test list_sites on synthetic directory with ETD and SE sites."""
    b1 = tmp_path / "Batch_1"
    b1.mkdir()

    # Site 1 with ETD and border mismatch
    make_tiff(b1 / "img_siteA_BSE.tif", (20, 32), base_val=80, border_mismatch=True)
    make_tiff(b1 / "img_siteA_Inlens.tif", (20, 32), base_val=90)
    make_tiff(b1 / "img_siteA_ETD.tif", (20, 32), base_val=100)

    # Site 2 with SE instead of ETD
    make_tiff(b1 / "img_siteB_BSE.tif", (22, 34), base_val=110)
    make_tiff(b1 / "img_siteB_Inlens.tif", (22, 34), base_val=120)
    make_tiff(b1 / "img_siteB_SE.tif", (22, 34), base_val=130)

    df = list_sites(tmp_path)
    assert len(df) == 2
    assert list(df.columns) == [
        "batch",
        "site",
        "se_detector",
        "height",
        "width",
        "nm_per_px",
        "path_bse",
        "path_inlens",
        "path_se_type",
    ]

    row_a = df[df["site"] == "siteA"].iloc[0]
    assert row_a["se_detector"] == "ETD"
    assert row_a["height"] == 20
    assert row_a["width"] == 32
    assert abs(row_a["nm_per_px"] - 25.0) < 0.01

    row_b = df[df["site"] == "siteB"].iloc[0]
    assert row_b["se_detector"] == "SE"
    assert row_b["height"] == 22
    assert row_b["width"] == 34


def test_list_sites_validation_failures(tmp_path: Path) -> None:
    """Test inventory validation failures: missing detector, shape mismatch, scale error, both ETD & SE."""
    b1 = tmp_path / "Batch_1"
    b1.mkdir()

    # Missing inlens
    make_tiff(b1 / "img_s1_BSE.tif", (20, 32))
    make_tiff(b1 / "img_s1_ETD.tif", (20, 32))
    with pytest.raises(ValueError, match="missing Inlens"):
        list_sites(tmp_path)

    # Add both ETD and SE
    make_tiff(b1 / "img_s1_Inlens.tif", (20, 32))
    make_tiff(b1 / "img_s1_SE.tif", (20, 32))
    with pytest.raises(ValueError, match="both ETD and SE"):
        list_sites(tmp_path)
    (b1 / "img_s1_SE.tif").unlink()

    # Shape mismatch
    make_tiff(b1 / "img_s2_BSE.tif", (20, 32))
    make_tiff(b1 / "img_s2_Inlens.tif", (20, 32))
    make_tiff(b1 / "img_s2_ETD.tif", (24, 32))
    with pytest.raises(ValueError, match="Shape mismatch"):
        list_sites(tmp_path)
    for f in (b1 / "img_s2_BSE.tif", b1 / "img_s2_Inlens.tif", b1 / "img_s2_ETD.tif"):
        f.unlink()

    # Scale error (outside 25 ± 0.01)
    make_tiff(b1 / "img_s3_BSE.tif", (20, 32), scale_nm=30.0)
    make_tiff(b1 / "img_s3_Inlens.tif", (20, 32))
    make_tiff(b1 / "img_s3_ETD.tif", (20, 32))
    with pytest.raises(ValueError, match="Scale outside 25 ± 0.01"):
        list_sites(tmp_path)


def test_normalise_image() -> None:
    """Test percentile normalisation maps p0.5 to 0 and p99.5 to 1."""
    # Gradient image from 0 to 255
    raw = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 256, 1), (10, 1, 3))
    norm = normalise_image(raw)

    assert norm.dtype == np.float32
    assert norm.shape == raw.shape
    assert np.min(norm) >= 0.0
    assert np.max(norm) <= 1.0
    assert np.isclose(norm[0, 0, 0], 0.0)
    assert np.isclose(norm[0, 255, 0], 1.0)


def test_downsample_to_half() -> None:
    """Test 2x2 block downsampling with cropped dimensions."""
    # Stacked raw image shape after 4-px margin crop on both sides: (20, 24, 3)
    # Downsample produces (10, 12, 3), which equals (H//2, (W-8)//2, 3)
    raw = np.full((20, 24, 3), 100, dtype=np.uint8)
    down = downsample_to_half(raw)
    assert down.shape == (10, 12, 3)
    assert down.dtype == np.uint8
    assert np.all(down == 100)


def test_load_site_synthetic(tmp_path: Path) -> None:
    """Test load_site with synthetic data (both full and half resolution)."""
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    b1 = data_dir / "Batch_1"
    b1.mkdir(parents=True)

    H, W = 20, 32  # after 4-px margin crop on both sides: (20, 24) -> half shape: (10, 12, 3)
    make_tiff(b1 / "img_syn1_BSE.tif", (H, W), base_val=50, border_mismatch=True)
    make_tiff(b1 / "img_syn1_Inlens.tif", (H, W), base_val=100)
    make_tiff(b1 / "img_syn1_ETD.tif", (H, W), base_val=150)

    # Test full resolution loading
    site_full = load_site("Batch_1", "syn1", resolution="full", normalise="none", data_root=data_dir)
    assert site_full.resolution == "full"
    assert site_full.image.shape == (20, 24, 3)
    assert site_full.image.dtype == np.uint8
    assert site_full.se_detector == "ETD"
    assert "BSE" in site_full.raw_stats

    site_full_norm = load_site("Batch_1", "syn1", resolution="full", normalise="percentile", data_root=data_dir)
    assert site_full_norm.image.dtype == np.float32

    # Test half resolution missing cache error
    with pytest.raises(FileNotFoundError, match="Run 'python scripts/build_cache.py'"):
        load_site("Batch_1", "syn1", resolution="half", cache_root=cache_dir)

    # Populate cache manually
    half_dir = cache_dir / "half"
    half_dir.mkdir(parents=True)
    raw_stacked = np.stack([
        read_detector_image(b1 / "img_syn1_BSE.tif"),
        read_detector_image(b1 / "img_syn1_Inlens.tif"),
        read_detector_image(b1 / "img_syn1_ETD.tif"),
    ], axis=-1)
    half_img = downsample_to_half(raw_stacked)
    assert half_img.shape == (10, 12, 3)
    np.savez_compressed(half_dir / "Batch_1__syn1.npz", image=half_img)

    # Also test manifest.csv lookup
    manifest_csv = half_dir / "manifest.csv"
    manifest_csv.write_text("batch,site,se_detector,height,width,nm_per_px,path\nBatch_1,syn1,ETD,10,12,50.0,dummy\n")

    site_half = load_site("Batch_1", "syn1", resolution="half", normalise="none", cache_root=cache_dir)
    assert site_half.resolution == "half"
    assert site_half.image.shape == (10, 12, 3)
    assert site_half.image.dtype == np.uint8
    assert site_half.nm_per_px == 50.0
    assert site_half.se_detector == "ETD"
