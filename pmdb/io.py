"""I/O module for PMDB microscopy datasets."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from skimage.transform import downscale_local_mean

import imagecodecs  # noqa: F401 - Must be imported before tifffile to register LZW and other codecs
import tifffile

from pmdb.stats import raw_intensity_stats
from pmdb import harmonise as _harm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_CACHE_ROOT = REPO_ROOT / "cache"

FILENAME_REGEX = re.compile(r"^img_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)\.tif$")


@dataclass
class Site:
    """Represents a single multi-detector imaging site.

    Attributes:
        image: Multi-channel image array (float32 [0, 1] if normalised, uint8 if raw).
        batch: Batch identifier string (e.g. 'Batch_1').
        site: Site identifier string (e.g. '4ih2ggld').
        se_detector: Specific detector used for the secondary electron channel ('ETD' or 'SE').
        nm_per_px: Spatial resolution in nanometres per pixel (e.g. 25.0 or 50.0).
        resolution: Resolution level loaded ('full' or 'half').
        raw_stats: Raw intensity statistics per channel on the uint8 array at this resolution,
            computed *before* any harmonisation so the imaging confound stays on the record.
        channels: Channel order tuple ('BSE', 'Inlens', 'SE_type').
        harmonise: Harmonisation method applied ('none' if the raw grey levels were kept).
        harmonised_stats: Intensity statistics after harmonisation (None when harmonise='none').
    """

    image: np.ndarray
    batch: str
    site: str
    se_detector: str
    nm_per_px: float
    resolution: Literal["full", "half"]
    raw_stats: dict[str, dict[str, float]]
    channels: tuple[str, str, str] = field(default=("BSE", "Inlens", "SE_type"))
    harmonise: str = "none"
    harmonised_stats: dict[str, dict[str, float]] | None = None


def get_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve data root directory respecting PMDB_DATA environment variable."""
    if data_root is not None:
        return Path(data_root)
    env_root = os.environ.get("PMDB_DATA")
    if env_root:
        return Path(env_root)
    return DEFAULT_DATA_ROOT


def get_cache_root(cache_root: str | Path | None = None) -> Path:
    """Resolve cache root directory respecting PMDB_CACHE environment variable."""
    if cache_root is not None:
        return Path(cache_root)
    env_cache = os.environ.get("PMDB_CACHE")
    if env_cache:
        return Path(env_cache)
    return DEFAULT_CACHE_ROOT


def get_scale_from_tiff(page: tifffile.TiffPage) -> float:
    """Extract physical scale in nm/pixel from TIFF XResolution and ResolutionUnit tags.

    Raises:
        ValueError: If tags are missing, invalid, or unit is unsupported.
    """
    xres_tag = page.tags.get("XResolution")
    if xres_tag is None:
        raise ValueError("Missing XResolution tag in TIFF file.")

    unit_tag = page.tags.get("ResolutionUnit")
    unit_val = unit_tag.value if unit_tag is not None else 2
    unit_code = getattr(unit_val, "value", unit_val)

    xres = xres_tag.value
    if isinstance(xres, tuple):
        xres_val = xres[0] / xres[1]
    else:
        xres_val = float(xres)

    if xres_val <= 0:
        raise ValueError(f"Invalid XResolution value: {xres_val}")

    if unit_code == 2:  # Inch (25.4 mm = 25,400,000 nm)
        nm_per_px = 25_400_000.0 / xres_val
    elif unit_code == 3:  # Centimetre (1 cm = 10,000,000 nm)
        nm_per_px = 10_000_000.0 / xres_val
    else:
        raise ValueError(f"Unsupported ResolutionUnit in TIFF: {unit_val}")

    return float(nm_per_px)


def list_sites(data_root: str | Path | None = None) -> pd.DataFrame:
    """Discover, validate, and index all imaging sites in the data directory.

    Args:
        data_root: Path to root data folder. Defaults to PMDB_DATA or <repo>/data.

    Returns:
        pandas.DataFrame with columns:
            batch, site, se_detector, height, width, nm_per_px, path_bse, path_inlens, path_se_type

    Raises:
        FileNotFoundError: If data_root does not exist.
        ValueError: If inventory is invalid (missing/duplicate detector, shape mismatch, or scale error).
    """
    root = get_data_root(data_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Data root directory not found: {root}")

    batch_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("Batch_")])
    if not batch_dirs:
        raise ValueError(f"No Batch_* directories found under data root: {root}")

    rows = []

    for b_dir in batch_dirs:
        batch_name = b_dir.name
        # Collect files by site
        sites_dict: dict[str, dict[str, Path]] = {}

        for file_path in sorted(b_dir.iterdir()):
            if file_path.name.startswith("."):
                continue
            if not file_path.name.endswith(".tif"):
                continue

            match = FILENAME_REGEX.match(file_path.name)
            if not match:
                raise ValueError(f"Unexpected filename format in {b_dir.name}: {file_path.name}")

            site_id, detector = match.group(1), match.group(2)
            if site_id not in sites_dict:
                sites_dict[site_id] = {}

            if detector in sites_dict[site_id]:
                raise ValueError(
                    f"Duplicate detector {detector} found for site {site_id} in {batch_name}: "
                    f"{file_path} and {sites_dict[site_id][detector]}"
                )

            sites_dict[site_id][detector] = file_path

        for site_id in sorted(sites_dict.keys()):
            dets = sites_dict[site_id]
            # Must have BSE and Inlens
            if "BSE" not in dets:
                raise ValueError(f"Site {batch_name}/{site_id} is missing BSE detector file.")
            if "Inlens" not in dets:
                raise ValueError(f"Site {batch_name}/{site_id} is missing Inlens detector file.")

            has_etd = "ETD" in dets
            has_se = "SE" in dets
            if has_etd and has_se:
                raise ValueError(f"Site {batch_name}/{site_id} has both ETD and SE detectors.")
            if not has_etd and not has_se:
                raise ValueError(f"Site {batch_name}/{site_id} has neither ETD nor SE detector.")

            se_detector = "ETD" if has_etd else "SE"
            se_path = dets[se_detector]
            bse_path = dets["BSE"]
            inlens_path = dets["Inlens"]

            # Validate shapes and scales from headers
            paths = [bse_path, inlens_path, se_path]
            shapes = []
            scales = []

            for p in paths:
                with tifffile.TiffFile(p) as tif:
                    page = tif.pages[0]
                    shapes.append(page.shape)
                    scale = get_scale_from_tiff(page)
                    scales.append(scale)

            # Check shape consistency across detectors
            if not (shapes[0] == shapes[1] == shapes[2]):
                raise ValueError(
                    f"Shape mismatch across detectors for site {batch_name}/{site_id}: "
                    f"BSE={shapes[0]}, Inlens={shapes[1]}, {se_detector}={shapes[2]}"
                )

            # Check scale within 25 ± 0.01 nm/px
            for p, sc in zip(paths, scales):
                if abs(sc - 25.0) > 0.01:
                    raise ValueError(
                        f"Scale outside 25 ± 0.01 nm/px for file {p}: {sc:.4f} nm/px"
                    )

            H, W = shapes[0][0], shapes[0][1]
            rows.append({
                "batch": batch_name,
                "site": site_id,
                "se_detector": se_detector,
                "height": H,
                "width": W,
                "nm_per_px": scales[0],
                "path_bse": str(bse_path),
                "path_inlens": str(inlens_path),
                "path_se_type": str(se_path),
            })

    columns = [
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
    df = pd.DataFrame(rows, columns=columns)
    df.sort_values(by=["batch", "site"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def read_detector_image(path: str | Path) -> np.ndarray:
    """Read a single detector TIFF file, crop a 4-px margin from both edges, assert R == G == B, and return 2D uint8 array.

    Per D-003 (rev 2):
        Crop a fixed 4-px margin from both the left and right edges (arr[:, 4:W-4]) before anything else.
        Then assert R == G == B on the cropped array and keep a single channel. If the assertion fails,
        raise an error naming the file and the offending columns; never average the channels silently.

    Args:
        path: Path to TIFF image.

    Returns:
        2D uint8 numpy array of shape (H, W - 8).

    Raises:
        ValueError: If the file is not (H, W, 3) or R == G == B fails on the cropped array.
    """
    arr = tifffile.imread(path)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3) in {path}, got shape {arr.shape}")

    H, W, _ = arr.shape
    if W <= 8:
        raise ValueError(f"Image width {W} in {path} is too small to crop 4 px from each side.")

    # Crop fixed 4-px margin from left and right edges (D-003 rev 2)
    cropped = arr[:, 4 : W - 4, :]
    r = cropped[:, :, 0]
    g = cropped[:, :, 1]
    b = cropped[:, :, 2]

    # Assert R == G == B and keep a single channel
    diff_rg = r != g
    diff_rb = r != b
    if np.any(diff_rg) or np.any(diff_rb):
        mismatched_mask = diff_rg | diff_rb
        # Find offending column indices in original coordinates
        offending_cols_cropped = np.unique(np.argwhere(mismatched_mask)[:, 1])
        offending_cols_orig = (offending_cols_cropped + 4).tolist()
        raise ValueError(
            f"R == G == B assertion failed in {path} after 4-px margin crop: "
            f"offending original columns {offending_cols_orig}."
        )

    return r


def normalise_fixed(raw_uint8: np.ndarray) -> np.ndarray:
    """Fixed scaling: grey level / 255 -> float32 in [0, 1], identical for every image.

    Unlike :func:`normalise_image`, this does not re-stretch each image to its own
    percentiles, so harmonised sites stay on one common grey scale.
    """
    return (raw_uint8.astype(np.float32) / 255.0).astype(np.float32)


def normalise_image(raw_uint8: np.ndarray) -> np.ndarray:
    """Perform robust percentile scaling per image and per channel (D-005).

    p0.5 maps to 0 and p99.5 maps to 1, clipped to [0, 1] and returned as float32.

    Args:
        raw_uint8: Input image array of shape (H, W, 3) and dtype uint8.

    Returns:
        float32 array of shape (H, W, 3) with values in [0.0, 1.0].
    """
    out = np.empty(raw_uint8.shape, dtype=np.float32)
    for c in range(3):
        ch = raw_uint8[..., c].astype(np.float32)
        p0_5 = float(np.percentile(ch, 0.5))
        p99_5 = float(np.percentile(ch, 99.5))
        denom = p99_5 - p0_5
        if denom > 0:
            norm_ch = (ch - p0_5) / denom
        else:
            norm_ch = np.zeros_like(ch)
        out[..., c] = np.clip(norm_ch, 0.0, 1.0)
    return out


def downsample_to_half(raw_uint8: np.ndarray) -> np.ndarray:
    """Downsample image by 2x2 block mean after cropping to even H and W (D-006).

    If W is odd after the 4-px margin crop, crop the last column before downsampling.

    Args:
        raw_uint8: Stacked image array of shape (H, W_cropped, 3) and dtype uint8.

    Returns:
        Downscaled image array of shape (H//2, W_cropped//2, 3) and dtype uint8.
    """
    H, W, C = raw_uint8.shape
    H_even = H - (H % 2)
    W_even = W - (W % 2)
    cropped = raw_uint8[:H_even, :W_even, :]
    down = downscale_local_mean(cropped, (2, 2, 1))
    return np.round(down).astype(np.uint8)


def load_site(
    batch: str,
    site: str,
    resolution: Literal["full", "half"] = "full",
    normalise: Literal["percentile", "fixed", "none"] = "percentile",
    data_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    harmonise: str = "none",
) -> Site:
    """Load an aligned, multi-detector imaging site.

    Args:
        batch: Batch identifier (e.g. 'Batch_1').
        site: Site identifier (e.g. '4ih2ggld').
        resolution: 'full' or 'half'. If 'half', reads from the cache.
        normalise: 'percentile' (per-image p0.5->0, p99.5->1, float32), 'fixed' (grey/255,
            float32, the same map for every image) or 'none' (uint8).
        data_root: Path to data directory (optional).
        cache_root: Path to cache directory (optional).
        harmonise: Grey-level harmonisation method from :data:`pmdb.harmonise.METHODS`
            ('none', 'offset', 'affine2', 'affine3', 'histmatch', 'hybrid'). The per-site LUT is read from
            ``<cache_root>/harmonised/<method>/luts.npz`` and applied before normalisation.
            Use with ``normalise='fixed'`` or ``'none'``; per-image percentile normalisation
            would re-stretch each image and undo most of the harmonisation.

    Returns:
        Site object.

    Raises:
        FileNotFoundError: If cache or data files are missing.
        ValueError: If inputs or configurations are invalid.
    """
    if resolution not in ("full", "half"):
        raise ValueError(f"Invalid resolution '{resolution}'. Must be 'full' or 'half'.")
    if normalise not in ("percentile", "fixed", "none"):
        raise ValueError(f"Invalid normalise '{normalise}'. Must be 'percentile', 'fixed' or 'none'.")
    if harmonise not in _harm.METHODS:
        raise ValueError(f"Invalid harmonise '{harmonise}'. Must be one of {_harm.METHODS}.")

    cache_dir = get_cache_root(cache_root)

    if resolution == "half":
        cache_file = cache_dir / "half" / f"{batch}__{site}.npz"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Half-resolution cache for site {batch}/{site} not found at {cache_file}. "
                f"Run 'python scripts/build_cache.py' to build the half-resolution cache."
            )

        with np.load(cache_file) as data:
            raw_uint8 = data["image"]

        # Determine se_detector from manifest.csv if available
        manifest_path = cache_dir / "half" / "manifest.csv"
        se_detector = None
        if manifest_path.exists():
            m_df = pd.read_csv(manifest_path)
            match = m_df[(m_df["batch"] == batch) & (m_df["site"] == site)]
            if not match.empty:
                se_detector = str(match.iloc[0]["se_detector"])

        if se_detector is None:
            # Fallback to reading data manifest
            df = list_sites(data_root=data_root)
            match = df[(df["batch"] == batch) & (df["site"] == site)]
            if match.empty:
                raise ValueError(f"Site {batch}/{site} not found in manifest.")
            se_detector = str(match.iloc[0]["se_detector"])

        nm_per_px = 50.0

    else:  # resolution == "full"
        df = list_sites(data_root=data_root)
        match = df[(df["batch"] == batch) & (df["site"] == site)]
        if match.empty:
            raise ValueError(f"Site {batch}/{site} not found in manifest.")

        row = match.iloc[0]
        se_detector = str(row["se_detector"])
        nm_per_px = float(row["nm_per_px"])

        bse = read_detector_image(row["path_bse"])
        inlens = read_detector_image(row["path_inlens"])
        se_type = read_detector_image(row["path_se_type"])

        # Stack detectors in fixed order [BSE, Inlens, SE_type] (D-001)
        raw_uint8 = np.stack([bse, inlens, se_type], axis=-1)

    # Compute raw statistics on the uint8 array at the loaded resolution
    raw_stats = {
        "BSE": raw_intensity_stats(raw_uint8[..., 0]),
        "Inlens": raw_intensity_stats(raw_uint8[..., 1]),
        "SE_type": raw_intensity_stats(raw_uint8[..., 2]),
    }

    harmonised_stats = None
    if harmonise != "none":
        lut = _harm.load_lut(cache_dir, harmonise, batch, site)
        raw_uint8 = _harm.apply_lut(raw_uint8, lut)
        harmonised_stats = {
            "BSE": raw_intensity_stats(raw_uint8[..., 0]),
            "Inlens": raw_intensity_stats(raw_uint8[..., 1]),
            "SE_type": raw_intensity_stats(raw_uint8[..., 2]),
        }

    if normalise == "percentile":
        image = normalise_image(raw_uint8)
    elif normalise == "fixed":
        image = normalise_fixed(raw_uint8)
    else:
        image = raw_uint8

    return Site(
        image=image,
        batch=batch,
        site=site,
        se_detector=se_detector,
        nm_per_px=nm_per_px,
        resolution=resolution,
        raw_stats=raw_stats,
        harmonise=harmonise,
        harmonised_stats=harmonised_stats,
    )
