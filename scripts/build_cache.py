"""Build half-resolution cache and compute full-resolution raw intensity statistics."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

import imagecodecs  # noqa: F401
import tifffile

from pmdb.io import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_ROOT,
    downsample_to_half,
    get_cache_root,
    get_data_root,
    list_sites,
    read_detector_image,
)
from pmdb.stats import raw_intensity_stats

DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"


def build_cache(
    data_root: Path | None = None,
    cache_root: Path | None = None,
    outputs_dir: Path | None = None,
    force: bool = False,
) -> None:
    """Build half-resolution cache and outputs/raw_intensity_stats.csv."""
    t_start = time.time()

    d_root = get_data_root(data_root)
    c_root = get_cache_root(cache_root)
    o_dir = outputs_dir if outputs_dir is not None else DEFAULT_OUTPUTS_DIR

    cache_half_dir = c_root / "half"
    cache_half_dir.mkdir(parents=True, exist_ok=True)
    o_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv_path = cache_half_dir / "manifest.csv"
    stats_csv_path = o_dir / "raw_intensity_stats.csv"

    # Step 1: Discover sites
    site_manifest = list_sites(data_root=d_root)
    total_sites = len(site_manifest)

    # Check idempotency: if not force, and all cache files + CSVs exist, exit early
    all_cached = (
        not force
        and manifest_csv_path.exists()
        and stats_csv_path.exists()
        and all(
            (cache_half_dir / f"{row['batch']}__{row['site']}.npz").exists()
            for _, row in site_manifest.iterrows()
        )
    )
    if all_cached:
        for idx, row in site_manifest.iterrows():
            print(f"[{idx+1}/{total_sites}] {row['batch']}/{row['site']}: already cached (skipped)")
        print(f"All {total_sites} sites already cached. Finished in {time.time() - t_start:.2f} s.")
        return

    # Process all sites
    cache_manifest_rows = []
    stats_rows = []

    for idx, row in site_manifest.iterrows():
        b = row["batch"]
        s = row["site"]
        se_det = row["se_detector"]
        npz_path = cache_half_dir / f"{b}__{s}.npz"

        # Read full-resolution detector images (4-px margin crop on both sides and asserting R == G == B)
        bse = read_detector_image(row["path_bse"])
        inlens = read_detector_image(row["path_inlens"])
        se_type = read_detector_image(row["path_se_type"])

        # Compute full-resolution stats per detector file (93 rows total across 31 sites)
        bse_stats = raw_intensity_stats(bse)
        stats_rows.append({
            "batch": b,
            "site": s,
            "detector": "BSE",
            "channel_slot": "BSE",
            **bse_stats,
        })

        inlens_stats = raw_intensity_stats(inlens)
        stats_rows.append({
            "batch": b,
            "site": s,
            "detector": "Inlens",
            "channel_slot": "Inlens",
            **inlens_stats,
        })

        se_stats = raw_intensity_stats(se_type)
        stats_rows.append({
            "batch": b,
            "site": s,
            "detector": se_det,
            "channel_slot": "SE_type",
            **se_stats,
        })

        # Stack into [BSE, Inlens, SE_type]
        raw_uint8 = np.stack([bse, inlens, se_type], axis=-1)

        # Downsample to half resolution
        half_uint8 = downsample_to_half(raw_uint8)

        # Write cache file if missing or force
        if force or not npz_path.exists():
            np.savez_compressed(npz_path, image=half_uint8)

        h_half, w_half, _ = half_uint8.shape
        cache_manifest_rows.append({
            "batch": b,
            "site": s,
            "se_detector": se_det,
            "height": h_half,
            "width": w_half,
            "nm_per_px": 50.0,
            "path": str(npz_path),
        })

        print(f"[{idx+1}/{total_sites}] {b}/{s}: processed -> half shape {half_uint8.shape}")

    # Write cache manifest
    cache_manifest_df = pd.DataFrame(cache_manifest_rows)
    cache_manifest_df.to_csv(manifest_csv_path, index=False)

    # Write stats CSV
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_csv_path, index=False)

    elapsed = time.time() - t_start
    print(f"Cache build complete: {total_sites} sites processed in {elapsed:.2f} s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PMDB half-resolution cache.")
    parser.add_argument("--force", action="store_true", help="Force rebuild existing cache files.")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory for raw data.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Root directory for cache.")
    parser.add_argument("--outputs-dir", type=Path, default=None, help="Output directory for stats.")
    args = parser.parse_args()

    build_cache(
        data_root=args.data_root,
        cache_root=args.cache_root,
        outputs_dir=args.outputs_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
