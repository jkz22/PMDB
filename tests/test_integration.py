"""Integration tests against real data (marked @pytest.mark.data)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pmdb.io import DEFAULT_DATA_ROOT, get_data_root, list_sites


@pytest.mark.data
def test_real_data_inventory_and_spec_criteria() -> None:
    """Validate acceptance criterion 2 against real microscopy data.

    Asserts:
      - 31 sites, with per-batch counts {Batch_1: 7, Batch_2: 7, Batch_3: 17}
      - se_detector counts of {ETD: 27, SE: 4}
      - every nm_per_px within 25 ± 0.01
      - R == G == B in all 93 files after cropping the 4-px left and right margins
      - all detectors at each site sharing one shape
      - a negative control: on the uncropped arrays, exactly 39 files have R != G,
        and every mismatched column satisfies x <= 3 or x >= W - 2
    """
    data_root = get_data_root()
    if not data_root.exists() or not (data_root / "Batch_1").exists():
        pytest.skip(f"Real data directory not found at {data_root}")

    df = list_sites(data_root=data_root)

    # 1. 31 sites, with per-batch counts {Batch_1: 7, Batch_2: 7, Batch_3: 17}
    assert len(df) == 31, f"Expected 31 sites, found {len(df)}"
    batch_counts = df["batch"].value_counts().to_dict()
    assert batch_counts == {"Batch_1": 7, "Batch_2": 7, "Batch_3": 17}, (
        f"Unexpected batch counts: {batch_counts}"
    )

    # 2. se_detector counts of {ETD: 27, SE: 4}
    se_counts = df["se_detector"].value_counts().to_dict()
    assert se_counts == {"ETD": 27, "SE": 4}, f"Unexpected se_detector counts: {se_counts}"

    # 3. every nm_per_px within 25 ± 0.01
    for _, row in df.iterrows():
        scale = row["nm_per_px"]
        assert 24.99 <= scale <= 25.01, (
            f"Scale out of bounds for {row['batch']}/{row['site']}: {scale}"
        )

    # 4. all detectors at each site sharing one shape
    for _, row in df.iterrows():
        for p in [row["path_bse"], row["path_inlens"], row["path_se_type"]]:
            with tifffile.TiffFile(p) as tif:
                sh = tif.pages[0].shape
                assert (sh[0], sh[1]) == (row["height"], row["width"]), (
                    f"Shape mismatch in {p}: expected {(row['height'], row['width'])}, got {sh}"
                )

    # Collect all 93 file paths
    all_files = []
    for _, row in df.iterrows():
        all_files.extend([row["path_bse"], row["path_inlens"], row["path_se_type"]])
    assert len(all_files) == 93

    # 5. R == G == B in all 93 files after cropping the 4-px left and right margins
    # 6. Negative control: on the uncropped arrays, exactly 39 files have R != G,
    #    and every mismatched column satisfies x <= 3 or x >= W - 2
    files_with_rg_mismatch = 0

    for p in all_files:
        arr = tifffile.imread(p)
        H, W, C = arr.shape
        diff_rg = arr[:, :, 0] != arr[:, :, 1]

        if np.any(diff_rg):
            files_with_rg_mismatch += 1
            mismatched_cols = np.unique(np.argwhere(diff_rg)[:, 1])
            for col in mismatched_cols:
                assert col <= 3 or col >= W - 2, (
                    f"Negative control failed in {p}: mismatched column {col} outside border (W={W})"
                )

        # Check cropped array (arr[:, 4:W-4])
        cropped = arr[:, 4 : W - 4, :]
        r = cropped[:, :, 0]
        g = cropped[:, :, 1]
        b = cropped[:, :, 2]
        assert np.array_equal(r, g), f"R != G in {p} after 4-px margin crop"
        assert np.array_equal(r, b), f"R != B in {p} after 4-px margin crop"

    assert files_with_rg_mismatch == 39, (
        f"Expected exactly 39 files with R != G on uncropped arrays, got {files_with_rg_mismatch}"
    )
