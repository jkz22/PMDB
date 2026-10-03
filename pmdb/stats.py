"""Raw intensity statistics module."""

from __future__ import annotations

import numpy as np


def raw_intensity_stats(arr_uint8: np.ndarray) -> dict[str, float]:
    """Compute raw intensity statistics on an unsigned 8-bit integer array.

    Args:
        arr_uint8: Input array of uint8 pixels (e.g. 2D single channel or flattened).

    Returns:
        dict containing:
            mean: Mean intensity (float)
            std: Standard deviation of intensity (float)
            p0_5: 0.5th percentile (float)
            p1: 1st percentile (float)
            p50: 50th percentile (median) (float)
            p99: 99th percentile (float)
            p99_5: 99.5th percentile (float)
            frac_zero: Fraction of pixels with value 0 (float)
            frac_255: Fraction of pixels with value 255 (float)
    """
    arr = np.asarray(arr_uint8)
    if arr.size == 0:
        raise ValueError("Cannot compute intensity statistics on an empty array.")

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p0_5": float(np.percentile(arr, 0.5)),
        "p1": float(np.percentile(arr, 1.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "p99_5": float(np.percentile(arr, 99.5)),
        "frac_zero": float(np.mean(arr == 0)),
        "frac_255": float(np.mean(arr == 255)),
    }
