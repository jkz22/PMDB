"""Image loading and per-image normalisation.

Data layout: <data_root>/<batch_name>/img_<id>_<detector>.tif
Each micrograph exists in three detector variants (BSE, ETD, Inlens);
the pipeline analyses only the configured detector (BSE).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from pmdb.io import read_detector_image

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def list_batches(cfg: dict) -> dict[str, list[Path]]:
    """Map batch name -> sorted list of image paths for the configured detector."""
    data_root = ROOT / cfg["data"]["root"]
    detector = cfg["data"]["detector"]
    batches = {}
    for batch_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        paths = sorted(batch_dir.glob(f"*_{detector}.tif"))
        if paths:
            batches[batch_dir.name] = paths
    return batches


def load(path: Path | str) -> np.ndarray:
    """Load one image as a 2D uint8 array.

    Delegates to pmdb.io.read_detector_image, which crops the 4-px
    microscope border artefact columns (AGENTS.md / D-003 rev 2) and
    asserts R == G == B before keeping one channel.
    """
    return read_detector_image(path)


def normalise(img: np.ndarray, p_low: float, p_high: float) -> tuple[np.ndarray, float, float]:
    """Clip to [p_low, p_high] percentiles and rescale to 0..255.

    Removes session-to-session brightness drift so segmentation thresholds
    transfer between images. Returns (normalised uint8 image, lo, hi) where
    lo/hi are the raw percentile values, kept as per-image diagnostics.
    """
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        raise ValueError(f"degenerate percentiles: lo={lo}, hi={hi}")
    out = (np.clip(img, lo, hi) - lo) / (hi - lo) * 255.0
    return out.astype(np.uint8), float(lo), float(hi)
