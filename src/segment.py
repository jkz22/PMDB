"""Phase segmentation: pore, graphite, bright, rim.

Multi-Otsu (4 classes) on a Gaussian-smoothed copy of the normalised
image. The two highest intensity classes (rim network and bright
particles) overlap in intensity, so they are separated by morphology:
an opening with a small disk removes thin rim structures; what survives
is the bright particle phase.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_multiotsu
from skimage.morphology import disk, opening


def _remove_small(mask: np.ndarray, max_px: int) -> np.ndarray:
    """Remove connected components with fewer than max_px pixels."""
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max_px
    keep[0] = False
    return keep[labels]


def segment(norm_img: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Segment a normalised uint8 image into boolean phase masks.

    Returns dict with keys pore, graphite, bright, rim. Masks are
    mutually exclusive and cover the full image.
    """
    scfg = cfg["segment"]
    smoothed = gaussian(norm_img, sigma=scfg["gaussian_sigma"], preserve_range=True)
    thresholds = threshold_multiotsu(smoothed, classes=scfg["n_classes"])
    classes = np.digitize(smoothed, thresholds)  # 0=darkest .. 3=brightest

    # graphite spans a wide intensity range, so with 4 Otsu classes it
    # occupies classes 1 AND 2; only the top class is bright+rim.
    # Mapping checked by eye on qc/segmentation overlays.
    pore = classes == 0
    graphite = (classes == 1) | (classes == 2)
    high = classes == 3  # rim + bright, overlapping in intensity

    # bright = thick structures in the high class; rim = thin remainder
    bright = opening(high, disk(scfg["rim_opening_radius"]))
    rim = high & ~bright

    # noise removal: drop tiny pore specks, fill small holes in graphite
    pore = _remove_small(pore, scfg["min_pore_px"])
    holes = ndi.binary_fill_holes(graphite) & ~graphite
    graphite = graphite | _remove_small_inverse(holes, scfg["fill_graphite_px"])

    # keep masks exclusive and exhaustive: pixels dropped from pore by
    # the speck filter join graphite (the surrounding majority phase)
    unassigned = ~(pore | graphite | bright | rim)
    graphite = graphite | unassigned

    return {"pore": pore, "graphite": graphite & ~bright & ~rim,
            "bright": bright, "rim": rim}


def _remove_small_inverse(mask: np.ndarray, max_px: int) -> np.ndarray:
    """Keep only components smaller than max_px (holes worth filling)."""
    labels, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes < max_px
    keep[0] = False
    return keep[labels]


# fixed colours for QC overlays: pore black, graphite grey,
# bright yellow, rim red
OVERLAY_COLOURS = {
    "pore": (0, 0, 0),
    "graphite": (110, 110, 110),
    "bright": (255, 220, 60),
    "rim": (220, 50, 50),
}


def overlay(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Render masks as an RGB uint8 image for eyeballing."""
    h, w = next(iter(masks.values())).shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for name, colour in OVERLAY_COLOURS.items():
        out[masks[name]] = colour
    return out
