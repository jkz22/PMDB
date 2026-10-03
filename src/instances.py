"""Particle instances.

Graphite: distance transform + peak_local_max + watershed, because
flakes touch each other and connected components would merge them.
Instances touching the image border are kept in the label array but
listed separately so size statistics can exclude them (border particles
are cut off, biasing size low).

Bright phase: connected components are enough (sparse, rarely touching).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def graphite_instances(masks: dict[str, np.ndarray], cfg: dict) -> tuple[np.ndarray, list[int]]:
    """Watershed the graphite mask into labelled particles.

    Returns (labels array, list of labels touching the image border).
    """
    mask = masks["graphite"]
    dist = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(
        dist,
        min_distance=cfg["instances"]["watershed_min_distance"],
        labels=mask,
        exclude_border=False,  # default excludes peaks within min_distance
                               # of the edge, silently dropping edge flakes
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-dist, markers, mask=mask)

    # a component whose only maxima were suppressed by min_distance gets no
    # marker and stays 0; give each such component its own label
    missed = mask & (labels == 0)
    if missed.any():
        extra, n_extra = ndi.label(missed)
        labels = labels + np.where(missed, extra + labels.max(), 0)

    border = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))
    border_labels = [int(b) for b in border if b != 0]
    return labels, border_labels


def bright_instances(masks: dict[str, np.ndarray]) -> np.ndarray:
    labels, _ = ndi.label(masks["bright"])
    return labels
