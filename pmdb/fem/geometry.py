"""Label maps, coarsening and cropping for the FEM mesh (P2-P4)."""

from __future__ import annotations

import numpy as np

from pmdb.fem.materials import ARTEFACT, BINDER, GRAPHITE, PORE, SI


def labels_from_masks(m) -> np.ndarray:
    """uint8 label map: BINDER default, then SI, GRAPHITE, PORE, ARTEFACT (later wins)."""
    lab = np.full(m.si.shape, BINDER, dtype=np.uint8)
    lab[m.si] = SI
    lab[m.graphite] = GRAPHITE
    lab[m.pore] = PORE
    lab[m.artefact] = ARTEFACT
    return lab


def coarsen_labels(labels: np.ndarray, factor: int = 2,
                   priority: tuple[int, ...] = (SI, PORE, GRAPHITE, BINDER, ARTEFACT)) -> np.ndarray:
    """Majority vote over factor x factor blocks; ties broken by ``priority`` (first wins)."""
    if factor == 1:
        return labels
    h, w = labels.shape
    hh, ww = (h // factor) * factor, (w // factor) * factor
    blocks = labels[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(hh // factor, ww // factor, factor * factor)
    n = len(priority)
    score = np.empty((hh // factor, ww // factor, n), dtype=np.int64)
    for rank, lab in enumerate(priority):
        score[..., rank] = (blocks == lab).sum(axis=-1) * 10 + (n - 1 - rank)
    best = np.argmax(score, axis=-1)
    return np.asarray(priority, dtype=np.uint8)[best]


def coarsen_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Block mean (float32) with the same trailing drop as ``coarsen_labels``."""
    if factor == 1:
        return img
    h, w = img.shape
    hh, ww = (h // factor) * factor, (w // factor) * factor
    b = img[:hh, :ww].astype(np.float64).reshape(hh // factor, factor, ww // factor, factor)
    return b.mean(axis=(1, 3)).astype(np.float32)


def central_cols(width_px: int, crop_um: float, nm_per_px: float) -> slice:
    n = 2 * int(np.floor(crop_um * 1000.0 / nm_per_px / 2.0))
    c0 = 2 * ((width_px - n) // 4)
    return slice(c0, c0 + n)
