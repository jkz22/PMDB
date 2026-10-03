"""Shared context and helpers for KPI functions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy import ndimage

from pmdb.segment import Masks

N_NULL_SIMS = 99
MIN_OBJECTS = 10
NAN = float("nan")


def stable_seed(key: str) -> int:
    """Stable 64-bit seed from a string such as ``batch/site/tile`` (D-013)."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


@dataclass
class KpiOutput:
    values: dict[str, float]
    curves: list[tuple[str, float, float]] = field(default_factory=list)


class TooFewObjects(ValueError):
    """Raised when a KPI needs at least MIN_OBJECTS Si objects."""


@dataclass
class KpiContext:
    """Masks + BSE + scale for one image or tile, with cached derived quantities.

    ``bse`` is the percentile-normalised BSE channel (only A02 uses it).
    """

    masks: Masks
    nm_per_px: float
    seed_key: str
    bse: np.ndarray | None = None
    n_null_sims: int = N_NULL_SIMS

    @property
    def px_um(self) -> float:
        return self.nm_per_px / 1000.0

    @property
    def px_area_um2(self) -> float:
        return self.px_um ** 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.masks.si.shape

    @cached_property
    def si_labels(self) -> tuple[np.ndarray, int]:
        lab, n = ndimage.label(self.masks.si, structure=np.ones((3, 3), dtype=bool))
        return lab, int(n)

    @property
    def n_objects(self) -> int:
        return self.si_labels[1]

    @cached_property
    def object_areas_px(self) -> np.ndarray:
        lab, n = self.si_labels
        return np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)

    @cached_property
    def centroids_rc(self) -> np.ndarray:
        """Si object centroids as (row, col) floats in pixel units, shape (n, 2)."""
        lab, n = self.si_labels
        if n == 0:
            return np.zeros((0, 2))
        return np.asarray(ndimage.center_of_mass(self.masks.si, lab, np.arange(1, n + 1)), dtype=np.float64)

    @cached_property
    def admissible_area_um2(self) -> float:
        return float(self.masks.admissible.sum()) * self.px_area_um2

    @cached_property
    def fraction_area_um2(self) -> float:
        """Denominator for Si area fractions and densities: everything except artefact."""
        return float(self.masks.fraction_space.sum()) * self.px_area_um2

    @cached_property
    def image_area_um2(self) -> float:
        return float(self.masks.si.size) * self.px_area_um2

    @cached_property
    def admissible_flat_idx(self) -> np.ndarray:
        return np.flatnonzero(self.masks.admissible.ravel())

    @cached_property
    def null_points_rc(self) -> list[np.ndarray]:
        """Random-labelling simulations: n points uniform over admissible pixels (D-013)."""
        rng = np.random.default_rng(stable_seed(self.seed_key))
        n = self.n_objects
        idx = self.admissible_flat_idx
        w = self.shape[1]
        sims = []
        for _ in range(self.n_null_sims):
            pick = rng.choice(idx, size=n, replace=False)
            sims.append(np.column_stack([pick // w, pick % w]).astype(np.float64))
        return sims

    def require_objects(self, k: int = MIN_OBJECTS) -> None:
        if self.n_objects < k:
            raise TooFewObjects(f"fewer than {k} Si objects (n={self.n_objects})")


def ecd_um(area_px: np.ndarray, px_um: float) -> np.ndarray:
    return 2.0 * np.sqrt(np.asarray(area_px, dtype=np.float64) / np.pi) * px_um


def depth_bands(n_rows: int, n_bands: int = 5) -> list[np.ndarray]:
    return np.array_split(np.arange(n_rows), n_bands)


def safe_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return NAN
    m = values.mean()
    if m <= 0:
        return NAN
    return float(values.std() / m)
