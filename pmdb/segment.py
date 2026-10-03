"""v0 phase segmentation of BSE cross-sections (spec 002, D-010 / D-011).

The KPI code consumes only :class:`Masks` (plus the BSE channel), so this module
can be replaced by a better segmenter without touching ``pmdb.kpis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

SEGMENTER_VERSION = "v0"

V0_PARAMS: dict = {
    "gauss_sigma_px": 1.0,
    "pore_anchor_frac": 0.25,  # T_pore = p1 + frac * (p50 - p1)
    "si_k_mad": 4.0,  # T_si = median_solid + k * MAD_solid
    "si_closing_radius_px": 2,
    "si_min_area_um2": 0.05,
    "graphite_opening_radius_px": 5,
    "graphite_min_area_um2": 4.0,
    "artefact_min_area_um2": 25.0,
}


@dataclass
class Masks:
    """Boolean phase masks for one image (or tile).

    ``si``, ``graphite``, ``pore`` and ``artefact`` are mutually exclusive; pixels in
    none of them are unassigned solid (binder / carbon black / small debris).
    ``admissible = ~graphite & ~artefact`` (D-010).
    """

    si: np.ndarray
    graphite: np.ndarray
    pore: np.ndarray
    artefact: np.ndarray
    admissible: np.ndarray
    version: str = SEGMENTER_VERSION
    params: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.si.shape

    def crop(self, rows: slice, cols: slice) -> "Masks":
        return Masks(
            si=self.si[rows, cols],
            graphite=self.graphite[rows, cols],
            pore=self.pore[rows, cols],
            artefact=self.artefact[rows, cols],
            admissible=self.admissible[rows, cols],
            version=self.version,
            params=self.params,
        )


def _um2_to_px(area_um2: float, nm_per_px: float) -> int:
    px_um = nm_per_px / 1000.0
    return int(np.ceil(area_um2 / (px_um * px_um)))


def segment_bse(bse: np.ndarray, nm_per_px: float, params: dict | None = None) -> Masks:
    """Apply the v0 rule (D-011) to a single un-normalised BSE image."""
    p = dict(V0_PARAMS)
    if params:
        p.update(params)

    g = ndimage.gaussian_filter(np.asarray(bse, dtype=np.float64), sigma=p["gauss_sigma_px"])

    p1, p50 = np.percentile(g, [1.0, 50.0])
    t_pore = p1 + p["pore_anchor_frac"] * (p50 - p1)
    dark = g < t_pore

    solid_vals = g[~dark]
    med = float(np.median(solid_vals))
    mad = float(np.median(np.abs(solid_vals - med)))
    t_si = med + p["si_k_mad"] * mad

    si = (g > t_si) & ~dark
    si = binary_closing(si, disk(p["si_closing_radius_px"]))
    si = ndimage.binary_fill_holes(si)
    si = remove_small_objects(si, min_size=_um2_to_px(p["si_min_area_um2"], nm_per_px))

    rest = ~dark & ~si
    graphite = binary_opening(rest, disk(p["graphite_opening_radius_px"]))
    graphite = remove_small_objects(graphite, min_size=_um2_to_px(p["graphite_min_area_um2"], nm_per_px))
    graphite &= ~si

    pore_all = dark & ~si
    artefact = remove_small_objects(pore_all, min_size=_um2_to_px(p["artefact_min_area_um2"], nm_per_px))
    pore = pore_all & ~artefact

    admissible = ~graphite & ~artefact
    used = dict(p)
    used.update({"T_pore": float(t_pore), "T_si": float(t_si), "median_solid": med, "mad_solid": mad,
                 "p1": float(p1), "p50": float(p50)})
    return Masks(si=si, graphite=graphite, pore=pore, artefact=artefact, admissible=admissible,
                 version=SEGMENTER_VERSION, params=used)


def segment(site) -> Masks:
    """Segment a :class:`pmdb.io.Site` (D-010).

    The site should be loaded with ``normalise="none"`` (D-011); the rule is
    affine-invariant, so only percentile clipping would change the result.
    """
    bse = np.asarray(site.image[..., 0], dtype=np.float64)
    return segment_bse(bse, site.nm_per_px)
