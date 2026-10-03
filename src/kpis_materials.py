"""Materials KPIs (prefix mat_): quantities a battery engineer already
names, measures and specifies on a datasheet.

Interface shared with kpis_geometry (colleague) and kpis_physics:
compute(masks, instances) -> dict[str, float]
masks: dict of boolean arrays (pore, graphite, bright, rim)
instances: labelled int array for graphite; border-touching labels in
instances_border are excluded from size statistics.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

try:
    # orientation is owned by the geometry module; imported, not recomputed
    from src.kpis_geometry import instance_orientations
except ImportError:
    instance_orientations = None


def _equivalent_diameters(instances: np.ndarray, exclude: list[int]) -> np.ndarray:
    """Equivalent-circle diameter per instance, excluding given labels."""
    areas = np.bincount(instances.ravel())
    areas[0] = 0
    areas[exclude] = 0
    areas = areas[areas > 0]
    return np.sqrt(4.0 * areas / np.pi)


def _crack_mask(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Pore components fully enclosed by graphite (intra-particle cracks).

    A pore component counts as a crack when every pixel of it lies inside
    a hole of the graphite mask, i.e. it has no contact with the main
    pore network between particles.
    """
    graphite = masks["graphite"]
    pore = masks["pore"]
    holes = ndi.binary_fill_holes(graphite) & ~graphite
    labels, n = ndi.label(pore)
    if n == 0:
        return np.zeros_like(pore)
    outside = np.bincount(labels.ravel(), weights=(~holes).ravel(), minlength=n + 1)
    enclosed = outside == 0
    enclosed[0] = False
    return enclosed[labels]


def compute(masks: dict[str, np.ndarray], instances: np.ndarray,
            instances_border: list[int]) -> dict[str, float]:
    total = masks["pore"].size
    graphite_px = masks["graphite"].sum()

    out = {
        # The number every QC engineer already measures from cross-section
        # SEM; shifts with calendering pressure and slurry solids content.
        "mat_porosity": masks["pore"].sum() / total,
        # Phase loading of the higher-Z component (SiOx); formulation
        # signature.
        "mat_bright_fraction": masks["bright"].sum() / total,
        # Solid active loading; complements porosity and catches
        # binder/rim changes.
        "mat_active_fraction": (graphite_px + masks["bright"].sum()) / total,
    }

    # Feedstock particle size distribution; first thing that moves when a
    # supplier changes milling or grade.
    diam = _equivalent_diameters(instances, instances_border)
    if diam.size:
        d10, d50, d90 = np.percentile(diam, [10, 50, 90])
        out.update({"mat_graphite_d10": d10, "mat_graphite_d50": d50,
                    "mat_graphite_d90": d90})

    # Intra-particle damage from over-calendering or weak feedstock.
    out["mat_crack_fraction"] = _crack_mask(masks).sum() / graphite_px

    # Binder and carbon-black distribution. Drop if rim segmentation is
    # noisy across batches.
    perimeter = masks["graphite"] & ~ndi.binary_erosion(masks["graphite"])
    rim_adjacent = ndi.binary_dilation(masks["rim"]) & perimeter
    per_px = perimeter.sum()
    if per_px:
        out["mat_rim_coverage"] = rim_adjacent.sum() / per_px

    # Degree of flake alignment produced by calendering; 1 = fully
    # in-plane, 0 = isotropic. Needs the geometry module's orientations.
    if instance_orientations is not None:
        theta, areas = instance_orientations(instances, instances_border)
        if len(theta):
            out["mat_orientation_anisotropy"] = float(
                np.average(np.abs(np.cos(2 * np.asarray(theta))),
                           weights=np.asarray(areas)))

    return {k: float(v) for k, v in out.items()}
