"""Si relative to graphite: K15, K16."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from pmdb.kpis.common import NAN, KpiContext, KpiOutput


def _four_neighbour_any(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, mode="edge")
    return p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def si_graphite_contact(si: np.ndarray, graphite: np.ndarray) -> float:
    boundary = si & _four_neighbour_any(~si)
    nb = boundary.sum()
    if nb == 0:
        return NAN
    return float((boundary & _four_neighbour_any(graphite)).sum() / nb)


def si_graphite_distance(si_labels: np.ndarray, n: int, graphite: np.ndarray, px_um: float) -> float:
    """Median over objects of the gap (edge to edge) to the nearest graphite pixel, in um."""
    if n == 0 or not graphite.any():
        return NAN
    d = ndimage.distance_transform_edt(~graphite)
    mins = np.asarray(ndimage.minimum(d, si_labels, np.arange(1, n + 1)), dtype=np.float64)
    gaps = np.maximum(mins - 1.0, 0.0) * px_um
    return float(np.median(gaps))


def k15_contact(ctx: KpiContext) -> KpiOutput:
    return KpiOutput({"K15_si_graphite_contact_frac": si_graphite_contact(ctx.masks.si, ctx.masks.graphite)})


def k16_distance(ctx: KpiContext) -> KpiOutput:
    ctx.require_objects()
    lab, n = ctx.si_labels
    return KpiOutput({"K16_si_graphite_dist_median_um": si_graphite_distance(lab, n, ctx.masks.graphite, ctx.px_um)})
