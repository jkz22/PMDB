"""Diagnostic (graphite, pore) and artefact KPIs: D01-D06, A01-A03."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import euler_number, label, regionprops_table

from pmdb.kpis.common import NAN, KpiContext, KpiOutput, ecd_um
from pmdb.kpis.fields import band_profile, band_summary

A02_MEDIAN_PX = 201


def d01_graphite_depth(ctx: KpiContext) -> KpiOutput:
    g = ctx.masks.graphite
    vals = band_profile(g, np.ones_like(g))
    m, maxdev, _ = band_summary(vals)
    curves = [("band_graphite_frac", float(i), float(v)) for i, v in enumerate(vals)]
    return KpiOutput({"D01_graphite_frac_mean": m, "D01_graphite_band_maxdev": maxdev}, curves)


def _graphite_props(ctx: KpiContext) -> dict[str, np.ndarray]:
    lab = label(ctx.masks.graphite, connectivity=2)
    return regionprops_table(lab, properties=("area", "orientation", "major_axis_length", "minor_axis_length"))


def d02_graphite_orientation(ctx: KpiContext) -> KpiOutput:
    p = _graphite_props(ctx)
    if len(p["area"]) == 0:
        return KpiOutput({"D02_graphite_orient_circsd_deg": NAN})
    w = p["area"].astype(np.float64)
    z = np.sum(w * np.exp(2j * p["orientation"])) / w.sum()
    r = min(abs(z), 1.0)
    sd = np.sqrt(-2.0 * np.log(r)) / 2.0 if r > 0 else np.pi / 2
    return KpiOutput({"D02_graphite_orient_circsd_deg": float(np.degrees(sd))})


def d03_graphite_size(ctx: KpiContext) -> KpiOutput:
    p = _graphite_props(ctx)
    if len(p["area"]) == 0:
        return KpiOutput({"D03_graphite_ecd_d50_um": NAN, "D03_graphite_aspect_median": NAN})
    ecd = ecd_um(p["area"], ctx.px_um)
    ok = p["minor_axis_length"] > 0
    aspect = p["major_axis_length"][ok] / p["minor_axis_length"][ok]
    return KpiOutput({
        "D03_graphite_ecd_d50_um": float(np.median(ecd)),
        "D03_graphite_aspect_median": float(np.median(aspect)) if aspect.size else NAN,
    })


def d04_porosity_depth(ctx: KpiContext) -> KpiOutput:
    vals = band_profile(ctx.masks.pore, ~ctx.masks.artefact)
    m, maxdev, _ = band_summary(vals)
    curves = [("band_pore_frac", float(i), float(v)) for i, v in enumerate(vals)]
    return KpiOutput({"D04_porosity_mean": m, "D04_porosity_band_maxdev": maxdev}, curves)


def d05_pore_size(ctx: KpiContext) -> KpiOutput:
    pore = ctx.masks.pore
    if not pore.any():
        return KpiOutput({"D05_pore_size_d50_um": NAN})
    d = 2.0 * ndimage.distance_transform_edt(pore)[pore] * ctx.px_um
    return KpiOutput({"D05_pore_size_d50_um": float(np.median(d))})


def d06_pore_connectivity(ctx: KpiContext) -> KpiOutput:
    area = float((~ctx.masks.artefact).sum()) * ctx.px_area_um2
    e = euler_number(ctx.masks.pore, connectivity=2)
    return KpiOutput({"D06_pore_euler_per_1000um2": float(e / area * 1000.0) if area > 0 else NAN})


def a01_large_voids(ctx: KpiContext) -> KpiOutput:
    return KpiOutput({"A01_large_void_frac": float(ctx.masks.artefact.mean())})


def a02_curtaining(ctx: KpiContext) -> KpiOutput:
    """SD of detrended BSE column means; BSE is percentile-normalised (D-005), so units are p0.5-p99.5 range."""
    if ctx.bse is None:
        raise ValueError("A02 needs the BSE channel")
    col = np.asarray(ctx.bse, dtype=np.float64).mean(axis=0)
    resid = col - ndimage.median_filter(col, size=A02_MEDIAN_PX, mode="reflect")
    return KpiOutput({"A02_curtaining_index": float(resid.std())})


def a03_height(ctx: KpiContext) -> KpiOutput:
    return KpiOutput({"A03_height_um": float(ctx.shape[0] * ctx.px_um)})
