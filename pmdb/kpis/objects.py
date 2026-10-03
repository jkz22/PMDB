"""Object-level Si KPIs: K01-K04."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from pmdb.kpis.common import NAN, KpiContext, KpiOutput, ecd_um

D_UM = 0.5
D_STAR_UM = 5.0
SWEEP_D_UM = (0.25, 0.5, 1.0)
SWEEP_D_STAR_UM = (3.0, 5.0, 8.0)


def k01_si_fraction(ctx: KpiContext) -> KpiOutput:
    den = ctx.masks.fraction_space.sum()
    val = float(ctx.masks.si.sum() / den) if den else NAN
    return KpiOutput({"K01_si_frac_adm": val})


def k02_si_density(ctx: KpiContext) -> KpiOutput:
    a = ctx.fraction_area_um2
    val = ctx.n_objects / a * 1000.0 if a > 0 else NAN
    return KpiOutput({"K02_si_density_per_1000um2": float(val)})


def k03_si_size(ctx: KpiContext) -> KpiOutput:
    ctx.require_objects()
    ecd = ecd_um(ctx.object_areas_px, ctx.px_um)
    return KpiOutput({
        "K03_ecd_d50_um": float(np.percentile(ecd, 50)),
        "K03_ecd_d90_um": float(np.percentile(ecd, 90)),
        "K03_ecd_max_um": float(ecd.max()),
    })


def agglomerate_stats(si: np.ndarray, px_um: float, area_um2: float,
                      d_um: float = D_UM, d_star_um: float = D_STAR_UM) -> dict[str, float]:
    """K04 (D-015): dilate Si by d/2; components are clusters; size = ECD of Si area inside."""
    if not si.any():
        return {"K04_agglom_frac": NAN, "K04_n_clusters_per_1000um2": NAN}
    radius_px = (d_um / 2.0) / px_um
    dist = ndimage.distance_transform_edt(~si)
    dilated = dist <= radius_px
    lab, n = ndimage.label(dilated, structure=np.ones((3, 3), dtype=bool))
    si_area = np.bincount(lab[si], minlength=n + 1)[1:].astype(np.float64)
    cluster_ecd = ecd_um(si_area, px_um)
    frac = float(si_area[cluster_ecd > d_star_um].sum() / si_area.sum())
    dens = n / area_um2 * 1000.0 if area_um2 > 0 else NAN
    return {"K04_agglom_frac": frac, "K04_n_clusters_per_1000um2": float(dens)}


def k04_agglomerates(ctx: KpiContext, d_um: float = D_UM, d_star_um: float = D_STAR_UM) -> KpiOutput:
    return KpiOutput(agglomerate_stats(ctx.masks.si, ctx.px_um, ctx.admissible_area_um2, d_um, d_star_um))


def k04_sweep(ctx: KpiContext) -> list[dict[str, float]]:
    rows = []
    for d in SWEEP_D_UM:
        for ds in SWEEP_D_STAR_UM:
            vals = agglomerate_stats(ctx.masks.si, ctx.px_um, ctx.admissible_area_um2, d, ds)
            rows.append({"d_um": d, "d_star_um": ds, **vals})
    return rows
