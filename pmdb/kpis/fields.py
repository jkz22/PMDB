"""Field-level localisation KPIs K10-K14 (D-017).

K10-K13 local Si fractions use ``Masks.fraction_space`` (non-artefact pixels) as denominator
and window filter instead of admissible pixels (deviation, see docs/kpis/kpi_status.md).
K14 still measures distances from admissible pixels.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from pmdb.kpis.common import NAN, KpiContext, KpiOutput, depth_bands, safe_cv

K10_WINDOWS_UM = (1.0, 2.0, 5.0, 10.0, 20.0)
K10_MIN_ADM_FRAC = 0.5
K11_WINDOW_UM = 10.0
N_DEPTH_BANDS = 5
K13_BIN_UM = 10.0


def window_counts(si: np.ndarray, adm: np.ndarray, wp: int) -> tuple[np.ndarray, np.ndarray]:
    """Si and admissible pixel counts in non-overlapping wp x wp windows (partial windows dropped)."""
    ny, nx = si.shape[0] // wp, si.shape[1] // wp
    if ny == 0 or nx == 0:
        return np.zeros(0), np.zeros(0)
    s = si[: ny * wp, : nx * wp].reshape(ny, wp, nx, wp).sum(axis=(1, 3), dtype=np.int64)
    a = adm[: ny * wp, : nx * wp].reshape(ny, wp, nx, wp).sum(axis=(1, 3), dtype=np.int64)
    keep = a >= K10_MIN_ADM_FRAC * wp * wp
    return s[keep].astype(np.float64), a[keep].astype(np.float64)


def _wpx(w_um: float, px_um: float) -> int:
    return max(1, int(round(w_um / px_um)))


def k10_scale_of_segregation(ctx: KpiContext) -> KpiOutput:
    si, adm = ctx.masks.si, ctx.masks.fraction_space
    ws, cvs = [], []
    curves = []
    for w in K10_WINDOWS_UM:
        s, a = window_counts(si, adm, _wpx(w, ctx.px_um))
        cv = safe_cv(s / a) if a.size else NAN
        curves.append(("cv", w, cv))
        if np.isfinite(cv) and cv > 0:
            ws.append(w)
            cvs.append(cv)
    cv10 = dict((c[1], c[2]) for c in curves)[10.0]
    slope = float(np.polyfit(np.log(ws), np.log(cvs), 1)[0]) if len(ws) >= 2 else NAN
    return KpiOutput({"K10_cv_w10": float(cv10), "K10_cv_slope": slope}, curves)


def k11_lacey(ctx: KpiContext) -> KpiOutput:
    """Lacey M at w = 10 um with N = expected Si objects per window (catalogue definition)."""
    s, a = window_counts(ctx.masks.si, ctx.masks.fraction_space, _wpx(K11_WINDOW_UM, ctx.px_um))
    if s.size < 2 or a.sum() == 0:
        return KpiOutput({"K11_lacey_w10": NAN})
    p = s.sum() / a.sum()
    s2 = float(np.var(s / a, ddof=1))
    s0 = p * (1.0 - p)
    total_adm = float(ctx.masks.fraction_space.sum())
    n_exp = ctx.n_objects / total_adm * a.mean()
    if n_exp <= 0:
        return KpiOutput({"K11_lacey_w10": NAN})
    sr = s0 / n_exp
    denom = s0 - sr
    val = (s0 - s2) / denom if denom != 0 else NAN
    return KpiOutput({"K11_lacey_w10": float(val)})


def band_profile(num: np.ndarray, den: np.ndarray, n_bands: int = N_DEPTH_BANDS) -> np.ndarray:
    vals = []
    for rows in depth_bands(num.shape[0], n_bands):
        d = den[rows].sum()
        vals.append(num[rows].sum() / d if d > 0 else np.nan)
    return np.asarray(vals, dtype=np.float64)


def band_summary(vals: np.ndarray) -> tuple[float, float, float]:
    """(mean, max |band - mean| / mean, |slope over normalised depth| / mean)."""
    m = float(np.nanmean(vals))
    if not np.isfinite(m) or m <= 0:
        return m, NAN, NAN
    maxdev = float(np.nanmax(np.abs(vals - m)) / m)
    centres = (np.arange(len(vals)) + 0.5) / len(vals)
    ok = np.isfinite(vals)
    slope = float(np.polyfit(centres[ok], vals[ok], 1)[0]) if ok.sum() >= 2 else NAN
    return m, maxdev, abs(slope) / m


def k12_depth_profile(ctx: KpiContext) -> KpiOutput:
    vals = band_profile(ctx.masks.si, ctx.masks.fraction_space)
    _, maxdev, absslope = band_summary(vals)
    curves = [("band_si_frac", float(i), float(v)) for i, v in enumerate(vals)]
    return KpiOutput({"K12_depth_maxdev": maxdev, "K12_depth_absslope": absslope}, curves)


def k13_lateral(ctx: KpiContext) -> KpiOutput:
    wp = _wpx(K13_BIN_UM, ctx.px_um)
    nb = ctx.shape[1] // wp
    si = ctx.masks.si[:, : nb * wp].reshape(ctx.shape[0], nb, wp).sum(axis=(0, 2))
    ad = ctx.masks.fraction_space[:, : nb * wp].reshape(ctx.shape[0], nb, wp).sum(axis=(0, 2))
    ok = ad > 0
    frac = si[ok] / ad[ok]
    centres = (np.arange(nb) + 0.5) * K13_BIN_UM
    curves = [("column_si_frac", float(x), float(v)) for x, v in zip(centres[ok], frac)]
    return KpiOutput({"K13_lateral_cv": safe_cv(frac)}, curves)


def k14_empty_space(ctx: KpiContext) -> KpiOutput:
    si = ctx.masks.si
    if not si.any():
        return KpiOutput({"K14_empty_p50_um": NAN, "K14_empty_p95_um": NAN})
    d = ndimage.distance_transform_edt(~si)[ctx.masks.admissible] * ctx.px_um
    p50, p95 = np.percentile(d, [50, 95])
    return KpiOutput({"K14_empty_p50_um": float(p50), "K14_empty_p95_um": float(p95)})
