"""Stretch KPIs S01-S04 (catalogue v2): cluster connectivity and crowding at every scale.

All four read the Si mask only and reuse ``KpiContext`` so they sit on the same masks as v1.
They are *not* part of ``compute_site_kpis`` (v1 is logged and locked); ``scripts/run_stretch.py``
produces them into ``outputs/stretch/``.

S01  two-point cluster function C2(r) along x and z: probability that two pixels r apart lie in
     the same 8-connected Si object. Reported as the correlation length  L = sum_r C2(r)/C2(0) dr
     (equals the decay length for an exponential, half the object length for a bar), per direction.
S02  Euler dilation merge radius: dilate the Si mask by eps; the radius at which the Euler number
     (objects - holes, 8-connected) first falls to half its undilated value. Smaller = objects closer.
S03  persistent homology H0 of the signed Euclidean distance filtration (negative inside Si).
     A component is born at -inradius and dies when it merges into an older one (elder rule);
     lifetime = death + inradius. Many short lifetimes = crowded, similar-sized objects.
S04  Minkowski tensor W1^{0,2} of the Si boundary (sum of n n^T over boundary length):
     beta = lambda_min / lambda_max (1 isotropic, -> 0 aligned) and the streak direction
     (0 deg along x / in-plane, 90 deg along z / through-thickness).

References: Torquato, Random Heterogeneous Materials (C2 and lineal path); Lu & Torquato 1992
(10.1103/physreva.45.922); Schroeder-Turk et al. 2011 (10.1002/adma.201100562, Minkowski tensors);
Edelsbrunner & Harer, Computational Topology (elder rule).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import euler_number

from pmdb.kpis.common import NAN, KpiContext, KpiOutput

S01_R_MAX_UM = 10.0
S02_EPS_MAX_UM = 5.0
S03_T_MAX_UM = 5.0
S04_SIGMA_PX = 1.0

_EIGHT = np.ones((3, 3), dtype=bool)


# ------------------------------------------------------------------ S01 two-point cluster function

def two_point_cluster(labels: np.ndarray, domain: np.ndarray, axis: int, r_max_px: int) -> np.ndarray:
    """C2(r), r = 0..r_max_px, along ``axis`` (0 = z/rows, 1 = x/cols).

    Pairs are counted only when both pixels are in ``domain`` (non-artefact); C2(0) = Si fraction.
    """
    lab = np.asarray(labels)
    dom = np.asarray(domain, dtype=bool)
    n = lab.shape[axis]
    out = np.full(r_max_px + 1, NAN)
    for r in range(min(r_max_px, n - 1) + 1):
        a = np.take(lab, np.arange(0, n - r), axis=axis)
        b = np.take(lab, np.arange(r, n), axis=axis)
        da = np.take(dom, np.arange(0, n - r), axis=axis)
        db = np.take(dom, np.arange(r, n), axis=axis)
        both = da & db
        n_pairs = int(both.sum())
        if n_pairs == 0:
            break
        same = (a == b) & (a > 0) & both
        out[r] = same.sum() / n_pairs
    return out


def c2_length_px(c2: np.ndarray) -> float:
    """Correlation length: integral of C2(r)/C2(0) over r (trapezoid, NaN tail dropped)."""
    c2 = np.asarray(c2, dtype=np.float64)
    ok = np.isfinite(c2)
    if not ok.any() or c2[0] <= 0:
        return NAN
    y = c2[ok] / c2[0]
    return float(np.trapz(y, dx=1.0))


def s01_two_point_cluster(ctx: KpiContext) -> KpiOutput:
    ctx.require_objects()
    lab, _ = ctx.si_labels
    r_max = int(round(S01_R_MAX_UM / ctx.px_um))
    values: dict[str, float] = {}
    curves: list[tuple[str, float, float]] = []
    for name, axis in (("x", 1), ("z", 0)):
        c2 = two_point_cluster(lab, ctx.masks.fraction_space, axis, r_max)
        values[f"S01_c2_length_{name}_um"] = c2_length_px(c2) * ctx.px_um
        curves.extend((f"c2_{name}", r * ctx.px_um, float(v)) for r, v in enumerate(c2) if np.isfinite(v))
    lx, lz = values["S01_c2_length_x_um"], values["S01_c2_length_z_um"]
    values["S01_c2_anisotropy"] = lx / lz if np.isfinite(lx) and np.isfinite(lz) and lz > 0 else NAN
    return KpiOutput(values, curves)


# ------------------------------------------------------------------ S02 Euler dilation merge radius

def euler_vs_dilation(si: np.ndarray, eps_px: np.ndarray) -> np.ndarray:
    dist = ndimage.distance_transform_edt(~si)
    return np.array([euler_number(si | (dist <= e), connectivity=2) for e in eps_px], dtype=np.float64)


def half_crossing(x: np.ndarray, y: np.ndarray) -> float:
    """First x at which y falls to y[0]/2 (linear interpolation), NaN if it never does or y[0] <= 0."""
    if y.size == 0 or y[0] <= 0:
        return NAN
    target = y[0] / 2.0
    below = np.flatnonzero(y <= target)
    if below.size == 0:
        return NAN
    i = int(below[0])
    if i == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return float(x0 + (y0 - target) / (y0 - y1) * (x1 - x0)) if y0 != y1 else float(x1)


def s02_euler_merge_radius(ctx: KpiContext) -> KpiOutput:
    ctx.require_objects()
    eps = np.arange(0, int(round(S02_EPS_MAX_UM / ctx.px_um)) + 1, dtype=np.float64)
    chi = euler_vs_dilation(ctx.masks.si, eps)
    r = half_crossing(eps, chi)
    curves = [("euler", float(e * ctx.px_um), float(c)) for e, c in zip(eps, chi)]
    return KpiOutput({"S02_euler_merge_radius_um": r * ctx.px_um if np.isfinite(r) else NAN,
                      "S02_euler_0": float(chi[0])}, curves)


# ------------------------------------------------------------------ S03 persistent homology H0

def h0_lifetimes(si: np.ndarray, t_max_px: float, step_px: float = 1.0) -> tuple[np.ndarray, np.ndarray, int]:
    """H0 (birth, death) pairs of the signed-distance sublevel filtration of ``si``.

    Births are exact (-inradius of each object); deaths are quantised to ``step_px``. Components
    still alive at ``t_max_px`` are returned with death = t_max_px and counted in the third output.
    """
    si = np.asarray(si, dtype=bool)
    if not si.any():
        return np.zeros(0), np.zeros(0), 0
    signed = ndimage.distance_transform_edt(~si) - ndimage.distance_transform_edt(si)
    lab0, n0 = ndimage.label(si, structure=_EIGHT)
    births = np.asarray(ndimage.minimum(signed, lab0, np.arange(1, n0 + 1)), dtype=np.float64)
    deaths = np.full(n0, np.inf)
    rep = ndimage.minimum_position(signed, lab0, np.arange(1, n0 + 1))
    rep_r = np.array([p[0] for p in rep], dtype=int)
    rep_c = np.array([p[1] for p in rep], dtype=int)
    alive = np.ones(n0, dtype=bool)
    for t in np.arange(step_px, t_max_px + 0.5 * step_px, step_px):
        lab, _ = ndimage.label(signed <= t, structure=_EIGHT)
        idx = np.flatnonzero(alive)
        owner = lab[rep_r[idx], rep_c[idx]]
        order = np.lexsort((births[idx], owner))
        o_sorted, i_sorted = owner[order], idx[order]
        first = np.r_[True, o_sorted[1:] != o_sorted[:-1]]
        dying = i_sorted[~first]
        deaths[dying] = t
        alive[dying] = False
        if alive.sum() <= 1:
            break
    n_censored = int(np.isinf(deaths).sum())
    deaths[np.isinf(deaths)] = t_max_px
    return births, deaths, n_censored


def s03_persistence(ctx: KpiContext) -> KpiOutput:
    ctx.require_objects()
    t_max = S03_T_MAX_UM / ctx.px_um
    births, deaths, n_cens = h0_lifetimes(ctx.masks.si, t_max)
    life = (deaths - births) * ctx.px_um
    q25, q50, q75 = np.percentile(life, [25, 50, 75])
    return KpiOutput({
        "S03_h0_life_p50_um": float(q50),
        "S03_h0_life_iqr_um": float(q75 - q25),
        "S03_h0_inradius_p50_um": float(np.median(-births) * ctx.px_um),
        "S03_h0_censored_frac": n_cens / life.size,
    })


# ------------------------------------------------------------------ S04 Minkowski tensor anisotropy

def minkowski_w102(mask: np.ndarray, sigma_px: float = S04_SIGMA_PX) -> np.ndarray:
    """2x2 tensor sum_boundary n n^T dl, from the gradient of the smoothed indicator (rows=z, cols=x)."""
    f = ndimage.gaussian_filter(np.asarray(mask, dtype=np.float64), sigma_px)
    gz, gx = np.gradient(f)
    mag = np.hypot(gz, gx)
    keep = mag > 1e-6
    gz, gx, mag = gz[keep], gx[keep], mag[keep]
    return np.array([[np.sum(gx * gx / mag), np.sum(gx * gz / mag)],
                     [np.sum(gx * gz / mag), np.sum(gz * gz / mag)]])


def tensor_anisotropy(t: np.ndarray) -> tuple[float, float]:
    """(beta = lambda_min/lambda_max, streak angle in degrees from x toward z, in [0, 180))."""
    if not np.all(np.isfinite(t)) or np.trace(t) <= 0:
        return NAN, NAN
    w, v = np.linalg.eigh(t)
    beta = float(w[0] / w[1]) if w[1] > 0 else NAN
    normal = v[:, 1]  # dominant boundary normal (x, z); streaks run perpendicular to it
    angle = (np.degrees(np.arctan2(normal[0], -normal[1]))) % 180.0
    return beta, float(angle)


def s04_minkowski(ctx: KpiContext) -> KpiOutput:
    ctx.require_objects()
    beta, angle = tensor_anisotropy(minkowski_w102(ctx.masks.si))
    return KpiOutput({"S04_si_beta": beta, "S04_si_streak_angle_deg": angle})
