"""Physics-based artefact patching and normalisation of the raw SEM detector images.

Companion to :mod:`pmdb.harmonise` (per-site grey-level LUTs).  This module works on the
full-resolution TIFFs and follows the "patch, then normalise on physical anchors" brief:

* **Patch** – mask (never fabricate) non-electrode content and imaging artefacts: border and
  colour-marker columns, copper current collector, coating free surface, charging; correct
  only what has a verifiable physical model (scan-line gain bands, smooth shading, detector
  offset and gain).
* **Normalise** – per image and detector, ``z = (I - D) / G(x, y)`` with the dark level ``D``
  estimated from deep pore interiors (left-censored Gaussian when pores clip at 0) and the
  graphite level ``G`` a smooth flat-field surface.  Pores ≈ 0, graphite = 1; the Si/graphite
  ratio is *not* forced to a reference.
* **Harmonise down** – blur sharper images to a common edge width and add calibrated noise up
  to a common Poisson–Gaussian noise level.  Never sharpen or denoise.

Every function is small and pure; the per-site driver is :func:`clean_site` and the CLI is
``scripts/build_clean.py``.  All randomness is seeded (:data:`SEED`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage, optimize, special
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, disk, remove_small_objects

SEED = 20261003
CODE_VERSION = "clean-v1"
DETECTORS: tuple[str, str, str] = ("BSE", "Inlens", "SE_type")
FULL_NM_PER_PX = 25.0
PX_PER_UM = 1000.0 / FULL_NM_PER_PX  # 40 px = 1 µm
Z_SCALE = 10_000  # uint16 fixed point for z

# ----------------------------------------------------------------------------------------------
# Mask bits (one uint16 mask per detector)
# ----------------------------------------------------------------------------------------------
BIT_BORDER = 1 << 0  # border or colour-marker column
BIT_COLLECTOR = 1 << 1  # copper current collector (+ delamination gap)
BIT_FREE_SURFACE = 1 << 2  # coating free surface
BIT_BAND_CORRECTED = 1 << 3  # scan band, corrected (informational)
BIT_BAND_BAD = 1 << 4  # scan band, uncorrectable
BIT_CHARGE_LOCAL = 1 << 5  # local charging (Inlens, SE_type)
BIT_CHARGE_BROAD = 1 << 6  # broad charging, Inlens unreliable
BIT_CLIP_HIGH = 1 << 7  # clipped at 255 (statistics only)
BIT_CLIP_LOW = 1 << 8  # clipped at 0 (statistics only)
BIT_CRACK = 1 << 9  # crack network (material, flag only)
BIT_PORE_BAND = 1 << 10  # pore-rich band (material, flag only)

BIT_NAMES = {
    BIT_BORDER: "border", BIT_COLLECTOR: "collector", BIT_FREE_SURFACE: "free_surface",
    BIT_BAND_CORRECTED: "band_corrected", BIT_BAND_BAD: "band_bad", BIT_CHARGE_LOCAL: "charge_local",
    BIT_CHARGE_BROAD: "charge_broad", BIT_CLIP_HIGH: "clip_high", BIT_CLIP_LOW: "clip_low",
    BIT_CRACK: "crack", BIT_PORE_BAND: "pore_band",
}
INVALID_KPI = BIT_BORDER | BIT_COLLECTOR | BIT_FREE_SURFACE | BIT_BAND_BAD | BIT_CHARGE_LOCAL | BIT_CHARGE_BROAD
INVALID_STATS = INVALID_KPI | BIT_CLIP_HIGH | BIT_CLIP_LOW

BORDER_PX = 8
DILATE_HALF_UM_PX = 20  # 0.5 µm


def valid_for_kpis(mask: np.ndarray) -> np.ndarray:
    return (mask & INVALID_KPI) == 0


def valid_for_stats(mask: np.ndarray) -> np.ndarray:
    return (mask & INVALID_STATS) == 0


# ----------------------------------------------------------------------------------------------
# Reading and sanitising (3.1)
# ----------------------------------------------------------------------------------------------
def read_rgb_tiff(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (R channel uint8, boolean colour-marker columns) for an 8-bit RGB TIFF."""
    arr = tifffile.imread(path)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB in {path}, got {arr.shape}")
    marker_cols = ((arr[..., 0] != arr[..., 1]) | (arr[..., 0] != arr[..., 2])).any(axis=0)
    return arr[..., 0], marker_cols


def sanitise(shape: tuple[int, int], marker_cols: np.ndarray | None = None, border_px: int = BORDER_PX) -> np.ndarray:
    """Border + colour-marker mask (bit 0) in original pixel coordinates."""
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint16)
    m[:border_px, :] |= BIT_BORDER
    m[h - border_px:, :] |= BIT_BORDER
    m[:, :border_px] |= BIT_BORDER
    m[:, w - border_px:] |= BIT_BORDER
    if marker_cols is not None and marker_cols.any():
        m[:, marker_cols] |= BIT_BORDER
    return m


def grey_step(img_uint8: np.ndarray, min_frac: float = 1e-4) -> int:
    """Grey-level step actually used by the image (1 for a native 8-bit image).

    Several sites were re-quantised after capture (e.g. only every 3rd DN is populated); the
    step is the median spacing of the populated levels (those holding ≥ ``min_frac`` of pixels).
    """
    sub = img_uint8[::4, ::4]
    lv, cnt = np.unique(sub, return_counts=True)
    used = lv[cnt >= min_frac * sub.size]
    if used.size < 3:
        return 1
    return int(max(1, np.median(np.diff(used.astype(int)))))


def dequantise(img_uint8: np.ndarray, rng: np.random.Generator, step: int = 1) -> np.ndarray:
    """Add U(-step/2, step/2) to unclipped pixels (4.1); clipped pixels (0, 255) stay put.

    ``step`` is the populated grey-level spacing (see :func:`grey_step`), so re-quantised images
    are spread over their true quantisation bin rather than ±0.5 DN around a sparse level.
    """
    x = img_uint8.astype(np.float32)
    half = 0.5 * float(step)
    noise = rng.uniform(-half, half, size=x.shape).astype(np.float32)
    unclipped = (img_uint8 > 0) & (img_uint8 < 255)
    x[unclipped] += noise[unclipped]
    return x


# ----------------------------------------------------------------------------------------------
# Provisional 3-class split on BSE (pores / graphite / Si)
# ----------------------------------------------------------------------------------------------
@dataclass
class PhaseMasks:
    pore: np.ndarray
    graphite: np.ndarray
    si: np.ndarray
    thresholds: tuple[float, float]


def provisional_phases(bse: np.ndarray, valid: np.ndarray | None = None, sigma: float = 1.0,
                       erode_px: int = 4) -> PhaseMasks:
    """Multi-Otsu 3-class split of a (smoothed) BSE image, each class eroded by ``erode_px``.

    Affine-invariant, so it works on raw DN as well as on normalised ``z``.
    """
    g = ndimage.gaussian_filter(np.asarray(bse, dtype=np.float32), sigma)
    sample = g if valid is None else g[valid]
    sample = sample[::max(1, sample.size // 2_000_000)]
    t1, t2 = threshold_multiotsu(sample, classes=3, nbins=256)
    pore = g < t1
    si = g > t2
    graphite = ~pore & ~si
    if valid is not None:
        pore &= valid
        si &= valid
        graphite &= valid
    se = disk(erode_px)
    return PhaseMasks(binary_erosion(pore, se), binary_erosion(graphite, se), binary_erosion(si, se), (float(t1), float(t2)))


# ----------------------------------------------------------------------------------------------
# Dark level D (4.2)
# ----------------------------------------------------------------------------------------------
@dataclass
class DarkLevel:
    value: float
    stderr: float
    n: int
    method: str  # 'gauss' (no clipping) | 'tobit' | 'tobit-weak' | 'fallback'
    frac_zero: float
    mode: float = float("nan")  # robust mode of the deep-pore pixels (diagnostic)
    median: float = float("nan")


def _robust_mode(x: np.ndarray) -> float:
    """Half-sample mode (Bickel 2002): robust to skew and to a clipped left tail."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    while x.size > 3:
        h = (x.size + 1) // 2
        widths = x[h - 1:] - x[: x.size - h + 1]
        i = int(np.argmin(widths))
        x = x[i: i + h]
    return float(np.mean(x))


def _tobit_fit(x: np.ndarray, censored: np.ndarray, censor: float = 0.5) -> tuple[float, float, float]:
    """MLE of a Gaussian left-censored at ``censor`` for the pixels flagged ``censored`` (raw 0).

    Returns (mu, sigma, se_mu).  Observed values are the dequantised pixels with raw > 0.
    """
    x = np.asarray(x, dtype=np.float64)
    cens = np.asarray(censored, dtype=bool)
    obs = x[~cens]
    n_c = int(cens.sum())

    def nll(theta):
        mu, log_s = theta
        s = np.exp(log_s)
        ll_obs = -0.5 * ((obs - mu) / s) ** 2 - log_s - 0.5 * np.log(2 * np.pi)
        ll_c = n_c * special.log_ndtr((censor - mu) / s) if n_c else 0.0
        return -(ll_obs.sum() + ll_c)

    mu0 = float(np.median(obs)) if obs.size else censor
    s_ref = 1.4826 * float(np.median(np.abs(obs - mu0))) if obs.size > 1 else 1.0
    s_ref = max(s_ref, 1e-6)
    res = optimize.minimize(nll, x0=[mu0, np.log(s_ref)], method="L-BFGS-B",
                            bounds=[(censor - 30.0 * s_ref, censor + 300.0 * s_ref), (np.log(0.1 * s_ref), np.log(30.0 * s_ref))])
    mu, s = float(res.x[0]), float(np.exp(res.x[1]))
    # numerical Hessian for the standard error of mu
    eps = 1e-3 * max(s, 1e-3)
    f0 = nll(res.x)
    fp = nll(res.x + np.array([eps, 0.0]))
    fm = nll(res.x - np.array([eps, 0.0]))
    h = (fp - 2 * f0 + fm) / eps**2
    se = float(1.0 / np.sqrt(h)) if h > 0 else float("nan")
    return mu, s, se


def dark_level(img_dq: np.ndarray, img_raw: np.ndarray, pore_eroded: np.ndarray, valid: np.ndarray,
               min_pixels: int = 2000, fallback: float | None = None, step: int = 1) -> DarkLevel:
    """Dark level from deep-pore pixels: robust mode, or a Tobit fit when pores clip at 0."""
    sel = pore_eroded & valid
    x = img_dq[sel]
    raw = img_raw[sel]
    n = int(x.size)
    if n < min_pixels:
        return DarkLevel(float(fallback) if fallback is not None else float("nan"), float("nan"), n, "fallback", float("nan"))
    frac_zero = float((raw == 0).mean())
    step = max(1, n // 400_000)
    sub, sub_raw = x[::step], raw[::step]
    mode = _robust_mode(sub[:: max(1, sub.size // 200_000)])
    median = float(np.median(x))
    # raw 0 means I < step/2 before quantisation, so that is the censoring point in dequantised units
    mu, _s, se = _tobit_fit(sub, sub_raw == 0, censor=0.5 * step)
    method = "tobit" if frac_zero >= 0.01 else "gauss"  # no censoring -> plain Gaussian mean
    if (raw > 0).sum() < 200:
        if fallback is not None:
            return DarkLevel(float(fallback), float("nan"), n, "fallback", frac_zero, mode, median)
        method = "tobit-weak"
    return DarkLevel(mu, se, n, method, frac_zero, mode, median)


def quantile_dark_level(img_dq: np.ndarray, img_raw: np.ndarray, pore_eroded: np.ndarray, valid: np.ndarray, q: float = 0.01,
                        min_pixels: int = 2000, fallback: float | None = None) -> DarkLevel:
    """Dark level for a topographic detector (Inlens) whose pore interiors are *not* uniformly dark.

    Measured Inlens deep-pore values are spread almost uniformly from the black level up to the
    graphite level (SE escape from the pore walls), so neither a Gaussian mean nor a mode is a
    physical anchor; the lower edge is.  The anchor is the ``q`` quantile of the deep-pore pixels;
    when more than ``q`` of them clip at 0 the edge is below the ADC range and 0 is reported with
    method ``quantile-clipped``.
    """
    sel = pore_eroded & valid
    x = img_dq[sel]
    n = int(x.size)
    if n < min_pixels:
        return DarkLevel(float(fallback) if fallback is not None else float("nan"), float("nan"), n, "fallback", float("nan"))
    frac_zero = float((img_raw[sel] == 0).mean())
    sub = x[:: max(1, n // 400_000)]
    mode = _robust_mode(sub[:: max(1, sub.size // 200_000)])
    median = float(np.median(x))
    if frac_zero >= q:
        return DarkLevel(0.0, float("nan"), n, "quantile-clipped", frac_zero, mode, median)
    value = float(np.quantile(x, q))
    # standard error of a sample quantile via the density at the quantile (histogram estimate)
    lo, hi = np.quantile(x, [max(q - 0.005, 0), q + 0.005])
    dens = 0.01 / max(hi - lo, 1e-6)
    se = float(np.sqrt(q * (1 - q) / n) / dens)
    return DarkLevel(value, se, n, "quantile", frac_zero, mode, median)


# ----------------------------------------------------------------------------------------------
# Graphite-level flat-field G(x, y) (4.3)
# ----------------------------------------------------------------------------------------------
@dataclass
class GraphiteMap:
    coef: np.ndarray  # 2-D second-order polynomial coefficients
    blocks: pd.DataFrame  # block medians used for the fit
    block_px: int
    shape: tuple[int, int]

    def evaluate(self, shape: tuple[int, int] | None = None) -> np.ndarray:
        shape = shape or self.shape
        yy, xx = np.mgrid[0: shape[0], 0: shape[1]].astype(np.float32)
        return _poly2_eval(self.coef, xx / shape[1], yy / shape[0]).astype(np.float32)

    @property
    def level(self) -> float:
        return float(np.median(self.blocks["median"]))


def _poly2_design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)


def _poly2_eval(coef: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = coef
    return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y


def block_medians(values: np.ndarray, sel: np.ndarray, block_px: int = 64, min_cover: float = 0.2,
                  stat: str = "hsm", max_px: int = 600) -> pd.DataFrame:
    """Robust level of ``values`` over ``sel`` in ``block_px`` squares with ≥ ``min_cover`` coverage.

    ``stat`` is the half-sample mode (``"hsm"``, default: robust to the bright topographic tail of
    Inlens/SE graphite), the ``"median"`` or the lower decile ``"q10"`` (the dark floor, used for
    broad-charging detection).  The level is reported in column ``median`` for all three.
    """
    h, w = values.shape
    rows = []
    for by in range(0, h - block_px + 1, block_px):
        for bx in range(0, w - block_px + 1, block_px):
            s = sel[by: by + block_px, bx: bx + block_px]
            cover = s.mean()
            if cover < min_cover:
                continue
            v = values[by: by + block_px, bx: bx + block_px][s]
            if stat == "hsm":
                level = _robust_mode(v[:: max(1, v.size // max_px)])
            elif stat == "q10":
                level = float(np.percentile(v, 10))
            else:
                level = float(np.median(v))
            rows.append((by + block_px / 2, bx + block_px / 2, cover, level, int(v.size)))
    return pd.DataFrame(rows, columns=["y", "x", "cover", "median", "n"])


def graphite_map(img_minus_d: np.ndarray, graphite_eroded: np.ndarray, valid: np.ndarray, block_px: int = 64,
                 min_cover: float = 0.2) -> GraphiteMap:
    """Smooth flat-field: block medians of (I - D) over graphite, fit by a robust 2-D quadratic."""
    h, w = img_minus_d.shape
    blocks = block_medians(img_minus_d, graphite_eroded & valid, block_px, min_cover)
    if len(blocks) < 12:
        raise ValueError(f"only {len(blocks)} graphite blocks; cannot fit the flat-field")
    X = _poly2_design(blocks["x"].to_numpy() / w, blocks["y"].to_numpy() / h)
    y = blocks["median"].to_numpy()
    wts = np.sqrt(blocks["n"].to_numpy(dtype=float))
    coef = np.linalg.lstsq(X * wts[:, None], y * wts, rcond=None)[0]
    for _ in range(3):  # Huber reweighting against charged / masked blocks
        r = y - X @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        hw = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9))
        ww = wts * np.sqrt(hw)
        coef = np.linalg.lstsq(X * ww[:, None], y * ww, rcond=None)[0]
    blocks = blocks.assign(fit=X @ coef)
    return GraphiteMap(coef, blocks, block_px, (h, w))


def normalise(img_dq: np.ndarray, d: float, gmap: np.ndarray) -> np.ndarray:
    """z = (I - D) / G(x, y): pores ≈ 0, graphite = 1."""
    return ((img_dq - np.float32(d)) / gmap).astype(np.float32)


def to_fixed_point(z: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(z * Z_SCALE), 0, 65535).astype(np.uint16)


def from_fixed_point(z16: np.ndarray) -> np.ndarray:
    return z16.astype(np.float32) / Z_SCALE


# ----------------------------------------------------------------------------------------------
# Scan-line bands (3.3)
# ----------------------------------------------------------------------------------------------
def row_profile(z: np.ndarray, graphite_eroded: np.ndarray, valid: np.ndarray, min_px: int = 50,
                return_se: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Per-row interquartile mean of z over graphite pixels (NaN where fewer than ``min_px`` pixels).

    The interquartile mean is used instead of the median because the raw images use coarse grey
    steps (often 3 DN): a row median then jumps between quantisation levels and fakes 2–5 % bands.
    With ``return_se`` the standard error of each row statistic is returned too.
    """
    sel = graphite_eroded & valid
    zz = np.where(sel, z, np.nan).astype(np.float32)
    n = sel.sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q1, q3 = np.nanpercentile(zz, [25, 75], axis=1)
        inner = np.where((zz >= q1[:, None]) & (zz <= q3[:, None]), zz, np.nan)
        prof = np.nanmean(inner, axis=1)
        sd = np.nanstd(zz, axis=1)
    bad = n < min_px
    prof[bad] = np.nan
    if not return_se:
        return prof.astype(np.float32)
    se = 1.2 * sd / np.sqrt(np.maximum(n, 1))  # IQ-mean is ~1.2× less efficient than the mean
    se[bad] = np.nan
    return prof.astype(np.float32), se.astype(np.float32)


def detrended_robust_z(prof: np.ndarray, window: int = 151, se_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(detrended ratio prof / trend, robust z of its deviation). NaNs are interpolated first.

    The scale is max(MAD of the detrended deviation, ``se_floor``) so a profile that is smoother
    than its own statistical error cannot produce inflated z-scores.
    """
    p = pd.Series(prof).interpolate(limit_direction="both").to_numpy(dtype=np.float64)
    trend = ndimage.median_filter(p, size=window, mode="nearest")
    ratio = p / trend
    dev = ratio - 1.0
    mad = 1.4826 * np.median(np.abs(dev - np.median(dev))) + 1e-9
    scale = max(mad, float(se_floor))
    return ratio.astype(np.float32), ((dev - np.median(dev)) / scale).astype(np.float32)


def _runs(flag: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Inclusive [start, end] runs of True with length ≥ min_len."""
    out = []
    f = np.concatenate([[False], flag, [False]])
    d = np.diff(f.astype(np.int8))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
        if e - s + 1 >= min_len:
            out.append((int(s), int(e)))
    return out


def detect_scan_bands(zprofiles: dict[str, np.ndarray], z_thr: float = 5.0, min_rows: int = 3) -> pd.DataFrame:
    """Rows with |z| > z_thr in runs ≥ min_rows; ``shared`` when overlapping runs appear in ≥ 2 detectors."""
    rows = []
    per_det = {d: _runs(np.abs(zp) > z_thr, min_rows) for d, zp in zprofiles.items()}
    for d, runs in per_det.items():
        for s, e in runs:
            shared = sum(any(not (e2 < s or s2 > e) for s2, e2 in per_det[o]) for o in per_det if o != d) >= 1
            rows.append({"detector": d, "start": s, "end": e, "shared": bool(shared),
                         "peak_z": float(np.max(np.abs(zprofiles[d][s: e + 1])))})
    return pd.DataFrame(rows, columns=["detector", "start", "end", "shared", "peak_z"])


def band_gain(ratio: np.ndarray, bands: Iterable[tuple[int, int]], pad: int = 2) -> np.ndarray:
    """Multiplicative row gain g(y): the detrended ratio inside each band ± pad with a linear taper, 1 elsewhere."""
    g = np.ones_like(ratio, dtype=np.float32)
    n = ratio.size
    for s, e in bands:
        lo, hi = max(0, s - pad), min(n - 1, e + pad)
        w = np.ones(hi - lo + 1, dtype=np.float32)
        for k in range(pad):  # taper the padding rows
            if k < w.size:
                w[k] = w[-1 - k] = (k + 1) / (pad + 1)
        g[lo: hi + 1] = 1.0 + (ratio[lo: hi + 1] - 1.0) * w
    return g


def correct_scan_bands(img_dq: np.ndarray, d: float, g_rows: np.ndarray) -> np.ndarray:
    """I' = D + (I - D) / g(y) — a gain correction about the dark level."""
    return (np.float32(d) + (img_dq - np.float32(d)) / g_rows[:, None]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Field-of-view intrusions (3.2)
# ----------------------------------------------------------------------------------------------
def _edge_connected(flag: np.ndarray, top: bool) -> np.ndarray:
    lab, _ = ndimage.label(flag)
    edge_row = 0 if top else flag.shape[0] - 1
    ids = np.unique(lab[edge_row])
    ids = ids[ids > 0]
    return np.isin(lab, ids)


def _mask_from_boundary(shape: tuple[int, int], boundary: np.ndarray, top: bool, dilate_px: int) -> np.ndarray:
    """Mask from a per-column boundary row to the top (or bottom) edge, extended by ``dilate_px``."""
    h, w = shape
    rows = np.arange(h)[:, None]
    b = boundary.astype(np.float32)
    if top:
        return rows <= (b + dilate_px)[None, :]
    return rows >= (b - dilate_px)[None, :]


def detect_collector(bse_raw: np.ndarray, valid: np.ndarray, sat_dn: int = 250, min_rows: int = 8,
                     col_frac: float = 0.5, dilate_px: int = DILATE_HALF_UM_PX) -> tuple[np.ndarray, dict]:
    """Copper current collector: a saturated BSE band connected to the top or bottom edge."""
    h, w = bse_raw.shape
    sat = (bse_raw >= sat_dn) & valid
    sat = ndimage.binary_opening(sat, structure=np.ones((3, 9)))
    out = np.zeros((h, w), dtype=bool)
    info: dict = {"found": False}
    for top in (True, False):
        band = _edge_connected(sat, top)
        if band.sum() < min_rows * 0.05 * w:
            continue
        cols = band.any(axis=0)
        if cols.mean() < 0.05:
            continue
        # per-column boundary = innermost saturated row; fill columns without saturation by interpolation
        rows = np.arange(h)[:, None]
        if top:
            bnd = np.where(cols, np.where(band, rows, -1).max(axis=0), np.nan)
        else:
            bnd = np.where(cols, np.where(band, rows, h).min(axis=0), np.nan)
        bnd = pd.Series(bnd).interpolate(limit_direction="both").to_numpy()
        bnd = ndimage.median_filter(bnd, size=101, mode="nearest")
        # include a dark delamination gap attached to the band: extend the boundary through
        # columns where the rows just inside the band are very dark
        depth = float(np.nanmedian(bnd if top else h - 1 - bnd))
        if top:
            inner = bse_raw[np.clip(bnd.astype(int) + 1, 0, h - 1) + np.arange(0, 1)[:, None], np.arange(w)]
        else:
            inner = bse_raw[np.clip(bnd.astype(int) - 1, 0, h - 1) + np.arange(0, 1)[:, None], np.arange(w)]
        del inner
        m = _mask_from_boundary((h, w), bnd, top, dilate_px)
        out |= m
        info = {"found": True, "edge": "top" if top else "bottom", "depth_px": depth,
                "frac_cols": float(cols.mean()), "depth_um": depth / PX_PER_UM}
    return out, info


def _changepoint_1d(y: np.ndarray, min_seg: int = 10) -> tuple[int, float, float]:
    """Single changepoint by binary segmentation (max standardised mean shift).

    Returns (index, score, shift) where shift is the raw mean difference between the segments.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 2 * min_seg:
        return -1, 0.0, 0.0
    cs = np.cumsum(y)
    best, best_i, best_shift = 0.0, -1, 0.0
    sd = 1.4826 * np.median(np.abs(y - np.median(y)))
    sd = max(sd, 0.1 * float(np.std(y)), 1e-9)  # a flat statistic (MAD = 0) must not explode the score
    for i in range(min_seg, n - min_seg):
        m1 = cs[i - 1] / i
        m2 = (cs[-1] - cs[i - 1]) / (n - i)
        score = abs(m1 - m2) / sd * np.sqrt(i * (n - i) / n)
        if score > best:
            best, best_i, best_shift = score, i, abs(m1 - m2)
    return best_i, float(best), float(best_shift)


def detect_free_surface(bse_raw: np.ndarray, inlens_raw: np.ndarray, valid: np.ndarray, strip_px: int = 250,
                        outer_frac: float = 0.15, score_thr: float = 12.0, dilate_px: int = DILATE_HALF_UM_PX,
                        row_step: int = 4, min_shift: tuple[float, float, float] = (0.05, 8.0, 0.03),
                        min_strips: int = 3) -> tuple[np.ndarray, dict]:
    """Coating free surface near the top (or bottom) edge: a robust changepoint in row statistics.

    Row statistics per vertical strip: Inlens saturation fraction, BSE row median, local edge
    density.  A strip is flagged when the standardised mean shift of ≥ 2 statistics exceeds
    ``score_thr`` and the outer segment is the abnormal one.
    """
    h, w = bse_raw.shape
    n_outer = int(h * outer_frac)
    out = np.zeros((h, w), dtype=bool)
    info: dict = {"found": False, "strips": []}
    grad = np.abs(np.diff(bse_raw.astype(np.float32), axis=1, append=bse_raw[:, -1:].astype(np.float32)))
    edge = grad > 25
    for top in (True, False):
        rows = np.arange(0, n_outer) if top else np.arange(h - n_outer, h)
        boundary = np.full(w, np.nan)
        for x0 in range(0, w, strip_px):
            sl = slice(x0, min(w, x0 + strip_px))
            v = valid[rows][:, sl]
            if v.mean() < 0.5:
                continue
            sat = np.where(v, inlens_raw[rows][:, sl] >= 254, np.nan)
            bse = np.where(v, bse_raw[rows][:, sl].astype(np.float32), np.nan)
            ed = np.where(v, edge[rows][:, sl], np.nan)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats = np.stack([np.nanmean(sat, 1), np.nanmedian(bse, 1), np.nanmean(ed, 1)], 1)
            stats = pd.DataFrame(stats).interpolate(limit_direction="both").to_numpy()[::row_step]
            # orient so index 0 is the image edge
            if not top:
                stats = stats[::-1]
            hits = []
            for k in range(3):
                i, sc, shift = _changepoint_1d(stats[:, k])
                if i > 0 and sc > score_thr and shift > min_shift[k]:
                    hits.append((i, sc))
            if len(hits) >= 2:
                i = int(np.median([hh[0] for hh in hits])) * row_step
                b = i if top else h - 1 - i
                boundary[sl] = b
                info["strips"].append({"edge": "top" if top else "bottom", "x0": x0, "boundary": int(b),
                                       "scores": [round(s, 1) for _, s in hits]})
        found_cols = np.isfinite(boundary)
        # a real surface spans many adjacent strips; isolated strips are particles / texture
        strip_hit = np.array([found_cols[x0: x0 + strip_px].any() for x0 in range(0, w, strip_px)])
        keep = np.zeros_like(strip_hit)
        for s0, e0 in _runs(strip_hit, min_strips):
            keep[s0: e0 + 1] = True
        for k, x0 in enumerate(range(0, w, strip_px)):
            if not keep[k]:
                boundary[x0: x0 + strip_px] = np.nan
        info["strips"] = [st for st in info["strips"] if keep[st["x0"] // strip_px] or st["edge"] != ("top" if top else "bottom")]
        found_cols = np.isfinite(boundary)
        if found_cols.sum() == 0:
            continue
        b = pd.Series(boundary).interpolate(limit_direction="both").to_numpy()
        b[~found_cols & ~_fill_gaps(found_cols, strip_px)] = -1 if top else h  # mask only detected strips (+ small gaps)
        m = _mask_from_boundary((h, w), b, top, dilate_px)
        m &= np.broadcast_to(found_cols | _fill_gaps(found_cols, strip_px), (h, w))
        out |= m
        depth = float(np.nanmedian(boundary if top else h - 1 - boundary))
        info.update({"found": True, "edge": "top" if top else "bottom", "depth_px": depth, "depth_um": depth / PX_PER_UM,
                     "frac_cols": float(found_cols.mean())})
    return out, info


def _fill_gaps(cols: np.ndarray, max_gap: int) -> np.ndarray:
    """True also inside gaps of ≤ max_gap columns between detected columns."""
    return ndimage.binary_closing(cols, structure=np.ones(max_gap + 1))


# ----------------------------------------------------------------------------------------------
# Charging (3.4)
# ----------------------------------------------------------------------------------------------
def detect_local_charging(inlens_raw: np.ndarray, bse_z: np.ndarray, valid: np.ndarray, si_z_thr: float = 1.4,
                          sat_dn: int = 254, min_area_um2: float = 1.0, min_solidity: float = 0.6,
                          dilate_px: int = DILATE_HALF_UM_PX) -> tuple[np.ndarray, pd.DataFrame]:
    """Saturated, compact, graphite-like (not high-Z) Inlens blobs ≥ min_area_um2."""
    min_px = int(min_area_um2 * PX_PER_UM**2)
    sat = (inlens_raw >= sat_dn) & valid
    sat = remove_small_objects(sat, min_size=min_px // 4)
    lab = label(sat, connectivity=2)
    out = np.zeros(inlens_raw.shape, dtype=bool)
    rows = []
    for rp in regionprops(lab):
        if rp.area < min_px or rp.solidity < min_solidity:
            continue
        coords = rp.coords
        bse_med = float(np.median(bse_z[coords[:, 0], coords[:, 1]]))
        if bse_med > si_z_thr:
            continue  # bright in BSE too: a real high-Z particle, not charging
        y0, x0, y1, x1 = rp.bbox
        out[lab == rp.label] = True
        rows.append({"y0": y0, "x0": x0, "y1": y1, "x1": x1, "area_um2": rp.area / PX_PER_UM**2,
                     "solidity": float(rp.solidity), "bse_z": bse_med})
    if out.any():
        out = binary_dilation(out, disk(dilate_px))
    return out, pd.DataFrame(rows, columns=["y0", "x0", "y1", "x1", "area_um2", "solidity", "bse_z"])


def detect_broad_charging(inlens_blocks: pd.DataFrame, shape: tuple[int, int], block_px: int, rel_thr: float = 0.75,
                          min_blocks: int = 20) -> tuple[np.ndarray, dict]:
    """Contiguous block region where the Inlens graphite *floor* is lifted > rel_thr above the site level.

    ``inlens_blocks`` holds the lower decile of provisional Inlens z over graphite per block
    (:func:`block_medians` with ``stat="q10"``).  Inlens graphite brightness is topographic — flake
    edges and small particles are 2–3× brighter than flat flake interiors — so the block median or
    mode against the flat-field flags topography on most sites (the brief's 25 % rule).  A broad
    charging glow lifts the dark floor of every block it covers, which topography does not, so the
    floor is compared with the site-wide median floor; positive deviations only, 3×3-median
    smoothed, regions of ≥ ``min_blocks`` contiguous blocks.
    """
    h, w = shape
    nby, nbx = h // block_px, w // block_px
    grid = np.full((nby, nbx), np.nan)
    iy = (inlens_blocks["y"] // block_px).astype(int).clip(0, nby - 1)
    ix = (inlens_blocks["x"] // block_px).astype(int).clip(0, nbx - 1)
    floor = inlens_blocks["median"].to_numpy()
    ref = float(np.nanmedian(floor)) if floor.size else np.nan
    grid[iy, ix] = floor / max(ref, 1e-6) - 1.0
    filled = np.where(np.isfinite(grid), grid, 0.0)
    smooth = ndimage.median_filter(filled, size=3, mode="nearest")
    dev = (smooth > rel_thr) & np.isfinite(grid)
    dev = ndimage.binary_closing(dev, structure=np.ones((3, 3)))
    lab, n = ndimage.label(dev)
    out = np.zeros((h, w), dtype=bool)
    info = {"found": False, "frac_blocks": float(dev.mean()) if dev.size else 0.0, "floor_ref": ref}
    for k in range(1, n + 1):
        reg = lab == k
        if reg.sum() < min_blocks:
            continue
        big = np.kron(reg, np.ones((block_px, block_px), dtype=bool))
        out[: big.shape[0], : big.shape[1]] |= big
        info["found"] = True
    return out, info


# ----------------------------------------------------------------------------------------------
# Clipping (3.5)
# ----------------------------------------------------------------------------------------------
def clipping_bits(img_raw: np.ndarray, d: float) -> np.ndarray:
    m = np.zeros(img_raw.shape, dtype=np.uint16)
    m[img_raw >= 255] |= BIT_CLIP_HIGH
    if d <= 0:
        m[img_raw <= 0] |= BIT_CLIP_LOW
    return m


# ----------------------------------------------------------------------------------------------
# Material flags (3.6) — per-site statistics; the across-site thresholds live in the CLI
# ----------------------------------------------------------------------------------------------
def pore_network_areas(pore: np.ndarray, valid: np.ndarray, top_k: int = 3) -> list[dict]:
    lab = label(pore & valid, connectivity=2)
    if lab.max() == 0:
        return []
    out = []
    for rp in sorted(regionprops(lab), key=lambda r: -r.area)[:top_k]:
        y0, x0, y1, x1 = rp.bbox
        out.append({"area_um2": rp.area / PX_PER_UM**2, "span_um": max(y1 - y0, x1 - x0) / PX_PER_UM,
                    "bbox": [int(y0), int(x0), int(y1), int(x1)], "label": int(rp.label)})
    return out


def eighth_porosity(pore: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    h = pore.shape[0]
    e = h // 8

    def por(sl):
        v = valid[sl]
        return float((pore[sl] & v).sum() / max(1, v.sum()))

    return {"top": por(slice(0, e)), "bottom": por(slice(h - e, h)), "all": por(slice(0, h))}


def robust_outlier_threshold(values: np.ndarray, k: float = 3.5) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return float(med + k * mad)


# ----------------------------------------------------------------------------------------------
# Edge width via edge-spread-function fits (4.6)
# ----------------------------------------------------------------------------------------------
def _erf_model(x, a, b, x0, s):
    return a + b * 0.5 * (1.0 + special.erf((x - x0) / (np.sqrt(2.0) * s)))


@dataclass
class EdgeWidth:
    sigma_px: float  # median ESF sigma
    n_fit: int
    n_kept: int
    width_10_90_nm: float
    sigmas: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


def edge_width(z: np.ndarray, pore: np.ndarray, valid: np.ndarray, n_edges: int = 10_000, half_len: int = 6,
               r2_min: float = 0.9, rng: np.random.Generator | None = None, nm_per_px: float = FULL_NM_PER_PX) -> EdgeWidth:
    """Median ESF sigma from erf fits across strong pore–solid edges sampled along the gradient normal."""
    rng = rng or np.random.default_rng(SEED)
    zf = np.asarray(z, dtype=np.float32)
    gy = ndimage.sobel(zf, axis=0)
    gx = ndimage.sobel(zf, axis=1)
    mag = np.hypot(gx, gy)
    # candidate edge pixels: boundary of the pore mask, strong gradient, away from invalid pixels
    boundary = ndimage.binary_dilation(pore, iterations=1) & ~pore
    valid3 = ndimage.binary_erosion(valid, structure=np.ones((3, 3)), border_value=0)  # Sobel footprint
    cand = valid3 & boundary
    ok = cand & (mag > np.percentile(mag[cand], 50)) if cand.any() else cand
    ok[:half_len + 2, :] = ok[-half_len - 2:, :] = False
    ok[:, :half_len + 2] = ok[:, -half_len - 2:] = False
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return EdgeWidth(float("nan"), 0, 0, float("nan"))
    idx = rng.choice(idx, size=min(n_edges, idx.size), replace=False)
    yy, xx = np.unravel_index(idx, zf.shape)
    nx, ny = gx.flat[idx] / (mag.flat[idx] + 1e-9), gy.flat[idx] / (mag.flat[idx] + 1e-9)
    t = np.arange(-half_len, half_len + 1, dtype=np.float32)
    ys = yy[:, None] + ny[:, None] * t[None, :]
    xs = xx[:, None] + nx[:, None] * t[None, :]
    # every bilinear sample must sit on valid pixels only (mask, don't fabricate)
    yi, xi = np.clip(np.floor(ys).astype(int), 0, zf.shape[0] - 2), np.clip(np.floor(xs).astype(int), 0, zf.shape[1] - 2)
    clean = (valid[yi, xi] & valid[yi + 1, xi] & valid[yi, xi + 1] & valid[yi + 1, xi + 1]).all(axis=1)
    ys, xs = ys[clean], xs[clean]
    prof = ndimage.map_coordinates(zf, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(ys.shape)
    sigmas = []
    for p in prof:
        a0, b0 = float(p[:2].mean()), float(p[-2:].mean() - p[:2].mean())
        if abs(b0) < 0.2:
            continue
        try:
            popt, _ = optimize.curve_fit(_erf_model, t, p, p0=[a0, b0, 0.0, 1.0], maxfev=200)
        except (RuntimeError, ValueError):
            continue
        res = p - _erf_model(t, *popt)
        r2 = 1.0 - float(np.sum(res**2) / (np.sum((p - p.mean()) ** 2) + 1e-12))
        if r2 > r2_min and 0.2 < abs(popt[3]) < half_len and abs(popt[2]) < half_len / 2:
            sigmas.append(abs(float(popt[3])))
    sig = np.asarray(sigmas)
    med = float(np.median(sig)) if sig.size else float("nan")
    return EdgeWidth(med, int(prof.shape[0]), int(sig.size), 2.563 * med * nm_per_px, sig)


def harmonise_resolution(z: np.ndarray, sigma_e: float, sigma_t: float) -> tuple[np.ndarray, float]:
    """Blur to the target edge width with sigma_k = sqrt(sigma_t² - sigma_e²); never sharpen."""
    if not np.isfinite(sigma_e) or sigma_e >= sigma_t:
        return np.asarray(z, dtype=np.float32), 0.0
    sk = float(np.sqrt(sigma_t**2 - sigma_e**2))
    return ndimage.gaussian_filter(np.asarray(z, dtype=np.float32), sk).astype(np.float32), sk


# ----------------------------------------------------------------------------------------------
# Poisson–Gaussian noise model var(z) = alpha * z + beta (4.7)
# ----------------------------------------------------------------------------------------------
@dataclass
class NoiseModel:
    alpha: float
    beta: float
    n_patches: int
    sigma_graphite: float  # sqrt(alpha*1 + beta): noise SD at the graphite level

    def var(self, z: np.ndarray | float) -> np.ndarray | float:
        return np.maximum(self.alpha * z + self.beta, 0.0)


def noise_model(z: np.ndarray, phases: PhaseMasks, valid: np.ndarray, patch: int = 7, grad_q: float = 0.3,
                max_patches: int = 200_000, rng: np.random.Generator | None = None) -> NoiseModel:
    """Fit var = alpha*mean + beta on flat graphite and Si patches.

    Pore patches are left out: their noise is censored by the clip at 0 (raw 0-clipping on Batch 1/2,
    and z < 0 clipping in the uint16 store), so they would bias beta low by a stage-dependent amount.
    """
    rng = rng or np.random.default_rng(SEED)
    zf = np.asarray(z, dtype=np.float32)
    mean = ndimage.uniform_filter(zf, patch, mode="nearest")
    sq = ndimage.uniform_filter(zf * zf, patch, mode="nearest")
    var = np.maximum(sq - mean * mean, 0.0) * (patch * patch / (patch * patch - 1))
    g_sigma, g_trunc = 2.0, 3.0
    smooth = ndimage.gaussian_filter(zf, g_sigma, truncate=g_trunc)
    grad = np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))
    r = int(np.ceil(g_sigma * g_trunc)) + 1  # gaussian + sobel footprint radius
    valid_fp = ndimage.binary_erosion(valid, structure=np.ones((2 * r + 1, 2 * r + 1)), border_value=0)
    xs, ys = [], []
    for m in (phases.graphite, phases.si):
        core = binary_erosion(m & valid_fp, disk(patch // 2 + 1))
        if core.sum() < 100:
            continue
        flat = core & (grad <= np.percentile(grad[core], grad_q * 100))
        idx = np.flatnonzero(flat)
        if idx.size == 0:
            continue
        # fixed-seed subsample so the same array always gives the same estimate (the normalisation and
        # harmonisation stages re-measure the same image and must agree)
        idx = np.random.default_rng(SEED).choice(idx, size=min(max_patches // 3, idx.size), replace=False)
        xs.append(mean.flat[idx])
        ys.append(var.flat[idx])
    if not xs:
        return NoiseModel(float("nan"), float("nan"), 0, float("nan"))
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    # robust: bin by mean level, take the median variance per bin, then weighted least squares
    bins = np.quantile(x, np.linspace(0, 1, 25))
    which = np.clip(np.searchsorted(bins, x, side="right") - 1, 0, len(bins) - 2)
    bx, by, bw = [], [], []
    for k in range(len(bins) - 1):
        sel = which == k
        if sel.sum() < 50:
            continue
        bx.append(np.median(x[sel]))
        by.append(np.median(y[sel]) / 0.4549)  # median of a chi²(k) variance estimate -> mean (k = 48 df ≈ 0.986); use empirical bias factor below
        bw.append(np.sqrt(sel.sum()))
    bx, by, bw = map(np.asarray, (bx, by, bw))
    by = by * 0.4549 / _median_bias(patch * patch - 1)
    A = np.stack([bx, np.ones_like(bx)], 1) * bw[:, None]
    alpha, beta = np.linalg.lstsq(A, by * bw, rcond=None)[0]
    alpha = max(float(alpha), 0.0)
    beta = max(float(beta), 0.0)
    return NoiseModel(alpha, beta, int(x.size), float(np.sqrt(alpha + beta)))


def _median_bias(df: int) -> float:
    """median / mean of a chi²_df / df variable (sample variance of Gaussian noise)."""
    from scipy import stats

    return float(stats.chi2.median(df) / df)


def harmonise_noise(z: np.ndarray, model: NoiseModel, target: NoiseModel, rng: np.random.Generator,
                    min_rel_gap: float = 0.10) -> tuple[np.ndarray, dict]:
    """Add zero-mean Gaussian noise with variance (alpha_t - alpha) z + (beta_t - beta), clipped at 0."""
    da, db = target.alpha - model.alpha, target.beta - model.beta
    zf = np.asarray(z, dtype=np.float32)
    v = np.maximum(da * np.maximum(zf, 0.0) + db, 0.0)
    # "only when needed": the alpha/beta split is poorly identified per image (they trade off), so the
    # decision uses the net variance gap at graphite (z = 1); skip when already within min_rel_gap of target
    needed = (da + db) > min_rel_gap * (target.alpha + target.beta)
    if not (np.isfinite(da) and np.isfinite(db)) or float(v.max()) <= 0 or not needed:
        return zf, {"d_alpha": float(da), "d_beta": float(db), "added": False}
    out = zf + rng.standard_normal(zf.shape, dtype=np.float32) * np.sqrt(v, dtype=np.float32)
    return out.astype(np.float32), {"d_alpha": float(da), "d_beta": float(db), "added": True}


# ----------------------------------------------------------------------------------------------
# Per-site driver
# ----------------------------------------------------------------------------------------------
@dataclass
class Targets:
    """Harmonisation targets derived from the accepted reference images (per detector)."""

    sigma_t: dict[str, float]  # ESF sigma target (px)
    noise: dict[str, NoiseModel]  # noisiest accepted reference image per detector
    reference_sites: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"sigma_t": self.sigma_t,
                "noise": {d: {"alpha": m.alpha, "beta": m.beta} for d, m in self.noise.items()},
                "reference_sites": self.reference_sites}

    @classmethod
    def from_dict(cls, d: dict) -> "Targets":
        return cls({k: float(v) for k, v in d["sigma_t"].items()},
                   {k: NoiseModel(v["alpha"], v["beta"], 0, float(np.sqrt(v["alpha"] + v["beta"]))) for k, v in d["noise"].items()},
                   list(d.get("reference_sites", [])))


@dataclass
class CleanResult:
    norm: dict[str, np.ndarray]  # float32 z per detector
    mask: dict[str, np.ndarray]  # uint16 per detector
    params: dict
    harm: dict[str, np.ndarray] = field(default_factory=dict)
    phases: PhaseMasks | None = None


def _row_z(z: np.ndarray, graphite: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prof, se = row_profile(z, graphite, valid, return_se=True)
    with np.errstate(all="ignore"):
        floor = float(np.nanmedian(se / np.abs(prof)))
    return detrended_robust_z(prof, se_floor=floor if np.isfinite(floor) else 0.0)


def _dark(detector: str, img_dq, img_raw, pore, valid, session_dark, step: int = 1) -> DarkLevel:
    fb = (session_dark or {}).get(detector)
    if detector == "Inlens":
        return quantile_dark_level(img_dq, img_raw, pore, valid, fallback=fb)
    return dark_level(img_dq, img_raw, pore, valid, fallback=fb, step=step)


def _np(x):
    """JSON-friendly scalars."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def clean_site(raw: dict[str, np.ndarray], marker_cols: np.ndarray | None = None, nm_per_px: float = FULL_NM_PER_PX,
               seed: int = SEED, session_dark: dict[str, float] | None = None, correct_bands: bool = True, max_single_rows: int = 8,
               detect_fov: bool = True, charging: bool = True) -> CleanResult:
    """Patch (3.1–3.6) and normalise (4.1–4.4) one site.  ``raw`` maps detector -> uint8 image."""

    rng = np.random.default_rng(seed)
    bse_raw = raw["BSE"]
    shape = bse_raw.shape
    params: dict = {"code_version": CODE_VERSION, "seed": seed, "shape": list(shape), "nm_per_px": nm_per_px}
    mask = {d: sanitise(shape, marker_cols) for d in DETECTORS}
    params["marker_cols"] = np.flatnonzero(marker_cols).tolist() if marker_cols is not None else []
    steps = {d: grey_step(raw[d]) for d in DETECTORS}
    dq = {d: dequantise(raw[d], rng, steps[d]) for d in DETECTORS}
    valid = valid_for_kpis(mask["BSE"])

    # --- provisional anchors ------------------------------------------------------------------
    ph = provisional_phases(dq["BSE"], valid)
    prov = {}
    for d in DETECTORS:
        dl = _dark(d, dq[d], raw[d], ph.pore, valid, session_dark, steps[d])
        gm = graphite_map(dq[d] - dl.value, ph.graphite, valid)
        prov[d] = (dl, gm, normalise(dq[d], dl.value, gm.evaluate()))

    # --- field of view (3.2) --------------------------------------------------------------------
    params["fov"] = {}
    if detect_fov:
        coll, cinfo = detect_collector(bse_raw, valid)
        fs, finfo = detect_free_surface(bse_raw, raw["Inlens"], valid & ~coll)
        for d in DETECTORS:
            mask[d][coll] |= BIT_COLLECTOR
            mask[d][fs] |= BIT_FREE_SURFACE
        params["fov"] = {"collector": cinfo, "free_surface": finfo}
        valid = valid_for_kpis(mask["BSE"])

    # --- scan-line bands (3.3) -------------------------------------------------------------------
    corrected = dict(dq)
    params["bands"] = {"events": [], "corrected": [], "masked": []}
    if correct_bands:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profiles = {d: _row_z(prov[d][2], ph.graphite, valid) for d in DETECTORS}
        events = detect_scan_bands({d: profiles[d][1] for d in DETECTORS})
        # a scan-line fault is a few rows wide and/or seen by more than one detector; a wide
        # single-detector run is Inlens/SE topography (flake-scale brightness), left untouched
        events["actionable"] = events["shared"] | ((events["end"] - events["start"] + 1) <= max_single_rows)
        params["bands"]["events"] = events.to_dict("records")
        for d in DETECTORS:
            ev = events[(events.detector == d) & events.actionable]
            if ev.empty:
                continue
            runs = [(int(r.start), int(r.end)) for r in ev.itertuples()]
            g = band_gain(profiles[d][0], runs)
            corrected[d] = correct_scan_bands(dq[d], prov[d][0].value, g)
            # verify on the corrected image
            z_new = normalise(corrected[d], prov[d][0].value, prov[d][1].evaluate())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _, zr = _row_z(z_new, ph.graphite, valid)
            for s, e in runs:
                rows = slice(max(0, s - 2), min(shape[0], e + 3))
                if float(np.nanmax(np.abs(zr[s: e + 1]))) < 3.0:
                    mask[d][rows, :] |= BIT_BAND_CORRECTED
                    params["bands"]["corrected"].append({"detector": d, "start": s, "end": e, "gain": float(np.mean(g[s: e + 1]))})
                else:
                    mask[d][rows, :] |= BIT_BAND_BAD
                    params["bands"]["masked"].append({"detector": d, "start": s, "end": e, "residual_z": float(np.nanmax(np.abs(zr[s: e + 1])))})

    # --- charging (3.4) -------------------------------------------------------------------------
    params["charging"] = {"local": [], "broad": {}}
    if charging:
        loc, ldf = detect_local_charging(raw["Inlens"], prov["BSE"][2], valid)
        for d in ("Inlens", "SE_type"):
            mask[d][loc] |= BIT_CHARGE_LOCAL
        params["charging"]["local"] = ldf.to_dict("records")
        bpx = prov["Inlens"][1].block_px
        floor_blocks = block_medians(prov["Inlens"][2], ph.graphite & valid, bpx, stat="q10")
        broad, binfo = detect_broad_charging(floor_blocks, shape, bpx)
        mask["Inlens"][broad] |= BIT_CHARGE_BROAD
        params["charging"]["broad"] = binfo

    # --- refine anchors on the patched image (4.2–4.4) -----------------------------------------
    valid = valid_for_kpis(mask["BSE"])
    ph = provisional_phases(corrected["BSE"], valid)
    norm, anchors = {}, {}
    for d in DETECTORS:
        v_d = valid_for_kpis(mask[d])
        dl = _dark(d, corrected[d], raw[d], ph.pore, v_d, session_dark, steps[d])
        gm = graphite_map(corrected[d] - dl.value, ph.graphite, v_d)
        G = gm.evaluate()
        norm[d] = normalise(corrected[d], dl.value, G)
        mask[d] |= clipping_bits(raw[d], dl.value)
        si_core = binary_erosion(ph.si, disk(3)) & v_d
        anchors[d] = {
            "D": dl.value, "D_se": dl.stderr, "D_method": dl.method, "D_n": dl.n, "D_frac_zero": dl.frac_zero, "D_mode": dl.mode, "D_median": dl.median,
            "G_level": gm.level, "G_coef": [float(c) for c in gm.coef], "G_blocks": int(len(gm.blocks)),
            "G_ptp_rel": float((G.max() - G.min()) / G.mean()),
            "si_graphite_ratio": float(np.median(norm[d][si_core])) if si_core.any() else float("nan"),
            "grey_levels_used": int(np.unique(raw[d]).size), "grey_step": steps[d],
        }
    params["anchors"] = anchors
    params["phase_thresholds_bse"] = list(ph.thresholds)

    # the fingerprint and the material statistics are measured on z exactly as it is stored (uint16 fixed
    # point: z < 0 clips to 0, 1e-4 steps) with phases re-derived on it, so that the harmonisation stage
    # and the evaluator, which read the stored arrays, see the same data and masks; ``norm`` itself stays
    # float (idempotence, z < 0 kept)
    stored = {d: from_fixed_point(to_fixed_point(norm[d])) for d in DETECTORS}
    ph = provisional_phases(stored["BSE"], valid)
    params["phase_thresholds_bse"] = list(ph.thresholds)

    # --- imaging fingerprint inputs (4.6, 4.7) ---------------------------------------------------
    fp = {}
    for d in DETECTORS:
        v_d = valid_for_stats(mask[d])
        ew = edge_width(stored[d], ph.pore, v_d, rng=rng, nm_per_px=nm_per_px)
        nm = noise_model(stored[d], ph, v_d, rng=rng)
        fp[d] = {"sigma_e_px": ew.sigma_px, "edge_10_90_nm": ew.width_10_90_nm, "edge_n_kept": ew.n_kept,
                 "noise_alpha": nm.alpha, "noise_beta": nm.beta, "noise_sigma_graphite": nm.sigma_graphite}
    params["fingerprint"] = fp

    # --- material statistics (3.6) ------------------------------------------------------------------
    v = valid_for_kpis(mask["BSE"])
    params["material"] = {"pore_networks": pore_network_areas(ph.pore, v), "eighth_porosity": eighth_porosity(ph.pore, v),
                          "porosity": float((ph.pore & v).sum() / max(1, v.sum())),
                          "si_fraction": float((ph.si & v).sum() / max(1, v.sum()))}
    params["mask_fractions"] = {d: {BIT_NAMES[b]: float(((mask[d] & b) != 0).mean()) for b in BIT_NAMES} for d in DETECTORS}
    params["masked_beyond_border"] = {d: float((((mask[d] & INVALID_KPI) != 0) & ((mask[d] & BIT_BORDER) == 0)).mean()) for d in DETECTORS}
    return CleanResult(norm, mask, _jsonable(params), phases=ph)


def harmonise_site(res: CleanResult, targets: Targets, seed: int = SEED, nm_per_px: float = FULL_NM_PER_PX) -> CleanResult:
    """4.6 + 4.7: blur to the target edge width, then add noise up to the target model; verify both."""
    rng = np.random.default_rng(seed + 1)
    ph = res.phases or provisional_phases(res.norm["BSE"], valid_for_kpis(res.mask["BSE"]))
    harm, info = {}, {}
    for d in DETECTORS:
        v = valid_for_stats(res.mask[d])
        fp = res.params["fingerprint"][d]
        sig_e, sig_t = fp["sigma_e_px"], targets.sigma_t[d]
        z, sk = harmonise_resolution(res.norm[d], sig_e, sig_t)
        ew_after = edge_width(z, ph.pore, v, rng=rng, nm_per_px=nm_per_px) if sk > 0 else None
        nm_before = noise_model(z, ph, v, rng=rng)
        z, ninfo = harmonise_noise(z, nm_before, targets.noise[d], rng)
        nm_after = noise_model(z, ph, v, rng=rng)
        harm[d] = z
        tgt = targets.noise[d]
        info[d] = {
            "sigma_e_px": sig_e, "sigma_t_px": sig_t, "sigma_k_px": sk,
            "resolution_outlier": bool(np.isfinite(sig_e) and sig_e > 1.1 * sig_t),
            "sigma_e_after_px": ew_after.sigma_px if ew_after else sig_e,
            "noise_before": {"alpha": nm_before.alpha, "beta": nm_before.beta},
            "noise_target": {"alpha": tgt.alpha, "beta": tgt.beta},
            "noise_after": {"alpha": nm_after.alpha, "beta": nm_after.beta},
            "noise_sigma_graphite_after": nm_after.sigma_graphite,
            "noise_sigma_graphite_target": tgt.sigma_graphite,
            "noise_within_10pct": bool(abs(nm_after.sigma_graphite / max(tgt.sigma_graphite, 1e-9) - 1) < 0.10),
            **ninfo,
        }
    params = dict(res.params)
    params["harmonisation"] = _jsonable(info)
    params["targets"] = targets.to_dict()
    return CleanResult(res.norm, res.mask, params, harm=harm, phases=ph)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ----------------------------------------------------------------------------------------------
# Storage (4.8)
# ----------------------------------------------------------------------------------------------
def write_site(out_dir: str | Path, res: CleanResult) -> Path:
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in DETECTORS:
        tifffile.imwrite(out_dir / f"{d}_norm.tif", to_fixed_point(res.norm[d]), compression="zlib")
        tifffile.imwrite(out_dir / f"{d}_mask.tif", res.mask[d], compression="zlib")
        if d in res.harm:
            tifffile.imwrite(out_dir / f"{d}_harm.tif", to_fixed_point(res.harm[d]), compression="zlib")
    (out_dir / "params.json").write_text(json.dumps(res.params, indent=1))
    return out_dir


def read_site(out_dir: str | Path, detector: str, kind: str = "norm") -> tuple[np.ndarray, np.ndarray]:
    """(z float32, mask uint16) for one detector; ``kind`` is 'norm' or 'harm'."""
    out_dir = Path(out_dir)
    z = from_fixed_point(tifffile.imread(out_dir / f"{detector}_{kind}.tif"))
    m = tifffile.imread(out_dir / f"{detector}_mask.tif")
    return z, m


def build_targets(params_by_site: dict[str, dict], reference_sites: Iterable[str]) -> Targets:
    """sigma_t = 75th percentile of sigma_e over the accepted references; noise target = noisiest reference."""
    ref = [s for s in reference_sites if s in params_by_site]
    sigma_t, noise = {}, {}
    for d in DETECTORS:
        fps = [params_by_site[s]["fingerprint"][d] for s in ref]
        sig = np.array([f["sigma_e_px"] for f in fps], dtype=float)
        sigma_t[d] = float(np.nanpercentile(sig, 75))
        noisiest = max(fps, key=lambda f: (f["noise_sigma_graphite"] if f["noise_sigma_graphite"] is not None else -1))
        noise[d] = NoiseModel(noisiest["noise_alpha"], noisiest["noise_beta"], 0, noisiest["noise_sigma_graphite"])
    return Targets(sigma_t, noise, list(ref))
