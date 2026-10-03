"""Synthetic SEM phantoms with known dark level, gain, shading, scan-band dips, charging blobs,
blur and Poisson–Gaussian noise, for ``tests/test_clean.py`` (acceptance check 7.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

PX_PER_UM = 40.0


@dataclass
class Phantom:
    raw: dict[str, np.ndarray]  # uint8 per detector
    truth_s: dict[str, np.ndarray]  # noiseless signal S per detector (pore 0, graphite 1, Si ratio)
    d: dict[str, float]
    gain: dict[str, float]
    gmap: np.ndarray  # relative shading field (mean 1)
    row_gain: np.ndarray  # multiplicative row gain (1 except in bands)
    bands: list[tuple[int, int]]
    charging_bboxes: list[tuple[int, int, int, int]]
    sigma_blur: float
    alpha: float
    beta: float
    phases: dict[str, np.ndarray]  # pore / graphite / si truth masks
    extra: dict = field(default_factory=dict)


def microstructure(shape: tuple[int, int], rng: np.random.Generator, porosity: float = 0.08, si_frac: float = 0.12,
                   si_ratio: float = 1.8) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Graphite background (1) with random pore discs (0) and Si discs (si_ratio)."""
    h, w = shape
    s = np.ones(shape, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    si = np.zeros(shape, dtype=bool)
    pore = np.zeros(shape, dtype=bool)
    target_si = si_frac * h * w
    while si.sum() < target_si:
        r = rng.uniform(12, 40)
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        si |= (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    target_pore = porosity * h * w
    while pore.sum() < target_pore:
        r = rng.uniform(6, 30)
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        pore |= (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    si &= ~pore
    s[si] = si_ratio
    s[pore] = 0.0
    graphite = ~si & ~pore
    return s, {"pore": pore, "graphite": graphite, "si": si}


def shading_field(shape: tuple[int, int], amp: float = 0.08) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xx / w - 0.5, yy / h - 0.5
    g = 1.0 + amp * (0.8 * x + 0.5 * y - 1.2 * x * x + 0.6 * x * y)
    return (g / g.mean()).astype(np.float32)


def make_phantom(shape: tuple[int, int] = (1024, 2048), seed: int = 0, d: dict[str, float] | None = None,
                 gain: dict[str, float] | None = None, shading_amp: float = 0.08,
                 bands: list[tuple[int, int, float]] | None = None, charging: int = 0, sigma_blur: float = 1.2,
                 alpha: float = 0.004, beta: float = 0.001, clip_low: bool = False, si_ratio: float = 1.8) -> Phantom:
    """Build a 3-detector phantom.  ``bands`` are (start_row, end_row, gain) dips applied to all detectors.

    I = clip(gain * S * G(x,y) * g(y) + D) with Poisson–Gaussian noise var = alpha*S + beta
    (in S units) added before the affine map and 8-bit quantisation.
    """
    rng = np.random.default_rng(seed)
    d = d or {"BSE": 20.0, "Inlens": 10.0, "SE_type": 15.0}
    gain = gain or {"BSE": 60.0, "Inlens": 90.0, "SE_type": 70.0}
    if clip_low:
        d = {k: -2.0 for k in d}  # ~90 % of pore pixels clip at 0 (Batches 1/2 clip ~50 %)
    s_bse, phases = microstructure(shape, rng, si_ratio=si_ratio)
    s = {"BSE": s_bse,
         "Inlens": np.where(phases["pore"], 0.0, np.where(phases["si"], 1.3, 1.0)).astype(np.float32),
         "SE_type": np.where(phases["pore"], 0.0, np.where(phases["si"], 1.4, 1.0)).astype(np.float32)}
    gmap = shading_field(shape, shading_amp)
    row_gain = np.ones(shape[0], dtype=np.float32)
    band_list = []
    for s0, e0, g0 in bands or []:
        row_gain[s0: e0 + 1] = g0
        band_list.append((s0, e0))
    bboxes = []
    raw, truth = {}, {}
    for k in range(charging):
        cy, cx = rng.integers(100, shape[0] - 100), rng.integers(100, shape[1] - 100)
        r = int(rng.uniform(25, 40))
        yy, xx = np.mgrid[0: shape[0], 0: shape[1]]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        s["Inlens"] = np.where(blob, 10.0, s["Inlens"]).astype(np.float32)  # saturates after the affine map
        s["SE_type"] = np.where(blob, 3.0, s["SE_type"]).astype(np.float32)
        bboxes.append((cy - r, cx - r, cy + r, cx + r))
    for det, sd in s.items():
        blurred = ndimage.gaussian_filter(sd, sigma_blur) if sigma_blur > 0 else sd
        truth[det] = blurred
        noisy = blurred + rng.standard_normal(shape, dtype=np.float32) * np.sqrt(alpha * np.maximum(blurred, 0) + beta)
        i = gain[det] * noisy * gmap * row_gain[:, None] + d[det]
        raw[det] = np.clip(np.rint(i), 0, 255).astype(np.uint8)
    return Phantom(raw, truth, d, gain, gmap, row_gain, band_list, bboxes, sigma_blur, alpha, beta, phases,
                   extra={"si_ratio": si_ratio})
