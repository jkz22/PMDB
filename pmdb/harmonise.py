"""Per-site grey-level harmonisation of the imaging black-level / gain artefact.

Several Batch_3 sites were imaged with a raised black level (and a slightly lower gain) on
all three detectors (see ``docs/harmonisation.md``). The artefact is spatially uniform and
per-image, so it is modelled as a per-site, per-channel monotone grey-level map stored as a
256-entry look-up table (LUT). LUTs are fitted once from *anchors* (grey levels of physically
meaningful features) and applied at load time by :func:`pmdb.io.load_site`; they are valid at
both resolutions because they act on grey levels, not pixels.

Anchors per site and channel (all estimated from the BSE phase masks of
:func:`pmdb.segment.segment_bse`, whose rule is affine-invariant):

* ``black``: 0.5th percentile of the channel (instrument black level; 0 when the image clips).
* ``pore`` / ``graphite`` / ``si``: median grey level inside the pore / graphite / Si masks.

Methods (``METHODS``):

* ``none``      identity.
* ``offset``    black-level subtraction only: ``y = x - black + ref.black`` (gain untouched).
* ``affine2``   line through (black -> ref.black) and (graphite -> ref.graphite). Si brightness
                is *not* anchored, so Si/graphite contrast (a material property) is preserved.
* ``affine3``   least-squares line through the black, graphite and Si anchors.
* ``histmatch`` full CDF matching of the channel histogram to the reference histogram
                (aggressive control: also equalises phase fractions, i.e. material information).
* ``hybrid``    ``affine2`` on BSE and SE_type (compositional channels, affine artefact) and
                ``histmatch`` on Inlens (topographic channel whose distortion is not affine).

The reference is the median anchor (and mean CDF) over all labelled sites, so the majority of
clean sites defines the target grey scale and the affected minority is pulled onto it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pmdb.segment import segment_bse

METHODS: tuple[str, ...] = ("none", "offset", "affine2", "affine3", "histmatch", "hybrid")
CHANNELS: tuple[str, str, str] = ("BSE", "Inlens", "SE_type")
ANCHOR_NAMES: tuple[str, ...] = ("black", "pore", "graphite", "si")
HARMONISED_DIRNAME = "harmonised"
N_GREY = 256


@dataclass(frozen=True)
class Reference:
    """Target grey scale: per-channel anchors and per-channel reference CDF."""

    anchors: dict[str, dict[str, float]]  # channel -> anchor -> value
    cdf: np.ndarray  # (3, 256) monotone in [0, 1]
    n_sites: int

    def to_json(self) -> str:
        return json.dumps(
            {"anchors": self.anchors, "cdf": self.cdf.tolist(), "n_sites": self.n_sites}, indent=1
        )

    @classmethod
    def from_json(cls, text: str) -> "Reference":
        d = json.loads(text)
        return cls(anchors=d["anchors"], cdf=np.asarray(d["cdf"], dtype=np.float64), n_sites=int(d["n_sites"]))


def estimate_anchors(raw_uint8: np.ndarray, nm_per_px: float) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """Return ``(anchors, hist)`` for one un-normalised ``(H, W, 3)`` uint8 site array.

    ``anchors[channel][name]`` for ``name`` in :data:`ANCHOR_NAMES`; ``hist`` is ``(3, 256)``
    normalised to sum 1 per channel.
    """
    if raw_uint8.ndim != 3 or raw_uint8.shape[2] != 3 or raw_uint8.dtype != np.uint8:
        raise ValueError(f"Expected (H, W, 3) uint8 array, got {raw_uint8.shape} {raw_uint8.dtype}")
    masks = segment_bse(raw_uint8[..., 0].astype(np.float64), nm_per_px)
    anchors: dict[str, dict[str, float]] = {}
    hist = np.zeros((3, N_GREY), dtype=np.float64)
    for c, name in enumerate(CHANNELS):
        ch = raw_uint8[..., c]
        anchors[name] = {
            "black": float(np.percentile(ch, 0.5)),
            "pore": float(np.median(ch[masks.pore])) if masks.pore.any() else float("nan"),
            "graphite": float(np.median(ch[masks.graphite])) if masks.graphite.any() else float("nan"),
            "si": float(np.median(ch[masks.si])) if masks.si.any() else float("nan"),
        }
        h = np.bincount(ch.ravel(), minlength=N_GREY).astype(np.float64)
        hist[c] = h / h.sum()
    return anchors, hist


def anchors_to_row(batch: str, site: str, anchors: dict[str, dict[str, float]]) -> dict[str, float | str]:
    row: dict[str, float | str] = {"batch": batch, "site": site}
    for ch in CHANNELS:
        for a in ANCHOR_NAMES:
            row[f"{ch}_{a}"] = anchors[ch][a]
    return row


def row_to_anchors(row: pd.Series | dict) -> dict[str, dict[str, float]]:
    return {ch: {a: float(row[f"{ch}_{a}"]) for a in ANCHOR_NAMES} for ch in CHANNELS}


def build_reference(anchor_rows: pd.DataFrame, hists: np.ndarray) -> Reference:
    """Median anchors and mean CDF over the given sites (``hists`` is ``(n_sites, 3, 256)``)."""
    if len(anchor_rows) == 0:
        raise ValueError("Need at least one site to build a reference.")
    anchors = {
        ch: {a: float(np.nanmedian(anchor_rows[f"{ch}_{a}"].to_numpy(dtype=float))) for a in ANCHOR_NAMES}
        for ch in CHANNELS
    }
    cdfs = np.cumsum(np.asarray(hists, dtype=np.float64), axis=-1)
    cdf = cdfs.mean(axis=0)
    cdf = np.minimum.accumulate(cdf[:, ::-1], axis=-1)[:, ::-1]  # enforce monotone, keep in [0, 1]
    cdf[:, -1] = 1.0
    return Reference(anchors=anchors, cdf=cdf, n_sites=int(len(anchor_rows)))


def _affine_lut(slope: float, intercept: float) -> np.ndarray:
    x = np.arange(N_GREY, dtype=np.float64)
    return np.clip(np.round(slope * x + intercept), 0, N_GREY - 1).astype(np.uint8)


def _fit_line(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if len(xa) < 2 or np.ptp(xa) < 1e-9:
        return 1.0, 0.0
    slope, intercept = np.polyfit(xa, ya, 1)
    return float(slope), float(intercept)


def _histmatch_lut(hist: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    # Map each grey level to the reference level with the same cumulative probability.
    # Use the midpoint of the source bin so clipped spikes (e.g. 2% at 0) map to a dark run, not to 0 alone.
    mid = cdf - hist / 2.0
    levels = np.interp(mid, ref_cdf, np.arange(N_GREY, dtype=np.float64))
    return np.clip(np.round(levels), 0, N_GREY - 1).astype(np.uint8)


def fit_lut(method: str, anchors: dict[str, dict[str, float]], hist: np.ndarray, ref: Reference) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit a ``(3, 256)`` uint8 LUT for one site. Also returns per-channel fit parameters."""
    if method not in METHODS:
        raise ValueError(f"Unknown harmonisation method '{method}'. Choose from {METHODS}.")
    luts = np.zeros((3, N_GREY), dtype=np.uint8)
    params = []
    for c, ch in enumerate(CHANNELS):
        a, r = anchors[ch], ref.anchors[ch]
        slope, intercept = 1.0, 0.0
        if method == "none":
            pass
        elif method == "offset":
            slope, intercept = 1.0, r["black"] - a["black"]
        elif method == "affine2" or (method == "hybrid" and ch != "Inlens"):
            slope, intercept = _fit_line([a["black"], a["graphite"]], [r["black"], r["graphite"]])
        elif method == "affine3":
            slope, intercept = _fit_line(
                [a["black"], a["graphite"], a["si"]], [r["black"], r["graphite"], r["si"]]
            )
        if method == "histmatch" or (method == "hybrid" and ch == "Inlens"):
            # Inlens is topographic: its grey-level response in the affected session is not
            # affine in the material anchors, so the full CDF is matched instead.
            luts[c] = _histmatch_lut(hist[c], ref.cdf[c])
            slope, intercept = float("nan"), float("nan")
        else:
            luts[c] = _affine_lut(slope, intercept)
        params.append({"channel": ch, "slope": slope, "intercept": intercept})
    return luts, pd.DataFrame(params)


def apply_lut(raw_uint8: np.ndarray, luts: np.ndarray) -> np.ndarray:
    """Apply a ``(3, 256)`` LUT channel-wise to an ``(H, W, 3)`` uint8 array."""
    if luts.shape != (3, N_GREY):
        raise ValueError(f"LUT must have shape (3, 256), got {luts.shape}")
    out = np.empty_like(raw_uint8)
    for c in range(3):
        out[..., c] = luts[c][raw_uint8[..., c]]
    return out


# --------------------------------------------------------------------------------------
# On-disk layout:  <cache_root>/harmonised/<method>/{luts.npz, params.csv, reference.json}
# --------------------------------------------------------------------------------------


def method_dir(cache_root: Path, method: str) -> Path:
    return Path(cache_root) / HARMONISED_DIRNAME / method


def lut_key(batch: str, site: str) -> str:
    return f"{batch}__{site}"


def save_luts(cache_root: Path, method: str, luts: dict[str, np.ndarray], params: pd.DataFrame, ref: Reference) -> Path:
    d = method_dir(cache_root, method)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "luts.npz", **luts)
    params.to_csv(d / "params.csv", index=False)
    (d / "reference.json").write_text(ref.to_json())
    return d


def load_lut(cache_root: Path, method: str, batch: str, site: str) -> np.ndarray:
    if method == "none":
        return np.tile(np.arange(N_GREY, dtype=np.uint8), (3, 1))
    path = method_dir(cache_root, method) / "luts.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No harmonisation LUTs for method '{method}' at {path}. "
            f"Run 'python scripts/build_harmonised.py' to fit them."
        )
    with np.load(path) as z:
        key = lut_key(batch, site)
        if key not in z.files:
            raise KeyError(f"Site {batch}/{site} has no '{method}' LUT in {path}.")
        return z[key]


def load_reference(cache_root: Path, method: str) -> Reference:
    return Reference.from_json((method_dir(cache_root, method) / "reference.json").read_text())
