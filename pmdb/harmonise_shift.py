"""Non-intensity ("shift") harmonisation routes: spectrum matching and Fourier Domain Adaptation.

Both methods leave the pixel *values* of the material essentially alone and act on the spatial
frequency content – the part of the acquisition fingerprint (focus/blur, noise texture) that
survives every intensity method (LUTs, N4 + Nyúl, BaSiC). Both are published methods from
other imaging domains and are applied here as published; neither is SEM-validated.

``spectrum`` – kernel-conversion / NPS-homogenisation filter
    Ohkubo et al., Med. Phys. 38 (2011) 3915, doi:10.1118/1.3590166 (MTF-ratio filter to convert
    a CT image reconstructed with one kernel into another); Mackin et al., Tomography 5 (2019) 61,
    doi:10.18383/j.tom.2018.00045 (noise-power-spectrum homogenisation filters). A radially
    averaged amplitude spectrum ``A_s(f)`` is measured per site/detector on valid interior tiles;
    the reference ``A_ref(f)`` is the median over the labelled sites; the site is filtered with
    ``H_s(f) = A_ref(f) / A_s(f)`` (smoothed, clamped). DC is kept, so the mean grey level and the
    per-pixel material contrast are untouched – only blur/noise texture move to the reference.

``fda`` – Fourier Domain Adaptation
    Yang & Soatto, CVPR 2020, "FDA: Fourier Domain Adaptation for Semantic Segmentation"
    (github.com/YanchaoYang/FDA). The low-frequency square window (half-width ``beta`` × image
    size) of the source amplitude spectrum is replaced by the reference amplitude; phase is
    kept. Reference amplitude = mean amplitude spectrum of the labelled reference sites. This is
    the published "style" baseline; it moves shading / black level (low frequencies) and not the
    high-frequency texture, so it is kept as a control for ``spectrum``.

Fitting (reference spectra) uses the labelled sites only; held-out sites are transform-only.
Inputs are the half-res raw grey + clean masks exactly as for ``pmdb.harmonise_ext``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from pmdb import clean as C
from pmdb.harmonise_ext import HALF_NM_PER_PX, REPO_ROOT, load_half_raw  # noqa: F401 (re-exported)

METHODS: tuple[str, ...] = ("spectrum", "fda")
DETECTORS = C.DETECTORS
DEFAULT_SHIFT_ROOT = REPO_ROOT / "cache" / "harmonised_shift"
DEFAULT_SHIFT_HELDOUT_ROOT = REPO_ROOT / "cache_heldout" / "harmonised_shift"

TILE = 512                 # spectra are averaged over TILE×TILE interior tiles (Hann window)
N_BINS = 64                # radial frequency bins between 0 and Nyquist (0.5 cycles/px)
SPECTRUM_CLAMP = (0.33, 3.0)   # bounds on H(f) = A_ref / A_s
SPECTRUM_SMOOTH = 3        # bins; moving-average smoothing of H(f)
FDA_BETA = 0.01            # Yang & Soatto's default low-frequency window half-width (fraction of size)
FDA_REFERENCE_SITES = ("vc2whyaq", "2tp5sryb", "x77cy643", "iv6g2oq0")  # fallback; overridden by build


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def fill_invalid(img: np.ndarray, valid: np.ndarray, size: int = 15) -> np.ndarray:
    """Replace excluded pixels by a local mean of the valid ones (then the global median), so
    masked saturation/charging cannot leak into the FFT."""
    f = np.asarray(img, dtype=np.float32).copy()
    if valid.all():
        return f
    glob = float(np.median(f[valid])) if valid.any() else 0.0
    # normalised convolution: local mean of the valid neighbours (falls back to the global median)
    w = gaussian_filter(valid.astype(np.float32), size / 3)
    loc = gaussian_filter(np.where(valid, f, 0.0).astype(np.float32), size / 3) / np.maximum(w, 1e-6)
    f[~valid] = np.where(w[~valid] > 0.05, loc[~valid], glob)
    return f


def _radial_bins(shape: tuple[int, int]) -> np.ndarray:
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    r = np.hypot(fy, fx)
    return np.minimum((r / 0.5 * N_BINS).astype(int), N_BINS - 1)


def _tiles(img: np.ndarray, valid: np.ndarray, tile: int = TILE, min_valid: float = 0.9):
    H, W = img.shape
    ys = range(0, H - tile + 1, tile)
    xs = range(0, W - tile + 1, tile)
    for y in ys:
        for x in xs:
            v = valid[y:y + tile, x:x + tile]
            if v.mean() >= min_valid:
                yield img[y:y + tile, x:x + tile]


def radial_amplitude(img: np.ndarray, valid: np.ndarray, tile: int = TILE) -> np.ndarray:
    """Radially averaged amplitude spectrum (N_BINS,) of the mean-removed, Hann-windowed valid tiles."""
    filled = fill_invalid(img, valid)
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    bins = _radial_bins((tile, tile))
    acc = np.zeros(N_BINS)
    n = 0
    for t in _tiles(filled, valid, tile):
        a = np.abs(np.fft.fft2((t - t.mean()) * win))
        acc += np.bincount(bins.ravel(), a.ravel(), minlength=N_BINS)
        n += 1
    if n == 0:  # short field: fall back to the whole image
        t = filled - filled.mean()
        a = np.abs(np.fft.fft2(t * np.outer(np.hanning(t.shape[0]), np.hanning(t.shape[1]))))
        bins = _radial_bins(t.shape)
        acc = np.bincount(bins.ravel(), a.ravel(), minlength=N_BINS)
        n = 1
    counts = np.bincount(bins.ravel(), minlength=N_BINS)
    return acc / np.maximum(counts, 1) / n


def bin_centres() -> np.ndarray:
    return (np.arange(N_BINS) + 0.5) / N_BINS * 0.5


# ----------------------------------------------------------------------------------------------
# spectrum: A_ref(f) / A_s(f) filter
# ----------------------------------------------------------------------------------------------
@dataclass
class SpectrumModel:
    reference: np.ndarray          # (N_BINS,) reference radial amplitude
    sites: dict[str, list[float]] = field(default_factory=dict)  # labelled per-site spectra (record)

    def to_json(self) -> dict:
        return {"reference": self.reference.tolist(), "sites": self.sites, "n_bins": N_BINS, "tile": TILE,
                "clamp": list(SPECTRUM_CLAMP), "smooth": SPECTRUM_SMOOTH}

    @classmethod
    def from_json(cls, d: dict) -> "SpectrumModel":
        return cls(np.asarray(d["reference"], dtype=np.float64), d.get("sites", {}))


def spectrum_fit(images: list[np.ndarray], valids: list[np.ndarray], keys: list[str] | None = None) -> SpectrumModel:
    """Reference = per-bin median of the labelled sites' radial amplitude spectra (log domain)."""
    specs = [radial_amplitude(i, v) for i, v in zip(images, valids)]
    ref = np.exp(np.median(np.log(np.maximum(specs, 1e-9)), axis=0))
    keys = keys or [str(i) for i in range(len(specs))]
    return SpectrumModel(ref, {k: s.tolist() for k, s in zip(keys, specs)})


def spectrum_filter(spec: np.ndarray, model: SpectrumModel) -> np.ndarray:
    """H(f) per radial bin: clamped, smoothed ratio of reference to site amplitude; H(DC) = 1."""
    h = model.reference / np.maximum(spec, 1e-9)
    k = SPECTRUM_SMOOTH
    if k > 1:
        pad = np.pad(h, (k // 2, k - 1 - k // 2), mode="edge")
        h = np.convolve(pad, np.ones(k) / k, mode="valid")
    h = np.clip(h, *SPECTRUM_CLAMP)
    h[0] = 1.0
    return h


def spectrum_apply(img: np.ndarray, valid: np.ndarray, model: SpectrumModel) -> tuple[np.ndarray, np.ndarray]:
    """Return the filtered image (float32, same grey scale) and the applied H(f) per bin."""
    spec = radial_amplitude(img, valid)
    h = spectrum_filter(spec, model)
    filled = fill_invalid(img, valid)
    mu = float(filled.mean())
    F = np.fft.fft2(filled - mu)
    H2 = h[_radial_bins(filled.shape)]
    out = np.real(np.fft.ifft2(F * H2)).astype(np.float32) + mu
    out[~valid] = np.asarray(img, dtype=np.float32)[~valid]
    return out, h


# ----------------------------------------------------------------------------------------------
# fda: Fourier Domain Adaptation (Yang & Soatto 2020), amplitude of the low-frequency window swapped
# ----------------------------------------------------------------------------------------------
@dataclass
class FdaModel:
    amplitude: np.ndarray      # (h, w) mean reference amplitude spectrum (fftshifted, full size)
    beta: float = FDA_BETA
    reference_sites: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"beta": self.beta, "reference_sites": self.reference_sites, "shape": list(self.amplitude.shape)}


def _resize_spectrum(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample an fftshifted amplitude spectrum to another image shape (zoom about the centre)."""
    if a.shape == tuple(shape):
        return a
    from scipy.ndimage import zoom
    z = zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)
    out = np.zeros(shape, dtype=a.dtype)
    h, w = min(z.shape[0], shape[0]), min(z.shape[1], shape[1])
    out[:h, :w] = z[:h, :w]
    return out


def fda_fit(images: list[np.ndarray], valids: list[np.ndarray], reference_sites: list[str], beta: float = FDA_BETA) -> FdaModel:
    """Mean fftshifted amplitude spectrum of the reference images (each resampled to the first one's shape)."""
    shape = images[0].shape
    acc = np.zeros(shape)
    for img, v in zip(images, valids):
        a = np.fft.fftshift(np.abs(np.fft.fft2(fill_invalid(img, v))))
        acc += _resize_spectrum(a, shape)
    return FdaModel(acc / len(images), beta, list(reference_sites))


def fda_apply(img: np.ndarray, valid: np.ndarray, model: FdaModel) -> tuple[np.ndarray, float]:
    """Yang & Soatto's ``FDA_source_to_target_np``: replace the centred low-frequency window of
    the source amplitude with the target's; keep the phase. Returns (image, window half-width px)."""
    src = fill_invalid(img, valid).astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(src))
    amp, pha = np.abs(F), np.angle(F)
    trg = _resize_spectrum(model.amplitude, src.shape)
    h, w = src.shape
    b = int(np.floor(np.amin(src.shape) * model.beta))
    cy, cx = h // 2, w // 2
    amp[cy - b:cy + b + 1, cx - b:cx + b + 1] = trg[cy - b:cy + b + 1, cx - b:cx + b + 1]
    out = np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha)))).astype(np.float32)
    out[~valid] = np.asarray(img, dtype=np.float32)[~valid]
    return out, float(b)


# ----------------------------------------------------------------------------------------------
# storage (same layout as pmdb.harmonise_ext)
# ----------------------------------------------------------------------------------------------
def method_dir(root: Path, method: str) -> Path:
    return Path(root) / method


def get_shift_root(root: str | Path | None = None, heldout: bool = False) -> Path:
    if root is not None:
        return Path(root)
    return DEFAULT_SHIFT_HELDOUT_ROOT if heldout else DEFAULT_SHIFT_ROOT


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def save_models(root: Path, method: str, models: dict) -> None:
    d = method_dir(root, method)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").write_text(json.dumps({k: m.to_json() for k, m in models.items()}, indent=1))
    if method == "fda":
        np.savez_compressed(d / "model.npz", **{f"{k}__amplitude": m.amplitude.astype(np.float32) for k, m in models.items()})


def load_models(root: Path, method: str) -> dict:
    d = method_dir(root, method)
    meta = json.loads((d / "model.json").read_text())
    if method == "spectrum":
        return {k: SpectrumModel.from_json(v) for k, v in meta.items()}
    z = np.load(d / "model.npz")
    return {k: FdaModel(z[f"{k}__amplitude"].astype(np.float64), v["beta"], v["reference_sites"]) for k, v in meta.items()}


def write_site(root: Path, method: str, batch: str, site: str, images: dict[str, np.ndarray],
               masks: dict[str, np.ndarray]) -> Path:
    d = method_dir(root, method) / "half"
    d.mkdir(parents=True, exist_ok=True)
    img = np.stack([to_uint8(images[k]) for k in DETECTORS], axis=-1)
    msk = np.stack([masks[k] for k in DETECTORS], axis=-1).astype(np.uint16)
    out = d / f"{batch}__{site}.npz"
    np.savez_compressed(out, image=img, mask=msk)
    return out


def load_shift(batch: str, site: str, method: str = "spectrum", root: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load one shift-harmonised site: ``(image (H, W, 3) uint8, mask (H, W, 3) uint16)`` at 50 nm/px.

    Channels BSE, Inlens, SE_type; grey scale is the raw one (no re-scaling). Always restrict
    statistics to ``clean.valid_for_stats(mask[..., c])``.
    """
    if method not in METHODS:
        raise ValueError(f"Invalid method '{method}'. Must be one of {METHODS}.")
    r = get_shift_root(root, heldout=batch.lower().startswith("batch_heldout"))
    p = method_dir(r, method) / "half" / f"{batch}__{site}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `python scripts/build_harmonise_shift.py` (docs/harmonisation_shift.md).")
    z = np.load(p)
    return z["image"], z["mask"]


def list_shift_sites(method: str = "spectrum", root: str | Path | None = None, heldout: bool = False) -> pd.DataFrame:
    p = method_dir(get_shift_root(root, heldout), method) / "half" / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run scripts/build_harmonise_shift.py first.")
    return pd.read_csv(p)
