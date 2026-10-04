"""Imported, literature-standard intensity-harmonisation pipelines applied to PMDB grey levels.

Two published pipelines, taken as implemented in their open-source packages and run unchanged on the
half-resolution grey levels; the only PMDB-specific input is the exclusion mask of :mod:`pmdb.clean`
(border/marker columns, bad scan bands, charging, clipping – "the crops"). No in-house anchors,
LUTs or dark-level/graphite normalisation are used.

``nyul`` – the neuroimaging multi-scanner recipe: N4 bias-field correction (Tustison et al., IEEE
TMI 2010, ``SimpleITK.N4BiasFieldCorrectionImageFilter``) followed by Nyúl–Udupa piecewise-linear
histogram standardisation (Nyúl, Udupa & Zhang, IEEE TMI 2000) as implemented in the
``intensity-normalization`` package (Reinhold et al., SPIE Medical Imaging 2019). The standard
scale (mean landmark positions at the 1, 10, 20 … 90, 99 percentiles) is learned on the labelled
sites only; every image – labelled or held-out – is mapped onto it with its own landmarks.

``basic`` – BaSiC (Peng et al., Nature Communications 2017, ``basicpy``): a flat-field S(x, y),
a dark-field D(x, y) and a per-image baseline b_i are estimated from the whole site collection
(low-rank + sparse model, masked pixels weighted out) and each image is corrected as
``(I − D) / S − b_i`` (BaSiC "time-lapse" baseline-drift mode).

Both operate per detector. Outputs are stored as compressed ``.npz`` (``image`` uint8 after a fixed
affine re-scaling recorded in ``params.json``, ``mask`` uint16 per detector) under
``cache/harmonised_ext/<method>/half/``; see :func:`load_ext`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from pmdb import clean as C
from pmdb.io import REPO_ROOT, downsample_clean, get_clean_root

METHODS: tuple[str, ...] = ("nyul", "basic")
DETECTORS = C.DETECTORS
HALF_NM_PER_PX = 50.0
DEFAULT_EXT_ROOT = REPO_ROOT / "cache" / "harmonised_ext"
DEFAULT_EXT_HELDOUT_ROOT = REPO_ROOT / "cache_heldout" / "harmonised_ext"

# Nyúl & Udupa (2000) default landmark set as used by intensity-normalization: 1 %, 10 % … 90 %, 99 %
NYUL_PARAMS = dict(min_percentile=1.0, max_percentile=99.0, percentile_after_min=10.0,
                   percentile_before_max=90.0, percentile_step=10.0,
                   output_min_value=0.0, output_max_value=255.0)
# N4ITK defaults (Tustison 2010) except for a 4× shrink for speed; 4 fitting levels × 50 iterations
N4_PARAMS = dict(shrink=4, iterations=(50, 50, 50, 50))
# BaSiC: dark-field on, default smoothness; the collection is fitted on a common working grid
BASIC_PARAMS = dict(working_shape=(256, 896), smoothness_flatfield=1.0, smoothness_darkfield=1.0,
                    fitting_mode="approximate", max_reweight_iterations=10)
# BaSiC output is raw − b_i (mean-centred grey); stored as uint8 after adding this offset
BASIC_STORE_OFFSET = 64.0


# ----------------------------------------------------------------------------------------------
# input: half-res raw grey + clean mask, in clean-pipeline coordinates
# ----------------------------------------------------------------------------------------------
def load_half_raw(batch: str, site: str, detector: str, clean_root: str | Path | None = None,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Raw grey levels (float32, mask-aware 2×2 mean) and uint16 mask at 50 nm/px.

    The mask is the one written by ``scripts/build_clean.py``; the raw TIFF path comes from that
    site's ``params.json`` (repo-relative). Coordinates follow :func:`pmdb.io.load_clean` (no 4-px
    column crop; border pixels are flagged ``BIT_BORDER`` instead).
    """
    root = get_clean_root(clean_root, heldout=batch.lower().startswith("batch_heldout"))
    site_dir = Path(root) / batch / site
    params = json.loads((site_dir / "params.json").read_text())
    p = Path(params["paths"][detector])
    if not p.is_absolute():
        p = REPO_ROOT / p
    raw, _ = C.read_rgb_tiff(p)
    mask = tifffile.imread(site_dir / f"{detector}_mask.tif")
    return downsample_clean(raw.astype(np.float32), mask)


# ----------------------------------------------------------------------------------------------
# nyul: N4 + Nyúl–Udupa
# ----------------------------------------------------------------------------------------------
def n4_correct(img: np.ndarray, valid: np.ndarray, shrink: int = N4_PARAMS["shrink"],
               iterations: tuple[int, ...] = N4_PARAMS["iterations"]) -> tuple[np.ndarray, np.ndarray]:
    """N4ITK bias-field correction; returns (corrected, multiplicative bias field).

    N4 works on log intensities, so the image is offset by +1 grey level before and after (0 is a
    legal BSE value). The field is estimated on a ``shrink``× subsampled image from ``valid``
    pixels only and resampled to full size.
    """
    import SimpleITK as sitk

    x = img.astype(np.float32) + 1.0
    im = sitk.GetImageFromArray(x)
    mk = sitk.GetImageFromArray(valid.astype(np.uint8))
    f = sitk.N4BiasFieldCorrectionImageFilter()
    f.SetMaximumNumberOfIterations(list(iterations))
    f.Execute(sitk.Shrink(im, [shrink, shrink]), sitk.Shrink(mk, [shrink, shrink]))
    bias = np.exp(sitk.GetArrayFromImage(f.GetLogBiasFieldAsImage(im))).astype(np.float32)
    return (x / bias - 1.0).astype(np.float32), bias


@dataclass
class NyulModel:
    percentiles: np.ndarray
    standard_scale: np.ndarray

    def to_json(self) -> dict:
        return {"percentiles": self.percentiles.tolist(), "standard_scale": self.standard_scale.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "NyulModel":
        return cls(np.asarray(d["percentiles"], float), np.asarray(d["standard_scale"], float))


def _nyul_normalizer(model: NyulModel | None = None):
    from intensity_normalization.normalize.nyul import NyulNormalize

    ny = NyulNormalize(**NYUL_PARAMS)
    if model is not None:
        ny.standard_scale = model.standard_scale
        ny._percentiles = model.percentiles
    return ny


def nyul_fit(images: list[np.ndarray], valids: list[np.ndarray]) -> NyulModel:
    """Learn the Nyúl–Udupa standard scale from (N4-corrected) labelled images."""
    ny = _nyul_normalizer()
    ny._fit(images, [v.astype(np.uint8) for v in valids])
    return NyulModel(np.asarray(ny.percentiles, float), np.asarray(ny.standard_scale, float))


def nyul_apply(img: np.ndarray, valid: np.ndarray, model: NyulModel) -> tuple[np.ndarray, np.ndarray]:
    """Map one image onto the standard scale with its own landmarks; returns (image, landmarks)."""
    ny = _nyul_normalizer(model)
    landmarks = ny.get_landmarks(img[valid])
    return np.asarray(ny.normalize_image(img, valid.astype(np.uint8)), dtype=np.float32), landmarks


# ----------------------------------------------------------------------------------------------
# basic: BaSiC flat-field / dark-field / per-image baseline
# ----------------------------------------------------------------------------------------------
@dataclass
class BasicModel:
    flatfield: np.ndarray
    darkfield: np.ndarray
    working_shape: tuple[int, int]
    settings: dict = field(default_factory=dict)


def _resize(a: np.ndarray, shape: tuple[int, int], anti_aliasing: bool) -> np.ndarray:
    from skimage.transform import resize

    return resize(a.astype(np.float32), shape, order=1, anti_aliasing=anti_aliasing,
                  preserve_range=True).astype(np.float32)


def _basic(model: BasicModel | None = None):
    from basicpy import BaSiC

    kw = {k: v for k, v in BASIC_PARAMS.items() if k != "working_shape"}
    b = BaSiC(get_darkfield=True, **kw)
    if model is not None:
        b.flatfield = model.flatfield
        b.darkfield = model.darkfield
    return b


def basic_fit(images: list[np.ndarray], valids: list[np.ndarray]) -> BasicModel:
    """Fit BaSiC flat-/dark-field on the labelled collection resampled to a common working grid."""
    ws = tuple(BASIC_PARAMS["working_shape"])
    stack = np.stack([_resize(x, ws, True) for x in images])
    wt = np.stack([_resize(v.astype(np.float32), ws, True) > 0.99 for v in valids]).astype(np.float32)
    b = _basic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.fit(stack, fitting_weight=wt)
    return BasicModel(np.asarray(b.flatfield, np.float32), np.asarray(b.darkfield, np.float32), ws,
                      {k: (list(v) if isinstance(v, tuple) else v) for k, v in BASIC_PARAMS.items()})


def basic_baseline(img: np.ndarray, valid: np.ndarray, model: BasicModel) -> float:
    """BaSiC per-image baseline b_i (time-lapse mode) on the working grid."""
    ws = model.working_shape
    # basicpy squeezes the per-image baseline vector, so a single image is passed as a pair of copies
    x = np.repeat(_resize(img, ws, True)[None], 2, axis=0)
    w = np.repeat((_resize(valid.astype(np.float32), ws, True) > 0.99).astype(np.float32)[None], 2, axis=0)
    b = _basic(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = b.fit_only_baseline(x, w, model.flatfield, model.darkfield)
    bl = bl.cpu().numpy() if hasattr(bl, "cpu") else np.asarray(bl)
    return float(np.ravel(bl)[0])


def basic_apply(img: np.ndarray, valid: np.ndarray, model: BasicModel) -> tuple[np.ndarray, float]:
    """``(I − D) / S − b_i`` with the fields resampled to the image grid; returns (image, b_i)."""
    S = _resize(model.flatfield, img.shape, False)
    D = _resize(model.darkfield, img.shape, False)
    bl = basic_baseline(img, valid, model)
    return ((img.astype(np.float32) - D) / S - bl).astype(np.float32), bl


# ----------------------------------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------------------------------
def method_dir(ext_root: Path, method: str) -> Path:
    return Path(ext_root) / method


def get_ext_root(ext_root: str | Path | None = None, heldout: bool = False) -> Path:
    if ext_root is not None:
        return Path(ext_root)
    return DEFAULT_EXT_HELDOUT_ROOT if heldout else DEFAULT_EXT_ROOT


def store_scale(method: str) -> tuple[float, float]:
    """Stored uint8 = clip(round(image * gain + offset)); the inverse is applied by :func:`load_ext`."""
    return (1.0, 0.0) if method == "nyul" else (1.0, BASIC_STORE_OFFSET)


def to_uint8(img: np.ndarray, method: str) -> np.ndarray:
    gain, off = store_scale(method)
    return np.clip(np.round(img * gain + off), 0, 255).astype(np.uint8)


def save_models(ext_root: Path, method: str, models: dict[str, NyulModel | BasicModel]) -> None:
    d = method_dir(ext_root, method)
    d.mkdir(parents=True, exist_ok=True)
    if method == "nyul":
        (d / "model.json").write_text(json.dumps({k: m.to_json() for k, m in models.items()}, indent=1))
    else:
        np.savez_compressed(d / "model.npz", **{f"{k}__flatfield": m.flatfield for k, m in models.items()},
                            **{f"{k}__darkfield": m.darkfield for k, m in models.items()})
        (d / "model.json").write_text(json.dumps(
            {k: {"working_shape": list(m.working_shape), "settings": m.settings} for k, m in models.items()},
            indent=1))


def load_models(ext_root: Path, method: str) -> dict[str, NyulModel | BasicModel]:
    d = method_dir(ext_root, method)
    meta = json.loads((d / "model.json").read_text())
    if method == "nyul":
        return {k: NyulModel.from_json(v) for k, v in meta.items()}
    z = np.load(d / "model.npz")
    return {k: BasicModel(z[f"{k}__flatfield"], z[f"{k}__darkfield"], tuple(v["working_shape"]), v["settings"])
            for k, v in meta.items()}


def write_site(ext_root: Path, method: str, batch: str, site: str, images: dict[str, np.ndarray],
               masks: dict[str, np.ndarray]) -> Path:
    d = method_dir(ext_root, method) / "half"
    d.mkdir(parents=True, exist_ok=True)
    img = np.stack([to_uint8(images[k], method) for k in DETECTORS], axis=-1)
    msk = np.stack([masks[k] for k in DETECTORS], axis=-1).astype(np.uint16)
    out = d / f"{batch}__{site}.npz"
    np.savez_compressed(out, image=img, mask=msk)
    return out


def load_ext(batch: str, site: str, method: str = "nyul", ext_root: str | Path | None = None,
             as_uint8: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load one harmonised site: ``(image (H, W, 3), mask (H, W, 3) uint16)`` at 50 nm/px.

    Channels are BSE, Inlens, SE_type. ``as_uint8=False`` undoes the storage re-scaling (for
    ``basic`` the stored values are ``raw − b_i + 64``). Always restrict statistics to
    ``clean.valid_for_stats(mask[..., c])``.
    """
    if method not in METHODS:
        raise ValueError(f"Invalid method '{method}'. Must be one of {METHODS}.")
    root = get_ext_root(ext_root, heldout=batch.lower().startswith("batch_heldout"))
    p = method_dir(root, method) / "half" / f"{batch}__{site}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `python scripts/build_harmonise_ext.py` (docs/harmonisation_ext.md).")
    z = np.load(p)
    img, msk = z["image"], z["mask"]
    if not as_uint8:
        gain, off = store_scale(method)
        img = (img.astype(np.float32) - off) / gain
    return img, msk


def list_ext_sites(method: str = "nyul", ext_root: str | Path | None = None, heldout: bool = False) -> pd.DataFrame:
    p = method_dir(get_ext_root(ext_root, heldout), method) / "half" / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run scripts/build_harmonise_ext.py first.")
    return pd.read_csv(p)
