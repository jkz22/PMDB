"""Shared helpers for the v2 representation study (paths, sites, groups, crop grids)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "outputs" / "v2"
CACHE = Path(os.environ.get("PMDB_CACHE", REPO / "cache")) / "half"

CROP = 256          # model crop size at 50 nm/px
STRIDE_TRAIN = 128  # overlapping training windows
NM_HALF = 50.0
NM_FULL = 25.0
DETECTORS = ("BSE", "Inlens", "SE_type")
SEED = 0


def manifest() -> pd.DataFrame:
    m = pd.read_csv(CACHE / "manifest.csv")[["batch", "site", "se_detector", "height", "width"]]
    m["group_id"] = m["batch"] + "/" + m["site"]
    m["group_idx"] = np.arange(len(m))
    return m


HARM_METHODS = ("hybrid", "affine2", "histmatch", "offset", "affine3")  # PR #16 per-site LUTs, materialised uint8
EXT_METHODS = ("nyul", "basic")  # PR #37 imported pipelines (N4ITK + Nyul-Udupa; BaSiC), cache/harmonised_ext/<m>/half uint8
CLEAN_METHODS = ("clean_norm", "clean_harm",  # PR #27 physical route (pmdb.clean), see src/v2/materialise_clean.py
                 "clean2_norm", "clean2_harm")  # same route rebuilt on PR #33 (edge-connected collector, contributor-only half-res mask bits)


def harm_method(h) -> str:
    """Normalise the ``harmonise`` config value: False/None/'none' -> 'none'; True -> 'gmm' (this
    pipeline's GMM-mean linear map); otherwise a pmdb.harmonise method name (see HARM_METHODS)."""
    if h is None or h is False or (isinstance(h, float) and np.isnan(h)) or str(h) in ("none", "False", "nan"):
        return "none"
    if h is True or str(h) == "True":
        return "gmm"
    assert h in HARM_METHODS or h in CLEAN_METHODS or h in EXT_METHODS, h
    return str(h)


def load_half_raw(batch: str, site: str, harm: str = "none") -> np.ndarray:
    """uint8 (H,W,3) at 50 nm/px. ``harm`` in HARM_METHODS reads cache/harmonised/<harm>/half/
    (byte-identical to pmdb.io.load_site(..., normalise='none', harmonise=harm))."""
    if harm in HARM_METHODS:
        return np.load(CACHE.parent / "harmonised" / harm / "half" / f"{batch}__{site}.npz")["image"]
    if harm in EXT_METHODS:  # stored uint8: nyul = standard scale, basic = raw - b_i + 64 (pmdb.harmonise_ext.store_scale)
        return np.load(CACHE.parent / "harmonised_ext" / harm / "half" / f"{batch}__{site}.npz")["image"]
    if harm in CLEAN_METHODS:
        return load_half_clean(batch, site, harm)[0]
    return np.load(CACHE / f"{batch}__{site}.npz")["image"]


def load_half_clean(batch: str, site: str, harm: str) -> tuple[np.ndarray, np.ndarray]:
    """Physical-clean arrays (PR #27) materialised by src.v2.materialise_clean: uint8 (H,W,3) with
    pore 0 / graphite 100 (Inlens 40) / Si free, fixed physical scale (z x 100, Inlens z x 40), invalid pixels zeroed, plus the
    per-detector validity mask (bool, H,W,3)."""
    tag, kind = harm.split("_", 1)  # clean_norm -> cache/clean/norm, clean2_harm -> cache/clean2/harm
    root = CACHE.parent / tag / kind / "half"
    z = np.load(root / f"{batch}__{site}.npz")
    im, valid = z["image"].copy(), z["valid"]
    im[~valid] = 0
    return im, valid


def load_full_raw(batch: str, site: str) -> np.ndarray:
    from pmdb.io import load_site
    return load_site(batch, site, resolution="full", normalise="none").image


def grid(h: int, w: int, size: int = CROP, stride: int | None = None) -> list[tuple[int, int]]:
    stride = stride or size
    return [(y, x) for y in range(0, h - size + 1, stride) for x in range(0, w - size + 1, stride)]


def git_hash(path: Path = REPO) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]

# Approved baseline batch (user, 2026-10-03: Batch_3 is the pure baseline; Batch_1/2 contain outliers).
BASELINE = os.environ.get("PMDB_BASELINE", "Batch_3")
