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


def load_half_raw(batch: str, site: str) -> np.ndarray:
    return np.load(CACHE / f"{batch}__{site}.npz")["image"]


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
