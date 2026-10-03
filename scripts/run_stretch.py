#!/usr/bin/env python3
"""Stretch KPI run (pmdb.kpis.stretch, catalogue v2 S01-S04) over all labelled sites, with 4 x-tiles per site.

    python scripts/run_stretch.py --jobs 8                 # outputs/stretch/{site,tile}_stretch.csv
    python scripts/run_stretch.py --heldout                # outputs/stretch/heldout_*.csv (read-only use)
    python scripts/run_stretch.py --perturb 0.05           # segmenter anchor fractions +-0.05, robustness check

Sites are loaded exactly as scripts/run_kpis.py does (half resolution, un-normalised BSE, v0r1
segmenter) so the S-columns sit on the same masks as the K/D/A KPIs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pmdb.segment as segment_mod  # noqa: E402
from pmdb.kpis import STRETCH_REGISTRY, catalogue_columns  # noqa: E402
from pmdb.io import list_sites, load_site  # noqa: E402
from pmdb.kpis.common import KpiContext, TooFewObjects  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "stretch"
HELDOUT_DATA = REPO_ROOT / "data_heldout"
HELDOUT_CACHE = REPO_ROOT / "cache_heldout"


def compute_stretch(masks, nm_per_px: float, key: str) -> dict:
    ctx = KpiContext(masks=masks, nm_per_px=nm_per_px, seed_key=key)
    site_cols, _ = catalogue_columns("v2")
    values: dict = {}
    for kpi_id, cols in site_cols.items():
        try:
            out = STRETCH_REGISTRY[kpi_id].site_fn(ctx).values
        except TooFewObjects as e:
            out = {c: float("nan") for c in cols}
            values[f"{kpi_id}_nan_reason"] = str(e)
        values.update({c: float(out[c]) for c in cols})
    return values


def process_site(args: tuple) -> dict:
    batch, site, heldout, perturb = args
    t0 = time.time()
    kw = {"data_root": HELDOUT_DATA, "cache_root": HELDOUT_CACHE} if heldout else {}
    raw = load_site(batch, site, resolution="half", normalise="none", **kw)
    params = None
    if perturb:
        params = {"si_anchor_frac": segment_mod.V0_PARAMS["si_anchor_frac"] + perturb,
                  "pore_anchor_frac": segment_mod.V0_PARAMS["pore_anchor_frac"] + perturb}
    bse = np.asarray(raw.image[..., 0], dtype=np.float64)
    masks = segment_mod.segment_bse(bse, raw.nm_per_px, params=params)
    meta = {"batch": batch, "site": site, "se_detector": raw.se_detector,
            "segmenter_version": masks.version, "kpi_version": "v2"}
    site_row = {**meta, **compute_stretch(masks, raw.nm_per_px, f"{batch}/{site}")}
    tiles: list[dict] = []  # S01-S04 are site-level only (catalogue per_tile = no)
    return {"site_row": site_row, "tiles": tiles, "seconds": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--heldout", action="store_true", help="run the held-out sites (never used for fitting)")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="add this to the segmenter's si/pore anchor fractions (robustness check)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    manifest = list_sites(HELDOUT_DATA) if a.heldout else list_sites()
    jobs = [(r.batch, r.site, a.heldout, a.perturb) for r in manifest.itertuples()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(process_site, jobs))

    tag = ("heldout_" if a.heldout else "") + (f"perturb{a.perturb:+.2f}_" if a.perturb else "")
    a.out.mkdir(parents=True, exist_ok=True)
    site_df = pd.DataFrame([r["site_row"] for r in results])
    tile_df = pd.DataFrame([t for r in results for t in r["tiles"]])
    site_df.to_csv(a.out / f"{tag}site_stretch.csv", index=False)
    tile_df.to_csv(a.out / f"{tag}tile_stretch.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    log = {"git_commit": commit, "kpi_version": "v2",
           "segmenter_version": segment_mod.SEGMENTER_VERSION, "kpis": sorted(catalogue_columns("v2")[0]),
           "perturb": a.perturb, "heldout": a.heldout, "n_sites": len(site_df), "n_tiles": len(tile_df),
           "elapsed_seconds": round(time.time() - t0, 1),
           "seconds_per_site": {f"{r['site_row']['batch']}/{r['site_row']['site']}": round(r["seconds"], 1) for r in results}}
    (a.out / f"{tag}run_log.json").write_text(json.dumps(log, indent=2))
    print(f"{len(site_df)} sites, {len(tile_df)} tiles -> {a.out} in {log['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
