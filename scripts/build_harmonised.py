"""Fit per-site harmonisation LUTs for every method and (optionally) materialise harmonised caches.

Outputs (per method ``m``):
    <cache_root>/harmonised/<m>/luts.npz        one (3, 256) uint8 LUT per site, key '<batch>__<site>'
    <cache_root>/harmonised/<m>/params.csv      per-site, per-channel slope/intercept (NaN for histmatch)
    <cache_root>/harmonised/<m>/reference.json  the reference anchors and CDF used
    <cache_root>/harmonised/anchors.csv         raw per-site anchors (black / pore / graphite / Si per channel)
    <cache_root>/harmonised/<m>/half/*.npz      only with --materialise: harmonised copies of cache/half

The reference is always fitted on the labelled sites in ``--cache-root``; ``--heldout-cache-root``
sites get LUTs against that same reference (they are never part of it).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from pmdb import harmonise as H

HALF_NM_PER_PX = 50.0


def _sites(cache_root: Path) -> pd.DataFrame:
    manifest = cache_root / "half" / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; build the half-res cache first.")
    return pd.read_csv(manifest)


def _collect_anchors(cache_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    rows, hists = [], []
    for _, r in _sites(cache_root).iterrows():
        arr = np.load(cache_root / "half" / f"{r.batch}__{r.site}.npz")["image"]
        anchors, hist = H.estimate_anchors(arr, HALF_NM_PER_PX)
        rows.append(H.anchors_to_row(r.batch, r.site, anchors))
        hists.append(hist)
        print(f"  anchors {r.batch}/{r.site}: BSE black={anchors['BSE']['black']:.0f} "
              f"graphite={anchors['BSE']['graphite']:.0f} si={anchors['BSE']['si']:.0f}", flush=True)
    return pd.DataFrame(rows), np.stack(hists)


def _fit_all(cache_root: Path, anchors: pd.DataFrame, hists: np.ndarray, ref: H.Reference, methods: list[str]) -> None:
    for m in methods:
        luts: dict[str, np.ndarray] = {}
        params = []
        for i, (_, r) in enumerate(anchors.iterrows()):
            lut, p = H.fit_lut(m, H.row_to_anchors(r), hists[i], ref)
            luts[H.lut_key(r.batch, r.site)] = lut
            p.insert(0, "site", r.site)
            p.insert(0, "batch", r.batch)
            params.append(p)
        H.save_luts(cache_root, m, luts, pd.concat(params, ignore_index=True), ref)


def _materialise(cache_root: Path, method: str) -> None:
    src = cache_root / "half"
    dst = H.method_dir(cache_root, method) / "half"
    dst.mkdir(parents=True, exist_ok=True)
    manifest = _sites(cache_root)
    for _, r in manifest.iterrows():
        arr = np.load(src / f"{r.batch}__{r.site}.npz")["image"]
        out = H.apply_lut(arr, H.load_lut(cache_root, method, r.batch, r.site))
        np.savez_compressed(dst / f"{r.batch}__{r.site}.npz", image=out)
    manifest.to_csv(dst / "manifest.csv", index=False)
    print(f"  materialised {len(manifest)} sites -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", default=str(REPO_ROOT / "cache"))
    ap.add_argument("--heldout-cache-root", default=str(REPO_ROOT / "cache_heldout"))
    ap.add_argument("--methods", nargs="+", default=[m for m in H.METHODS if m != "none"])
    ap.add_argument("--materialise", nargs="*", default=[], metavar="METHOD",
                    help="also write harmonised npz copies of cache/half for these methods")
    args = ap.parse_args()

    t0 = time.time()
    cache_root = Path(args.cache_root)
    print(f"Estimating anchors for labelled sites in {cache_root} ...")
    anchors, hists = _collect_anchors(cache_root)
    (cache_root / H.HARMONISED_DIRNAME).mkdir(parents=True, exist_ok=True)
    anchors.to_csv(cache_root / H.HARMONISED_DIRNAME / "anchors.csv", index=False)
    ref = H.build_reference(anchors, hists)
    print("Reference anchors:", {ch: {k: round(v, 1) for k, v in a.items()} for ch, a in ref.anchors.items()})
    _fit_all(cache_root, anchors, hists, ref, args.methods)

    heldout = Path(args.heldout_cache_root)
    if heldout.exists() and (heldout / "half" / "manifest.csv").exists():
        print(f"Fitting held-out sites in {heldout} against the labelled reference ...")
        h_anchors, h_hists = _collect_anchors(heldout)
        (heldout / H.HARMONISED_DIRNAME).mkdir(parents=True, exist_ok=True)
        h_anchors.to_csv(heldout / H.HARMONISED_DIRNAME / "anchors.csv", index=False)
        _fit_all(heldout, h_anchors, h_hists, ref, args.methods)

    for m in args.materialise:
        _materialise(cache_root, m)
        if heldout.exists() and (heldout / "half" / "manifest.csv").exists():
            _materialise(heldout, m)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
