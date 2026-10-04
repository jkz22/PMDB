"""Run the shift (non-intensity) harmonisation routes (``pmdb.harmonise_shift``: spectrum, fda) on all sites.

Inputs are the raw TIFFs plus the exclusion masks written by ``scripts/build_clean.py`` (run that
first). Reference spectra are fitted on the labelled sites only and applied to the held-out sites.

Outputs (per method ``m``):
    cache/harmonised_shift/<m>/model.json (+ model.npz)  reference radial spectrum / FDA amplitude
    cache/harmonised_shift/<m>/half/<batch>__<site>.npz  image (H, W, 3) uint8, mask (H, W, 3) uint16
    cache/harmonised_shift/<m>/half/manifest.csv         repo-relative paths
    cache/harmonised_shift/<m>/params.csv                per-site/detector filter H(f) summary
    cache_heldout/harmonised_shift/<m>/...                same for the held-out sites
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

from pmdb import clean as C
from pmdb import harmonise_shift as S
from pmdb.io import get_clean_root, list_clean_sites

REPORT_BINS = (4, 13, 26, 38, 51, 63)   # bins of H(f) recorded in params.csv (≈0.035, 0.1, 0.2, 0.3, 0.4, 0.5 c/px)


def _portable(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p.resolve())


def _load_all(sites: pd.DataFrame, clean_root: Path | None):
    data = {}
    t0 = time.time()
    for b, s in zip(sites.batch, sites.site):
        for d in S.DETECTORS:
            data[(b, s, d)] = S.load_half_raw(b, s, d, clean_root)
    print(f"  loaded {len(sites)} sites × 3 detectors in {time.time() - t0:.0f}s", flush=True)
    return data


def _lut_all(sites: pd.DataFrame, data) -> dict:
    """hybrid_spectrum input: the same grey after the per-site hybrid LUT (masks unchanged)."""
    return {(b, s, d): (S.hybrid_lut_grey(b, s, d, data[(b, s, d)][0]), data[(b, s, d)][1])
            for b, s in zip(sites.batch, sites.site) for d in S.DETECTORS}


def _fit(method: str, sites: pd.DataFrame, data, fda_sites: list[str]) -> dict:
    models = {}
    for d in S.DETECTORS:
        t0 = time.time()
        keys = [f"{b}/{s}" for b, s in zip(sites.batch, sites.site)]
        imgs = [data[(b, s, d)][0] for b, s in zip(sites.batch, sites.site)]
        vals = [C.valid_for_stats(data[(b, s, d)][1]) for b, s in zip(sites.batch, sites.site)]
        if method in S.SPECTRUM_METHODS:
            models[d] = S.spectrum_fit(imgs, vals, keys)
            print(f"  {method}/{d}: reference amplitude at f=0.1/0.3/0.5 c/px = "
                  f"{models[d].reference[13]:.1f}/{models[d].reference[38]:.1f}/{models[d].reference[63]:.1f} ({time.time() - t0:.0f}s)", flush=True)
        else:
            sel = [i for i, s in enumerate(sites.site) if s in fda_sites]
            models[d] = S.fda_fit([imgs[i] for i in sel], [vals[i] for i in sel], [sites.site.iloc[i] for i in sel])
            print(f"  fda/{d}: reference amplitude from {len(sel)} sites, beta={models[d].beta} ({time.time() - t0:.0f}s)", flush=True)
    return models


def _apply(method: str, sites: pd.DataFrame, data, models: dict, root: Path) -> pd.DataFrame:
    rows, paths = [], []
    f = S.bin_centres()
    for b, s in zip(sites.batch, sites.site):
        images, masks = {}, {}
        row: dict = {"batch": b, "site": s}
        for d in S.DETECTORS:
            img, mask = data[(b, s, d)]
            valid = C.valid_for_stats(mask)
            if method in S.SPECTRUM_METHODS:
                out, h = S.spectrum_apply(img, valid, models[d])
                for i in REPORT_BINS:
                    row[f"{d}_H_f{f[i]:.2f}"] = float(h[i])
                row[f"{d}_H_min"], row[f"{d}_H_max"] = float(h.min()), float(h.max())
            else:
                out, bpx = S.fda_apply(img, valid, models[d])
                row[f"{d}_window_px"] = bpx
            row[f"{d}_frac_clipped_store"] = float((out > 255).mean() + (out < 0).mean())
            row[f"{d}_mean_abs_change"] = float(np.abs(out - img)[valid].mean())
            images[d], masks[d] = out, mask
        paths.append(_portable(S.write_site(root, method, b, s, images, masks)))
        rows.append(row)
        print(f"  {method} {b}/{s} written", flush=True)
    params = pd.DataFrame(rows)
    manifest = sites[["batch", "site"]].copy()
    manifest["path"] = paths
    manifest["nm_per_px"] = S.HALF_NM_PER_PX
    manifest.to_csv(S.method_dir(root, method) / "half" / "manifest.csv", index=False)
    params.to_csv(S.method_dir(root, method) / "params.csv", index=False)
    return params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(S.METHODS), choices=S.METHODS)
    ap.add_argument("--clean-root", default=None)
    ap.add_argument("--clean-heldout-root", default=None)
    ap.add_argument("--out", default=str(S.DEFAULT_SHIFT_ROOT))
    ap.add_argument("--out-heldout", default=str(S.DEFAULT_SHIFT_HELDOUT_ROOT))
    ap.add_argument("--fda-reference", nargs="+", default=None,
                    help="labelled sites whose mean amplitude is the FDA target (default: all labelled sites)")
    ap.add_argument("--no-heldout", action="store_true")
    args = ap.parse_args()

    sites = list_clean_sites(args.clean_root)
    data = _load_all(sites, args.clean_root)
    fda_sites = args.fda_reference or list(sites.site)
    heldout = None
    if not args.no_heldout:
        hroot = get_clean_root(args.clean_heldout_root, heldout=True)
        if (hroot / "summary.csv").exists():
            heldout = list_clean_sites(args.clean_heldout_root, heldout=True)
            hdata = _load_all(heldout, args.clean_heldout_root)
        else:
            print(f"  no held-out clean outputs under {hroot}; skipping", flush=True)
    for m in args.methods:
        print(f"== {m}", flush=True)
        dm, hm = (data, hdata if heldout is not None else None)
        if m == "hybrid_spectrum":
            dm = _lut_all(sites, data)
            hm = _lut_all(heldout, hdata) if heldout is not None else None
        models = _fit(m, sites, dm, fda_sites)
        S.save_models(Path(args.out), m, models)
        _apply(m, sites, dm, models, Path(args.out))
        if heldout is not None:
            S.save_models(Path(args.out_heldout), m, models)
            _apply(m, heldout, hm, models, Path(args.out_heldout))
    print("done")


if __name__ == "__main__":
    main()
