"""Run the imported harmonisation pipelines (``pmdb.harmonise_ext``: N4 + Nyúl–Udupa, BaSiC) on all sites.

Inputs are the raw TIFFs plus the exclusion masks written by ``scripts/build_clean.py`` (run that
first). Models are fitted on the labelled sites only and applied to the held-out sites.

Outputs (per method ``m``):
    cache/harmonised_ext/<m>/model.json (+ model.npz)   fitted standard scale / flat- and dark-fields
    cache/harmonised_ext/<m>/half/<batch>__<site>.npz   image (H, W, 3) uint8, mask (H, W, 3) uint16
    cache/harmonised_ext/<m>/half/manifest.csv          repo-relative paths
    cache/harmonised_ext/<m>/params.csv                 per-site/detector landmarks, bias range, b_i
    cache_heldout/harmonised_ext/<m>/...                 same for the held-out sites
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
from pmdb import harmonise_ext as X
from pmdb.io import get_clean_root, list_clean_sites


def _portable(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p.resolve())


def _load_all(sites: pd.DataFrame, clean_root: Path | None) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    data = {}
    t0 = time.time()
    for b, s in zip(sites.batch, sites.site):
        for d in X.DETECTORS:
            data[(b, s, d)] = X.load_half_raw(b, s, d, clean_root)
    print(f"  loaded {len(sites)} sites × 3 detectors in {time.time() - t0:.0f}s", flush=True)
    return data


def _fit(method: str, sites: pd.DataFrame, data) -> dict:
    models = {}
    for d in X.DETECTORS:
        t0 = time.time()
        imgs = [data[(b, s, d)][0] for b, s in zip(sites.batch, sites.site)]
        vals = [C.valid_for_stats(data[(b, s, d)][1]) for b, s in zip(sites.batch, sites.site)]
        if method == "nyul":
            n4 = [X.n4_correct(i, v)[0] for i, v in zip(imgs, vals)]
            models[d] = X.nyul_fit(n4, vals)
            print(f"  nyul/{d}: standard scale {np.round(models[d].standard_scale, 1).tolist()} ({time.time() - t0:.0f}s)", flush=True)
        else:
            models[d] = X.basic_fit(imgs, vals)
            m = models[d]
            print(f"  basic/{d}: flatfield {m.flatfield.min():.3f}–{m.flatfield.max():.3f}, "
                  f"darkfield {m.darkfield.min():.2f}–{m.darkfield.max():.2f} ({time.time() - t0:.0f}s)", flush=True)
    return models


def _apply(method: str, sites: pd.DataFrame, data, models: dict, ext_root: Path) -> pd.DataFrame:
    rows, paths = [], []
    for b, s in zip(sites.batch, sites.site):
        images, masks = {}, {}
        row: dict = {"batch": b, "site": s}
        for d in X.DETECTORS:
            img, mask = data[(b, s, d)]
            valid = C.valid_for_stats(mask)
            if method == "nyul":
                n4, bias = X.n4_correct(img, valid)
                out, lm = X.nyul_apply(n4, valid, models[d])
                row[f"{d}_bias_min"], row[f"{d}_bias_max"] = float(bias.min()), float(bias.max())
                for p, v in zip(models[d].percentiles, lm):
                    row[f"{d}_landmark_p{p:g}"] = float(v)
            else:
                out, bl = X.basic_apply(img, valid, models[d])
                row[f"{d}_baseline"] = bl
            row[f"{d}_frac_clipped_store"] = float(((out * X.store_scale(method)[0] + X.store_scale(method)[1]) > 255).mean()
                                                   + ((out * X.store_scale(method)[0] + X.store_scale(method)[1]) < 0).mean())
            images[d], masks[d] = out, mask
        paths.append(_portable(X.write_site(ext_root, method, b, s, images, masks)))
        rows.append(row)
        print(f"  {method} {b}/{s} written", flush=True)
    params = pd.DataFrame(rows)
    manifest = sites[["batch", "site"]].copy()
    manifest["path"] = paths
    manifest["nm_per_px"] = X.HALF_NM_PER_PX
    manifest["store_gain"], manifest["store_offset"] = X.store_scale(method)
    manifest.to_csv(X.method_dir(ext_root, method) / "half" / "manifest.csv", index=False)
    params.to_csv(X.method_dir(ext_root, method) / "params.csv", index=False)
    return params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(X.METHODS), choices=X.METHODS)
    ap.add_argument("--clean-root", default=None, help="labelled clean outputs (default outputs/clean)")
    ap.add_argument("--clean-heldout-root", default=None, help="held-out clean outputs (default outputs/clean_heldout)")
    ap.add_argument("--out", default=str(X.DEFAULT_EXT_ROOT))
    ap.add_argument("--out-heldout", default=str(X.DEFAULT_EXT_HELDOUT_ROOT))
    ap.add_argument("--no-heldout", action="store_true")
    args = ap.parse_args()

    sites = list_clean_sites(args.clean_root)
    data = _load_all(sites, args.clean_root)
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
        models = _fit(m, sites, data)
        X.save_models(Path(args.out), m, models)
        _apply(m, sites, data, models, Path(args.out))
        if heldout is not None:
            X.save_models(Path(args.out_heldout), m, models)
            _apply(m, heldout, hdata, models, Path(args.out_heldout))
    print("done")


if __name__ == "__main__":
    main()
