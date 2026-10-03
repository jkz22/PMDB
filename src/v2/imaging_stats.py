"""Phase 0.2: imaging stats on RAW (uint8, border-cropped, native 25 nm/px) data, per image and per crop.

Crops are the non-overlapping 256-px evaluation grid at 50 nm/px, i.e. 512-px windows at native resolution.
Sharpness = var(Laplacian) / (p99 - p1)^2; the raw Laplacian variance is also kept (it tracks contrast).
Noise = Immerkaer (1996) fast noise sigma estimate.
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, laplace
from sklearn.mixture import GaussianMixture

from src.v2.common import CROP, OUT, grid, load_full_raw, manifest

IM_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def stats(a: np.ndarray, rng: np.random.Generator, n_gmm: int) -> dict:
    a = a.astype(np.float32)
    p1, p99 = np.percentile(a, [1, 99])
    dr = float(p99 - p1)
    lap = laplace(a)
    lapvar = float(lap[1:-1, 1:-1].var())
    h, w = a.shape
    noise = float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(convolve(a, IM_K)[1:-1, 1:-1]).sum())
    v = a.ravel()
    sub = v[rng.choice(v.size, min(n_gmm, v.size), replace=False)].reshape(-1, 1)
    sub = sub + rng.uniform(-0.5, 0.5, sub.shape)  # dequantise uint8
    g = GaussianMixture(3, random_state=0, n_init=2).fit(sub)
    o = np.argsort(g.means_.ravel())
    d = dict(p1=float(p1), p99=float(p99), dyn_range=dr, mean=float(a.mean()), std=float(a.std()),
             noise_sigma=noise, lap_var_raw=lapvar, sharpness=lapvar / max(dr, 1.0) ** 2)
    for k, idx in enumerate(o):
        d[f"gmm_mu{k}"] = float(g.means_.ravel()[idx])
        d[f"gmm_sd{k}"] = float(np.sqrt(g.covariances_.ravel()[idx]))
        d[f"gmm_w{k}"] = float(g.weights_[idx])
    return d


def site_stats(args):
    batch, site, se, gid = args
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(gid.encode()).digest()[:8], 'little'))
    img = load_full_raw(batch, site)
    names = ["BSE", "Inlens", se]
    hh, wh = img.shape[0] // 2, img.shape[1] // 2
    per_img, per_crop = [], []
    for c, det in enumerate(names):
        ch = img[..., c]
        base = dict(batch=batch, site=site, group_id=gid, channel=c, detector=det,
                    view=["BSE", "Inlens", "SE_type"][c])
        per_img.append({**base, **stats(ch, rng, 200_000)})
        for ci, (y, x) in enumerate(grid(hh, wh, CROP)):
            win = ch[2 * y:2 * y + 2 * CROP, 2 * x:2 * x + 2 * CROP]
            per_crop.append({**base, "crop_idx": ci, "y_half": y, "x_half": x, **stats(win, rng, 20_000)})
    return per_img, per_crop


def main():
    m = manifest()
    args = list(m[["batch", "site", "se_detector", "group_id"]].itertuples(index=False, name=None))
    res = []
    with ProcessPoolExecutor(int(os.environ.get('NWORK', 6))) as ex:
        for i, r in enumerate(ex.map(site_stats, args), 1):
            res.append(r); print(f'{i}/{len(args)} sites', flush=True)
    d = OUT / "imaging_stats"
    d.mkdir(parents=True, exist_ok=True)
    pi = pd.DataFrame([r for a, _ in res for r in a]); pc = pd.DataFrame([r for _, b in res for r in b])
    pi.to_csv(d / "per_image.csv", index=False); pc.to_csv(d / "per_crop.csv", index=False)
    print(pi.groupby(["view", "batch"])[["p1", "p99", "dyn_range", "noise_sigma", "sharpness", "lap_var_raw"]].mean().round(4))
    print("crops:", len(pc))


if __name__ == "__main__":
    main()
