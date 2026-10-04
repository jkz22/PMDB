"""Materialise the physical-clean route (PR #27, pmdb.clean) as half-res uint8 arrays aligned to
cache/half, so the Modal pipeline can read them like the PR #16 LUT caches.

cache/clean/<kind>/half/<Batch>__<site>.npz  (kind in norm|harm)
  image : uint8 (H,W,3) = round(clip(z, 0, 255/scale) * scale), scale = 100 (BSE, SE_type) / 40 (Inlens)
          -> pore 0, graphite 100 (Inlens 40), Si free (~170-255 on BSE). Inlens needs the smaller
          scale because its topographic highlights reach z ~ 4-6 (19-23 % of pixels > 2.55 on some sites).
  valid : bool  (H,W,3) = clean.valid_for_kpis(mask) per detector
The full-res clean arrays keep the raw TIFF width; the 4-px side margins are cropped before the
mask-aware 2x2 mean so (y, x) crop coordinates match cache/half exactly. Values are a fixed
physical scale (no per-image normalisation); masked pixels keep their real values here and are
zeroed by src.v2.common.load_half_raw.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pmdb import clean
from pmdb.io import downsample_clean

from src.v2.common import CACHE, REPO

DETS = ("BSE", "Inlens", "SE_type")
Z_SCALE = {"BSE": 100.0, "Inlens": 40.0, "SE_type": 100.0}


def materialise_site(site_dir: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    ims, vals = [], []
    for d in DETS:
        z, m = clean.read_site(site_dir, d, kind)
        W = z.shape[1]
        z, m = downsample_clean(z[:, 4:W - 4], m[:, 4:W - 4])
        sc = Z_SCALE[d]
        ims.append(np.round(np.clip(z, 0, 255.0 / sc) * sc).astype(np.uint8))
        vals.append(clean.valid_for_kpis(m))
    return np.stack(ims, -1), np.stack(vals, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", default=str(REPO / "outputs" / "clean"))
    ap.add_argument("--cache-root", default=str(CACHE.parent))
    ap.add_argument("--kinds", nargs="+", default=["norm", "harm"])
    ap.add_argument("--only-kind", default=None)
    ap.add_argument("--tag", default="clean", help="cache sub-directory: clean (PR #27/#28 build) or clean2 (PR #33 rebuild)")
    a = ap.parse_args()
    root = Path(a.clean_root)
    for kind in a.kinds:
        out = Path(a.cache_root) / a.tag / kind / "half"
        out.mkdir(parents=True, exist_ok=True)
        for site_dir in sorted(p for p in root.glob("Batch_*/*") if (p / f"BSE_{kind}.tif").exists()):
            f = out / f"{site_dir.parent.name}__{site_dir.name}.npz"
            im, va = materialise_site(site_dir, kind)
            tmp = f.with_suffix(".tmp.npz")
            np.savez(tmp, image=im, valid=va)
            tmp.rename(f)
            print(kind, f.name, im.shape, f"valid={va.mean():.3f}", f"graphite_med={np.median(im[..., 0][va[..., 0]]):.0f}")


if __name__ == "__main__":
    main()
