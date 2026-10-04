"""Composite route ``spechyb`` = hybrid LUT (grey level / gain) followed by the fitted ``spectrum`` filter (texture).

Uses the per-detector SpectrumModel fitted by scripts/build_harmonise_shift.py (labelled sites only) and applies it
to the hybrid-harmonised arrays, so both the DC (black level, gain) and the radial amplitude spectrum (blur, noise
texture) are moved to the labelled reference. Written with the 2-px _ext border so src.v2.common.trim_ext applies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pmdb import clean as C
from pmdb import harmonise_shift as S
from src.v2.common import EXT_TRIM, REPO, manifest

DETS = ("BSE", "Inlens", "SE_type")
HELDOUT = ("3e122cbj", "fn0mhxef", "xrv9xvzb")


def main() -> None:
    models = S.load_models(REPO / "cache" / "harmonised_shift", "spectrum")
    jobs = [(b, s, REPO / "cache", REPO / "cache") for b, s in zip(manifest().batch, manifest().group_id.str.split("/").str[1])]
    jobs += [("Batch_heldout", s, REPO / "cache_heldout", REPO / "cache_heldout") for s in HELDOUT]
    rows = []
    for b, s, root, _ in jobs:
        hyb = np.load(root / "harmonised" / "hybrid" / "half" / f"{b}__{s}.npz")["image"]  # (H, W, 3) uint8, cache grid
        z = np.load(root / "harmonised_shift" / "spectrum" / "half" / f"{b}__{s}.npz")
        mask = z["mask"][:, EXT_TRIM:-EXT_TRIM]
        out = np.zeros_like(hyb)
        for d, name in enumerate(DETS):
            valid = C.valid_for_stats(mask[..., d])
            img = hyb[..., d].astype(np.float32)
            f, h = S.spectrum_apply(img, valid, models[name])
            out[..., d] = S.to_uint8(f)
            rows.append(dict(batch=b, site=s, det=name, h_lo=round(float(h[2]), 3), h_hi=round(float(h[-1]), 3), mean_abs_change=round(float(np.abs(f - img)[valid].mean()), 3)))
        od = root / "harmonised_shift" / "spechyb" / "half"; od.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(od / f"{b}__{s}.npz", image=np.pad(out, ((0, 0), (EXT_TRIM, EXT_TRIM), (0, 0)), mode="edge"), mask=z["mask"])
        print(b, s, "written", flush=True)
    pd.DataFrame(rows).to_csv(REPO / "cache" / "harmonised_shift" / "spechyb" / "params.csv", index=False)


if __name__ == "__main__":
    main()
