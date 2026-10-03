"""Phase 0 -> Phase 1: forward-model nuisance ranges measured from raw imaging stats.

Gain a: per-image dynamic range / median dynamic range of that view (5th-95th pct over images).
Offset b: (p1 - median p1) / 255 (5th-95th pct, widened to include 0).
Added noise: 0 .. spread (95th-5th pct) of per-image noise sigma, converted to 50 nm/px
(2x2 mean halves white-noise sigma; assumption) and [0,1] units.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.v2.augment import ForwardRanges
from src.v2.common import OUT


def main():
    p = pd.read_csv(OUT / "imaging_stats" / "per_image.csv")
    g = p.groupby("view")
    a = p.dyn_range / g.dyn_range.transform("median")
    b = (p.p1 - g.p1.transform("median")) / 255.0
    n = p.noise_sigma / 255.0 / 2.0
    lo, hi = np.percentile(a, [5, 95]), np.percentile(b, [5, 95])
    fr = ForwardRanges(a_min=float(lo[0]), a_max=float(lo[1]), b_min=float(min(hi[0], 0)), b_max=float(max(hi[1], 0)),
                       noise_min=0.0, noise_max=float(np.percentile(n, 95) - np.percentile(n, 5)))
    d = fr.__dict__
    (OUT / "imaging_stats" / "forward_ranges.json").write_text(json.dumps(d, indent=2))
    print(json.dumps(d, indent=2))
    cols = ["p1", "p99", "dyn_range", "noise_sigma", "sharpness", "gmm_mu0", "gmm_mu1", "gmm_mu2", "gmm_w0", "gmm_w1", "gmm_w2"]
    s = p.groupby(["view", "batch"])[cols].mean().round(3)
    s.to_csv(OUT / "imaging_stats" / "by_view_batch.csv")
    with pd.option_context("display.width", 250):
        print(s)


if __name__ == "__main__":
    main()
