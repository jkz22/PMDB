"""Phase 0.3: characterise the teammate normalisation (pmdb.io.load_site normalise='percentile').

For every image/channel, fit normalised = a * raw + b on pixels the normaliser did not clip (0 < y < 1),
and test alternatives (polynomial degree 3, monotone lookup). Report whether parameters are per image,
per batch or global by comparing their spread at each level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pmdb.io import load_site
from src.v2.common import OUT, manifest


def main():
    rows = []
    for b, s, se in manifest()[["batch", "site", "se_detector"]].itertuples(index=False, name=None):
        raw = load_site(b, s, resolution="half", normalise="none").image.astype(np.float64)
        nrm = load_site(b, s, resolution="half", normalise="percentile").image.astype(np.float64)
        for c, det in enumerate(["BSE", "Inlens", se]):
            x, y = raw[..., c].ravel(), nrm[..., c].ravel()
            keep = (y > 0) & (y < 1)
            A = np.vstack([x[keep], np.ones(keep.sum())]).T
            (a, bb), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
            res = y[keep] - A @ [a, bb]
            ss = ((y[keep] - y[keep].mean()) ** 2).sum()
            p3 = np.polyfit(x[keep], y[keep], 3)
            res3 = y[keep] - np.polyval(p3, x[keep])
            # one-to-one check: does each raw grey level map to exactly one normalised value?
            lut_spread = pd.Series(y).groupby(x).agg(lambda v: v.max() - v.min()).max()
            p05, p995 = np.percentile(x, [0.5, 99.5])
            rows.append(dict(batch=b, site=s, channel=c, detector=det, view=["BSE", "Inlens", "SE_type"][c],
                             gain=a, offset=bb, r2_linear=1 - (res ** 2).sum() / ss, max_abs_res_linear=np.abs(res).max(),
                             r2_cubic=1 - (res3 ** 2).sum() / ss, lut_max_spread=lut_spread,
                             raw_p0_5=p05, raw_p99_5=p995, gain_pred=1 / (p995 - p05), offset_pred=-p05 / (p995 - p05),
                             frac_clipped_low=(y <= 0).mean(), frac_clipped_high=(y >= 1).mean()))
    df = pd.DataFrame(rows)
    d = OUT / "normalisation"; d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "raw_to_normalised_fit.csv", index=False)
    lvl = []
    for v, g in df.groupby("view"):
        lvl.append(dict(view=v, gain_cv_global=g.gain.std() / g.gain.mean(),
                        gain_cv_within_batch_mean=g.groupby("batch").gain.agg(lambda z: z.std() / z.mean()).mean(),
                        offset_sd_global=g.offset.std(),
                        offset_sd_within_batch_mean=g.groupby("batch").offset.std().mean(),
                        min_r2_linear=g.r2_linear.min(), max_lut_spread=g.lut_max_spread.max()))
    pd.DataFrame(lvl).to_csv(d / "parameter_levels.csv", index=False)
    print(pd.DataFrame(lvl).round(4).to_string())
    print(df.groupby(["view", "batch"])[["gain", "offset", "frac_clipped_low", "frac_clipped_high"]].mean().round(4))


if __name__ == "__main__":
    main()
