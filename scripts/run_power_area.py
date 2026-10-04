#!/usr/bin/env python3
"""Sectioning-aware power: fields per batch *and* area per field.

Variance components from `outputs/reliability/icc.csv` (one-way ANOVA on 16 tiles x 31 fields):
sd_within_tile (one tile), sd_between_sites (true field-to-field SD). A field imaged at k x the current area
has site-mean sampling SD = sd_within_tile / sqrt(16 k), so a field value has variance
sd_between^2 + sd_within^2 / (16 k). Power for a two-sample test (alpha 0.05, normal approximation) to detect
the largest observed batch-median difference (`batch_median_range`) with n fields per group at k x area.
Writes outputs/reliability/power_area.csv and figures/power_area.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reliability"
N_TILES = 16
NS = (5, 7, 10, 15, 20, 30, 50, 100)
KS = (0.5, 1, 2, 4, 8, 1e9)
ALPHA = 0.05


def power(d: float, sd_between: float, sd_within: float, n: int, k: float) -> float:
    var_field = sd_between**2 + sd_within**2 / (N_TILES * k)
    se = np.sqrt(2.0 * var_field / n)
    z = norm.ppf(1 - ALPHA / 2)
    return float(1 - norm.cdf(z - d / se) + norm.cdf(-z - d / se))


def main() -> int:
    icc = pd.read_csv(OUT / "icc.csv").dropna(subset=["sd_within_tile", "sd_between_sites", "batch_median_range"])
    rows = []
    for r in icc.itertuples():
        for k in KS:
            for n in NS:
                rows.append(dict(kpi=r.kpi, area_factor=k, n_per_group=n,
                                 power=power(r.batch_median_range, r.sd_between_sites, r.sd_within_tile, n, k)))
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "power_area_grid.csv", index=False)

    def need_n(kpi: str, k: float) -> str:
        g = P[(P.kpi == kpi) & (P.area_factor == k)]
        ok = g[g.power >= 0.8]
        return str(int(ok.n_per_group.min())) if len(ok) else f">{max(NS)}"

    S = pd.DataFrame({"kpi": icc.kpi,
                      "icc_tile": icc.icc_tile.round(2),
                      "sampling_share_of_field_var_now": (icc.sd_within_tile**2 / N_TILES / (icc.sd_between_sites**2 + icc.sd_within_tile**2 / N_TILES)).round(2),
                      "power_n7_now": [round(power(r.batch_median_range, r.sd_between_sites, r.sd_within_tile, 7, 1), 2) for r in icc.itertuples()],
                      "power_n7_4x_area": [round(power(r.batch_median_range, r.sd_between_sites, r.sd_within_tile, 7, 4), 2) for r in icc.itertuples()],
                      "power_n7_infinite_area": [round(power(r.batch_median_range, r.sd_between_sites, r.sd_within_tile, 7, 1e9), 2) for r in icc.itertuples()],
                      "n_for_80pct_half_area": [need_n(k, 0.5) for k in icc.kpi],
                      "n_for_80pct_now": [need_n(k, 1) for k in icc.kpi],
                      "n_for_80pct_4x_area": [need_n(k, 4) for k in icc.kpi],
                      "n_for_80pct_infinite_area": [need_n(k, 1e9) for k in icc.kpi]})
    S = S.sort_values("power_n7_now", ascending=False)
    S.to_csv(OUT / "power_area.csv", index=False)
    print(S.to_string(index=False))

    show = [k for k in ("K15_si_graphite_contact_frac", "K01_si_frac_adm", "K02_si_density_per_1000um2", "K03_ecd_d50_um") if k in set(P.kpi)]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.6), sharey=True)
    for ax, kpi in zip(np.atleast_1d(axes), show):
        M = P[(P.kpi == kpi) & (P.area_factor < 1e8)].pivot(index="n_per_group", columns="area_factor", values="power")
        im = ax.imshow(M.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(M.shape[1]), [f"{c:g}x" for c in M.columns])
        ax.set_yticks(range(M.shape[0]), M.index)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M.iat[i, j] < 0.6 else "k")
        ax.set_title(kpi.split("_")[0] + " " + " ".join(kpi.split("_")[1:3]), fontsize=9)
        ax.set_xlabel("area per field (x current)")
    np.atleast_1d(axes)[0].set_ylabel("fields per batch")
    fig.colorbar(im, ax=axes, label="power to detect the largest observed batch difference", shrink=0.8)
    fig.suptitle("More fields or bigger fields? Power at alpha 0.05 (variance components from 16 tiles x 31 fields)", fontsize=10)
    fig.savefig(OUT / "figures" / "power_area.png", dpi=150, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
