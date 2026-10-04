#!/usr/bin/env python3
"""KPI reliability: within-field (tile) noise vs between-field spread, per KPI.

ICC(1) from the 16-tile table (`outputs/overnight/features/tile_kpis_rich.csv`): the share of a
site-level KPI's variance that is site, not sampling. A KPI with low ICC cannot carry a batch signal at
one field per site whatever the model; the attenuation factor sqrt(ICC) caps the observable effect.

    python scripts/run_kpi_reliability.py -> outputs/reliability/{icc.csv, figures/icc.png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kruskal  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reliability"
N_TILES = 16
MIN_TILES = 8


def icc1(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """One-way random-effects ICC(1) with unequal group sizes; returns (icc, sd_within, sd_between)."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    n = np.array([len(g) for g in groups])
    k = len(groups)
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n.sum() - k)
    n0 = (n.sum() - (n**2).sum() / n.sum()) / (k - 1)
    var_b = max((msb - msw) / n0, 0.0)
    icc = var_b / (var_b + msw) if var_b + msw > 0 else np.nan
    return float(icc), float(np.sqrt(msw)), float(np.sqrt(var_b))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    T = pd.read_csv(ROOT / "outputs" / "overnight" / "features" / "tile_kpis_rich.csv")
    T = T[(T.n_tiles == N_TILES) & (T.batch != "Batch_heldout")]
    S = pd.read_csv(ROOT / "outputs" / "kpis" / "site_kpis.csv")
    kpis = [c for c in T.columns if c.startswith("K")]
    rows = []
    for c in kpis:
        groups = [g[c].dropna().to_numpy() for _, g in T.groupby(["batch", "site"]) if g[c].notna().sum() >= MIN_TILES]
        if len(groups) < 10:
            continue
        icc, sdw, sdb = icc1(groups)
        site_sd_1tile = sdw
        site_sd_16 = sdw / np.sqrt(N_TILES)
        kw = np.nan
        if c in S and S[c].nunique() > 1:
            kw = kruskal(*[g[c].dropna() for _, g in S.groupby("batch")]).pvalue
        bmed = S.groupby("batch")[c].median() if c in S else None
        batch_range = float(bmed.max() - bmed.min()) if bmed is not None else np.nan
        rows.append({"kpi": c, "n_sites": len(groups), "icc_tile": icc,
                     "icc_site_16tiles": icc * N_TILES / (1 + (N_TILES - 1) * icc) if icc == icc else np.nan,
                     "sd_within_tile": site_sd_1tile, "sd_site_mean_of_16": site_sd_16, "sd_between_sites": sdb,
                     "batch_median_range": batch_range, "range_over_sampling_sd": batch_range / site_sd_16 if site_sd_16 > 0 else np.nan,
                     "kw_p_sites": kw})
    R = pd.DataFrame(rows).sort_values("icc_tile", ascending=False)
    R.to_csv(OUT / "icc.csv", index=False)
    print(R.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(R))
    ax.barh(y, R.icc_tile, color="tab:gray", label="ICC(1), one tile (1/16 field)")
    ax.barh(y, R.icc_site_16tiles, color="none", edgecolor="k", label="ICC of the 16-tile site mean (Spearman-Brown)")
    ax.set_yticks(y)
    ax.set_yticklabels(R.kpi, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of variance that is the site, not where you looked")
    ax.axvline(0.5, ls=":", c="k", lw=0.8)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("KPI reliability from 16 tiles x 31 fields", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "icc.png", dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
