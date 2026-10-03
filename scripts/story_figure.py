#!/usr/bin/env python3
"""One-page story figure: imaging artefact -> composition (same) -> arrangement (differs) -> function
(consequence), with the three held-out sites marked. Reads committed outputs only.

    python scripts/story_figure.py  -> docs/figures/story.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
COL = {"Batch_1": "#1f77b4", "Batch_2": "#2ca02c", "Batch_3": "#d62728"}
HELD = {"3e122cbj": "^", "fn0mhxef": "s", "xrv9xvzb": "D"}
rng = np.random.default_rng(0)


def strip(ax, df, col, held=None, label=None):
    for i, b in enumerate(BATCHES):
        y = df.loc[df.batch == b, col].to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, y.size), y, s=22, color=COL[b], alpha=0.8, zorder=3)
        ax.hlines(np.median(y), i - 0.3, i + 0.3, color=COL[b], lw=2.5, zorder=4)
    if held is not None:
        for site, mk in HELD.items():
            v = held.loc[held.site == site, col]
            if len(v):
                ax.scatter([3.0], v.to_numpy(), marker=mk, s=70, facecolor="none", edgecolor="k", lw=1.5, zorder=5)
    ax.set_xticks([0, 1, 2, 3] if held is not None else [0, 1, 2])
    ax.set_xticklabels(["B1", "B2", "B3", "held-out"] if held is not None else ["B1", "B2", "B3"])
    ax.set_ylabel(label or col)
    ax.grid(axis="y", alpha=0.3)


def main() -> int:
    hm = pd.read_csv(O / "harmonisation" / "site_metrics.csv")
    kp = pd.read_csv(O / "kpis" / "site_kpis.csv")
    kh = pd.read_csv(O / "heldout" / "kpis" / "site_kpis.csv")
    fp = pd.read_csv(O / "fingerprint" / "features.csv")
    fh = pd.read_csv(O / "fingerprint" / "heldout_features.csv")
    fu = pd.read_csv(O / "functional" / "site_functional.csv")
    fuh = pd.read_csv(O / "functional" / "heldout_site_functional.csv")
    nl = pd.read_csv(O / "functional" / "null_relocation.csv")

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    ax = axes[0]
    for meth, mk, lab in (("none", "o", "raw"), ("hybrid", "x", "harmonised")):
        d = hm[hm.method == meth]
        for i, b in enumerate(BATCHES):
            y = d.loc[d.batch == b, "BSE_p1"].to_numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, y.size) + (0.0 if meth == "none" else 0.0), y, marker=mk,
                       s=28, color=COL[b], alpha=0.85, label=lab if i == 2 else None)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["B1", "B2", "B3"])
    ax.set_ylabel("BSE black level p1 (DN)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("1. Imaging artefact\nBatch 3 offset/gain -> harmonise first", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    strip(axes[1], kp, "K01_si_frac_adm", kh, "K01 Si area fraction (admissible)")
    p_k01 = stats.kruskal(*[kp.loc[kp.batch == b, "K01_si_frac_adm"] for b in BATCHES]).pvalue
    axes[1].set_title(f"2. Composition: the same\nK01 Kruskal-Wallis p = {p_k01:.2f}", fontsize=10)

    strip(axes[2], fp, "si_depth_mid_dip", fh, "Si mid-depth dip (fingerprint)")
    axes[2].set_title("3. Arrangement: differs\nLOSO 0.68 vs 0.55 majority, perm p = 0.003", fontsize=10)

    ax = axes[3]
    strip(ax, fu, "F02_soc100_into_graphite", fuh, "SOC-1 Si growth landing on graphite")
    for i, b in enumerate(BATCHES):
        y = nl.loc[nl.batch == b, "null_mean_into_graphite"].to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, y.size), y, s=14, color="grey", alpha=0.6, zorder=2,
                   label="random-relocation null" if i == 0 else None)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("4. Consequence on lithiation\nconstrained share: B3 highest (p = 0.009 @ SOC 0.5)\nall fields above null; excess is batch-free", fontsize=10)

    ax = axes[4]
    pw = pd.read_csv(O / "functional" / "power.csv")
    pw = pw.sort_values("power_n7", ascending=False).drop_duplicates("column").head(8)
    ax.barh(pw["column"].str.slice(0, 24) + "  (" + pw["contrast"].astype(str) + ")", pw["power_n7"], color="#555")
    ax.axvline(0.8, color="k", ls="--", lw=1)
    ax.set_xlabel("power with 7 fields per batch")
    ax.set_title("5. The limit is fields, not models\n(dashed: power 0.8)", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(0, 1)

    fig.suptitle("PMDB in one line: correct the imaging, composition is equal, arrangement differs, and that arrangement sets "
                 "how Si swells against graphite  (open markers: held-out sites ^ 3e122cbj, s fn0mhxef, D xrv9xvzb)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ROOT / "docs" / "figures" / "story.png", dpi=140)
    return 0


if __name__ == "__main__":
    sys.exit(main())
