"""Small-multiples figure: one panel per KPI, one dot per image grouped
by batch, baseline mean line and 95% interval band behind. Feeds the
report and slides. Reads only results/kpis.parquet.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import ROOT, load_config

# categorical slots 1-3 (validated all-pairs), fixed per batch identity
BATCH_COLOURS = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a"}
SURFACE, GRID, MUTED, INK = "#fcfcfb", "#e1e0d9", "#898781", "#0b0b0b"


def main() -> None:
    cfg = load_config()
    baseline = cfg["data"]["baseline_batch"]
    df = pd.read_parquet(ROOT / "results" / "kpis.parquet")
    kpis = sorted(df.kpi.unique())
    batches = sorted(df.batch.unique())
    rng = np.random.default_rng(0)

    ncols = 4
    nrows = -(-len(kpis) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows),
                             facecolor=SURFACE)
    for ax, kpi in zip(axes.ravel(), kpis):
        sub = df[df.kpi == kpi]
        base = sub[sub.batch == baseline].value
        lo, hi = np.percentile(base, [2.5, 97.5])
        ax.axhspan(lo, hi, color=GRID, alpha=0.55, zorder=0)
        ax.axhline(base.mean(), color=MUTED, lw=1, zorder=1)
        for i, batch in enumerate(batches):
            v = sub[sub.batch == batch].value
            x = i + rng.uniform(-0.13, 0.13, len(v))
            ax.scatter(x, v, s=26, color=BATCH_COLOURS[batch], zorder=2,
                       edgecolors=SURFACE, linewidths=0.8)
        ax.set_title(kpi, fontsize=9.5, color=INK, pad=4)
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels([b.replace("Batch_", "B") for b in batches],
                           fontsize=8.5, color=MUTED)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED, length=0)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes.ravel()[len(kpis):]:
        ax.set_visible(False)

    handles = [plt.Line2D([], [], marker="o", ls="", color=c,
                          label=b + (" (baseline)" if b == baseline else ""))
               for b, c in BATCH_COLOURS.items() if b in batches]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
               ncol=len(handles))
    fig.suptitle("KPIs per image: batch clouds vs baseline mean and 95% interval",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / "results" / "kpis_small_multiples.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
