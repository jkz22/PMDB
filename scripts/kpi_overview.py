"""Plain QC plots of the KPI outputs (spec 002): inspection only, no statistics.

Writes to outputs/kpis/figures/:
  - verdict_kpis_by_batch.png: strip plot of every verdict-tier site column, by batch
  - k10_curves_by_batch.png:   K10 CV vs window size, one line per site, coloured by batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb.kpis import catalogue_columns, load_catalogue  # noqa: E402


def batch_colours(batches: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {b: cmap(i) for i, b in enumerate(sorted(batches))}


def strip_plot(site_df: pd.DataFrame, out: Path) -> Path:
    site_cols, _ = catalogue_columns()
    verdict = [r["id"] for r in load_catalogue() if r["version"] == "v1" and r["tier"] == "verdict"]
    cols = [c for k in verdict for c in site_cols[k]]
    batches = sorted(site_df["batch"].unique())
    colours = batch_colours(batches)
    ncol = 6
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, c in zip(axes.ravel(), cols):
        for i, b in enumerate(batches):
            v = site_df.loc[site_df["batch"] == b, c].to_numpy(dtype=float)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=colours[b], alpha=0.85)
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels([b.replace("Batch_", "B") for b in batches], fontsize=8)
        ax.set_title(c, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Verdict-tier KPIs by batch (site level; inspection only, no statistics)", fontsize=11)
    fig.tight_layout()
    path = out / "verdict_kpis_by_batch.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def k10_plot(curves: pd.DataFrame, out: Path) -> Path:
    k10 = curves[curves["kpi_id"] == "K10"]
    colours = batch_colours(list(k10["batch"].unique()))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    seen = set()
    for (b, s), g in k10.groupby(["batch", "site"]):
        g = g.sort_values("x")
        ax.plot(g["x"], g["value"], "-o", ms=3, lw=1, color=colours[b], alpha=0.7,
                label=b if b not in seen else None)
        seen.add(b)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("window w (um)")
    ax.set_ylabel("CV of local Si fraction")
    ax.set_title("K10 scale-of-segregation curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "k10_curves_by_batch.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kpi-dir", default=str(ROOT / "outputs" / "kpis"))
    args = ap.parse_args(argv)
    kdir = Path(args.kpi_dir)
    out = kdir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    site_df = pd.read_csv(kdir / "site_kpis.csv")
    curves = pd.read_csv(kdir / "curves.csv")
    for p in (strip_plot(site_df, out), k10_plot(curves, out)):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
