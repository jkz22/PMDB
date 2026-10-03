"""Leave-one-site-out outlier test on site-level KPIs (spec 003 preliminary).

Reads outputs/kpis/site_kpis.csv and writes to outputs/outliers/:
  - site_scores.csv, kpi_contributions.csv, batch_scores.csv, run_log.json
  - outlier_summary.png: robust distance per site + attribution heatmap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb.outliers import run_pipeline  # noqa: E402

HEAT_COLS = ["comp_loading", "comp_clustering", "comp_localisation", "comp_contact", "idx_process", "idx_artefact"]
HEAT_TITLES = [
    "Si loading",
    "Si clustering",
    "Si localisation",
    "Si–graphite contact (−)",
    "Process diag. (|z|)",
    "Artefact (|z|)",
]


def batch_colours(batches: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {b: cmap(i) for i, b in enumerate(sorted(batches))}


def summary_plot(sc: pd.DataFrame, out: Path) -> Path:
    n = len(sc)
    y = np.arange(n)[::-1]
    colours = batch_colours(list(sc["batch"].unique()))
    labels = [f"{b.replace('Batch_', 'B')} {s}" for b, s in zip(sc["batch"], sc["site"])]
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1]})
    for yi, (_, r) in zip(y, sc.iterrows()):
        ax.errorbar(r["d"], yi, xerr=[[r["d"] - r["d_lo"]], [r["d_hi"] - r["d"]]], fmt="o",
                    color=colours[r["batch"]], ms=5, lw=1)
        if r["q_bh"] < 0.05:
            ax.plot(r["d_hi"] + 0.05 * sc["d_hi"].max(), yi, marker="*", color="k", ms=10)
    ax.axvline(np.sqrt(stats.f.ppf(0.95, 7, 23) * 7 * 31 * 29 / (30 * 23)), ls="--", color="grey")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Distance from the other 30 sites (Mahalanobis, block-PCA KPIs)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=b) for b, c in colours.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", color="k", label="q_BH < 0.05"))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    mat = sc[HEAT_COLS].to_numpy(dtype=float)
    im = axh.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5)
    for i in range(n):
        for j in range(mat.shape[1]):
            axh.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6)
    axh.set_xticks(range(len(HEAT_COLS)))
    axh.set_xticklabels(HEAT_TITLES, rotation=45, ha="right", fontsize=7)
    axh.set_yticks([])
    fig.colorbar(im, ax=axh, fraction=0.04)
    fig.tight_layout()
    path = out / "outlier_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kpis", default=str(ROOT / "outputs" / "kpis" / "site_kpis.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "outliers"))
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kpi_path = Path(args.kpis)
    t0 = time.time()
    df = pd.read_csv(kpi_path)
    sc, contrib, batch_df, log = run_pipeline(df, n_boot=args.n_boot)
    sc.round(4).to_csv(out / "site_scores.csv", index=False)
    contrib.round(4).to_csv(out / "kpi_contributions.csv", index=False)
    batch_df.round(4).to_csv(out / "batch_scores.csv", index=False)
    summary_plot(sc, out)
    log["input"] = {"path": str(kpi_path), "sha256": hashlib.sha256(kpi_path.read_bytes()).hexdigest()}
    log["runtime_s"] = round(time.time() - t0, 1)
    (out / "run_log.json").write_text(json.dumps(log, indent=2))
    print(f"wrote {out} in {log['runtime_s']} s")


if __name__ == "__main__":
    main()
