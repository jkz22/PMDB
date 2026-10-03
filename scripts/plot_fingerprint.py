"""Demo figures for the batch fingerprint model.

    python scripts/plot_fingerprint.py

Reads outputs/fingerprint/ (written by run_fingerprint.py) and writes to
outputs/fingerprint/figures/:

    depth_profiles.png    per-batch Si depth-profile small multiples with the
                          held-out sites overlaid on the batch medians
    card_<site>.png       one fingerprint card per held-out site: the most
                          decisive features as strip plots (labelled batches
                          as dots, the held-out site as a black diamond) plus
                          the conformal p-value verdict
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FP_DIR = ROOT / "outputs" / "fingerprint"
FIG_DIR = FP_DIR / "figures"

# Reference dataviz palette: categorical slots 1-3 (validated all-pairs, light
# mode), chart chrome from the same reference instance.
BATCH_COLOR = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

FEATURE_LABEL = {
    "si_depth_rel_band0": "Si fraction, top band (rel.)",
    "si_depth_rel_band1": "Si fraction, band 1 (rel.)",
    "si_depth_rel_band2": "Si fraction, mid band (rel.)",
    "si_depth_rel_band3": "Si fraction, band 3 (rel.)",
    "si_depth_rel_band4": "Si fraction, bottom band (rel.)",
    "si_depth_slope": "Si depth slope (bottom - top)",
    "si_depth_mid_dip": "Si mid-depth dip",
    "k15_contact_tilestd": "Si-graphite contact, tile spread",
}


def _label(feature: str) -> str:
    if feature in FEATURE_LABEL:
        return FEATURE_LABEL[feature]
    if feature.startswith(("gx_", "gz_")):
        ax = "x" if feature[1] == "x" else "z"
        lo, hi = feature.split("_")[1:3]
        return f"pair corr. g{ax}, {lo}-{hi} um"
    return feature


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def plot_depth_profiles(curves: pd.DataFrame, held: pd.DataFrame, out: Path):
    """Small multiples: relative Si depth profile per batch + held-out overlay."""
    def rel_profiles(c):
        piv = c[c["curve"] == "band_si_frac"].pivot_table(
            index=["batch", "site"], columns="x", values="value")
        return piv.div(piv.mean(axis=1), axis=0)

    rel = rel_profiles(curves)
    rel_h = rel_profiles(held)
    bands = rel.columns.to_numpy(float)
    batches = ["Batch_1", "Batch_2", "Batch_3"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, b in zip(axes[:3], batches):
        _style_axes(ax)
        sub = rel.loc[rel.index.get_level_values("batch") == b]
        for _, row in sub.iterrows():
            ax.plot(bands, row.to_numpy(), color=BATCH_COLOR[b], alpha=0.30,
                    linewidth=1.2)
        ax.plot(bands, sub.median().to_numpy(), color=BATCH_COLOR[b],
                linewidth=2.6, solid_capstyle="round")
        ax.axhline(1.0, color=BASELINE, linewidth=0.8, linestyle=(0, (4, 3)))
        n = len(sub)
        title = {"Batch_1": "Batch 1 - bottom-heavy, variable",
                 "Batch_2": "Batch 2 - top-heavy, mid-depth depleted",
                 "Batch_3": "Batch 3 (baseline) - uniform"}[b]
        ax.set_title(f"{title}  (n={n})", fontsize=9.5, color=INK, pad=8)
        ax.set_xlabel("depth band (top -> bottom)", fontsize=9, color=INK_2)
        ax.set_xticks(bands)
    axes[0].set_ylabel("Si fraction / site mean", fontsize=9, color=INK_2)

    ax = axes[3]
    _style_axes(ax)
    for b in batches:
        sub = rel.loc[rel.index.get_level_values("batch") == b]
        ax.plot(bands, sub.median().to_numpy(), color=BATCH_COLOR[b],
                linewidth=2.0, alpha=0.9, label=f"{b.replace('_', ' ')} median")
    markers = ["D", "s", "o"]
    for (idx, row), mk in zip(rel_h.iterrows(), markers):
        ax.plot(bands, row.to_numpy(), color=INK, linewidth=1.6,
                linestyle=(0, (4, 2)), marker=mk, markersize=5,
                markerfacecolor=SURFACE, label=f"held-out {idx[1]}")
    ax.axhline(1.0, color=BASELINE, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_title("Held-out sites vs batch medians", fontsize=9.5, color=INK, pad=8)
    ax.set_xlabel("depth band (top -> bottom)", fontsize=9, color=INK_2)
    ax.set_xticks(bands)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="lower left")

    fig.suptitle("Si through-thickness distribution is the batch fingerprint",
                 fontsize=12, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_card(site: str, X: pd.DataFrame, held: pd.DataFrame,
              pred_row: pd.Series, explain: pd.DataFrame, out: Path,
              n_features: int = 6):
    """Fingerprint card: decisive feature strip plots + conformal verdict."""
    batches = ["Batch_1", "Batch_2", "Batch_3"]
    ex = explain[explain["site_index"].astype(str).str.contains(site)].copy()
    assigned = pred_row["assigned"]
    others = [b for b in batches if b != assigned]
    # decisiveness: how much closer to the assigned batch than to the best other
    ex["decisive"] = ex[[f"dev_{b}" for b in others]].min(axis=1) - ex[f"dev_{assigned}"]
    top = ex.sort_values("decisive", ascending=False).head(n_features)["feature"]

    fig, axes = plt.subplots(n_features, 1, figsize=(7.6, 1.05 * n_features + 1.6))
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    ybatch = X.index.get_level_values("batch")
    for ax, f in zip(axes, top):
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        for j, b in enumerate(batches):
            vals = X.loc[ybatch == b, f].to_numpy()
            jitter = (rng.random(len(vals)) - 0.5) * 0.30
            ax.scatter(vals, np.full(len(vals), j) + jitter, s=26,
                       color=BATCH_COLOR[b], alpha=0.75, edgecolors="none")
        hv = float(held.loc[held.index.get_level_values("site") == site, f].iloc[0])
        ax.axvline(hv, color=INK, linewidth=1.0, alpha=0.55, zorder=4)
        ax.scatter([hv], [1.0], marker="D", s=70, color=INK, zorder=5,
                   facecolor=SURFACE, linewidths=1.8)
        ax.set_yticks(range(3))
        ax.set_yticklabels([b.replace("_", " ") for b in batches], fontsize=8.5)
        ax.set_ylim(-0.6, 2.6)
        ax.invert_yaxis()
        ax.set_title(_label(f), fontsize=9, color=INK_2, loc="left", pad=3)

    ps = " / ".join(f"p({b.replace('_', ' ')}) = {pred_row[f'p_{b}']:.2f}"
                    for b in batches)
    verdict = (f"assigned {assigned.replace('_', ' ')}"
               f"   credibility {pred_row['credibility']:.2f}"
               f"   confidence {pred_row['confidence']:.2f}"
               + ("   OUT OF DISTRIBUTION" if pred_row["ood"] else ""))
    fig.suptitle(f"Held-out site {site}: {verdict}\n{ps}",
                 fontsize=10.5, color=INK, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(ROOT / "outputs/kpis/curves.csv",
                         dtype={"batch": str, "site": str})
    held_curves = pd.read_csv(ROOT / "outputs/heldout/kpis/curves.csv",
                              dtype={"batch": str, "site": str})
    X = pd.read_csv(FP_DIR / "features.csv", dtype={"batch": str, "site": str}
                    ).set_index(["batch", "site"])
    H = pd.read_csv(FP_DIR / "heldout_features.csv",
                    dtype={"batch": str, "site": str}).set_index(["batch", "site"])
    pred = pd.read_csv(FP_DIR / "heldout_predictions.csv",
                       dtype={"batch": str, "site": str}).set_index(["batch", "site"])
    explain = pd.read_csv(FP_DIR / "heldout_explain.csv")

    plot_depth_profiles(curves, held_curves, FIG_DIR / "depth_profiles.png")
    for (b, site), row in pred.iterrows():
        plot_card(site, X, H, row, explain, FIG_DIR / f"card_{site}.png")
    print(f"wrote {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
