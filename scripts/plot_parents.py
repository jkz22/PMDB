"""Figure for docs/parents.md: LOSO vs LOPO, pixel adjacency evidence, dark level per parent."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
COL = {"Batch_1": "#1f77b4", "Batch_2": "#ff7f0e", "Batch_3": "#2ca02c"}


def main() -> int:
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 1.4]})

    # A: LOSO vs LOPO, 34 sites
    d = pd.read_csv(O / "parents" / "lopo_rescore.csv")
    d = d[d.sites == 34]
    names = list(dict.fromkeys(d.method))
    y = np.arange(len(names))
    for off, cv, c in ((-0.18, "LOSO", "#bbbbbb"), (0.18, "LOPO", "#444444")):
        v = [d[(d.method == n) & (d.cv == cv)].acc.iloc[0] for n in names]
        ax[0].barh(y + off, v, height=0.34, color=c, label=cv)
    ax[0].axvline(18 / 34, color="k", ls=":", lw=1)
    ax[0].text(18 / 34 + 0.005, len(names) - 0.4, "majority", fontsize=8)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([n.replace("nearest mean: ", "NM: ").replace("XGBoost: ", "XGB: ").replace(" (acquisition)", "*")
                           .replace(" (material)", "") for n in names], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 0.8)
    ax[0].set_xlabel("accuracy, 34 sites (31 + released truths)")
    ax[0].set_title("A  leave-one-site-out vs leave-one-parent-out\n(* = grey-level / acquisition reader)", fontsize=10)
    ax[0].legend(loc="lower right", fontsize=8)

    # B: edge adjacency
    p = pd.read_csv(O / "parents" / "overlap_pairs.csv")
    r = np.maximum(p.edge_r_ab, p.edge_r_ba)
    same = p.same_key.astype(bool)
    bins = np.linspace(-0.3, 0.85, 24)
    ax[1].hist(r[~same], bins=bins, color="#bbbbbb", label=f"different key (n={(~same).sum()})", density=True)
    ax[1].hist(r[same], bins=bins, color="#d62728", alpha=0.7, label=f"same key (n={same.sum()})", density=True)
    ax[1].axvline(np.percentile(r[~same], 99), color="k", ls=":", lw=1)
    ax[1].text(np.percentile(r[~same], 99) + 0.01, ax[1].get_ylim()[1] * 0.9, "control 99th pct", fontsize=8)
    ax[1].set_xlabel("edge-strip Pearson r (best of the two directions)")
    ax[1].set_ylabel("density")
    ax[1].set_title("B  are same-key sites adjacent tiles?\n15/34 same-key pairs above 0.5, 0/527 controls", fontsize=10)
    ax[1].legend(fontsize=8)

    # C: dark level per parent
    pg = pd.read_csv(O / "parent_groups.csv", dtype=str).set_index("site")
    truth = pd.read_csv(O / "heldout_labels.csv", dtype=str).set_index("site")["batch"]
    pg.loc[truth.index, "batch"] = truth
    s = pd.concat([pd.read_csv(O / "clean" / "summary.csv", dtype={"site": str}),
                   pd.read_csv(O / "clean_heldout" / "summary.csv", dtype={"site": str})]).set_index("site")
    pg["D"] = s.loc[pg.index, "SE_type_D"].to_numpy(float)
    order = pg.groupby("parent_id").D.mean().sort_values().index
    xpos = {k: i for i, k in enumerate(order)}
    rng = np.random.default_rng(0)
    for site, row in pg.iterrows():
        x = xpos[row.parent_id] + rng.uniform(-0.18, 0.18)
        held = site in truth.index
        ax[2].scatter(x, row.D, s=70 if held else 40, color=COL[row.batch], edgecolor="k" if held else "none",
                      linewidth=1.5, zorder=3)
    for k in order:
        g = pg[pg.parent_id == k]
        ax[2].plot([xpos[k] - 0.3, xpos[k] + 0.3], [g.D.mean()] * 2, color="#999999", lw=1, zorder=1)
        if g.batch.nunique() > 1:
            ax[2].text(xpos[k], g.D.max() + 1.2, "mixed", ha="center", fontsize=7, color="#555555")
    for c1, c2 in ((-8.84, 0.546),):
        ax[2].axhline(c1, color="k", ls="--", lw=0.8)
        ax[2].axhline(c2, color="k", ls="--", lw=0.8)
    ax[2].set_xticks(range(len(order)))
    ax[2].set_xticklabels([k.replace("_ETD", "").replace("_SE", "·SE") for k in order], rotation=60, fontsize=7, ha="right")
    ax[2].set_ylabel("SE-detector dark level D (DN, from deep-pore pixels)")
    for b, c in COL.items():
        ax[2].scatter([], [], color=c, label=b.replace("_", " "))
    ax[2].scatter([], [], color="w", edgecolor="k", linewidth=1.5, label="released held-out")
    ax[2].legend(fontsize=8, loc="upper left")
    ax[2].set_title("C  the best single-feature label rule is the dark level\n(27/34 with the dashed cuts; within parents 12/13 pairs ordered B1<B2<B3)", fontsize=10)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "parents.png"
    fig.savefig(out, dpi=140)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
