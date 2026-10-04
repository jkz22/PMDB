"""Pitch figure: simulated swelling at full charge vs Si area fraction (K01), coloured by batch.

Reads outputs/fem/site_curves.csv (sym, s = 1) and outputs/kpis/site_kpis.csv; held-out sites use
outputs/heldout/kpis/site_kpis.csv. Writes outputs/fem/figures/swelling_vs_si.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Batch_1": "#1f77b4", "Batch_2": "#ff7f0e", "Batch_3": "#2ca02c", "Batch_heldout": "#000000"}


def main() -> None:
    sc = pd.read_csv(ROOT / "outputs/fem/site_curves.csv")
    sw = sc[(sc["orientation"] == "sym") & np.isclose(sc["s"], 1.0)][["batch", "site", "swelling"]]
    kpi = pd.concat([pd.read_csv(ROOT / "outputs/kpis/site_kpis.csv"),
                     pd.read_csv(ROOT / "outputs/heldout/kpis/site_kpis.csv")])[["batch", "site", "K01_si_frac_adm"]]
    d = sw.merge(kpi, on=["batch", "site"], how="inner")
    lab = d[d["batch"] != "Batch_heldout"]
    slope, icpt = np.polyfit(lab["K01_si_frac_adm"], lab["swelling"], 1)
    r2 = np.corrcoef(lab["K01_si_frac_adm"], lab["swelling"])[0, 1] ** 2

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    for b, g in d.groupby("batch"):
        held = b == "Batch_heldout"
        ax.scatter(g["K01_si_frac_adm"], g["swelling"] * 100, c=COLORS[b], s=60 if held else 36,
                   marker="*" if held else "o", label="held-out" if held else b.replace("_", " "), zorder=3)
        if held:
            for _, r in g.iterrows():
                ax.annotate(r["site"], (r["K01_si_frac_adm"], r["swelling"] * 100), fontsize=7,
                            xytext=(4, 3), textcoords="offset points")
    x = np.linspace(d["K01_si_frac_adm"].min(), d["K01_si_frac_adm"].max(), 50)
    ax.plot(x, (slope * x + icpt) * 100, "k--", lw=1, label=f"linear fit, labelled sites (R² = {r2:.2f})")
    ax.set_xlabel("Si area fraction (K01)")
    ax.set_ylabel("simulated swelling at 100% SOC (%)")
    ax.set_title("Swelling is set by Si content")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "outputs/fem/figures/swelling_vs_si.png"
    fig.savefig(out)
    print(f"wrote {out}  n={len(d)}  R2={r2:.3f}")


if __name__ == "__main__":
    main()
