"""One 16:9 slide: per-batch distributions of shared KPIs, analytical benchmark vs materials classifier."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SURFACE, INK1, INK2, MUTED, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COL = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a"}
a = pd.read_csv("analytical_benchmarks/site_kpis.csv")
m = pd.read_csv("outputs/kpis/materials_site_kpis.csv")
SOURCES = [("Analytical benchmark", a, {"Porosity": "porosity", "Si fraction": "si_frac"}),
           ("Materials classifier", m, {"Porosity": "mat_porosity", "Si fraction": "mat_bright_fraction"})]
KPIS = ["Porosity", "Si fraction"]

plt.rcParams.update({"font.family": "sans-serif", "text.color": INK1})
fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor=SURFACE, sharey=False)
fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.17, hspace=0.42, wspace=0.08)
rng = np.random.default_rng(0)
for r, kpi in enumerate(KPIS):
    for c, (src, df, cols) in enumerate(SOURCES):
        ax = axes[r, c]; ax.set_facecolor(SURFACE)
        for i, (b, g) in enumerate(df.groupby("batch")):
            v = g[cols[kpi]].values / df[cols[kpi]].mean()
            ax.scatter(i + rng.uniform(-.16, .16, len(v)), v, s=70, color=COL[b],
                       edgecolor=SURFACE, linewidth=1.5, alpha=.9, zorder=3)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i - .3, i + .3], [med, med], color=INK1, lw=2, zorder=4)
            ax.plot([i, i], [q1, q3], color=INK2, lw=1, alpha=.5, zorder=2)
            ax.text(i + .33, med, f"{med:.2f}", va="center", fontsize=12, color=INK2)
        ax.set_xticks(range(3)); ax.set_xlim(-.5, 2.7)
        ax.set_xticklabels([f"Batch {k}\n(n={n})" for k, n in enumerate(df.groupby("batch").size(), 1)],
                           fontsize=12, color=INK2)
        ax.set_ylim(0, 1.08 * max((d[c_[kpi]] / d[c_[kpi]].mean()).max() for _, d, c_ in SOURCES)); ax.tick_params(colors=MUTED, length=0, labelsize=11)
        ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1); ax.grid(axis="y", color=HAIR, lw=.8); ax.set_axisbelow(True)
        for s in ax.spines.values(): s.set_visible(False)
        if c == 0: ax.set_ylabel(f"{kpi} (index)", fontsize=13, color=INK2)
        if r == 0: ax.set_title(src, fontsize=16, fontweight="bold", loc="left", pad=12)
fig.text(.07, .93, "Both methods: Batch 1 skews Si-rich, porosity is similar",
         fontsize=26, fontweight="bold", va="top")
fig.text(.07, .865, "Per-site KPIs by batch, 31 sites, each divided by its own method's all-site mean (dashed = 1). "
         "black bar = batch median (labelled), thin line = IQR.", fontsize=14, color=INK2, va="top")
fig.text(.07, .03, "Sources: analytical_benchmarks/site_kpis.csv (porosity, si_frac); outputs/kpis/materials_site_kpis.csv "
         "(mat_porosity, mat_bright_fraction).\nRaw levels differ by segmentation (porosity ~6% vs ~9%); indexing removes that offset so batch patterns compare directly.",
         fontsize=10.5, color=MUTED)
fig.savefig("outputs/slides/kpi_distributions_by_batch.png", dpi=150, facecolor=SURFACE)
