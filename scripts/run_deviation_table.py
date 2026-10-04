#!/usr/bin/env python3
"""Noise-aware "why is this field off?" table against the Batch 3 baseline.

For every labelled and held-out field and every v1 site KPI (plus the two headline functional columns):
robust z vs Batch 3 (leave-one-out for Batch 3 fields) AND the deviation in units of the field's own
sampling SD (the SD of a 16-tile site mean from `outputs/reliability/icc.csv`). A deviation counts as
evidence only if it clears both: |z| > 2 and |dev| > 2 sampling SDs. Writes
outputs/acceptance/deviation_{long,z,flags}.csv, deviation_why.md and figures/deviation_heatmap.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
OUT = O / "acceptance"
Z_T, NOISE_T = 2.0, 2.0
FUNC_COLS = {"F02_soc100_pore_loss": "pore_loss", "F02_soc100_into_graphite": "into_graphite"}


def md(df: pd.DataFrame) -> str:
    df = df.copy()
    df.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    head = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [f"{v:g}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v) for v in r]) + " |"
            for i, r in zip(df.index, df.to_numpy())]
    return "\n".join([head, sep] + body)


def key(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(field=df.batch + "/" + df.site).set_index("field")


def load() -> tuple[pd.DataFrame, pd.Series]:
    kp = pd.concat([key(pd.read_csv(O / "kpis" / "site_kpis.csv")), key(pd.read_csv(O / "heldout" / "kpis" / "site_kpis.csv"))])
    fu = pd.concat([key(pd.read_csv(O / "functional" / "site_functional.csv")), key(pd.read_csv(O / "functional" / "heldout_site_functional.csv"))])
    for c in FUNC_COLS:
        src = c if c in fu.columns else FUNC_COLS[c]
        if src in fu.columns:
            kp[c] = fu[src].reindex(kp.index)
    cols = [c for c in kp.columns if c[:1] in "KF" and c[1:3].isdigit() and pd.api.types.is_numeric_dtype(kp[c]) and kp[c].nunique() > 1]
    return kp[cols], kp["batch"]


def main() -> int:
    X, batch = load()
    icc = pd.read_csv(O / "reliability" / "icc.csv").set_index("kpi")
    noise = icc["sd_site_mean_of_16"].reindex(X.columns)
    is_b3 = (batch == "Batch_3").to_numpy()
    rows = []
    for c in X.columns:
        x = X[c]
        for f in X.index:
            base = x[is_b3 & (X.index != f)].dropna()
            med, mad = base.median(), 1.4826 * (base - base.median()).abs().median()
            dev = x[f] - med
            z = dev / mad if mad > 0 else np.nan
            nz = dev / noise[c] if pd.notna(noise[c]) and noise[c] > 0 else np.nan
            if np.isnan(z):
                flag = "undefined"
            elif abs(z) <= Z_T:
                flag = "inside"
            elif np.isnan(nz):
                flag = "off_noise_unknown"
            elif abs(nz) > NOISE_T:
                flag = "off"
            else:
                flag = "within_sampling_noise"
            rows.append(dict(field=f, batch=batch[f], kpi=c, value=x[f], b3_median=med, b3_mad=mad, robust_z=z,
                             sampling_sd=noise[c], dev_over_sampling_sd=nz, flag=flag))
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "deviation_long.csv", index=False)
    Z = L.pivot(index="field", columns="kpi", values="robust_z").loc[X.index, X.columns]
    F = L.pivot(index="field", columns="kpi", values="flag").loc[X.index, X.columns]
    Z.to_csv(OUT / "deviation_z.csv")
    F.to_csv(OUT / "deviation_flags.csv")

    lines = ["# Why is this field off? (vs Batch 3 baseline, noise-aware)", "",
             f"Flag `off` = |robust z| > {Z_T:g} vs Batch 3 *and* deviation > {NOISE_T:g} × the SD of a 16-tile site mean "
             "(`outputs/reliability/icc.csv`); `within_sampling_noise` = |z| > 2 but not resolvable at this field size; "
             "`off_noise_unknown` = no tile-level noise estimate for that KPI. Batch 3 fields are scored leave-one-out.", "",
             "| field | batch | n off | n unresolved | off columns (z) |", "|---|---|---|---|---|"]
    counts = []
    for f in X.index:
        g = L[L.field == f]
        off = g[g.flag == "off"].assign(a=lambda d: d.robust_z.abs()).sort_values("a", ascending=False)
        unres = (g.flag == "within_sampling_noise").sum()
        counts.append(dict(field=f, batch=batch[f], n_off=len(off), n_unresolved=unres, n_off_unknown=(g.flag == "off_noise_unknown").sum()))
        lines.append(f"| {f} | {batch[f]} | {len(off)} | {unres} | " + ", ".join(f"{r.kpi} ({r.robust_z:+.1f})" for r in off.head(5).itertuples()) + " |")
    C = pd.DataFrame(counts)
    lines += ["", "## Per batch", "", md(C.groupby("batch")[["n_off", "n_unresolved", "n_off_unknown"]].agg(["median", "max"]).round(1))]
    lines += ["", "## Per column: how many fields are `off` / `within_sampling_noise`", "",
              md(L.groupby("kpi").flag.value_counts().unstack(fill_value=0).reindex(X.columns))]
    (OUT / "deviation_why.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(0.45 * len(X.columns) + 3, 0.28 * len(X.index) + 2))
    zc = Z.clip(-4, 4)
    ax.imshow(zc.to_numpy(float), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    for i, f in enumerate(X.index):
        for j, c in enumerate(X.columns):
            fl = F.loc[f, c]
            if fl == "off":
                ax.text(j, i, "●", ha="center", va="center", fontsize=6, color="k")
            elif fl == "within_sampling_noise":
                ax.text(j, i, "○", ha="center", va="center", fontsize=6, color="k")
            elif fl == "off_noise_unknown":
                ax.text(j, i, "?", ha="center", va="center", fontsize=6, color="k")
    ax.set_xticks(range(len(X.columns)), [c.replace("_", "\n", 1) for c in X.columns], fontsize=6, rotation=90)
    ax.set_yticks(range(len(X.index)), X.index, fontsize=6)
    for i in np.flatnonzero(np.diff(pd.factorize(batch)[0])):
        ax.axhline(i + 0.5, color="k", lw=0.6)
    ax.set_title("Robust z vs Batch 3 (LOO for Batch 3). ● off beyond sampling noise, ○ |z|>2 but within sampling noise, ? no noise estimate", fontsize=8)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "deviation_heatmap.png", dpi=160)
    print(C.groupby("batch")[["n_off", "n_unresolved"]].median())
    print(C[C.batch == "Batch_heldout"])
    print(L.groupby("kpi").flag.value_counts().unstack(fill_value=0).reindex(X.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
