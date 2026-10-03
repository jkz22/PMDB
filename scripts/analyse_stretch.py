#!/usr/bin/env python3
"""Analyse outputs/stretch/site_stretch.csv (v2 S01-S04) against the v1 KPIs, the F columns and batch.

Writes outputs/stretch/{batch_comparison,redundancy,robustness}.csv, loso.json, heldout_predictions.csv
and figures/stretch_by_batch.png. Reuses the helpers of scripts/analyse_functional.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import analyse_functional as af  # noqa: E402
from pmdb import fingerprint as fp  # noqa: E402

OUT = ROOT / "outputs" / "stretch"
S_SET = ["S01_c2_length_x_um", "S01_c2_length_z_um", "S01_c2_anisotropy", "S02_euler_merge_radius_um",
         "S03_h0_life_p50_um", "S03_h0_life_iqr_um", "S04_si_beta", "S04_si_streak_angle_deg"]


def main() -> int:
    S = pd.read_csv(OUT / "site_stretch.csv")
    S.index = af._key(S)
    K = pd.read_csv(af.KPI_FILE)
    K.index = af._key(K)
    K = K.loc[S.index]
    F = pd.read_csv(af.OUT / "site_functional.csv")
    F.index = af._key(F)
    F = F.loc[S.index]
    cols = [c for c in S.columns if c.startswith("S0") and not c.endswith("nan_reason")]

    bc = af.batch_comparison(S, cols)
    bc.to_csv(OUT / "batch_comparison.csv", index=False)

    # redundancy: strongest partner among v1 K/D KPIs and among F columns
    kcols = [c for c in K.columns if c[:1] in "KD" and c[1:3].isdigit() and K[c].std() > 0]
    fcols = [c for c in F.columns if c.startswith("F0") and F[c].std() > 0]
    rows = []
    for c in cols:
        rk = K[kcols].corrwith(S[c], method="spearman")
        rf = F[fcols].corrwith(S[c], method="spearman")
        bk, bf = rk.abs().idxmax(), rf.abs().idxmax()
        rows.append({"column": c, "best_kpi": bk, "rho_kpi": float(rk[bk]), "best_functional": bf,
                     "rho_functional": float(rf[bf]),
                     "rho_K01": float(stats.spearmanr(S[c], K["K01_si_frac_adm"]).statistic),
                     "partial_rho_best_kpi_given_K01": af._partial_spearman(S[c], K[bk], K[["K01_si_frac_adm"]])})
    rd = pd.DataFrame(rows)
    rd.to_csv(OUT / "redundancy.csv", index=False)

    rb_rows = []
    for tag in ("perturb+0.05", "perturb-0.05"):
        path = OUT / f"{tag}_site_stretch.csv"
        if not path.exists():
            continue
        P = pd.read_csv(path)
        P.index = af._key(P)
        P = P.loc[S.index]
        for c in cols:
            rb_rows.append({"column": c, "perturbation": tag,
                            "spearman_sites": float(stats.spearmanr(S[c], P[c], nan_policy="omit").statistic),
                            "median_rel_shift": float(np.nanmedian((P[c] - S[c]) / S[c].abs()))})
    rb = pd.DataFrame(rb_rows)
    rb.to_csv(OUT / "robustness.csv", index=False)

    FPX = pd.read_csv(af.FP_FEATURES)
    FPX.index = af._key(FPX)
    fp_feats = [c for c in FPX.columns if c not in ("batch", "site")]
    X = pd.concat([S[S_SET], FPX.loc[S.index, fp_feats]], axis=1)
    y = S["batch"]
    n_perm = 300
    loso = {"n_perm": n_perm, "majority_baseline": float(y.value_counts(normalize=True).max()),
            "stretch_only": af.loso_block(X, y, S_SET, n_perm),
            "fingerprint_only": af.loso_block(X, y, fp_feats, n_perm),
            "fingerprint_plus_stretch": af.loso_block(X, y, fp_feats + S_SET, n_perm)}
    (OUT / "loso.json").write_text(json.dumps(loso, indent=2))

    H = pd.read_csv(OUT / "heldout_site_stretch.csv")
    H.index = af._key(H)
    pred = fp.predict(fp.fit(S[S_SET], y, features=S_SET), H[S_SET])
    pred.to_csv(OUT / "heldout_predictions.csv")

    (OUT / "figures").mkdir(exist_ok=True)
    show = ["S01_c2_length_x_um", "S01_c2_anisotropy", "S02_euler_merge_radius_um", "S03_h0_life_p50_um",
            "S03_h0_life_iqr_um", "S04_si_beta"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    colors = {"Batch_1": "tab:blue", "Batch_2": "tab:orange", "Batch_3": "tab:green"}
    for ax, c in zip(axes.ravel(), show):
        data = [S.loc[S.batch == b, c].to_numpy() for b in af.BATCHES]
        ax.boxplot(data, tick_labels=[b.replace("Batch_", "B") for b in af.BATCHES], widths=0.5)
        for i, d in enumerate(data):
            ax.scatter(np.full(d.size, i + 1) + np.random.default_rng(i).uniform(-0.12, 0.12, d.size), d, s=14,
                       color=colors[af.BATCHES[i]])
        p = float(bc.loc[bc.column == c, "kruskal_p"].iloc[0])
        ax.set_title(f"{c}  (KW p = {p:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "stretch_by_batch.png", dpi=150)

    print(bc[["column", "median_1", "median_2", "median_3", "kruskal_p", "delta_B1_vs_B2", "delta_B12_vs_B3"]].round(3).to_string())
    print(rd.round(2).to_string())
    if len(rb):
        print(rb.groupby("column")["spearman_sites"].min().round(2).to_string())
    for k in ("stretch_only", "fingerprint_only", "fingerprint_plus_stretch"):
        print(k, "LOO acc", round(loso[k]["loo"]["accuracy"], 3), "perm p", round(loso[k]["permutation"]["p_value"], 4),
              "recall", {b: round(v, 2) for b, v in loso[k]["loo"]["recall"].items()})
    print(pred[["assigned", "credibility", "confidence"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
