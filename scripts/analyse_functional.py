#!/usr/bin/env python3
"""Analyse outputs/functional/*.csv against the existing KPIs, fingerprint and batch labels.

Writes to outputs/functional/:
    batch_comparison.csv   per F column: batch medians, Kruskal-Wallis p, B1-B2 / B12-B3 Cliff's delta
    redundancy.csv         per F column: strongest Spearman partner among K/D KPIs, partial rho | K01 & pore
    robustness.csv         per F column: Spearman of site ranking, baseline vs segmenter anchors +-0.05
    loso.json              leave-one-site-out + permutation for F-only, fingerprint-only, fingerprint+F
    heldout_predictions.csv F-only fingerprint model applied to the three held-out sites (read-only use)
    power.csv              sites per batch needed for 80 % power (Mann-Whitney) at the observed effect
    figures/*.png
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
from pmdb import fingerprint as fp  # noqa: E402

OUT = ROOT / "outputs" / "functional"
FIG = OUT / "figures"
KPI_FILE = ROOT / "outputs" / "kpis" / "site_kpis.csv"
FP_FEATURES = ROOT / "outputs" / "fingerprint" / "features.csv"
BATCHES = ("Batch_1", "Batch_2", "Batch_3")
SEED = 0


def _key(df: pd.DataFrame) -> pd.Index:
    return pd.Index(df["batch"] + "/" + df["site"], name="key")


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff)).mean())


def batch_comparison(F: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        g = {b: F.loc[F.batch == b, c].dropna().to_numpy() for b in BATCHES}
        if all(np.ptp(v) == 0 for v in g.values() if v.size):
            kw_p = float("nan")
        else:
            kw_p = float(stats.kruskal(*g.values()).pvalue)
        rows.append({
            "column": c,
            **{f"median_{b[-1]}": float(np.median(v)) if v.size else np.nan for b, v in g.items()},
            "kruskal_p": kw_p,
            "delta_B1_vs_B2": cliffs_delta(g["Batch_1"], g["Batch_2"]),
            "delta_B12_vs_B3": cliffs_delta(np.concatenate([g["Batch_1"], g["Batch_2"]]), g["Batch_3"]),
            "mannwhitney_p_B1_B2": float(stats.mannwhitneyu(g["Batch_1"], g["Batch_2"]).pvalue)
            if np.ptp(np.concatenate([g["Batch_1"], g["Batch_2"]])) > 0 else np.nan,
            "mannwhitney_p_B12_B3": float(stats.mannwhitneyu(np.concatenate([g["Batch_1"], g["Batch_2"]]), g["Batch_3"]).pvalue)
            if np.ptp(F[c].dropna()) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def _partial_spearman(x: pd.Series, y: pd.Series, controls: pd.DataFrame) -> float:
    """Spearman correlation of rank residuals after linear regression on ranked controls."""
    df = pd.concat([x, y, controls], axis=1).dropna().rank()
    if len(df) < 6:
        return float("nan")
    z = np.column_stack([np.ones(len(df)), df.iloc[:, 2:].to_numpy()])
    rx = df.iloc[:, 0].to_numpy() - z @ np.linalg.lstsq(z, df.iloc[:, 0].to_numpy(), rcond=None)[0]
    ry = df.iloc[:, 1].to_numpy() - z @ np.linalg.lstsq(z, df.iloc[:, 1].to_numpy(), rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def redundancy(F: pd.DataFrame, K: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    kcols = [c for c in K.columns if c[:1] in "KD" and c[1:3].isdigit() and K[c].std() > 0]
    controls = pd.concat([K["K01_si_frac_adm"], F["F01_pore_frac"]], axis=1)
    rows = []
    for c in cols:
        if F[c].std() == 0 or F[c].isna().all():
            rows.append({"column": c, "best_kpi": None, "rho": np.nan, "rho_K01": np.nan,
                         "rho_pore": np.nan, "partial_rho_best_given_K01_pore": np.nan})
            continue
        rho = K[kcols].corrwith(F[c], method="spearman")
        best = rho.abs().idxmax()
        rows.append({
            "column": c, "best_kpi": best, "rho": float(rho[best]),
            "rho_K01": float(stats.spearmanr(F[c], K["K01_si_frac_adm"], nan_policy="omit").statistic),
            "rho_pore": float(stats.spearmanr(F[c], F["F01_pore_frac"], nan_policy="omit").statistic)
            if c != "F01_pore_frac" else 1.0,
            "partial_rho_best_given_K01_pore": _partial_spearman(F[c], K[best], controls),
        })
    return pd.DataFrame(rows)


def robustness(F: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for tag in ("perturb+0.05", "perturb-0.05"):
        P = pd.read_csv(OUT / f"{tag}_site_functional.csv")
        P.index = _key(P)
        P = P.loc[F.index]
        for c in cols:
            if F[c].std() == 0 or P[c].std() == 0:
                rho, med_shift = np.nan, float(np.nanmedian(P[c] - F[c]))
            else:
                rho = float(stats.spearmanr(F[c], P[c], nan_policy="omit").statistic)
                med_shift = float(np.nanmedian((P[c] - F[c]) / F[c].abs().replace(0, np.nan)))
            rows.append({"column": c, "perturbation": tag, "spearman_sites": rho, "median_rel_shift": med_shift})
    return pd.DataFrame(rows)


def loso_block(X: pd.DataFrame, y: pd.Series, features: list[str], n_perm: int) -> dict:
    pred, metrics = fp.loo_evaluate(X, y, features=features)
    perm = fp.permutation_test(X, y, n_perm=n_perm, seed=SEED, features=features)
    return {"features": features, "loo": metrics, "permutation": perm}


def power(F: pd.DataFrame, K: pd.DataFrame, cols_f: list[str], n_boot: int = 400,
          ns: tuple[int, ...] = (5, 7, 10, 15, 20, 30, 50)) -> pd.DataFrame:
    """How many sites per group would a Mann-Whitney test (alpha 0.05) need to reach 80 % power
    at the *observed* separation? Resample with replacement from each group's observed values
    (plug-in bootstrap); the answer is optimistic when the observed effect is itself noise."""
    rng = np.random.default_rng(SEED)
    both = pd.concat([F[cols_f], K[[c for c in K.columns if c[:1] in "K" and c[1:3].isdigit()]]], axis=1)
    rows = []
    for c in both.columns:
        if both[c].std() == 0 or both[c].isna().any():
            continue
        for name, ga, gb in (("B1_vs_B2", ["Batch_1"], ["Batch_2"]), ("B12_vs_B3", ["Batch_1", "Batch_2"], ["Batch_3"])):
            a = both.loc[F.batch.isin(ga), c].to_numpy()
            b = both.loc[F.batch.isin(gb), c].to_numpy()
            delta = cliffs_delta(a, b)
            res = {"column": c, "contrast": name, "cliffs_delta": delta, "n_a": a.size, "n_b": b.size}
            for n in ns:
                hits = 0
                for _ in range(n_boot):
                    sa = rng.choice(a, n, replace=True)
                    sb = rng.choice(b, n, replace=True)
                    if np.ptp(np.concatenate([sa, sb])) == 0:
                        continue
                    hits += stats.mannwhitneyu(sa, sb).pvalue < 0.05
                res[f"power_n{n}"] = hits / n_boot
            need = [n for n in ns if res[f"power_n{n}"] >= 0.8]
            res["n_per_group_for_80pct"] = need[0] if need else f">{ns[-1]}"
            rows.append(res)
    return pd.DataFrame(rows)


def figures(F: pd.DataFrame, K: pd.DataFrame, H: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"Batch_1": "tab:blue", "Batch_2": "tab:orange", "Batch_3": "tab:green"}

    # 1. swelling budget per batch (median) at each SOC
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, soc in zip(axes, ("025", "050", "100")):
        med = F.groupby("batch")[[f"F02_soc{soc}_into_graphite", f"F02_soc{soc}_into_binder", f"F02_soc{soc}_into_pore"]].median()
        bottom = np.zeros(len(med))
        for col, lab, colr in zip(med.columns, ("graphite (constrained)", "binder / CB", "pore"), ("0.35", "0.7", "white")):
            ax.bar(med.index, med[col], bottom=bottom, color=colr, edgecolor="k", label=lab)
            bottom += med[col].to_numpy()
        ax.set_title(f"SOC {int(soc) / 100:.2f}: where Si growth lands")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("share of intended Si growth (batch median)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "swelling_budget.png", dpi=150)
    plt.close(fig)

    # 2. per-site scatter: pore loss at full lithiation vs Si fraction, coloured by batch
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for b, g in F.groupby("batch"):
        ax.scatter(K.loc[g.index, "K01_si_frac_adm"], g["F02_soc100_pore_loss"], label=b, color=colors[b], s=35)
    ax.scatter(H["K01"], H["F02_soc100_pore_loss"], marker="x", color="k", label="held-out")
    ax.set_xlabel("K01 Si area fraction")
    ax.set_ylabel("pore lost at full lithiation (F02_soc100_pore_loss)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "pore_loss_vs_si_fraction.png", dpi=150)
    plt.close(fig)

    # 3. Si -> pore access distance and constrained-growth share per batch
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, c, lab in zip(axes, ("F01_si_pore_dist_p50_um", "F02_soc100_into_graphite", "F02_soc100_si_objects_ratio"),
                          ("median Si-to-pore distance (um)", "growth landing on graphite, SOC 1", "Si objects after / before swelling")):
        data = [F.loc[F.batch == b, c].to_numpy() for b in BATCHES]
        ax.boxplot(data, tick_labels=[b.replace("Batch_", "B") for b in BATCHES], widths=0.5)
        for i, d in enumerate(data):
            ax.scatter(np.full(d.size, i + 1) + np.random.default_rng(i).uniform(-0.12, 0.12, d.size), d, s=14,
                       color=colors[BATCHES[i]])
        ax.set_title(lab, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "access_and_constraint_by_batch.png", dpi=150)
    plt.close(fig)


def main() -> int:
    F = pd.read_csv(OUT / "site_functional.csv")
    F.index = _key(F)
    K = pd.read_csv(KPI_FILE)
    K.index = _key(K)
    K = K.loc[F.index]
    cols = [c for c in F.columns if c.startswith("F0")]
    informative = [c for c in cols if F[c].std() > 0 and not F[c].isna().any()]

    bc = batch_comparison(F, cols)
    bc.to_csv(OUT / "batch_comparison.csv", index=False)
    rd = redundancy(F, K, cols)
    rd.to_csv(OUT / "redundancy.csv", index=False)
    rb = robustness(F, informative)
    rb.to_csv(OUT / "robustness.csv", index=False)

    # LOSO: F-only (compact, non-redundant set), fingerprint-only, fingerprint + F
    f_set = ["F01_si_pore_dist_p50_um", "F01_si_pore_access_frac", "F01_pore_chord_anisotropy",
             "F01_pore_clusters_per_1000um2", "F02_soc100_into_graphite", "F02_soc100_pore_loss",
             "F02_soc100_si_objects_ratio"]
    FPX = pd.read_csv(FP_FEATURES)
    FPX.index = _key(FPX)
    fp_feats = [c for c in FPX.columns if c not in ("batch", "site")]
    X = pd.concat([F[f_set], FPX.loc[F.index, fp_feats]], axis=1)
    y = F["batch"]
    n_perm = 300
    loso = {
        "n_perm": n_perm,
        "majority_baseline": float(y.value_counts(normalize=True).max()),
        "functional_only": loso_block(X, y, f_set, n_perm),
        "fingerprint_only": loso_block(X, y, fp_feats, n_perm),
        "fingerprint_plus_functional": loso_block(X, y, fp_feats + f_set, n_perm),
    }
    (OUT / "loso.json").write_text(json.dumps(loso, indent=2))

    # held-out: F-only model, read-only use of the held-out sites
    H = pd.read_csv(OUT / "heldout_site_functional.csv")
    H.index = _key(H)
    model = fp.fit(F[f_set], y, features=f_set)
    pred = fp.predict(model, H[f_set])
    pred.to_csv(OUT / "heldout_predictions.csv")
    HK = pd.read_csv(ROOT / "outputs" / "heldout" / "kpis" / "site_kpis.csv")
    HK.index = _key(HK)
    H["K01"] = HK.loc[H.index, "K01_si_frac_adm"]

    pw = power(F, K, informative)
    pw.to_csv(OUT / "power.csv", index=False)
    figures(F, K, H)

    print(bc[["column", "median_1", "median_2", "median_3", "kruskal_p", "delta_B1_vs_B2", "delta_B12_vs_B3"]].round(3).to_string())
    print(rd.round(2).to_string())
    print(rb.groupby("column")["spearman_sites"].min().round(2).to_string())
    for k in ("functional_only", "fingerprint_only", "fingerprint_plus_functional"):
        print(k, "LOO acc", round(loso[k]["loo"]["accuracy"], 3), "perm p", round(loso[k]["permutation"]["p_value"], 4),
              "recall", {b: round(v, 2) for b, v in loso[k]["loo"]["recall"].items()})
    print(pred[["assigned", "credibility", "confidence"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
