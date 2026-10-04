#!/usr/bin/env python3
"""XGBoost on the reliable feature set: can a non-linear learner do better than the robust naive-Bayes
fingerprint on 31 fields?

Feature families (all held-out-free for fitting):
  reliable_kpis   v1 site KPIs with 16-tile ICC >= 0.75 (docs/reliability.md §1): K01, K02, K03 d50/d90/max,
                  K05, K09 m/sigma, K14 p95, K15.  Drops K07, K04 agglomerate, K14 p50, K16 (noise / constant).
  fingerprint     the 16 locked arrangement features (outputs/fingerprint/features.csv).
  functional      SOC-1 pore loss and its mid-depth contrast (segmenter-robust, docs/reliability.md §3);
                  the constrained share is excluded on purpose.

Three evaluations, each leave-one-site-out with a label-permutation null on the whole LOSO pipeline:
  site-level 3-way XGBoost  (31 rows, depth-2 trees)
  site-level Batch 3 vs rest XGBoost -> out-of-fold AUC
  tile-level 3-way XGBoost on reliable tile KPIs (16 tiles x 31 sites), site posterior = mean tile log-prob

Writes outputs/xgb/{results.json, loso_predictions.csv, heldout_predictions.csv, importance.csv} and
figures/importance.png.  CPU only; ~5 min on 8 cores.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
OUT = O / "xgb"
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
HELD = ["3e122cbj", "fn0mhxef", "xrv9xvzb"]

RELIABLE_KPIS = [
    "K01_si_frac_adm", "K02_si_density_per_1000um2", "K03_ecd_d50_um", "K03_ecd_d90_um", "K03_ecd_max_um",
    "K05_voronoi_sigma", "K09_mst_m_norm", "K09_mst_sigma_norm", "K14_empty_p95_um", "K15_si_graphite_contact_frac",
]
FUNCTIONAL = ["F02_soc100_pore_loss", "F02_pore_loss_mid_contrast"]
PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=2, reg_lambda=1.0, n_jobs=1, verbosity=0, tree_method="hist")


def rd(p: str) -> pd.DataFrame:
    return pd.read_csv(O / p, dtype={"batch": str, "site": str})


def mid_contrast(d: pd.DataFrame) -> pd.Series:
    """Mid-depth pore loss relative to the site mean (the Batch 2 'dip')."""
    piv = d.pivot_table(index="site", columns="band", values="F02_pore_loss")
    mid = piv.columns[len(piv.columns) // 2]
    return (piv[mid] / piv.mean(axis=1)).rename("F02_pore_loss_mid_contrast")


def site_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    k = pd.concat([rd("kpis/site_kpis.csv"), rd("heldout/kpis/site_kpis.csv")]).set_index("site")
    f = pd.concat([rd("fingerprint/features.csv"), rd("fingerprint/heldout_features.csv")]).set_index("site")
    fu = pd.concat([rd("functional/site_functional.csv"), rd("functional/heldout_site_functional.csv")]).set_index("site")
    d = rd("functional/depth_swelling.csv")
    X = k[["batch"] + RELIABLE_KPIS].join(f.drop(columns=["batch"])).join(fu[["F02_soc100_pore_loss"]]).join(mid_contrast(d))
    lab = X[X.batch.isin(BATCHES)]
    held = X[X.index.isin(HELD)]
    assert len(lab) == 31 and len(held) == 3, (len(lab), len(held))
    return lab, held


def tile_table() -> pd.DataFrame:
    t = rd("overnight/features/tile_kpis_rich.csv")
    t = t[(t.n_tiles == 16) & t.batch.isin(BATCHES)]
    return t[["batch", "site", "tile"] + RELIABLE_KPIS].reset_index(drop=True)


def fillna_train_median(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)


def loso_site(X: np.ndarray, y: np.ndarray, sites: np.ndarray, seed: int = 0) -> np.ndarray:
    """Out-of-fold class log-probabilities, one row per site."""
    out = np.full((len(X), 3), np.nan)
    for i in range(len(X)):
        tr = np.arange(len(X)) != i
        a, b = fillna_train_median(X[tr], X[[i]])
        m = XGBClassifier(objective="multi:softprob", num_class=3, random_state=seed, **PARAMS)
        m.fit(a, y[tr])
        out[i] = np.log(np.clip(m.predict_proba(b)[0], 1e-9, 1))
    return out


def loso_binary(X: np.ndarray, yb: np.ndarray, seed: int = 0) -> np.ndarray:
    out = np.empty(len(X))
    for i in range(len(X)):
        tr = np.arange(len(X)) != i
        a, b = fillna_train_median(X[tr], X[[i]])
        m = XGBClassifier(objective="binary:logistic", random_state=seed, **PARAMS)
        m.fit(a, yb[tr])
        out[i] = m.predict_proba(b)[0, 1]
    return out


def loso_tile(X: np.ndarray, y_site: np.ndarray, site_of_row: np.ndarray, sites: np.ndarray, seed: int = 0) -> np.ndarray:
    """Tile-level model; site posterior = mean log-prob over its 16 tiles."""
    y_row = np.array([y_site[list(sites).index(s)] for s in site_of_row])
    out = np.full((len(sites), 3), np.nan)
    for i, s in enumerate(sites):
        tr = site_of_row != s
        a, b = fillna_train_median(X[tr], X[~tr])
        m = XGBClassifier(objective="multi:softprob", num_class=3, random_state=seed, **PARAMS)
        m.fit(a, y_row[tr])
        out[i] = np.log(np.clip(m.predict_proba(b), 1e-9, 1)).mean(axis=0)
    return out


def perm_pvalue(stat_fn, y: np.ndarray, n_perm: int, seed: int, n_jobs: int) -> dict:
    obs = stat_fn(y)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(n_perm)]
    null = np.asarray(Parallel(n_jobs=n_jobs)(delayed(stat_fn)(p) for p in perms))
    return {"observed": float(obs), "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + (null >= obs).sum()) / (n_perm + 1)), "n_perm": int(n_perm)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    lab, held = site_table()
    sites = lab.index.to_numpy()
    y = np.array([BATCHES.index(b) for b in lab.batch])
    FAMILIES = {
        "reliable_kpis": RELIABLE_KPIS,
        "fingerprint16": [c for c in lab.columns if c not in RELIABLE_KPIS + FUNCTIONAL + ["batch"]],
        "reliable_kpis+functional": RELIABLE_KPIS + FUNCTIONAL,
        "all_reliable": [c for c in lab.columns if c != "batch"],
    }
    results: dict = {"params": PARAMS, "families": {k: v for k, v in FAMILIES.items()}, "site_level": {}, "tile_level": {}}
    preds = []
    for name, cols in FAMILIES.items():
        X = lab[cols].to_numpy(float)
        lp = loso_site(X, y, sites)
        acc = float((lp.argmax(1) == y).mean())
        recall = {b: float((lp.argmax(1)[y == i] == i).mean()) for i, b in enumerate(BATCHES)}
        perm = perm_pvalue(lambda yy: float((loso_site(X, yy, sites).argmax(1) == yy).mean()), y, a.n_perm, 0, a.n_jobs)
        pb = loso_binary(X, (y == 2).astype(int))
        auc = float(roc_auc_score((y == 2).astype(int), pb))
        perm_auc = perm_pvalue(lambda yy: float(roc_auc_score((yy == 2).astype(int), loso_binary(X, (yy == 2).astype(int)))),
                               y, a.n_perm, 1, a.n_jobs)
        results["site_level"][name] = {"n_features": len(cols), "loso_accuracy": acc, "recall": recall,
                                       "permutation": perm, "b3_vs_rest_auc": auc, "b3_vs_rest_permutation": perm_auc}
        print(f"site {name:28s} acc {acc:.3f} p {perm['p_value']:.3f} (null95 {perm['null_p95']:.3f}) | "
              f"B3-vs-rest AUC {auc:.3f} p {perm_auc['p_value']:.3f}", flush=True)
        df = pd.DataFrame(np.exp(lp), columns=[f"p_{b}" for b in BATCHES], index=sites)
        df.insert(0, "family", name)
        df.insert(1, "true", lab.batch.values)
        df.insert(2, "assigned", [BATCHES[i] for i in lp.argmax(1)])
        df["p_batch3_binary"] = pb
        preds.append(df)
    pd.concat(preds).rename_axis("site").to_csv(OUT / "loso_predictions.csv")

    # tile level
    T = tile_table()
    Xt = T[RELIABLE_KPIS].to_numpy(float)
    lp = loso_tile(Xt, y, T.site.to_numpy(), sites)
    acc = float((lp.argmax(1) == y).mean())
    perm = perm_pvalue(lambda yy: float((loso_tile(Xt, yy, T.site.to_numpy(), sites).argmax(1) == yy).mean()),
                       y, max(50, a.n_perm // 2), 2, a.n_jobs)
    results["tile_level"]["reliable_kpis"] = {
        "n_rows": int(len(T)), "loso_accuracy": acc,
        "recall": {b: float((lp.argmax(1)[y == i] == i).mean()) for i, b in enumerate(BATCHES)}, "permutation": perm}
    print(f"tile reliable_kpis acc {acc:.3f} p {perm['p_value']:.3f} (null95 {perm['null_p95']:.3f})", flush=True)

    # importance (gain) on the full fit, all_reliable, averaged over 20 seeds
    cols = FAMILIES["all_reliable"]
    X = lab[cols].to_numpy(float)
    X, _ = fillna_train_median(X, X)
    imp = np.zeros(len(cols))
    for s in range(20):
        m = XGBClassifier(objective="multi:softprob", num_class=3, random_state=s, **PARAMS).fit(X, y)
        imp += m.feature_importances_
    imp = pd.Series(imp / 20, index=cols).sort_values(ascending=False)
    imp.rename("gain_share").to_csv(OUT / "importance.csv")
    fig, ax = plt.subplots(figsize=(6, 6))
    imp.head(15)[::-1].plot.barh(ax=ax, color="#444")
    ax.set_xlabel("XGBoost gain share (mean of 20 seeds, all_reliable, fit on all 31 sites)")
    ax.set_title("What the trees split on")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "importance.png", dpi=150)

    # held-out: fit on all 31, every family
    rows = []
    for name, cols in FAMILIES.items():
        Xl, Xh = fillna_train_median(lab[cols].to_numpy(float), held[cols].to_numpy(float))
        P = np.zeros((3, 3))
        for s in range(20):
            m = XGBClassifier(objective="multi:softprob", num_class=3, random_state=s, **PARAMS).fit(Xl, y)
            P += m.predict_proba(Xh)
        P /= 20
        for j, site in enumerate(held.index):
            rows.append({"family": name, "site": site, "assigned": BATCHES[int(P[j].argmax())],
                         **{f"p_{b}": float(P[j, i]) for i, b in enumerate(BATCHES)}})
    pd.DataFrame(rows).to_csv(OUT / "heldout_predictions.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
