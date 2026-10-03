#!/usr/bin/env python3
"""The organiser's actual question: is a new field inside the Batch 3 (supplier baseline) distribution?

One-class acceptance test. For every candidate column (v1 KPIs, fingerprint features, functional F,
stretch S) score each labelled field by |robust z| against the Batch 3 fields (leave-one-out when the
field is itself Batch 3), then ask how well that score separates Batch 1+2 from Batch 3 (ROC AUC).
A permutation test on the *maximum* AUC over all columns controls for the search.

    python scripts/run_acceptance.py  -> outputs/acceptance/{per_feature.csv, summary.json, fingerprint_ovr.csv}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
OUT = O / "acceptance"
SEED = 0
N_PERM = 2000
BASE = "Batch_3"


def load_features() -> pd.DataFrame:
    def key(df):
        return df.assign(key=df["batch"] + "/" + df["site"]).set_index("key")
    kp = key(pd.read_csv(O / "kpis" / "site_kpis.csv"))
    fp = key(pd.read_csv(O / "fingerprint" / "features.csv"))
    fu = key(pd.read_csv(O / "functional" / "site_functional.csv"))
    st = key(pd.read_csv(O / "stretch" / "site_stretch.csv"))
    X = kp[[c for c in kp.columns if c[:1] in "KA" and c[1:3].isdigit()]].copy()
    for df, pref in ((fp, None), (fu, "F"), (st, "S")):
        cols = [c for c in df.columns if (pref is None and c not in ("batch", "site"))
                or (pref is not None and c.startswith(pref))]
        X = X.join(df[cols], how="left")
    X.insert(0, "batch", kp["batch"])
    X.insert(1, "site", kp["site"])
    return X


def loo_abs_z(x: np.ndarray, is_base: np.ndarray) -> np.ndarray:
    """|robust z| of each value against the baseline fields, excluding itself if it is baseline."""
    out = np.full(x.size, np.nan)
    for i in range(x.size):
        ref = x[is_base & (np.arange(x.size) != i)]
        ref = ref[np.isfinite(ref)]
        if ref.size < 3 or not np.isfinite(x[i]):
            continue
        med = np.median(ref)
        mad = 1.4826 * np.median(np.abs(ref - med))
        out[i] = abs(x[i] - med) / mad if mad > 0 else np.nan
    return out


def auc_for(X: pd.DataFrame, cols: list[str], is_base: np.ndarray) -> pd.Series:
    y = (~is_base).astype(int)
    res = {}
    for c in cols:
        s = loo_abs_z(X[c].to_numpy(dtype=float), is_base)
        ok = np.isfinite(s)
        res[c] = roc_auc_score(y[ok], s[ok]) if ok.sum() > 5 and len(set(y[ok])) == 2 else np.nan
    return pd.Series(res)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    n_perm = ap.parse_args().n_perm
    OUT.mkdir(exist_ok=True)
    X = load_features()
    is_base = (X["batch"] == BASE).to_numpy()
    cols = [c for c in X.columns if c not in ("batch", "site") and X[c].notna().sum() >= 25
            and X[c].nunique() > 3]
    auc = auc_for(X, cols, is_base)
    per = pd.DataFrame({"column": auc.index, "auc_reject_vs_B3": auc.values}).dropna()
    per["family"] = per["column"].str.extract(r"^([KASF]\d\d|si_depth|gx|gz|k15)", expand=False).fillna("other")
    per = per.sort_values("auc_reject_vs_B3", ascending=False)
    per.to_csv(OUT / "per_feature.csv", index=False)

    rng = np.random.default_rng(SEED)
    obs_max = float(per["auc_reject_vs_B3"].max())
    better = 0
    for _ in range(n_perm):
        perm_base = np.zeros_like(is_base)
        perm_base[rng.choice(is_base.size, is_base.sum(), replace=False)] = True
        if auc_for(X, cols, perm_base).max() >= obs_max:
            better += 1
    p_max = (better + 1) / (n_perm + 1)

    # the fingerprint classifier's own one-vs-rest view (LOO), from its committed predictions
    lp = pd.read_csv(O / "fingerprint" / "loo_predictions.csv")
    ovr = []
    for b in ("Batch_1", "Batch_2", "Batch_3"):
        yb = (lp["true"] == b).astype(int)
        ovr.append({"batch": b, "auc_p": roc_auc_score(yb, lp[f"p_{b}"]),
                    "auc_score": roc_auc_score(yb, -lp[f"score_{b}"]),  # score is a distance: lower = more typical
                    "recall": float(((lp["true"] == b) & (lp["assigned"] == b)).sum() / (lp["true"] == b).sum()),
                    "precision": float(((lp["true"] == b) & (lp["assigned"] == b)).sum() / max(1, (lp["assigned"] == b).sum()))})
    ovr = pd.DataFrame(ovr)
    ovr.to_csv(OUT / "fingerprint_ovr.csv", index=False)

    summary = {"n_columns": int(len(per)), "best_column": per.iloc[0]["column"], "best_auc": obs_max,
               "perm_p_max_auc": p_max, "n_perm": n_perm,
               "median_auc": float(per["auc_reject_vs_B3"].median()),
               "n_auc_above_0.7": int((per["auc_reject_vs_B3"] > 0.7).sum()),
               "fingerprint_ovr": ovr.round(3).to_dict(orient="records")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(per.head(12).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
