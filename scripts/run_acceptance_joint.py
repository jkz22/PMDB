#!/usr/bin/env python3
"""Pre-specified multivariate acceptance score vs the Batch 3 baseline (no column search).

Three families fixed in advance: v1 KPIs (K*/A*), fingerprint features, functional F*. For each
family the score of a field is the RMS of its leave-one-out |robust z| against Batch 3 over all
columns of the family. AUC for rejecting Batch 1+2, permutation p per family (Bonferroni x3), and
the same score for the held-out sites with its percentile among the LOO Batch 3 scores.

    python scripts/run_acceptance_joint.py  -> outputs/acceptance/{joint.csv, heldout_joint.csv, joint_summary.json}
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
sys.path.insert(0, str(ROOT / "scripts"))
from run_acceptance import BASE, O, OUT, SEED, load_features, loo_abs_z  # noqa: E402

FAMILIES = {
    "kpi_v1": lambda c: c[:1] in "KA" and c[1:3].isdigit(),
    "fingerprint": lambda c: c.startswith(("si_depth", "gx_", "gz_", "k15_")),
    "functional": lambda c: c.startswith("F"),
}


def family_score(X: pd.DataFrame, cols: list[str], is_base: np.ndarray) -> np.ndarray:
    Z = np.column_stack([loo_abs_z(X[c].to_numpy(dtype=float), is_base) for c in cols])
    return np.sqrt(np.nanmean(Z ** 2, axis=1))


def heldout_score(X: pd.DataFrame, H: pd.DataFrame, cols: list[str], is_base: np.ndarray) -> np.ndarray:
    z = []
    for c in cols:
        ref = X.loc[is_base, c].to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        med = np.median(ref)
        mad = 1.4826 * np.median(np.abs(ref - med))
        z.append(np.abs(H[c].to_numpy(dtype=float) - med) / mad if mad > 0 else np.full(len(H), np.nan))
    return np.sqrt(np.nanmean(np.column_stack(z) ** 2, axis=1))


def load_heldout() -> pd.DataFrame:
    def key(df):
        return df.assign(key=df["batch"] + "/" + df["site"]).set_index("key")
    kp = key(pd.read_csv(O / "heldout" / "kpis" / "site_kpis.csv"))
    fp = key(pd.read_csv(O / "fingerprint" / "heldout_features.csv"))
    fu = key(pd.read_csv(O / "functional" / "heldout_site_functional.csv"))
    H = kp
    for df in (fp, fu):
        H = H.join(df[[c for c in df.columns if c not in H.columns]], how="left")
    return H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    n_perm = ap.parse_args().n_perm
    X = load_features()
    H = load_heldout()
    is_base = (X["batch"] == BASE).to_numpy()
    y = (~is_base).astype(int)
    usable = [c for c in X.columns if c not in ("batch", "site") and X[c].notna().sum() >= 25
              and X[c].nunique() > 3 and c in H.columns]
    rng = np.random.default_rng(SEED)
    rows, hrows, summary = [], [], {}
    for fam, pred in FAMILIES.items():
        cols = [c for c in usable if pred(c)]
        s = family_score(X, cols, is_base)
        auc = roc_auc_score(y, s)
        better = 0
        for _ in range(n_perm):
            pb = np.zeros_like(is_base)
            pb[rng.choice(is_base.size, is_base.sum(), replace=False)] = True
            if roc_auc_score((~pb).astype(int), family_score(X, cols, pb)) >= auc:
                better += 1
        p = (better + 1) / (n_perm + 1)
        base_scores = s[is_base]
        hs = heldout_score(X, H, cols, is_base)
        for site, v in zip(H["site"], hs):
            hrows.append({"family": fam, "site": site, "score": float(v),
                          "pct_among_B3_loo": float((base_scores < v).mean() * 100),
                          "pct_among_B12": float((s[~is_base] < v).mean() * 100)})
        for k, v in zip(X.index, s):
            rows.append({"family": fam, "key": k, "batch": X.loc[k, "batch"], "score": float(v)})
        summary[fam] = {"n_columns": len(cols), "auc_reject_vs_B3": float(auc), "perm_p": p,
                        "perm_p_bonferroni3": min(1.0, 3 * p),
                        "median_score_B3": float(np.median(base_scores)),
                        "median_score_B12": float(np.median(s[~is_base]))}
    pd.DataFrame(rows).to_csv(OUT / "joint.csv", index=False)
    hj = pd.DataFrame(hrows)
    hj.to_csv(OUT / "heldout_joint.csv", index=False)
    summary["n_perm"] = n_perm
    (OUT / "joint_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(hj.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
