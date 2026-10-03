"""D10 check 3: compare site-level pooling rules on KPI tiles (plus the KPIs-alone baseline).

Leave-one-site-out over the labelled sites, one fixed model (median impute, standard scale,
LogisticRegression balanced C=1.0), four pooling rules, three tasks. Writes to outputs/pooling_checks/:
  check3_metrics.csv, check3_site_predictions.csv, check3_calibration.csv, check3_results.md
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pmdb.kpis import catalogue_columns  # noqa: E402
from tile_signal_checks import (  # noqa: E402
    BATCHES, P_COLS, SEED, VARIANTS, load_tiles, make_pipeline, md_table,
)

RULES = ["R1_mean_prob", "R2_feat_mean", "R3_feat_mean_max_std", "R4_site_kpis"]
BINS = [(0.33, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
N_BOOT = 1000


def pool_features(df: pd.DataFrame, features: list[str], rule: str) -> pd.DataFrame:
    g = df.groupby("site", sort=True)
    meta = g["batch"].first()
    stats = [("mean", np.nanmean)] if rule == "R2_feat_mean" else [
        ("mean", np.nanmean), ("max", np.nanmax), ("std", np.nanstd)]
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, fn in stats:
            part = g[features].agg(lambda s, fn=fn: np.nan if s.isna().all() else fn(s.to_numpy(float)))
            part.columns = features if rule == "R2_feat_mean" else [f"{c}__{name}" for c in features]
            parts.append(part)
    X = pd.concat(parts, axis=1)
    X.insert(0, "batch", meta)
    return X.reset_index()


def site_kpi_table(path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    site_cols, _ = catalogue_columns()
    cols = [c for group in site_cols.values() for c in group]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"site_kpis missing catalogue columns: {missing}")
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in num if np.nanstd(df[c].to_numpy(float)) > 0]
    return df[["batch", "site"] + feats], feats


def prob_record(site, batch, rule, task, P, cl) -> dict:
    rec = {"site": site, "batch": batch, "rule": rule, "task": task}
    for c in BATCHES:
        rec["p_" + c] = float(P[cl.index(c)]) if c in cl else 0.0
    rec["predicted"] = cl[int(np.argmax(P))]
    return rec


def loso_site(table: pd.DataFrame, feats: list[str], rule: str, task: str) -> pd.DataFrame:
    sub = table[table.batch.isin(VARIANTS[task])].sort_values("site").reset_index(drop=True)
    X, y, g = sub[feats].to_numpy(float), sub.batch.to_numpy(), sub.site.to_numpy()
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = make_pipeline().fit(X[tr], y[tr])
        P = m.predict_proba(X[te])[0]
        rows.append(prob_record(g[te[0]], y[te[0]], rule, task, P, list(m.classes_)))
    return pd.DataFrame(rows)


def loso_tile_mean(df: pd.DataFrame, features: list[str], task: str) -> pd.DataFrame:
    sub = df[df.batch.isin(VARIANTS[task])].reset_index(drop=True)
    X, y, g = sub[features].to_numpy(float), sub.batch.to_numpy(), sub.site.to_numpy()
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = make_pipeline().fit(X[tr], y[tr])
        P = m.predict_proba(X[te]).mean(axis=0)
        rows.append(prob_record(g[te[0]], y[te[0]], "R1_mean_prob", task, P, list(m.classes_)))
    return pd.DataFrame(rows).sort_values("site").reset_index(drop=True)


def metrics_for(t: pd.DataFrame, task: str) -> dict:
    classes = VARIANTS[task]
    y, pred = t.batch.to_numpy(), t.predicted.to_numpy()
    P = t[["p_" + c for c in classes]].to_numpy(float)
    Y = np.array([[float(b == c) for c in classes] for b in y])
    return {
        "site_acc": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=classes, zero_division=0),
        "log_loss": log_loss(y, np.clip(P, 1e-15, 1), labels=classes),
        "brier": float(((P - Y) ** 2).mean()),
    }


def bootstrap_ci(t: pd.DataFrame) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    y, pred = t.batch.to_numpy(), t.predicted.to_numpy()
    n = len(y)
    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            vals.append(balanced_accuracy_score(y[idx], pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def calibration(t: pd.DataFrame, task: str, rule: str) -> list[dict]:
    classes = VARIANTS[task]
    pmax = t[["p_" + c for c in classes]].to_numpy(float).max(axis=1)
    ok = t.predicted.to_numpy() == t.batch.to_numpy()
    rows = []
    for k, (lo, hi) in enumerate(BINS):
        m = (pmax >= lo) & ((pmax < hi) if k < len(BINS) - 1 else (pmax <= hi))
        rows.append({"rule": rule, "task": task, "bin": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()),
                     "accuracy": float(ok[m].mean()) if m.any() else np.nan})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile-kpis", default=str(ROOT / "outputs/kpis/tile_kpis.csv"))
    ap.add_argument("--site-kpis", default=str(ROOT / "outputs/kpis/site_kpis.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs/pooling_checks"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, features, dropped = load_tiles(args.tile_kpis)
    tables = {r: pool_features(df, features, r) for r in ["R2_feat_mean", "R3_feat_mean_max_std"]}
    feats_by_rule = {r: [c for c in t.columns if c not in ("batch", "site")] for r, t in tables.items()}
    tables["R4_site_kpis"], feats_by_rule["R4_site_kpis"] = site_kpi_table(args.site_kpis)

    preds = []
    for task in VARIANTS:
        for rule in RULES:
            if rule == "R1_mean_prob":
                preds.append(loso_tile_mean(df, features, task))
            else:
                preds.append(loso_site(tables[rule], feats_by_rule[rule], rule, task))
    sp = pd.concat(preds, ignore_index=True)[["site", "batch", "rule", "task"] + P_COLS + ["predicted"]]
    if not np.allclose(sp[P_COLS].sum(axis=1), 1.0, atol=1e-9, rtol=0):
        raise ValueError("probabilities do not sum to 1")

    mrows, crows = [], []
    for task in VARIANTS:
        for rule in RULES:
            t = sp[(sp.task == task) & (sp.rule == rule)]
            lo, hi = bootstrap_ci(t)
            mrows.append({"rule": rule, "task": task, "n_sites": len(t), **metrics_for(t, task),
                          "bacc_ci_lo": lo, "bacc_ci_hi": hi})
            crows += calibration(t, task, rule)
    metrics, cal = pd.DataFrame(mrows), pd.DataFrame(crows)
    sp.to_csv(out / "check3_site_predictions.csv", index=False)
    metrics.to_csv(out / "check3_metrics.csv", index=False)
    cal.to_csv(out / "check3_calibration.csv", index=False)

    L = ["# Check 3: site-level pooling rules (D10)\n"]
    L.append(f"- Inputs: `{args.tile_kpis}`, `{args.site_kpis}`")
    L.append(f"- Tile features ({len(features)}); dropped constant: {dropped}")
    L.append(f"- Feature counts: R1 {len(features)}; R2 {len(feats_by_rule['R2_feat_mean'])}; "
             f"R3 {len(feats_by_rule['R3_feat_mean_max_std'])}; R4 {len(feats_by_rule['R4_site_kpis'])}")
    L.append("- Model: median impute, StandardScaler, LogisticRegression(class_weight=balanced, C=1.0, "
             "max_iter=5000, random_state=0), refitted per leave-one-site-out fold; no tuning.")
    L.append("- Rules: R1 mean of tile probabilities; R2 per-feature tile mean; R3 per-feature [mean, max, "
             "std ddof=0]; R4 site-level KPIs (KPIs-alone baseline).")
    L.append(f"- Bootstrap: {N_BOOT} site resamples, seed 0, percentile 95% CI of balanced accuracy.")
    L.append("- Binary tasks: log-loss and Brier use the two task classes only.\n")
    L.append("## Metrics\n")
    L.append(md_table(metrics) + "\n")
    L.append("## Calibration (predicted max-prob bin -> n, accuracy)\n")
    L.append(md_table(cal) + "\n")
    L.append("## Plain reading\n")
    for task in VARIANTS:
        mt = metrics[metrics.task == task]
        best = mt.loc[mt.balanced_acc.idxmax()]
        L.append(f"- {task}: best by balanced accuracy is {best.rule} ({best.balanced_acc:.3f}, "
                 f"95% CI {best.bacc_ci_lo:.3f}-{best.bacc_ci_hi:.3f}); "
                 f"range across rules {mt.balanced_acc.min():.3f}-{mt.balanced_acc.max():.3f}.")
    L.append("")
    L.append("Caveat: with 31 labelled sites, differences below about 0.1 in balanced accuracy are within "
             "noise (see the bootstrap CIs).")
    (out / "check3_results.md").write_text("\n".join(L) + "\n")
    print(metrics[["rule", "task", "balanced_acc"]].to_string(index=False))


if __name__ == "__main__":
    main()
