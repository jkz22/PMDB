"""Batch classifier on the combined KPI lanes (Batch_1 / Batch_2 / Batch_3).

Classic classifiers predict `batch` from per-site KPIs (geometric, materials
and analytical lanes). With n = 31 sites (7 / 7 / 17) every score is reported
against a label-shuffle null.

Regenerate:
    python scripts/batch_classifier.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import ROOT

KEYS = ["batch", "site"]
SENTINEL_COLS = ["K08_pcf_rpeak_x_um", "K08_pcf_rpeak_z_um"]
ANNA_DROP = ["se_detector", "n_tiles", "bse_black", "bse_graphite", "si_thr", "si_peak"]
ARTEFACT_RE = re.compile(
    r"^A\d|^mat_porosity$|^mat_rim_coverage$|^ab_inlens_texture.*|"
    r"^ab_inlens_edge_binder$|^ab_porosity.*"
)


def load_combined(kevin_csv, leo_csv, anna_csv) -> pd.DataFrame:
    kev = pd.read_csv(kevin_csv).drop(columns=["se_detector", "segmenter_version"])
    for c in SENTINEL_COLS:
        if c in kev.columns:
            kev[c] = kev[c].where(kev[c] != -1, np.nan)
    leo = pd.read_csv(leo_csv)
    leo = leo[KEYS + [c for c in leo.columns if c.startswith("mat_")]]
    anna = pd.read_csv(anna_csv)
    anna = anna.drop(columns=[c for c in ANNA_DROP if c in anna.columns])
    anna = anna.drop(
        columns=[c for c in anna.columns if c.endswith("__err") or c.endswith("__naive_err")]
    )
    anna = anna.rename(columns={c: f"ab_{c}" for c in anna.columns if c not in KEYS})

    out = kev.merge(leo, on=KEYS, how="inner").merge(anna, on=KEYS, how="inner")
    if any(len(out) != len(d) for d in (kev, leo, anna)):
        raise ValueError(
            f"site mismatch across lanes: kevin={len(kev)} leo={len(leo)} "
            f"anna={len(anna)} joined={len(out)}"
        )
    if out.columns.duplicated().any():
        raise ValueError("duplicated column names after join")
    return out.sort_values(KEYS).reset_index(drop=True)


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in df.columns if c not in KEYS]
    kevin = [c for c in cols if re.match(r"^(K\d|D\d|A\d)", c)]
    leo = [c for c in cols if c.startswith("mat_")]
    anna = [c for c in cols if c.startswith("ab_")]
    combined = kevin + leo + anna
    return {
        "kevin": kevin,
        "leo": leo,
        "anna": anna,
        "combined": combined,
        "combined_no_artefact": [c for c in combined if not ARTEFACT_RE.match(c)],
    }


def _pipe(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


MODELS = {
    "logreg": lambda: _pipe(
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
    ),
    "lda": lambda: _pipe(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "rf": lambda: _pipe(
        RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0)
    ),
    # Shallow, heavily subsampled trees: n = 31 sites. No class_weight equivalent.
    "xgb": lambda: _pipe(
        XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, random_state=0, n_jobs=1,
        )
    ),
}


def evaluate(df, cols, model, seed=0) -> dict:
    X = df[cols].to_numpy(dtype=float)
    # Integer labels: XGBClassifier rejects strings. Balanced accuracy is label-invariant.
    classes, y = np.unique(df["batch"].to_numpy(), return_inverse=True)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
    score, perm, pval = permutation_test_score(
        model,
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        n_permutations=200,
        scoring="balanced_accuracy",
        random_state=seed,
        n_jobs=-1,
    )
    loo = classes[cross_val_predict(model, X, y, cv=LeaveOneOut())]
    return {
        "n_features": len(cols),
        "cv_bal_acc_mean": scores.mean(),
        "cv_bal_acc_std": scores.std(),
        "perm_score": score,
        "null_mean": perm.mean(),
        "null_p95": np.percentile(perm, 95),
        "perm_p": pval,
        "loo_pred": loo,
    }


def importance(df, cols) -> pd.DataFrame:
    X = df[cols].to_numpy(dtype=float)
    y = df["batch"].to_numpy()
    lr = MODELS["logreg"]().fit(X, y)
    coef = np.abs(lr[-1].coef_)
    classes = lr[-1].classes_
    rf = MODELS["rf"]().fit(X, y)
    groups = [df.loc[df["batch"] == b] for b in classes]
    rows = []
    for i, c in enumerate(cols):
        vals = [g[c].dropna().to_numpy() for g in groups]
        try:
            p = kruskal(*vals).pvalue
        except ValueError:
            p = np.nan
        rows.append(
            {
                "feature": c,
                "logreg_max_abs_coef": coef[:, i].max(),
                "logreg_max_class": classes[coef[:, i].argmax()],
                "rf_importance": rf[-1].feature_importances_[i],
                "kruskal_p": p,
            }
        )
    return pd.DataFrame(rows).sort_values("rf_importance", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kevin", type=Path, default=ROOT / "outputs/kpis/site_kpis.csv")
    ap.add_argument("--leo", type=Path, default=ROOT / "outputs/kpis/materials_site_kpis.csv")
    ap.add_argument("--anna", type=Path, default=ROOT / "outputs/kpis/anna_site_kpis.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/batch_classifier")
    args = ap.parse_args()

    df = load_combined(args.kevin, args.leo, args.anna)
    fsets = feature_sets(df)
    out = args.out_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    loo = df[KEYS].copy()
    loo_combined = {}
    for fname, cols in fsets.items():
        for mname, make in MODELS.items():
            r = evaluate(df, cols, make())
            pred = r.pop("loo_pred")
            loo[f"{fname}__{mname}"] = pred
            if fname == "combined":
                loo_combined[mname] = pred
            rows.append({"feature_set": fname, "model": mname, **r})
    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    loo.to_csv(out / "loo_predictions.csv", index=False)

    imp = importance(df, fsets["combined"])
    imp.to_csv(out / "importance.csv", index=False)

    labels = sorted(df["batch"].unique())
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.3 * len(MODELS), 4))
    for ax, (mname, pred) in zip(axes, loo_combined.items()):
        cm = confusion_matrix(df["batch"], pred, labels=labels)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{mname} (LOO, combined)")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out / "figures/confusion_combined.png", dpi=150)
    plt.close(fig)

    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top["feature"], top["rf_importance"])
    ax.set_xlabel("RF importance (combined)")
    fig.tight_layout()
    fig.savefig(out / "figures/importance_top20.png", dpi=150)
    plt.close(fig)

    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
