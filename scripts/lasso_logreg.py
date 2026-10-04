"""L1 (lasso) logistic regression on the full combined KPI list -> Batch_1 / 2 / 3.

Train on the 31 labelled sites, validate on the 3 held-back sites whose batches the
organisers released (3e122cbj = Batch_2, fn0mhxef = Batch_1, xrv9xvzb = Batch_3).
C is picked by grouped 5-fold CV on the training sites only (groups = parent image, pmdb.parents), with
imputer and scaler refitted inside every fold. The headline accuracy is nested leave-one-PARENT-out: every
crop of the held-out parent is removed, and C is re-selected on the remaining parents, so neither the
left-out labels nor sibling crops of the same image influence the prediction or its penalty.

    python scripts/heldout_materials_kpis.py   # once, writes held-out mat_ KPIs
    python scripts/lasso_logreg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from batch_classifier import feature_sets, load_combined  # noqa: E402
from pmdb.parents import parent_groups  # noqa: E402

HELDOUT_TRUTH = {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}
OUT = ROOT / "outputs" / "kpis" / "lasso_logreg"
CS = np.logspace(-2, 2, 30)


def pipe(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def l1_logreg(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, penalty="l1", solver="saga", class_weight="balanced",
                              max_iter=20000, random_state=0)


def main() -> None:
    k = ROOT / "outputs" / "kpis"
    h = ROOT / "outputs" / "heldout" / "kpis"
    train = load_combined(k / "site_kpis.csv", k / "materials_site_kpis.csv", k / "anna_site_kpis.csv")
    test = load_combined(h / "site_kpis.csv", h / "materials_site_kpis.csv", h / "anna_site_kpis.csv")
    test["batch"] = test["site"].map(HELDOUT_TRUTH)
    cols = [c for c in feature_sets(train)["combined"] if train[c].notna().any()]
    Xtr, ytr = train[cols].to_numpy(float), train["batch"].to_numpy()
    Xte = test[cols].to_numpy(float)

    pg = parent_groups(include_test=False).drop_duplicates(["batch", "site"]).set_index(["batch", "site"])
    groups = pg.loc[list(zip(train["batch"], train["site"])), "parent_id"].to_numpy()

    def search(g: np.ndarray) -> GridSearchCV:
        # imputer + scaler inside the searched pipeline; inner folds never split a parent image
        return GridSearchCV(pipe(l1_logreg(1.0)), {"logisticregression__C": CS},
                            cv=GroupKFold(min(5, len(np.unique(g)))),
                            scoring="balanced_accuracy", n_jobs=-1)

    C = float(search(groups).fit(Xtr, ytr, groups=groups).best_params_["logisticregression__C"])

    # nested leave-one-parent-out: drop every crop of one parent, rerun the whole C search on the rest
    loo = np.empty_like(ytr)
    for tr_i, te_i in LeaveOneGroupOut().split(Xtr, ytr, groups):
        loo[te_i] = search(groups[tr_i]).fit(Xtr[tr_i], ytr[tr_i], groups=groups[tr_i]).predict(Xtr[te_i])
    loo_bacc = np.mean([np.mean(loo[ytr == c] == c) for c in np.unique(ytr)])

    fit = pipe(l1_logreg(C)).fit(Xtr, ytr)
    lr = fit[-1]
    coef = pd.DataFrame(lr.coef_.T, index=cols, columns=lr.classes_)
    coef = coef[(coef != 0).any(axis=1)]
    coef = coef.loc[coef.abs().max(axis=1).sort_values(ascending=False).index]

    proba = fit.predict_proba(Xte)
    res = test[["site", "batch"]].rename(columns={"batch": "true"})
    res["pred"] = lr.classes_[proba.argmax(1)]
    for i, c in enumerate(lr.classes_):
        res[f"p_{c}"] = proba[:, i].round(3)

    OUT.mkdir(parents=True, exist_ok=True)
    coef.to_csv(OUT / "coefficients.csv")
    res.to_csv(OUT / "heldout_predictions.csv", index=False)
    print(f"features: {len(cols)}  chosen C: {C:.3g}  non-zero KPIs: {len(coef)}")
    print(f"train leave-one-parent-out balanced accuracy: {loo_bacc:.3f} ({len(np.unique(groups))} parents)")
    print(res.to_string(index=False))
    print(f"held-out correct: {(res.true == res.pred).sum()}/3")
    print("\nnon-zero coefficients:\n" + coef.round(3).to_string())


if __name__ == "__main__":
    main()
