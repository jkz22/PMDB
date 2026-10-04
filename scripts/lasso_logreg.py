"""L1 (lasso) logistic regression on the full combined KPI list -> Batch_1 / 2 / 3.

Train on the 31 labelled sites, validate on the 3 held-back sites whose batches the
organisers released (3e122cbj = Batch_2, fn0mhxef = Batch_1, xrv9xvzb = Batch_3).
C is picked by stratified 5-fold CV on the training sites only, with imputer and scaler refitted inside
every fold. The LOO accuracy is nested: C is re-selected inside each leave-one-out fold, so a left-out
site's label never influences its own penalty.

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
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from batch_classifier import feature_sets, load_combined  # noqa: E402

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

    def search() -> GridSearchCV:
        # imputer + scaler inside the searched pipeline: refitted on each CV training split
        return GridSearchCV(pipe(l1_logreg(1.0)), {"logisticregression__C": CS},
                            cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            scoring="balanced_accuracy", n_jobs=-1)

    C = float(search().fit(Xtr, ytr).best_params_["logisticregression__C"])

    # nested LOO: the whole C search is rerun without the left-out site
    loo = cross_val_predict(search(), Xtr, ytr, cv=LeaveOneOut())
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
    print(f"train LOO balanced accuracy: {loo_bacc:.3f}")
    print(res.to_string(index=False))
    print(f"held-out correct: {(res.true == res.pred).sum()}/3")
    print("\nnon-zero coefficients:\n" + coef.round(3).to_string())


if __name__ == "__main__":
    main()
