"""KPI-PCA model: median impute -> standardise -> 5 PCs -> balanced logistic regression, evaluated leave-one-parent-out."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pmdb import patch_mil as pm

BATCHES = ("Batch_1", "Batch_2", "Batch_3")
assert tuple(pm.BATCHES) == BATCHES
# Batch_1 frames with ~2.5x Si fraction, excluded from training only.
OUTLIERS = ("5n1q8atc", "4ih2ggld")
N_PCS = 5


def make_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(N_PCS, random_state=0),
                         LogisticRegression(class_weight="balanced", max_iter=5000))


def lopo_proba(X, y, sites, groups) -> np.ndarray:
    X, y, sites, groups = np.asarray(X, float), np.asarray(y), np.asarray(sites), np.asarray(groups)
    out = np.zeros((len(y), len(BATCHES)))
    keep = ~np.isin(sites, OUTLIERS)
    for g in np.unique(groups):
        test = groups == g
        train = ~test & keep
        missing = set(BATCHES) - set(y[train])
        if missing:
            raise ValueError(f"fold {g}: classes missing from training: {sorted(missing)}")
        model = make_model().fit(X[train], y[train])
        proba = model.predict_proba(X[test])
        order = [list(model.classes_).index(b) for b in BATCHES]
        out[test] = proba[:, order]
    return out


def fit_full(X, y, sites):
    X, y, sites = np.asarray(X, float), np.asarray(y), np.asarray(sites)
    keep = ~np.isin(sites, OUTLIERS)
    return make_model().fit(X[keep], y[keep])
