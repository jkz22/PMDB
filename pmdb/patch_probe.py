"""Supervised linear probe on frozen patch embeddings, leave-one-parent-out (see docs/patch_mil.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
N_PCA = 64
C = 0.1


def _prepare_folds(X_by_site, site_parent, seed):
    """Per parent: train/test site lists and PCA-projected patches (label independent, reusable)."""
    sites = list(X_by_site)
    folds = []
    for par in sorted(set(site_parent[s] for s in sites)):
        test = [s for s in sites if site_parent[s] == par]
        train = [s for s in sites if site_parent[s] != par]
        assert not set(test) & set(train)
        assert all(site_parent[s] != par for s in train)
        Xtr = np.concatenate([X_by_site[s] for s in train]).astype(np.float32)
        sc = StandardScaler().fit(Xtr)
        pca = PCA(n_components=N_PCA, svd_solver="randomized", random_state=seed).fit(sc.transform(Xtr))
        Z = {s: pca.transform(sc.transform(X_by_site[s].astype(np.float32))) for s in sites}
        folds.append((par, test, train, Z))
    return folds


def _site_weights(train, n_patches, site_batch):
    """Per-patch weights: each site sums to 1/(sites in its batch), so batches weigh equally."""
    n_sites = {}
    for s in train:
        n_sites[site_batch[s]] = n_sites.get(site_batch[s], 0) + 1
    w = np.concatenate([np.full(n, 1.0 / (n_sites[site_batch[s]] * n)) for s, n in zip(train, n_patches)])
    return w * len(w) / w.sum()


def _fit_predict(folds, site_batch, strict=True):
    rows = []
    for par, test, train, Z in folds:
        Xtr = np.concatenate([Z[s] for s in train])
        ytr = np.concatenate([[site_batch[s]] * len(Z[s]) for s in train])
        w = _site_weights(train, [len(Z[s]) for s in train], site_batch)
        present = set(ytr)
        if strict:
            assert present == set(BATCHES), f"fold {par}: training lacks {set(BATCHES) - present}"
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(Xtr, ytr, sample_weight=w)
        for s in test:
            lp = np.full(len(BATCHES), -50.0)
            lpp = clf.predict_log_proba(Z[s]).mean(axis=0)
            for c, v in zip(clf.classes_, lpp):
                lp[BATCHES.index(c)] = v
            p = np.exp(lp - lp.max())
            p /= p.sum()
            rows.append({"parent_id": par, "site": s, "true": site_batch[s], "call": BATCHES[int(p.argmax())],
                         **{f"p_{b}": float(p[i]) for i, b in enumerate(BATCHES)}})
    return pd.DataFrame(rows)


def lopo_probe(X_by_site, site_batch, site_parent, seed=0, _folds=None, strict=True) -> pd.DataFrame:
    folds = _folds or _prepare_folds(X_by_site, site_parent, seed)
    return _fit_predict(folds, site_batch, strict)


def site_metrics(df: pd.DataFrame) -> dict:
    y = df["true"].to_numpy()
    c = df["call"].to_numpy()
    correct = y == c
    rec = {b: float(np.mean(c[y == b] == b)) if (y == b).any() else float("nan") for b in BATCHES}
    f1 = []
    for b in BATCHES:
        tp = np.sum((c == b) & (y == b))
        d = np.sum(c == b) + np.sum(y == b)
        f1.append(2 * tp / d if d else 0.0)
    pmax = df[[f"p_{b}" for b in BATCHES]].max(axis=1).to_numpy()
    high = pmax >= 0.5
    return {"n": int(len(df)), "accuracy": float(correct.mean()),
            "balanced_accuracy": float(np.nanmean(list(rec.values()))), "macro_f1": float(np.mean(f1)),
            "recall": rec,
            "confusion": {t: {p: int(np.sum((y == t) & (c == p))) for p in BATCHES} for t in BATCHES},
            "rubric_all_high": float(2 * correct.mean()),
            "rubric_flag_pmax_ge_0.5": float(np.mean(np.where(high, np.where(correct, 2, 0), 1))),
            "n_high": int(high.sum())}


def permutation_test(X_by_site, site_batch, site_parent, n_perm=200, seed=0) -> dict:
    folds = _prepare_folds(X_by_site, site_parent, seed)
    obs = site_metrics(_fit_predict(folds, site_batch))["accuracy"]
    rng = np.random.default_rng(seed)
    sites = list(X_by_site)
    labs = [site_batch[s] for s in sites]
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(sites))
        pb = {s: labs[perm[i]] for i, s in enumerate(sites)}
        null.append(_perm_acc(folds, pb))
    null = np.array(null)
    return {"observed_accuracy": obs, "n_perm": n_perm, "null_mean": float(null.mean()),
            "p_value": float((1 + np.sum(null >= obs)) / (1 + n_perm))}


def _perm_acc(folds, perm_batch):
    """Train on permuted labels, score calls against the permuted labels of the test sites."""
    df = _fit_predict(folds, perm_batch, strict=False)
    return float(np.mean(df["call"].to_numpy() == df["true"].to_numpy()))
