"""Feature-level harmonisation with ComBat (neuroHarmonize) – the modelling-side counterpart of the image routes.

ComBat (Johnson, Li & Rabinovic, Biostatistics 2007) removes additive + multiplicative "site" effects
from a feature table while protecting covariates; `neuroHarmonize` (Pomponio et al., NeuroImage 2020,
github.com/rpomponio/neuroHarmonize) is the multi-scanner imaging implementation used here as shipped.

Design for PMDB: SITE = acquisition session fingerprint (``strong`` = the four 1030-px Batch-3 fields,
``mild`` = the re-quantised / offset Batch-3 fields, ``clean`` = everything else), and *batch* is the
protected covariate (one-hot), so that the material batch signal is kept and only the imaging-session
effect is regressed out. Models are learnt on the labelled sites and applied to the held-out features.

Inputs:  outputs/fingerprint/features.csv (+ heldout_features.csv), outputs/kpis/site_kpis.csv
Outputs: outputs/harmonisation_shift/combat/{fingerprint,kpis}_combat.csv (+ heldout), summary.json,
         session-shortcut / batch accuracies before, after (in-sample) and nested leave-one-parent-out.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pmdb import fingerprint as fp

STRONG = ("71vgq3fw", "kbdh4tri", "tuy3zymq", "x7u69zsw")
MILD = ("9luzk4jm", "hzumfsms", "ufdvpb81", "ptg8lmto", "xgj4xftb", "xrv9xvzb")


def session(site: str) -> str:
    return "strong" if site in STRONG else "mild" if site in MILD else "clean"


def _covars(df: pd.DataFrame) -> pd.DataFrame:
    cov = pd.DataFrame({"SITE": [session(s) for s in df.site]})
    for b in ("Batch_1", "Batch_2", "Batch_3"):
        cov[b] = (df.batch == b).astype(float).to_numpy()
    cov = cov.drop(columns=["Batch_1"])  # reference level
    return cov


def combat_fit_apply(train: pd.DataFrame, held: pd.DataFrame | None, cols: list[str]):
    from neuroHarmonize import harmonizationApply, harmonizationLearn
    X = train[cols].to_numpy(dtype=float)
    keep = np.isfinite(X).all(axis=0) & (np.nanstd(X, axis=0) > 0)
    cols = [c for c, k in zip(cols, keep) if k]
    X = X[:, keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, Xh = harmonizationLearn(X, _covars(train), eb=True)
    out = train.copy()
    out[cols] = Xh
    hout = None
    if held is not None and len(held):
        # harmonizationApply builds its design from the SITE levels present, so the held-out rows are
        # applied together with the training rows (same design columns) and sliced out afterwards.
        # Policy for a SITE level absent from training (no session parameters exist; neuroHarmonize
        # would return NaN): the row is left uncorrected.
        cov_h = _covars(held)
        cov_h[["Batch_2", "Batch_3"]] = 0.0  # held-out batch unknown -> reference level, nothing protected
        seen = cov_h["SITE"].isin(set(_covars(train)["SITE"])).to_numpy()
        hout = held.copy()
        hout["combat_applied"] = seen
        if seen.any():
            cov = pd.concat([_covars(train), cov_h[seen]], ignore_index=True)
            Xa = np.vstack([X, held.loc[seen, cols].to_numpy(dtype=float)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hout.loc[seen, cols] = harmonizationApply(Xa, cov, model)[len(train):]
    return out, hout, cols


def _parents(df: pd.DataFrame) -> np.ndarray:
    pg = pd.read_csv(REPO_ROOT / "outputs/parent_groups.csv").set_index("site")["parent_id"]
    return np.array([pg.get(s, s) for s in df.site])


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def _nested_lopo(train: pd.DataFrame, cols: list[str]) -> dict:
    """Leave-one-parent-out: ComBat is refitted on the training parents only and the held-out parent
    is transformed with its batch *unknown* (reference level), exactly as a real held-out site is
    (a session level unseen in training - the strong parent fold - stays uncorrected, see combat_fit_apply);
    classifiers are then fitted on the fold's transformed training rows. No test label enters the
    transform."""
    parents = _parents(train)
    y_sess = np.array([session(s) for s in train.site])
    y_batch = train.batch.to_numpy()
    pred = {k: np.empty(len(train), dtype=object) for k in ("session", "strong", "batch")}
    n_uncorrected = 0
    for p in np.unique(parents):
        te, tr = parents == p, parents != p
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_tr, a_te, c = combat_fit_apply(train[tr].reset_index(drop=True), train[te].reset_index(drop=True), cols)
        n_uncorrected += int((~a_te["combat_applied"]).sum())
        Xtr, Xte = np.nan_to_num(a_tr[c].to_numpy(float)), np.nan_to_num(a_te[c].to_numpy(float))
        for key, y in (("session", y_sess), ("strong", y_sess == "strong"), ("batch", y_batch)):
            if len(np.unique(y[tr])) < 2:
                pred[key][te] = y[tr][0]
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred[key][te] = _clf().fit(Xtr, y[tr]).predict(Xte)
    return {"nested_folds": len(np.unique(parents)),
            "nested_rows_uncorrected": int(n_uncorrected),
            "session_lopo_acc_nested": float((pred["session"] == y_sess).mean()),
            "strong_vs_rest_lopo_acc_nested": float((pred["strong"].astype(bool) == (y_sess == "strong")).mean()),
            "batch_lopo_acc_logreg_nested": float((pred["batch"] == y_batch).mean())}


def _lopo_acc(df: pd.DataFrame, cols: list[str], y: np.ndarray) -> float:
    parents = _parents(df)
    X = np.nan_to_num(df[cols].to_numpy(float))
    pred = np.empty(len(df), dtype=object)
    for p in np.unique(parents):
        te, tr = parents == p, parents != p
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred[te] = _clf().fit(X[tr], y[tr]).predict(X[te])
    return float((pred == y).mean())


def _evaluate(name: str, before: pd.DataFrame, after: pd.DataFrame, cols: list[str]) -> dict:
    """'before' = raw features; 'insample' = full-table ComBat (labels of every row used in the
    transform, so these scores are descriptive only); 'nested' = leave-one-parent-out refit, the
    honest generalisation estimate."""
    res = {"table": name, "n_features": len(cols)}
    for tag, df in (("before", before), ("insample", after)):
        y_sess = np.array([session(s) for s in df.site])
        res[f"session_lopo_acc_{tag}"] = _lopo_acc(df, cols, y_sess)
        res[f"strong_vs_rest_lopo_acc_{tag}"] = _lopo_acc(df, cols, y_sess == "strong")
        res[f"batch_lopo_acc_logreg_{tag}"] = _lopo_acc(df, cols, df.batch.to_numpy())
        if name == "fingerprint":
            Xi = df.set_index(["batch", "site"])
            _, m = fp.loo_evaluate(Xi, Xi.index.get_level_values("batch").to_series(index=Xi.index), features=cols)
            res[f"fingerprint_loo_acc_{tag}"] = m["accuracy"]
    res.update(_nested_lopo(before, cols))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "harmonisation_shift" / "combat"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "fingerprint": (REPO_ROOT / "outputs/fingerprint/features.csv", REPO_ROOT / "outputs/fingerprint/heldout_features.csv"),
        "kpis": (REPO_ROOT / "outputs/kpis/site_kpis.csv", REPO_ROOT / "outputs/heldout/site_kpis.csv"),
    }
    results = []
    for name, (p_train, p_held) in tables.items():
        if not p_train.exists():
            print(f"  {name}: {p_train} missing, skipped")
            continue
        train = pd.read_csv(p_train)
        held = pd.read_csv(p_held) if p_held.exists() else None
        cols = [c for c in train.columns if c not in ("batch", "site") and pd.api.types.is_numeric_dtype(train[c])]
        after, hafter, cols = combat_fit_apply(train, held, cols)
        after.to_csv(out / f"{name}_combat.csv", index=False)
        if hafter is not None:
            hafter.to_csv(out / f"{name}_combat_heldout.csv", index=False)
        r = _evaluate(name, train, after, cols)
        r["session_sizes"] = {k: int(v) for k, v in pd.Series([session(s) for s in train.site]).value_counts().items()}
        results.append(r)
        print(json.dumps(r, indent=1))
    (out / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
