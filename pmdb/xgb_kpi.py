"""D22 pre-registered XGBoost-on-screened-KPIs test (see .claude/plans/fem-swelling.md, D22).

Pure functions (no I/O beyond the input loaders) so the permutation chunks can run on Modal.
Arms: X1 = 24 screened KPIs; X2 = X1 + swell_resid, si_stress_spread (in-fold OLS);
X3 = X1 + top-2 raw FEM metrics selected in-fold (D21 A2). Suffix 'e' = edge5 FEM table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pmdb.fem_features import load_fem, select_features, subset, swell_resid_columns

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_KPIS = ("A02_curtaining_index", "A03_height_um")
XGB_PARAMS = dict(objective="multi:softprob", n_estimators=300, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0,
                  tree_method="hist", random_state=0, n_jobs=1)
MIN_CORRECT = 22
ALPHA = 0.05
FINGERPRINT_CORRECT = 21
FEATURE_COUNT_ORDER = ("X1", "X2", "X3")


def base_arm(arm: str) -> str:
    """X2e -> X2 (the 'e' suffix only switches the FEM table)."""
    return arm[:2]


def training_fingerprint(tr: dict, y) -> str:
    """sha256 over every training input the permutation null depends on (KPIs, FEM features, labels).

    Any change to the screened KPI table, the FEM table or the label set changes this hash.
    """
    import hashlib

    h = hashlib.sha256()
    for key in ("index", "kpi", "swell", "spread", "k01", "cand"):
        if key in tr:
            a = np.asarray(tr[key])
            h.update(key.encode())
            h.update(a.astype(str).tobytes() if a.dtype == object else np.ascontiguousarray(a, dtype=float).tobytes())
    h.update("|".join(tr.get("cand_names", [])).encode())
    h.update("|".join(map(str, y)).encode())
    return h.hexdigest()


def load_inputs(fem_path=None):
    """Return (tr, te, y, kpi_cols). tr/te are dicts with 'index' (batch, site), 'kpi'
    (the 24 KPI columns), and, when fem_path is given, 'swell', 'spread', 'k01', 'cand',
    'cand_names' (candidates finite on all labelled sites; same screen applied to held-out)."""
    rd = dict(dtype={"batch": str, "site": str})
    kp = pd.read_csv(ROOT / "outputs/kpis/screen/site_kpis_filtered.csv", **rd).set_index(["batch", "site"])
    kp = kp.drop(columns=list(EXCLUDE_KPIS))
    kph = pd.read_csv(ROOT / "outputs/heldout/kpis/site_kpis.csv", **rd).set_index(["batch", "site"])
    kph = kph[list(kp.columns)]
    if not np.isfinite(kp.to_numpy(float)).all() or not np.isfinite(kph.to_numpy(float)).all():
        raise ValueError("non-finite KPI")
    cols = list(kp.columns)

    def pack(K):
        return {"index": np.array(list(K.index), dtype=object).reshape(-1, 2), "kpi": K.to_numpy(float)}

    tr, te = pack(kp), pack(kph)
    y = pd.Series(kp.index.get_level_values("batch"), index=kp.index, name="batch")
    if fem_path is not None:
        fem, fem_h = load_fem(fem_path, False), load_fem(fem_path, True)
        for d, K, F in ((tr, kp, fem), (te, kph, fem_h)):
            sites = K.index.get_level_values("site")
            cand = F.drop(columns=["swell", "si_stress_spread"]).reindex(sites)
            d.update(swell=F["swell"].reindex(sites).to_numpy(float),
                     spread=F["si_stress_spread"].reindex(sites).to_numpy(float),
                     k01=K["K01_si_frac_adm"].to_numpy(float),
                     cand=cand.to_numpy(float), cand_names=list(cand.columns))
        keep = np.isfinite(tr["cand"]).all(axis=0)
        for d in (tr, te):
            d["cand"] = d["cand"][:, keep]
            d["cand_names"] = [n for n, k in zip(d["cand_names"], keep) if k]
            for name in ("swell", "spread", "k01"):
                if not np.isfinite(d[name]).all():
                    raise ValueError(f"non-finite {name}")
    return tr, te, y, cols


def fold_matrices(arm: str, tr: dict, te: dict, codes_tr: np.ndarray):
    """Train/test matrices for one fold; every fitted quantity uses the training rows only.
    Returns (xtr, xte, extra feature names, selection info or None)."""
    b = base_arm(arm)
    if b == "X1":
        return tr["kpi"], te["kpi"], [], None
    if b == "X2":
        c_tr, c_te, names = swell_resid_columns(tr, te)
        return np.column_stack([tr["kpi"], c_tr]), np.column_stack([te["kpi"], c_te]), names, None
    if b == "X3":
        chosen, p, skipped = select_features(tr["cand"], codes_tr)
        names = [tr["cand_names"][j] for j in chosen]
        return (np.column_stack([tr["kpi"], tr["cand"][:, chosen]]),
                np.column_stack([te["kpi"], te["cand"][:, chosen]]), names,
                {"chosen": chosen, "p": p, "skipped": skipped})
    raise ValueError(arm)


def class_weights(codes: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse class frequency sample weights, n / (K * n_c)."""
    counts = np.bincount(codes, minlength=n_classes).astype(float)
    return (len(codes) / (n_classes * counts))[codes]


def fit_predict(xtr, codes_tr, xte, n_classes):
    """Fit the fixed D22 XGBClassifier; return (test probabilities, fitted model)."""
    from xgboost import XGBClassifier
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(xtr, codes_tr, sample_weight=class_weights(codes_tr, n_classes))
    return model.predict_proba(xte), model


def loo_codes(arm, tr, codes, n_classes):
    n = len(codes)
    out = np.empty(n, dtype=int)
    for i in range(n):
        m = np.arange(n) != i
        xtr, xte, _, _ = fold_matrices(arm, subset(tr, m), subset(tr, ~m), codes[m])
        out[i] = fit_predict(xtr, codes[m], xte, n_classes)[0].argmax(axis=1)[0]
    return out


def perm_chunk(arm, tr, codes, n_classes, seed, ks):
    """LOO accuracy under label permutation k (seeded default_rng([seed, k])), whole
    pipeline (in-fold selection and fit) rerun; identical to a serial run."""
    out = []
    for k in ks:
        cp = np.random.default_rng([seed, int(k)]).permutation(codes)
        out.append(float((loo_codes(arm, tr, cp, n_classes) == cp).mean()))
    return out


def perm_summary(null, observed: float, seed: int) -> dict:
    null = np.asarray(null, float)
    n = len(null)
    return {"observed_accuracy": float(observed), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1.0 + (null >= observed - 1e-12).sum()) / (n + 1.0)),
            "n_perm": int(n), "seed": int(seed)}


def decide(results: dict) -> tuple[str, dict]:
    """D22 rule: BEATS iff LOO correct >= 22/31 and perm p <= 0.05; highest correct wins,
    tie -> fewer features (X1 < X2 < X3); none -> the fingerprint stays."""
    passes = {a: bool(r["n_correct"] >= MIN_CORRECT and r["perm_p"] <= ALPHA) for a, r in results.items()}
    ok = [a for a in passes if passes[a]]
    if not ok:
        return "fingerprint", passes
    best = max(results[a]["n_correct"] for a in ok)
    top = [a for a in ok if results[a]["n_correct"] == best]
    return min(top, key=lambda a: FEATURE_COUNT_ORDER.index(base_arm(a))), passes
