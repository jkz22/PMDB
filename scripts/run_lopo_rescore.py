"""Re-score this branch's batch classifiers under leave-one-parent-out (LOPO).

main's `outputs/parent_groups.csv` shows the 31 labelled sites are crops of 13 parent images
(shared height, detector, grey step). Leave-one-site-out leaves sibling crops of the test site
in the training set; anything that reads the parent (acquisition session or electrode) is
therefore inflated. Every model family on this branch is re-run with the parent as the CV
group, next to its LOSO number, on the 31 labelled sites and on 34 (31 + the released
held-out truths). Permutation nulls permute site labels with the parent groups fixed.

Families: fingerprint16 naive Bayes (pmdb.fingerprint), nearest batch mean on the
`run_mean_predictor` feature families (grey-level/acquisition vs overlay/material), XGBoost on
the reliable-KPI sets (`run_xgb_reliable`). Writes outputs/parents/lopo_rescore.csv / .md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pmdb import fingerprint as fp  # noqa: E402
from run_mean_predictor import families  # noqa: E402
from run_xgb_reliable import FUNCTIONAL, PARAMS, RELIABLE_KPIS, fillna_train_median, site_table  # noqa: E402

O = ROOT / "outputs"
OUT = O / "parents"
B = ["Batch_1", "Batch_2", "Batch_3"]
N_PERM = {"fingerprint": 300, "nearest_mean": 500, "xgb": 100}
FP16 = ["si_depth_rel_band0", "si_depth_rel_band1", "si_depth_rel_band2", "si_depth_rel_band3", "si_depth_rel_band4",
        "si_depth_slope", "si_depth_mid_dip", "gx_0.5_2.0", "gx_2.0_4.0", "gx_4.0_7.0", "gx_7.0_10.0",
        "gz_0.5_2.0", "gz_2.0_4.0", "gz_4.0_7.0", "gz_7.0_10.0", "k15_contact_tilestd"]


def folds(groups: np.ndarray) -> list[np.ndarray]:
    return [groups == g for g in pd.unique(groups)]


# --- learners: each returns out-of-fold assigned codes ------------------------------------
def nearest_mean(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), int)
    for te in folds(groups):
        tr = ~te
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-12
        Z = np.nan_to_num((X - mu) / sd)
        cents = np.stack([Z[tr & (y == k)].mean(0) for k in range(3)])
        pred[te] = np.argmin(((Z[te][:, None, :] - cents[None]) ** 2).sum(2), 1)
    return pred


def fingerprint_nb(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), int)
    for te in folds(groups):
        tr = ~te
        center, scale, ok, mu, sb = fp._fit_core(X[tr], y[tr], 3)
        z = np.clip((X[te][:, ok] - center[ok]) / scale[ok], -fp.MAX_Z, fp.MAX_Z)
        pred[te] = np.argmin(fp._scores(z, mu, sb), 1)
    return pred


def xgb(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    pred = np.empty(len(y), int)
    for te in folds(groups):
        tr = ~te
        a, b = fillna_train_median(X[tr], X[te])
        m = XGBClassifier(objective="multi:softprob", num_class=3, random_state=seed, **PARAMS)
        m.fit(a, y[tr])
        pred[te] = m.predict_proba(b).argmax(1)
    return pred


def evaluate(fn, X, y, groups, n_perm, seed=0, n_jobs=1) -> dict:
    obs = fn(X, y, groups)
    acc = float((obs == y).mean())
    bal = float(np.mean([(obs[y == k] == k).mean() for k in range(3)]))
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(n_perm)]
    null = Parallel(n_jobs=n_jobs)(delayed(lambda yp: float((fn(X, yp, groups) == yp).mean()))(yp) for yp in perms)
    null = np.asarray(null)
    return dict(acc=acc, balanced_acc=bal, perm_p=(1 + (null >= acc).sum()) / (n_perm + 1), null_mean=null.mean(),
                recall_B1=float((obs[y == 0] == 0).mean()), recall_B2=float((obs[y == 1] == 1).mean()),
                recall_B3=float((obs[y == 2] == 2).mean()))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pg = pd.read_csv(O / "parent_groups.csv", dtype=str).set_index("site")
    truth = pd.read_csv(O / "heldout_labels.csv", dtype=str).set_index("site")["batch"]
    lab, held = site_table()
    S = pd.concat([lab, held])
    S.loc[truth.index, "batch"] = truth
    mp = pd.read_csv(O / "meanpred" / "site_features.csv", dtype={"site": str}).set_index("site").drop(columns=["batch"])
    S = S.join(mp)
    S["parent"] = pg.loc[S.index, "parent_id"]
    fam = families(list(mp.columns))

    sets = {
        ("fingerprint", "fingerprint16 NB"): (fingerprint_nb, FP16),
        ("nearest_mean", "nearest mean: fingerprint16"): (nearest_mean, FP16),
        ("nearest_mean", "nearest mean: overlay fractions (material)"): (nearest_mean, fam["overlay_frac"]),
        ("nearest_mean", "nearest mean: overlay depth profiles (material)"): (nearest_mean, fam["overlay_depth"]),
        ("nearest_mean", "nearest mean: raw BSE grey (acquisition)"): (nearest_mean, fam["grey_raw"]),
        ("nearest_mean", "nearest mean: harmonised BSE grey (acquisition)"): (nearest_mean, fam["grey_harm"]),
        ("nearest_mean", "nearest mean: graphite-only grey (acquisition)"): (nearest_mean, fam["grey_harm_graphite_only"]),
        ("nearest_mean", "nearest mean: pore-only grey (acquisition)"): (nearest_mean, fam["grey_harm_pore_only"]),
        ("xgb", "XGBoost: reliable KPIs"): (xgb, RELIABLE_KPIS),
        ("xgb", "XGBoost: reliable KPIs + functional"): (xgb, RELIABLE_KPIS + FUNCTIONAL),
        ("xgb", "XGBoost: fingerprint16"): (xgb, FP16),
        ("xgb", "XGBoost: all reliable (KPIs + fingerprint16 + functional)"): (xgb, RELIABLE_KPIS + FP16 + FUNCTIONAL),
    }
    rows = []
    for n_sites, T in (("31", S[S.index.isin(lab.index)]), ("34", S)):
        y = np.array([B.index(b) for b in T.batch])
        for (kind, name), (fn, cols) in sets.items():
            X = T[cols].to_numpy(float)
            for cv, groups in (("LOSO", np.arange(len(T))), ("LOPO", T.parent.to_numpy())):
                r = evaluate(fn, X, y, groups, N_PERM[kind], n_jobs=(8 if kind == "xgb" else 1))
                rows.append(dict(sites=n_sites, method=name, cv=cv, n_features=len(cols), **r))
                print(n_sites, name, cv, f"acc {r['acc']:.3f} p {r['perm_p']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lopo_rescore.csv", index=False)

    lines = ["# LOSO vs leave-one-parent-out (LOPO), every family on this branch", "",
             "Majority baseline: 17/31 = 0.548, 18/34 = 0.529. Permutation p: site labels permuted, parent groups fixed.", ""]
    for n_sites in ("31", "34"):
        d = df[df.sites == n_sites]
        lines += [f"## {n_sites} sites", "", "| method | LOSO acc (p) | LOPO acc (p) | LOPO balanced | LOPO recall B1/B2/B3 |", "|---|---|---|---|---|"]
        for name in dict.fromkeys(d.method):
            a = d[(d.method == name) & (d.cv == "LOSO")].iloc[0]
            b = d[(d.method == name) & (d.cv == "LOPO")].iloc[0]
            lines.append(f"| {name} | {a.acc:.3f} ({a.perm_p:.3f}) | {b.acc:.3f} ({b.perm_p:.3f}) | {b.balanced_acc:.3f} | "
                         f"{b.recall_B1:.2f}/{b.recall_B2:.2f}/{b.recall_B3:.2f} |")
        lines.append("")
    (OUT / "lopo_rescore.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
