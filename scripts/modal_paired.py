"""Paired-contrast feature selection + naive Bayes vs fingerprint NB baseline, on Modal.

    python scripts/paired_contrasts.py        # builds outputs/paired_contrasts/features.csv first
    modal run scripts/modal_paired.py         # 200 site-level label permutations
    modal run scripts/modal_paired.py --n-perm 10   # smoke test

Methods: baseline = robust NB (pmdb.fingerprint) on the 16 fingerprint features;
paired = same NB on K features chosen by pmdb.paired.select from the 104 candidates,
re-selected inside every CV fold (training crops only) and inside every permutation.
Schemes: LOSO (31 folds) and leave-one-parent-out (13 folds). Permutations shuffle the
31 site labels (seeds 0..n_perm-1, same seeds for every config); parents stay fixed.

Writes outputs/paired_contrasts/{evaluation.json, cv_predictions.csv, heldout_predictions.csv,
selection_frequency.csv}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(  # pmdb/__init__ imports pmdb.io, which needs the imaging stack
        "numpy==1.26.4", "scipy==1.14.1", "scikit-image==0.25.2", "pandas==2.1.4",
        "Pillow==12.3.0", "tifffile==2025.5.10", "imagecodecs==2025.3.30",
    )
    .add_local_python_source("pmdb")
)
app = modal.App("pmdb-paired", image=image)
FN_KW = dict(cpu=1.0, memory=2048, timeout=3600)

METHODS = ("baseline", "paired")
SCHEMES = ("loso", "lopo")


def _folds(scheme, sites, parents):
    import numpy as np
    return np.asarray(sites if scheme == "loso" else parents)


@app.function(**FN_KW)
def observed(method: str, scheme: str, x: list, codes: list, sites: list, parents: list,
             n_batches: int) -> dict:
    import numpy as np
    from pmdb import paired
    x, codes, parents = np.asarray(x, float), np.asarray(codes), np.asarray(parents)
    a, proba, sel = paired.cv_assign(x, codes, parents, _folds(scheme, sites, parents),
                                     n_batches, method)
    return {"assigned": a.tolist(), "proba": proba.tolist(), "selected": sel}


@app.function(**FN_KW)
def null_bacc(method: str, scheme: str, x: list, codes: list, sites: list, parents: list,
              n_batches: int, seeds: list) -> list:
    import numpy as np
    from pmdb import paired
    x, codes, parents = np.asarray(x, float), np.asarray(codes), np.asarray(parents)
    folds = _folds(scheme, sites, parents)
    out = []
    for seed in seeds:
        cp = np.random.default_rng(seed).permutation(codes)
        a, _, _ = paired.cv_assign(x, cp, parents, folds, n_batches, method)
        out.append(paired.balanced_accuracy(cp, a, n_batches))
    return out


@app.function(**FN_KW)
def heldout(X: dict, H: dict, y: dict, parents: list, n_batches: int, base_cols: list) -> dict:
    import numpy as np
    import pandas as pd
    from pmdb import fingerprint as fp
    from pmdb import paired
    X, H, y = pd.DataFrame(X), pd.DataFrame(H), pd.Series(y)
    batches = sorted(y.unique())
    codes = np.array([batches.index(b) for b in y.reindex(X.index)])
    sel = paired.select(X.to_numpy(float), codes, np.asarray(parents), n_batches)
    feats = {"baseline": base_cols, "paired": [X.columns[j] for j in sel]}
    out = {}
    for m, f in feats.items():
        pred = fp.predict(fp.fit(X, y, features=f), H)
        out[m] = {"features": f, "pred": pred.reset_index().to_dict(orient="list")}
    return out


def _metrics(codes, a, batches):
    import numpy as np
    k = len(batches)
    return {
        "accuracy": float((a == codes).mean()),
        "balanced_accuracy": float(np.mean([(a[codes == c] == c).mean() for c in range(k)])),
        "recall": {b: float((a[codes == c] == c).mean()) for c, b in enumerate(batches)},
        "confusion": {bt: {ba: int(((codes == i) & (a == j)).sum()) for j, ba in enumerate(batches)}
                      for i, bt in enumerate(batches)},
    }


@app.local_entrypoint()
def main(n_perm: int = 200, chunk: int = 25):
    import numpy as np
    import pandas as pd
    from pmdb import paired

    out = ROOT / "outputs/paired_contrasts"
    X = pd.read_csv(out / "features.csv", index_col=[0, 1])
    H = pd.read_csv(out / "heldout_features.csv", index_col=[0, 1])
    parents = X.pop("parent").tolist()
    H.pop("parent")
    sites = X.index.get_level_values("site").tolist()
    assert parents == paired.parent_of(sites, require=set(sites))
    batches = sorted(X.index.get_level_values("batch").unique())
    codes = np.array([batches.index(b) for b in X.index.get_level_values("batch")])
    k = len(batches)
    base_cols = [c for c in X.columns if c.startswith("fp:")]
    assert len(base_cols) == 16, base_cols
    xs = {"baseline": X[base_cols].to_numpy(float).tolist(), "paired": X.to_numpy(float).tolist()}

    configs = [(m, s) for s in SCHEMES for m in METHODS]
    seed_chunks = [list(range(s, min(s + chunk, n_perm))) for s in range(0, n_perm, chunk)]
    common = lambda m: (xs[m], codes.tolist(), sites, parents, k)  # noqa: E731
    obs_calls = [observed.spawn(m, s, *common(m)) for m, s in configs]
    Xs = X.reset_index(drop=True)
    Xs.index = sites
    Hs = H.reset_index(drop=True)
    Hs.index = H.index.get_level_values("site").tolist()
    held_call = heldout.spawn(Xs.to_dict(), Hs.to_dict(),
                              dict(zip(sites, [batches[c] for c in codes])), parents, k, base_cols)
    perm_args = [(m, s, *common(m), sc) for m, s in configs for sc in seed_chunks]
    null_lists = list(null_bacc.starmap(perm_args)) if n_perm else []
    obs = [c.get() for c in obs_calls]
    held = held_call.get()

    evaluation = {"n_perm": n_perm, "k_select": paired.K_SELECT, "batches": batches,
                  "n_candidates": X.shape[1], "configs": {}}
    cvp = pd.DataFrame({"parent": parents, "true": [batches[c] for c in codes]}, index=X.index)
    freq = {}
    nc = len(seed_chunks)
    for j, ((m, s), o) in enumerate(zip(configs, obs)):
        name = f"{m}_{s}"
        a = np.array(o["assigned"])
        r = _metrics(codes, a, batches)
        if n_perm:
            null = np.concatenate([null_lists[j * nc + i] for i in range(nc)])
            r["perm_null_mean_bacc"] = float(null.mean())
            r["perm_null_p95_bacc"] = float(np.percentile(null, 95))
            r["perm_p_bacc"] = float((1 + (null >= r["balanced_accuracy"]).sum()) / (len(null) + 1))
        proba = np.array(o["proba"])
        p_pred = proba[np.arange(len(a)), a]
        correct = a == codes
        rub = paired.rubric_score(correct, p_pred)
        r["rubric_mean"] = float(rub.mean())
        r["n_high_conf"] = int((p_pred > 0.5).sum())
        bins = {"<0.5": p_pred < 0.5, "0.5-0.7": (p_pred >= 0.5) & (p_pred <= 0.7),
                ">0.7": p_pred > 0.7}
        r["calibration_bins"] = {b: {"n": int(m_.sum()),
                                     "accuracy": float(correct[m_].mean()) if m_.any() else None}
                                 for b, m_ in bins.items()}
        cvp[name] = [batches[c] for c in a]
        cvp[f"{name}_p_pred"] = p_pred
        cvp[f"{name}_rubric"] = rub
        if m == "paired":
            cols = X.columns
            cnt = pd.Series([cols[i] for f in o["selected"] for i in f]).value_counts()
            freq[name] = cnt / len(o["selected"])
        evaluation["configs"][name] = r
        print(f"{name:16s} acc {r['accuracy']:.3f}  bacc {r['balanced_accuracy']:.3f}  "
              f"recall {[round(v, 2) for v in r['recall'].values()]}  "
              f"perm p {r.get('perm_p_bacc', float('nan')):.3f}  rubric {r['rubric_mean']:.3f}")
    evaluation["heldout_features"] = {m: v["features"] for m, v in held.items()}
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    cvp.to_csv(out / "cv_predictions.csv")
    pd.DataFrame(freq).fillna(0).sort_values("paired_loso", ascending=False).to_csv(
        out / "selection_frequency.csv", float_format="%.3f")
    hp = []
    for m, v in held.items():
        d = pd.DataFrame(v["pred"])
        d.insert(0, "method", m)
        sc = d[[f"score_{b}" for b in batches]].to_numpy(float)
        pr = paired.softmax_proba(sc, float(len(v["features"])))
        for c, b in enumerate(batches):
            d[f"P_{b}"] = pr[:, c]
        d["P_pred"] = pr.max(axis=1)
        d["conf_label"] = np.where(d["P_pred"] > 0.5, "high", "low")
        hp.append(d)
    hp = pd.concat(hp).rename(columns={"index": "site"})
    hp.insert(2, "parent", paired.parent_of(hp["site"]))
    hp.to_csv(out / "heldout_predictions.csv", index=False, float_format="%.4f")
    print(hp.to_string())
    # Held-out feature values vs Batch 3 (z = (value - B3 mean) / B3 SD over the 17 B3 crops).
    b3 = X[X.index.get_level_values("batch") == "Batch_3"]
    rows = []
    for m, v in held.items():
        for f in v["features"]:
            for site, val in zip(Hs.index, Hs[f]):
                rows.append({"method": m, "site": site, "feature": f, "value": float(val),
                             "b3_mean": float(b3[f].mean()), "b3_sd": float(b3[f].std()),
                             "z_vs_b3": float((val - b3[f].mean()) / b3[f].std()),
                             **{f"mean_{b}": float(X[X.index.get_level_values("batch") == b][f].mean())
                                for b in batches}})
    pd.DataFrame(rows).to_csv(out / "heldout_vs_batch3.csv", index=False, float_format="%.4f")
    print(f"wrote {out}")
