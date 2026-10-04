"""Soft one-electrode mixture vs fingerprint naive Bayes, with and without FEM features, on Modal.

    modal run scripts/modal_mixture.py            # 200 label permutations per config
    modal run scripts/modal_mixture.py --n-perm 20

Configs: model {nb, soft} x features {fp, fem, fp_fem}. fp = the 15 fingerprint
features (outputs/fingerprint/features.csv); fem = the 16 FEM tile features
(pmdb.classify.features, sym orientation) as the median over a site's 6 tiles.
Leave-one-site-out on the 31 labelled sites; the permutation test shuffles labels
with the same seeds for every config. Feature tables are built locally (CSV reads);
all fitting runs on Modal.

Writes outputs/mixture_poc/soft/{evaluation.json, loo_predictions.csv, heldout_predictions.csv}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # local pmdb for add_local_python_source + feature tables

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.4", "scipy==1.14.1", "scikit-image==0.25.2", "pandas==2.1.4",
        "Pillow==12.3.0", "tifffile==2025.5.10", "imagecodecs==2025.3.30",
    )
    .add_local_python_source("pmdb")
)
app = modal.App("pmdb-mixture", image=image)
FN_KW = dict(cpu=1.0, memory=2048, timeout=3600)

MODELS = ("nb", "soft")
FEATURE_SETS = ("fp", "fem", "fp_fem")


def _loo_assign(model, x, codes, batches):
    from pmdb import fingerprint as fp
    from pmdb import mixture
    if model == "nb":
        return fp._loo_assignments(x, codes, len(batches)), None
    post = mixture.loo_posterior(x, codes, batches)
    return post.argmax(1), post


def _bacc(codes, assigned, k):
    return float(sum((assigned[codes == c] == c).mean() for c in range(k)) / k)


@app.function(**FN_KW)
def observed(model: str, x: list, codes: list, batches: list, xh: list) -> dict:
    import numpy as np
    from pmdb import fingerprint as fp
    from pmdb import mixture
    x, codes, xh = np.asarray(x), np.asarray(codes), np.asarray(xh)
    assigned, post = _loo_assign(model, x, codes, batches)
    out = {"assigned": assigned.tolist(), "post": None if post is None else post.tolist()}
    if model == "nb":
        center, scale, ok, mu, sb = fp._fit_core(x, codes, len(batches))
        z = np.clip((xh[:, ok] - center[ok]) / scale[ok], -fp.MAX_Z, fp.MAX_Z)
        out["heldout_assigned"] = fp._scores(z, mu, sb).argmin(1).tolist()
        out["heldout_post"] = None
    else:
        m = mixture.fit(x, codes, batches)
        hp = mixture.posterior(m, xh)
        out["heldout_assigned"] = hp.argmax(1).tolist()
        out["heldout_post"] = hp.tolist()
        out["full_fit"] = {b: {"pi": float(c["pi"]), "resp": c["resp"].tolist(),
                               "rows": c["rows"].tolist()} for b, c in m["comps"].items()}
    return out


@app.function(**FN_KW)
def null_bacc(model: str, x: list, codes: list, batches: list, seeds: list) -> list:
    import numpy as np
    x, codes = np.asarray(x), np.asarray(codes)
    res = []
    for seed in seeds:
        cp = np.random.default_rng(seed).permutation(codes)
        a, _ = _loo_assign(model, x, cp, batches)
        res.append(_bacc(cp, a, len(batches)))
    return res


def _feature_tables():
    import pandas as pd
    from pmdb.classify.features import fem_tile_features, load_fem_tile_curves
    X = pd.read_csv(ROOT / "outputs/fingerprint/features.csv", index_col=[0, 1])
    H = pd.read_csv(ROOT / "outputs/fingerprint/heldout_features.csv", index_col=[0, 1])
    t = fem_tile_features(load_fem_tile_curves(ROOT / "outputs/fem/tile_curves.csv"))
    fcols = [c for c in t.columns if c.startswith("fem_")]
    F = t.groupby(["batch", "site"])[fcols].median()
    for T in (X, H):
        missing = set(T.index) - set(F.index)
        if missing:
            raise ValueError(f"no FEM features for {sorted(missing)}")
    sets = {
        "fp": (X, H),
        "fem": (F.loc[X.index], F.loc[H.index]),
        "fp_fem": (X.join(F), H.join(F)),
    }
    return X.index, H.index, sets


@app.local_entrypoint()
def main(n_perm: int = 200, chunk: int = 25):
    import numpy as np
    import pandas as pd

    idx, hidx, sets = _feature_tables()
    batches = sorted(idx.get_level_values("batch").unique())
    codes = np.array([batches.index(b) for b in idx.get_level_values("batch")])
    k = len(batches)
    configs = [(m, f) for f in FEATURE_SETS for m in MODELS]

    obs_args = [(m, sets[f][0].to_numpy(float).tolist(), codes.tolist(), batches,
                 sets[f][1].to_numpy(float).tolist()) for m, f in configs]
    seed_chunks = [list(range(s, min(s + chunk, n_perm))) for s in range(0, n_perm, chunk)]
    perm_args = [(m, sets[f][0].to_numpy(float).tolist(), codes.tolist(), batches, sc)
                 for m, f in configs for sc in seed_chunks]

    obs_calls = [observed.spawn(*a) for a in obs_args]
    null_lists = list(null_bacc.starmap(perm_args))
    obs = [c.get() for c in obs_calls]

    out = ROOT / "outputs/mixture_poc/soft"
    out.mkdir(parents=True, exist_ok=True)
    evaluation = {"n_perm": n_perm, "batches": batches, "configs": {}}
    loo = pd.DataFrame({"true": [batches[c] for c in codes]}, index=idx)
    held = pd.DataFrame(index=hidx)
    nc = len(seed_chunks)
    for j, ((m, f), o) in enumerate(zip(configs, obs)):
        name = f"{m}_{f}"
        a = np.array(o["assigned"])
        null = np.concatenate([null_lists[j * nc + i] for i in range(nc)])
        bacc = _bacc(codes, a, k)
        r = {
            "n_features": int(sets[f][0].shape[1]),
            "accuracy": float((a == codes).mean()),
            "balanced_accuracy": bacc,
            "recall": {b: float((a[codes == c] == c).mean()) for c, b in enumerate(batches)},
            "confusion": {bt: {ba: int(((codes == i) & (a == jj)).sum())
                               for jj, ba in enumerate(batches)} for i, bt in enumerate(batches)},
            "perm_null_mean_bacc": float(null.mean()),
            "perm_p_bacc": float((1 + (null >= bacc).sum()) / (len(null) + 1)),
        }
        loo[name] = [batches[c] for c in a]
        held[name] = [batches[c] for c in o["heldout_assigned"]]
        if o["post"] is not None:
            post = np.array(o["post"])
            r["log_loss"] = float(-np.mean(np.log(post[np.arange(len(codes)), codes])))
            for c, b in enumerate(batches):
                loo[f"{name}_P_{b}"] = post[:, c]
                held[f"{name}_P_{b}"] = np.array(o["heldout_post"])[:, c]
            r["full_fit"] = {b: {"pi": v["pi"],
                                 "resp": {idx[row][1]: round(rr, 3) for row, rr in zip(v["rows"], v["resp"])}}
                             for b, v in o["full_fit"].items()}
        evaluation["configs"][name] = r
        print(f"{name:12s} nfeat {r['n_features']:2d}  acc {r['accuracy']:.3f}  bacc {bacc:.3f}  "
              f"recall {[round(v, 2) for v in r['recall'].values()]}  perm p {r['perm_p_bacc']:.3f}"
              + (f"  logloss {r['log_loss']:.3f}" if "log_loss" in r else ""))
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    loo.to_csv(out / "loo_predictions.csv")
    held.to_csv(out / "heldout_predictions.csv")
    print(held[[c for c in held.columns if "_P_" not in c]].to_string())
    print(f"wrote {out}")
