"""Proof of concept: one-electrode-per-batch mixture model vs the fingerprint naive Bayes.

Assumption: each batch is one electrode imaged at several places. Batch 3 (supplier
baseline) is one "normal" population N. Batch 1 and Batch 2 are mixtures: a fraction
pi_b of sites carry that batch's defect D_b, the rest look like N:

    p(x | B3) = N(x)
    p(x | Bb) = pi_b * D_b(x) + (1 - pi_b) * N(x),   b in {B1, B2}

Within each training fold: N = Batch 3 robust Laplace params; a Batch 1/2 site is a
"defect site" if its Laplace score against N exceeds the 90th percentile of Batch 3's
own leave-one-out scores; D_b is fitted on b's defect sites; pi_b = (n_def + 0.5) /
(n_b + 1). Same features, standardisation and per-feature Laplace score as
pmdb/fingerprint.py, so the only change is the mixture structure. Uniform batch prior.

    python scripts/poc_mixture.py

Writes outputs/mixture_poc/{evaluation.json, loo_predictions.csv, heldout_predictions.csv}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmdb import fingerprint as fp  # noqa: E402

DEFECT_PCT = 90.0   # defect threshold: percentile of Batch 3 LOO scores against N
MIN_SITES_FOR_SCALE = 3  # below this, D_b uses the pooled scale (1 in z space)
N_PERM = 200
BASE = "Batch_3"


def _laplace_params(z):
    mu = np.median(z, axis=0)
    if len(z) < MIN_SITES_FOR_SCALE:
        return mu, np.ones_like(mu)
    s = fp.SCALE_SHRINK + (1 - fp.SCALE_SHRINK) * fp._robust_scale(z - mu)
    return mu, np.where(s > 0, s, 1.0)


def _loglik(z, mu, s):
    """Per-feature-mean Laplace log-likelihood (same temperature as fp._scores)."""
    return -fp._scores(z, mu[None], s[None])[:, 0]


def fit(x, codes, batches):
    center, scale, ok, _, _ = fp._fit_core(x, codes, len(batches))
    z = np.clip((x[:, ok] - center[ok]) / scale[ok], -fp.MAX_Z, fp.MAX_Z)
    ib = batches.index(BASE)
    zN = z[codes == ib]
    muN, sN = _laplace_params(zN)
    tau = np.percentile(fp._loo_scores(zN), DEFECT_PCT)
    comps = {}
    for b, name in enumerate(batches):
        if b == ib:
            continue
        zb = z[codes == b]
        defect = -_loglik(zb, muN, sN) > tau
        n_def = int(defect.sum())
        # no defect site in this fold: fall back to the whole batch as D_b
        muD, sD = _laplace_params(zb[defect] if n_def else zb)
        comps[name] = dict(pi=(n_def + 0.5) / (len(zb) + 1), mu=muD, s=sD, n_def=n_def,
                           defect_rows=np.flatnonzero(codes == b)[defect])
    return dict(center=center, scale=scale, ok=ok, muN=muN, sN=sN, tau=tau,
                comps=comps, batches=batches)


def posterior(m, x):
    ok = m["ok"]
    z = np.clip((x[:, ok] - m["center"][ok]) / m["scale"][ok], -fp.MAX_Z, fp.MAX_Z)
    lN = _loglik(z, m["muN"], m["sN"])
    like = {}
    for name in m["batches"]:
        if name == BASE:
            like[name] = np.exp(lN)
        else:
            c = m["comps"][name]
            like[name] = c["pi"] * np.exp(_loglik(z, c["mu"], c["s"])) + (1 - c["pi"]) * np.exp(lN)
    L = np.column_stack([like[b] for b in m["batches"]])
    return L / L.sum(axis=1, keepdims=True)


def loo(x, codes, batches):
    n = len(x)
    post = np.empty((n, len(batches)))
    for i in range(n):
        tr = np.arange(n) != i
        post[i] = posterior(fit(x[tr], codes[tr], batches), x[i:i + 1])[0]
    return post


def metrics(codes, assigned, batches):
    rec = {b: float((assigned[codes == k] == k).mean()) for k, b in enumerate(batches)}
    return {
        "accuracy": float((assigned == codes).mean()),
        "balanced_accuracy": float(np.mean(list(rec.values()))),
        "recall": rec,
        "confusion": {bt: {ba: int(((codes == i) & (assigned == j)).sum())
                           for j, ba in enumerate(batches)} for i, bt in enumerate(batches)},
    }


def main() -> int:
    out = ROOT / "outputs/mixture_poc"
    out.mkdir(parents=True, exist_ok=True)
    X = pd.read_csv(ROOT / "outputs/fingerprint/features.csv", index_col=[0, 1])
    H = pd.read_csv(ROOT / "outputs/fingerprint/heldout_features.csv", index_col=[0, 1])
    batches = sorted(X.index.get_level_values("batch").unique())
    codes = np.array([batches.index(b) for b in X.index.get_level_values("batch")])
    x = X.to_numpy(float)

    base_assigned = fp._loo_assignments(x, codes, len(batches))
    post = loo(x, codes, batches)
    mix_assigned = post.argmax(1)
    res = {"baseline_naive_bayes": metrics(codes, base_assigned, batches),
           "mixture": metrics(codes, mix_assigned, batches)}
    res["mixture"]["log_loss"] = float(-np.mean(np.log(post[np.arange(len(codes)), codes])))
    res["mixture"]["brier"] = float(np.mean(np.sum((post - np.eye(len(batches))[codes]) ** 2, 1)))

    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_PERM):
        cp = rng.permutation(codes)
        null.append(metrics(cp, loo(x, cp, batches).argmax(1), batches)["balanced_accuracy"])
    null = np.array(null)
    obs = res["mixture"]["balanced_accuracy"]
    res["mixture"]["permutation"] = {
        "n_perm": N_PERM, "null_mean_bacc": float(null.mean()),
        "p_value": float((1 + (null >= obs).sum()) / (N_PERM + 1))}

    full = fit(x, codes, batches)
    res["defect_sites_full_fit"] = {b: [X.index[r][1] for r in c["defect_rows"]]
                                    for b, c in full["comps"].items()}
    res["pi_full_fit"] = {b: c["pi"] for b, c in full["comps"].items()}
    res["settings"] = {"defect_percentile": DEFECT_PCT, "min_sites_for_scale": MIN_SITES_FOR_SCALE,
                       "features": list(X.columns)}
    (out / "evaluation.json").write_text(json.dumps(res, indent=2))

    loo_df = pd.DataFrame(post, index=X.index, columns=[f"P_{b}" for b in batches])
    loo_df.insert(0, "mixture", [batches[k] for k in mix_assigned])
    loo_df.insert(0, "baseline", [batches[k] for k in base_assigned])
    loo_df.insert(0, "true", [batches[k] for k in codes])
    loo_df.to_csv(out / "loo_predictions.csv")
    hp = pd.DataFrame(posterior(full, H.to_numpy(float)), index=H.index,
                      columns=[f"P_{b}" for b in batches])
    hp.insert(0, "assigned", hp.idxmax(axis=1).str[2:])
    hp.to_csv(out / "heldout_predictions.csv")

    for k in ("baseline_naive_bayes", "mixture"):
        r = res[k]
        print(f"{k}: acc {r['accuracy']:.3f}  bacc {r['balanced_accuracy']:.3f}  recall {r['recall']}")
        for bt, row in r["confusion"].items():
            print(f"   {bt}: {row}")
    print("mixture log-loss", round(res["mixture"]["log_loss"], 3), "brier", round(res["mixture"]["brier"], 3),
          "perm", res["mixture"]["permutation"])
    print("defect sites (full fit):", res["defect_sites_full_fit"], "pi:", res["pi_full_fit"])
    print(hp.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
