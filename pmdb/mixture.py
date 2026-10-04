"""Soft one-electrode-per-batch mixture classifier (proof of concept).

Assumption: each batch is one electrode imaged at several places. The baseline
batch (Batch 3) is one "normal" population N; every other batch b is a mixture
of N and a batch-specific defect component D_b:

    p(x | base) = N(x)
    p(x | b)    = pi_b * D_b(x) + (1 - pi_b) * N(x)

N uses the same robust per-batch Laplace params as `pmdb.fingerprint`. D_b and
pi_b are fitted by EM on batch b's sites: each site gets a responsibility r_i
(probability it is a defect site) instead of a hard in/out cut. Likelihoods use
the fingerprint's per-feature-mean (tempered) Laplace score, so the only change
from the naive Bayes is the mixture structure. Uniform batch prior.
"""

from __future__ import annotations

import numpy as np

from pmdb import fingerprint as fp

EM_ITERS = 200
EM_TOL = 1e-8
#: Pseudo-sites pulling the defect component's per-feature scale toward the
#: pooled scale (1 in standardized space); a component carried by 2-3 sites
#: cannot estimate its own spread.
SCALE_PRIOR_N = 3.0
#: Pseudo-count on each side of pi_b (Beta(0.5, 0.5) style), so pi stays in (0, 1).
PI_PRIOR = 0.5


def _loglik(z: np.ndarray, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-feature-mean Laplace log-likelihood (same temperature as fp._scores)."""
    return -fp._scores(z, mu[None], s[None])[:, 0]


def _weighted_median(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    order = np.argsort(z, axis=0)
    zs = np.take_along_axis(z, order, axis=0)
    cw = np.cumsum(w[order], axis=0)
    k = np.argmax(cw >= 0.5 * cw[-1], axis=0)
    return zs[k, np.arange(z.shape[1])]


def _weighted_laplace(z: np.ndarray, w: np.ndarray):
    """Weighted Laplace fit: weighted median, mean abs deviation shrunk toward 1."""
    mu = _weighted_median(z, w)
    n_eff = float(w.sum())
    b_hat = (w[:, None] * np.abs(z - mu)).sum(axis=0) / max(n_eff, 1e-12)
    s = (n_eff * b_hat + SCALE_PRIOR_N) / (n_eff + SCALE_PRIOR_N)
    return mu, np.where(s > 0, s, 1.0)


def _em(zb: np.ndarray, lN: np.ndarray):
    """EM for pi * D + (1 - pi) * N on one batch, N fixed. Init: anomaly rank."""
    n = len(zb)
    r = np.argsort(np.argsort(-lN)) / max(n - 1, 1)  # most anomalous -> 1
    for _ in range(EM_ITERS):
        mu, s = _weighted_laplace(zb, r)
        pi = (r.sum() + PI_PRIOR) / (n + 2 * PI_PRIOR)
        a = np.log(pi) + _loglik(zb, mu, s)
        c = np.log1p(-pi) + lN
        r_new = np.exp(a - np.logaddexp(a, c))
        done = np.max(np.abs(r_new - r)) < EM_TOL
        r = r_new
        if done:
            break
    mu, s = _weighted_laplace(zb, r)
    return dict(pi=(r.sum() + PI_PRIOR) / (n + 2 * PI_PRIOR), mu=mu, s=s, resp=r)


def fit(x: np.ndarray, codes: np.ndarray, batches: list[str], base: str = "Batch_3") -> dict:
    center, scale, ok, mu, sb = fp._fit_core(x, codes, len(batches))
    z = np.clip((x[:, ok] - center[ok]) / scale[ok], -fp.MAX_Z, fp.MAX_Z)
    ib = batches.index(base)
    muN, sN = mu[ib], sb[ib]
    comps = {}
    for b, name in enumerate(batches):
        if b == ib:
            continue
        zb = z[codes == b]
        comps[name] = _em(zb, _loglik(zb, muN, sN))
        comps[name]["rows"] = np.flatnonzero(codes == b)
    return dict(center=center, scale=scale, ok=ok, muN=muN, sN=sN, comps=comps,
                batches=batches, base=base)


def posterior(m: dict, x: np.ndarray) -> np.ndarray:
    """P(batch | x) per row, uniform prior, columns in m['batches'] order."""
    ok = m["ok"]
    z = np.clip((x[:, ok] - m["center"][ok]) / m["scale"][ok], -fp.MAX_Z, fp.MAX_Z)
    lN = _loglik(z, m["muN"], m["sN"])
    cols = []
    for name in m["batches"]:
        if name == m["base"]:
            cols.append(lN)
        else:
            c = m["comps"][name]
            cols.append(np.logaddexp(np.log(c["pi"]) + _loglik(z, c["mu"], c["s"]),
                                     np.log1p(-c["pi"]) + lN))
    L = np.column_stack(cols)
    return np.exp(L - np.logaddexp.reduce(L, axis=1, keepdims=True))


def loo_posterior(x: np.ndarray, codes: np.ndarray, batches: list[str]) -> np.ndarray:
    n = len(x)
    post = np.empty((n, len(batches)))
    for i in range(n):
        tr = np.arange(n) != i
        post[i] = posterior(fit(x[tr], codes[tr], batches), x[i:i + 1])[0]
    return post
