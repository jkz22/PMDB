"""Within-parent paired contrasts: discover what the organisers designed into the batches.

The 31 labelled crops (+3 held-out) were cut from ~15 parent electrode images and
regrouped into artificial batches. Five parents are split across batches; inside a
parent the imaging is identical, so a feature difference between crops of different
batches in the same parent is free of acquisition confounds.

Selection rule (pre-stated, k fixed in advance)
-----------------------------------------------
For every batch pair (a, b) and every parent containing both a and b, the contrast is
d_p = (mean_b - mean_a) / s, with s the pooled within-batch SD (residuals from batch
means, all training crops). Per pair: consistency = |sum sign(d_p)| / n_p,
score = consistency * |mean d_p|. Features are picked round-robin over the pairs
(B1vB2, B1vB3, B2vB3), each pair taking its best not-yet-chosen feature with score > 0,
until K_SELECT are chosen. Everything is computed from the crops passed in, so calling
it on the training crops of a CV fold keeps the selection nested.

Numpy only (runs on Modal); the parent grouping is the organiser-recovered one.
"""
from __future__ import annotations

import numpy as np

#: Number of features selected (fixed in advance, k <= 8).
K_SELECT = 6

#: Parent electrode image -> crops (organiser-recovered from acquisition fingerprints).
PARENT_SITES: dict[str, list[str]] = {
    "G1612": ["ptg8lmto", "xgj4xftb"],
    "G1780": ["iv6g2oq0"],
    "G1880": ["uhdslk0o"],
    "G1904": ["0grcilhi", "hawkfj64", "mgxahqnk"],
    "G2048": ["avn74qx1", "3806gxp0", "fn0mhxef"],
    "G2060": ["71vgq3fw", "tuy3zymq", "x7u69zsw", "kbdh4tri"],
    "G2068SE": ["rxax5ozo", "x77cy643", "utfgcjfa", "vc2whyaq"],
    "G2080": ["ffwubibz", "r17byphk", "cfe5vt7s"],
    "G2088": ["ufdvpb81", "hzumfsms", "9luzk4jm", "xrv9xvzb"],
    "G2148": ["f1vzngrs", "epqdaau9"],
    "G2156": ["fzrt2k6r", "b3esycq1"],
    "G2272": ["pl8uabbv", "i9jiqjwl"],
    "G2316": ["4ih2ggld", "5n1q8atc", "3e122cbj"],
}
HELDOUT_SITES = ("3e122cbj", "fn0mhxef", "xrv9xvzb")
SITE_PARENT: dict[str, str] = {s: p for p, ss in PARENT_SITES.items() for s in ss}


def parent_of(sites, require: set[str] | None = None) -> list[str]:
    """Parent per site; if `require` is given, assert the grouping covers exactly those sites."""
    sites = list(sites)
    if require is not None:
        labelled = set(SITE_PARENT) - set(HELDOUT_SITES)
        assert labelled == set(require), (
            f"parent grouping mismatch: missing {sorted(set(require) - labelled)}, "
            f"extra {sorted(labelled - set(require))}")
    return [SITE_PARENT[s] for s in sites]


def pairs(n_batches: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(n_batches) for b in range(a + 1, n_batches)]


def pooled_within_sd(x: np.ndarray, codes: np.ndarray, n_batches: int) -> np.ndarray:
    """Pooled within-batch SD per feature (residuals from batch means, df = n - #batches)."""
    resid = np.empty_like(x, dtype=float)
    present = 0
    for b in range(n_batches):
        m = codes == b
        if m.any():
            present += 1
            resid[m] = x[m] - x[m].mean(axis=0)
    df = max(len(x) - present, 1)
    return np.sqrt((resid ** 2).sum(axis=0) / df)


def within_parent_diffs(x: np.ndarray, codes: np.ndarray, parents: np.ndarray,
                        n_batches: int) -> dict[tuple[int, int], tuple[list[str], np.ndarray]]:
    """Per batch pair: (parent names, array n_parents x F of standardised mean_b - mean_a)."""
    s = pooled_within_sd(x, codes, n_batches)
    s = np.where(s > 0, s, np.nan)
    out = {}
    for a, b in pairs(n_batches):
        names, rows = [], []
        for p in sorted(set(parents)):
            ma = (parents == p) & (codes == a)
            mb = (parents == p) & (codes == b)
            if ma.any() and mb.any():
                names.append(p)
                rows.append((x[mb].mean(axis=0) - x[ma].mean(axis=0)) / s)
        out[(a, b)] = (names, np.array(rows).reshape(len(rows), x.shape[1]))
    return out


def pair_scores(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(signed mean d, consistency, score) per feature from an n_parents x F array."""
    if len(d) == 0:
        z = np.zeros(d.shape[1])
        return z, z, z
    mean = np.nanmean(d, axis=0)
    cons = np.abs(np.nansum(np.sign(d), axis=0)) / len(d)
    score = np.nan_to_num(cons * np.abs(mean), nan=0.0)
    return np.nan_to_num(mean, nan=0.0), cons, score


def select(x: np.ndarray, codes: np.ndarray, parents: np.ndarray, n_batches: int,
           k: int = K_SELECT) -> list[int]:
    """Round-robin top-k feature indices by within-parent pair score (see module doc)."""
    diffs = within_parent_diffs(x, codes, parents, n_batches)
    orders = []
    for pr in pairs(n_batches):
        _, _, score = pair_scores(diffs[pr][1])
        order = [int(j) for j in np.argsort(-score, kind="stable") if score[j] > 0]
        orders.append(order)
    chosen: list[int] = []
    while len(chosen) < k and any(orders):
        for order in orders:
            while order and order[0] in chosen:
                order.pop(0)
            if order and len(chosen) < k:
                chosen.append(order.pop(0))
    return chosen


def softmax_proba(scores: np.ndarray, temperature: float) -> np.ndarray:
    """P(batch) = softmax(-T * score) per row (score = mean Laplace NLL, so T = #features
    turns the mean back into the summed log-likelihood)."""
    a = -temperature * scores
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def _nb_proba(xtr, ctr, xte, n_batches, feats):
    """NB fit on training rows, softmax(-T * score) probabilities for test rows, T = #features used."""
    from pmdb import fingerprint as fp
    center, scale, ok, mu, sb = fp._fit_core(xtr[:, feats], ctr, n_batches)
    z = np.clip((xte[:, feats][:, ok] - center[ok]) / scale[ok], -fp.MAX_Z, fp.MAX_Z)
    return softmax_proba(fp._scores(z, mu, sb), float(ok.sum()))


def cv_assign(x: np.ndarray, codes: np.ndarray, parents: np.ndarray, folds: np.ndarray,
              n_batches: int, method: str, k: int = K_SELECT):
    """Cross-validated assignments. folds: fold id per row (site id for LOSO, parent for LOPO).

    method 'baseline' = NB on all columns; 'paired' = NB on features selected inside the fold.
    Returns (assigned codes, n x batches probabilities, selected-feature lists per fold).
    """
    proba = np.empty((len(x), n_batches))
    sel = []
    for f in sorted(set(folds)):
        te = folds == f
        tr = ~te
        if method == "baseline":
            feats = list(range(x.shape[1]))
        else:
            feats = select(x[tr], codes[tr], parents[tr], n_batches, k)
        sel.append(feats)
        proba[te] = _nb_proba(x[tr], codes[tr], x[te], n_batches, feats)
    return proba.argmax(axis=1), proba, sel


def rubric_score(correct: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    """Organiser rubric per site: high (P > 0.5) correct 2, low correct 1, low wrong 1, high wrong 0."""
    high = p_pred > 0.5
    return np.where(correct, np.where(high, 2, 1), np.where(high, 0, 1)).astype(float)


def balanced_accuracy(codes: np.ndarray, assigned: np.ndarray, n_batches: int) -> float:
    return float(np.mean([(assigned[codes == c] == c).mean() for c in range(n_batches)]))
