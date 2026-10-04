"""Tests for the D21 FEM-on-fingerprint script: in-fold selection uses training rows only."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import kruskal

_spec = importlib.util.spec_from_file_location(
    "fem_fingerprint_eval", Path(__file__).resolve().parents[1] / "scripts" / "fem_fingerprint_eval.py")
ffe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ffe)


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    codes = np.repeat([0, 1, 2], 8)
    good = codes + rng.normal(0, 0.3, 24)
    dup = good * 2 + 1                      # |rho| = 1 with good
    second = (codes == 1) * 1.0 + rng.normal(0, 0.3, 24)
    trap = rng.normal(0, 1, 24)             # noise in training
    cand = np.column_stack([good, dup, second, trap])
    return cand, codes


def test_kw_matches_scipy():
    cand, codes = _toy()
    p = ffe.kw_pvalues(cand, codes)
    for j in range(cand.shape[1]):
        assert np.isclose(p[j], kruskal(*[cand[codes == g, j] for g in range(3)]).pvalue)


def test_duplicate_skipped_and_constant_ignored():
    cand, codes = _toy()
    cand = np.column_stack([cand, np.ones(24)])
    chosen, _, skipped = ffe.select_features(cand, codes)
    assert chosen[0] in (0, 1) and not ({0, 1} <= set(chosen))
    assert any(abs(r) > 0.9 for _, r, _ in skipped)
    assert 4 not in chosen


def test_selection_uses_training_fold_only():
    cand, codes = _toy()
    # trap is constant on the training rows, informative-looking only on the test site
    tr = {"cand": cand[1:].copy(), "cand_names": list("abcd"), "leo": np.zeros((23, 1))}
    tr["cand"][:, 3] = 5.0
    te_a = {"cand": cand[:1].copy(), "leo": np.zeros((1, 1))}
    te_b = {"cand": cand[:1].copy(), "leo": np.zeros((1, 1))}
    te_a["cand"][0, 3] = 1e6
    te_b["cand"][0, 3] = -1e6
    xa, _, na, ia = ffe.fold_matrices("A2", tr, te_a, codes[1:])
    xb, _, nb, ib = ffe.fold_matrices("A2", tr, te_b, codes[1:])
    assert na == nb and "d" not in na and ia["chosen"] == ib["chosen"]


def test_decide_never_promotes_sensitivity_arm():
    res = {"A0": {"n_correct": 21, "perm_p": 0.5}, "A1": {"n_correct": 30, "perm_p": 0.001},
           "A2": {"n_correct": 30, "perm_p": 0.001}}
    assert ffe.decide(res)[0] in ("A1", "A2")
    assert ffe.decide(res, sensitivity=True)[0] == "A0"


def test_availability_screen_uses_training_rows_only():
    cand, codes = _toy()
    tr = {"cand": cand[1:].copy(), "cand_names": list("abcd"), "leo": np.zeros((23, 1))}
    te = {"cand": cand[:1].copy(), "leo": np.zeros((1, 1))}
    cand_nan = cand.copy()
    cand_nan[0, 1] = np.nan  # NaN only at the held-out site must not change the training selection
    tr2 = {**tr, "cand": cand_nan[1:].copy()}
    _, _, n1, _ = ffe.fold_matrices("A2", tr, te, codes[1:])
    _, _, n2, _ = ffe.fold_matrices("A2", tr2, te, codes[1:])
    assert n1 == n2
    tr3 = {**tr, "cand": tr["cand"].copy()}
    tr3["cand"][0, 0] = np.nan  # NaN at a training site drops that column in-fold
    _, _, n3, _ = ffe.fold_matrices("A2", tr3, te, codes[1:])
    assert "a" not in n3


def test_test_site_nan_never_reaches_prediction_matrix():
    cand, codes = _toy()
    tr = {"cand": cand[1:].copy(), "cand_names": list("abcd"), "leo": np.zeros((23, 1))}
    for j in range(cand.shape[1]):
        te = {"cand": cand[:1].copy(), "leo": np.zeros((1, 1))}
        te["cand"][0, j] = np.nan
        _, xte, names, _ = ffe.fold_matrices("A2", tr, te, codes[1:])
        assert np.isfinite(xte).all() and "abcd"[j] not in names
