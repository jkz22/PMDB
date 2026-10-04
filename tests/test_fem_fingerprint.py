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
