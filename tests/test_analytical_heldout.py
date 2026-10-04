import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analytical_benchmarks"))
C = pytest.importorskip("classify_heldout")


def toy(sep, n=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(C.B, n)
    X = rng.normal(size=(3 * n, 6))
    X[:, 0] += sep * np.repeat([0, 1, 2], n)  # only feature 0 carries the batch
    return X, y


def test_separable_batches_are_recovered_and_feature_selected():
    X, y = toy(6.0)
    acc, bal, _ = C.score(C.loso(X, y, 3), y)
    assert acc == 1.0 and bal == 1.0
    sel, _, _ = C.fit(X, y, 1)
    assert list(sel) == [0]
    m = C.fit(X, y, 3)
    assert C.B[int(C.loglik(m, np.r_[12.0, np.zeros(5)]).argmax())] == "Batch_3"


def test_unrelated_features_stay_near_chance_and_calibrate_flat():
    X, y = toy(0.0, n=10, seed=1)
    LL = C.loso(X, y, 3)
    assert C.score(LL, y)[1] < 0.6
    T, nll = C.best_T(LL, y)
    assert nll == pytest.approx(np.log(3), abs=0.08)
    p = C.post(LL[0], T)
    assert p.sum() == pytest.approx(1.0) and p.max() < 0.6
