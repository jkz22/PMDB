"""Tests for the D22 XGBoost-on-KPIs pieces: in-fold steps never see the test site."""

import numpy as np

from pmdb import xgb_kpi as xk
from pmdb.fem_features import subset


def _toy(n=24, seed=0):
    rng = np.random.default_rng(seed)
    codes = np.repeat([0, 1, 2], n // 3)
    cand = np.column_stack([codes + rng.normal(0, 0.3, n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
    d = {"index": np.array([["B", f"s{i}"] for i in range(n)], dtype=object),
         "kpi": rng.normal(size=(n, 4)), "cand": cand, "cand_names": ["c0", "c1", "c2"],
         "k01": rng.normal(size=n), "spread": rng.normal(size=n)}
    d["swell"] = 2 * d["k01"] + rng.normal(0, 0.1, n)
    return d, codes


def test_selection_never_sees_test_site():
    d, codes = _toy()
    i = 5
    m = np.arange(len(codes)) != i
    base = xk.fold_matrices("X3", subset(d, m), subset(d, ~m), codes[m])
    d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    d2["cand"][i] = 1e6 * (codes[i] + 5)            # wildly change the test site's candidates
    d2["swell"][i] += 100.0
    d2["k01"][i] += 50.0
    for arm in ("X3", "X2"):
        a = xk.fold_matrices(arm, subset(d, m), subset(d, ~m), codes[m])
        b = xk.fold_matrices(arm, subset(d2, m), subset(d2, ~m), codes[m])
        assert np.array_equal(a[0], b[0]) and a[2] == b[2]           # train matrix + chosen names unchanged
    assert base[2][0] == "c0"


def test_x1_has_24_kpis_without_a02_a03():
    tr, te, y, cols = xk.load_inputs()
    assert len(cols) == 24 and tr["kpi"].shape == (31, 24) and te["kpi"].shape == (3, 24)
    assert "A02_curtaining_index" not in cols and "A03_height_um" not in cols
    xtr, xte, extra, _ = xk.fold_matrices("X1", tr, te, np.zeros(31, int))
    assert xtr.shape[1] == 24 and extra == []


def test_class_weights_inverse_frequency():
    w = xk.class_weights(np.array([0, 0, 0, 1, 2, 2]), 3)
    assert np.allclose(w, [6 / 9] * 3 + [6 / 3] + [6 / 6] * 2)


def test_decide_rule_and_tiebreak():
    r = {"X1": {"n_correct": 22, "perm_p": 0.01}, "X2": {"n_correct": 22, "perm_p": 0.02},
         "X3": {"n_correct": 24, "perm_p": 0.2}}
    assert xk.decide(r)[0] == "X1"                   # X3 fails the permutation, tie -> fewer features
    r["X3"]["perm_p"] = 0.01
    assert xk.decide(r)[0] == "X3"
    r["X3"]["n_correct"] = 21
    r["X1"]["n_correct"] = r["X2"]["n_correct"] = 21
    assert xk.decide(r)[0] == "fingerprint"


def test_perm_chunk_matches_serial_and_is_seeded():
    d, codes = _toy(18)
    a = xk.perm_chunk("X1", d, codes, 3, 0, [0, 1])
    assert a == xk.perm_chunk("X1", d, codes, 3, 0, [0]) + xk.perm_chunk("X1", d, codes, 3, 0, [1])
