import numpy as np

from pmdb import patch_probe as pp


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X, sb, sp = {}, {}, {}
    for bi, b in enumerate(pp.BATCHES):
        for k in range(2):
            s = f"{b}_{k}"
            mu = np.zeros(80)
            mu[bi * 5:(bi + 1) * 5] = 6.0
            X[s] = (rng.normal(size=(30, 80)) + mu).astype(np.float32)
            sb[s] = b
            sp[s] = f"p{bi}{k}"
    return X, sb, sp


def test_separable_accuracy_one():
    X, sb, sp = _data()
    df = pp.lopo_probe(X, sb, sp)
    assert len(df) == 6
    assert pp.site_metrics(df)["accuracy"] == 1.0


def test_no_leakage():
    X, sb, sp = _data()
    sp["Batch_1_1"] = sp["Batch_1_0"] = "shared"
    folds = pp._prepare_folds(X, sp, 0)
    for par, test, train, _ in folds:
        assert all(sp[s] != par for s in train)
        assert set(test) == {s for s in sp if sp[s] == par}


def test_permutation_runs():
    X, sb, sp = _data()
    r = pp.permutation_test(X, sb, sp, n_perm=3)
    assert 0 < r["p_value"] <= 1 and r["observed_accuracy"] == 1.0
