import numpy as np
import pandas as pd

from pmdb import patch_probe as pp
from pmdb import probe_explain as pe

COLS = ["si_frac", "porosity", "si_mean_area_um2", "depth_frac"]


def test_pc_label_recovers_kpi():
    rng = np.random.default_rng(0)
    K = pd.DataFrame(rng.normal(size=(600, 4)), columns=COLS)
    Z = 0.1 * rng.normal(size=(600, 64))
    Z[:, 0] += 2 * K["porosity"]
    pc, kr = pe.pc_kpi_regression(Z, K, list(K.columns))
    assert pc.loc[0, "r2"] > 0.9
    assert pc.loc[0, "explained"]
    assert pc.loc[0, "label"].startswith("+porosity")
    assert not pc.loc[1, "explained"]
    assert kr.set_index("kpi").loc["porosity", "r2"] > 0.9


def test_contributions_sum_to_margin():
    rng = np.random.default_rng(0)
    X, sb = {}, {}
    for bi, b in enumerate(pp.BATCHES):
        for k in range(2):
            s = f"{b}_{k}"
            mu = np.zeros(80)
            mu[bi * 5:(bi + 1) * 5] = 6.0
            X[s] = (rng.normal(size=(30, 80)) + mu).astype(np.float32)
            sb[s] = b
    sc, pca, clf = pp.fit_full_probe(X, sb)
    Zs = {s: pca.transform(sc.transform(X[s])) for s in X}
    K = pd.DataFrame(rng.normal(size=(180, 4)), columns=COLS)
    pc, _ = pe.pc_kpi_regression(np.concatenate(list(Zs.values())), K, COLS)
    for s in X:
        e = pe.explain_site(Zs[s], clf, pc)
        assert abs(e["contrib"].sum() + e["intercept_term"] - e["margin"]) < 1e-6
        assert e["call"] == sb[s]
        assert 0 <= e["explained_share"] <= 1
        assert "PC" not in e["sentence"]


class _Clf:
    def __init__(self, coef, icpt):
        self.coef_, self.intercept_ = np.asarray(coef, float), np.asarray(icpt, float)

    def predict_log_proba(self, Z):
        s = Z @ self.coef_.T + self.intercept_
        return s - np.log(np.exp(s).sum(1, keepdims=True))


def _meanings(b, explained):
    n = len(b)
    d = {"pc": [f"PC{i + 1}" for i in range(n)], "sd": [1.0] * n, "r2": [0.5 if e else 0.05 for e in explained],
         "explained": explained}
    for k, kp in enumerate(["si_frac", "porosity"]):
        d[f"b_{kp}"] = [row[k] for row in b]
        d[f"B_{kp}"] = [row[k] for row in b]
    return pd.DataFrame(d)


def test_opposing_pcs_give_no_contradiction():
    # PC1 and PC2 both push Batch_1 over Batch_2 but load Si fraction in opposite directions
    coef = [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
    clf = _Clf(coef, [0, 0, -9])
    pc = _meanings([[0.9, 0.0], [-0.9, 0.5], [0.0, 0.0]], [True, True, False])
    e = pe.explain_site(np.array([[1.0, 1.0, 0.0]]), clf, pc)
    s = e["sentence"]
    assert not ("higher Si area fraction" in s and "lower Si area fraction" in s)
    assert "Si area fraction" not in s  # cancels exactly: dropped as ambiguous
    assert s.count("porosity") <= 1


def test_unexplained_pcs_carry_no_kpi_direction():
    clf = _Clf([[1, 3], [-1, -3], [0, 0]], [0, 0, 0])
    pc = _meanings([[0.9, 0.0], [0.0, 0.9]], [True, False])
    bd = pe.batch_directions(clf, pc)
    b1 = bd[bd["batch"] == "Batch_1"].set_index("kpi")
    assert abs(b1.loc["porosity", "logit_per_sd"]) < 1e-12  # only PC2 loads porosity, and it is unexplained
    assert b1.loc["si_frac", "logit_per_sd"] != 0
    assert 0 < b1["unexplained_share"].iloc[0] < 1
    assert np.isclose(b1["unexplained_share"].iloc[0] + b1["explainable_share"].iloc[0], 1)


def test_kpi_effect_zero_at_training_mean():
    w = np.array([1.0, -2.0])
    B = np.array([[0.5, 0.5], [0.3, -0.3]])
    eff = pe.kpi_effects(w, B, [True, True], [1.0, 3.0], [1.0, 2.0], [1.0, 1.0])
    assert abs(eff[0]) < 1e-12 and eff[1] != 0
    assert np.allclose(eff[1], 1.0 * (1.0 * 0.5 + -2.0 * -0.3))
