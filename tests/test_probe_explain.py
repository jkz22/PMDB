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
