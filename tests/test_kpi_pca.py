import numpy as np

from pmdb import kpi_pca


def _data():
    rng = np.random.default_rng(0)
    y = np.array(list(kpi_pca.BATCHES) * 10)
    X = rng.normal(size=(30, 10)) + 2.0 * np.eye(3)[np.arange(30) % 3].dot(rng.normal(size=(3, 10)))
    sites = np.array([f"s{i}" for i in range(30)])
    groups = np.array([f"g{i % 6}" for i in range(30)])
    return X, y, sites, groups


def test_lopo_shape_and_sum():
    X, y, sites, groups = _data()
    p = kpi_pca.lopo_proba(X, y, sites, groups)
    assert p.shape == (30, 3)
    assert np.allclose(p.sum(1), 1.0)


def test_outlier_never_trained(monkeypatch):
    X, y, sites, groups = _data()
    X[0] = 1e6
    monkeypatch.setattr(kpi_pca, "OUTLIERS", ("s0",))
    seen = []
    orig = kpi_pca.make_model

    class Rec:
        def __init__(self):
            self.m = orig()

        def fit(self, Xt, yt):
            seen.append(np.abs(Xt).max())
            self.m.fit(Xt, yt)
            self.classes_ = self.m.classes_
            return self

        def predict_proba(self, Xt):
            return self.m.predict_proba(Xt)

    monkeypatch.setattr(kpi_pca, "make_model", Rec)
    p = kpi_pca.lopo_proba(X, y, sites, groups)
    assert p.shape == (30, 3)
    assert len(seen) == 6 and max(seen) < 1e5
