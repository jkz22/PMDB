"""Synthetic tests for pmdb.fingerprint (no real data required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmdb import fingerprint as fp

BATCH_PROFILES = {
    # relative Si fraction per depth band: flat / top-heavy / bottom-heavy
    "Batch_A": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "Batch_B": np.array([1.6, 1.1, 0.5, 0.8, 1.0]),
    "Batch_C": np.array([0.7, 0.8, 1.0, 1.2, 1.3]),
}


def make_tables(n_per_batch=8, seed=0, noise=0.05):
    """Synthetic curves.csv / tile_kpis.csv long tables with batch structure."""
    rng = np.random.default_rng(seed)
    curve_rows, tile_rows = [], []
    for batch, prof in BATCH_PROFILES.items():
        for s in range(n_per_batch):
            site = f"s{s}"
            base = 0.06 * (1 + 0.1 * rng.standard_normal())
            for b in range(5):
                curve_rows.append((batch, site, "K12", "band_si_frac", float(b),
                                   base * prof[b] * (1 + noise * rng.standard_normal())))
            for ax in ("x", "z"):
                for lag in np.arange(0.25, 10.25, 0.25):
                    g = 1 - np.exp(-lag / (2 + 0.5 * list(BATCH_PROFILES).index(batch)))
                    curve_rows.append((batch, site, "K08", f"g_obs_{ax}", float(lag),
                                       g * (1 + noise * rng.standard_normal())))
            spread = 0.02 * (1 + list(BATCH_PROFILES).index(batch))
            for t in range(4):
                tile_rows.append((batch, site, t,
                                  0.7 + spread * rng.standard_normal()))
    curves = pd.DataFrame(curve_rows,
                          columns=["batch", "site", "kpi_id", "curve", "x", "value"])
    tiles = pd.DataFrame(tile_rows,
                         columns=["batch", "site", "tile",
                                  "K15_si_graphite_contact_frac"])
    return curves, tiles


@pytest.fixture(scope="module")
def features():
    curves, tiles = make_tables()
    X = fp.build_features(curves, tiles)
    y = pd.Series(X.index.get_level_values("batch"), index=X.index)
    return X, y


def test_build_features_shape(features):
    X, _ = features
    assert len(X) == 3 * 8
    assert "si_depth_rel_band2" in X.columns
    assert "si_depth_mid_dip" in X.columns
    assert "k15_contact_tilestd" in X.columns
    for lo, hi in fp.G_OBS_BINS:
        assert f"gx_{lo}_{hi}" in X.columns
        assert f"gz_{lo}_{hi}" in X.columns
    assert X.notna().all().all()


def test_fit_predict_recovers_batches(features):
    X, y = features
    model = fp.fit(X, y)
    pred = fp.predict(model, X)
    # training-set assignment should be near-perfect on well-separated data
    assert (pred["assigned"] == y).mean() >= 0.9
    # p-values live on the conformal grid
    for b in model.batches:
        n_b = len(model.calibration[b])
        assert ((pred[f"p_{b}"] * (n_b + 1)).round() ==
                pred[f"p_{b}"] * (n_b + 1)).all()


def test_loo_beats_chance(features):
    X, y = features
    _, metrics = fp.loo_evaluate(X, y)
    assert metrics["accuracy"] >= 0.8
    assert set(metrics["confusion"]) == set(BATCH_PROFILES)


def test_ood_site_is_rejected(features):
    X, y = features
    model = fp.fit(X, y)
    weird = X.iloc[[0]].copy()
    weird.loc[:, :] = X.to_numpy().max(axis=0) * 5  # like no batch at all
    pred = fp.predict(model, weird)
    assert bool(pred["ood"].iloc[0])
    # every batch p-value sits at its achievable floor 1/(n_b + 1)
    for b in model.batches:
        floor = 1.0 / (len(model.calibration[b]) + 1.0)
        assert pred[f"p_{b}"].iloc[0] == pytest.approx(floor)


def test_in_distribution_site_is_not_rejected(features):
    X, y = features
    model = fp.fit(X, y)
    pred = fp.predict(model, X)
    # typical training sites must not be flagged OOD
    assert pred["ood"].mean() < 0.2


def test_permutation_test_significant(features):
    X, y = features
    res = fp.permutation_test(X, y, n_perm=49, seed=1)
    assert res["p_value"] <= 0.05
    assert res["observed_accuracy"] >= 0.8


def test_conformal_p_calibration():
    """Conformal p-values of held-out same-batch sites are ~uniform: the
    fraction below alpha must not exceed alpha by much."""
    curves, tiles = make_tables(n_per_batch=12, seed=3)
    X = fp.build_features(curves, tiles)
    y = pd.Series(X.index.get_level_values("batch"), index=X.index)
    hits, total = 0, 0
    for i in range(len(X)):
        model = fp.fit(X.drop(X.index[i]), y.drop(y.index[i]))
        p = fp.conformal_p(model, X.iloc[[i]])
        hits += float(p.iloc[0][y.iloc[i]]) <= 0.25
        total += 1
    assert hits / total <= 0.40  # 0.25 nominal + small-sample slack


def test_missing_feature_column_raises(features):
    X, y = features
    model = fp.fit(X, y)
    with pytest.raises(ValueError, match="missing feature columns"):
        fp.predict(model, X.drop(columns=["si_depth_mid_dip"]))


def test_explain_structure(features):
    X, y = features
    model = fp.fit(X, y)
    ex = fp.explain(model, X.iloc[[0]])
    assert len(ex) == len(model.features)
    for b in model.batches:
        assert f"dev_{b}" in ex.columns
        assert (ex[f"scale_{b}"] > 0).all()
