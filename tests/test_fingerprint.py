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
        n_b = int((model.train_codes == model.batches.index(b)).sum())
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
        floor = 1.0 / (int((model.train_codes == model.batches.index(b)).sum()) + 1.0)
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


def test_full_conformal_swap_ranks_are_exact():
    """#1: hold out each of the n_b + 1 batch-A sites in turn; full conformal
    scores all swaps from one multiset, so the p-values are exactly 1..n_b+1
    over (n_b + 1). The old jackknife calibration failed this (ties at 4/9)."""
    X = fp.build_features(*make_tables())
    extra = fp.build_features(*make_tables(seed=1)).loc[[("Batch_A", "s0")]]
    extra = extra.rename(index={"s0": "extra"})
    XA = pd.concat([X, extra])
    a_sites = [i for i in XA.index if i[0] == "Batch_A"]
    ranks = []
    for j in a_sites:
        tr = XA.drop([j])
        y = pd.Series(tr.index.get_level_values("batch"), index=tr.index)
        p = fp.conformal_p(fp.fit(tr, y), XA.loc[[j]])["Batch_A"].iloc[0]
        ranks.append(round(p * len(a_sites)))
    assert sorted(ranks) == list(range(1, len(a_sites) + 1))


def test_confidence_is_one_minus_best_rival_p(features):
    """#2: confidence = 1 - max p over the non-assigned batches, row by row."""
    X, y = features
    model = fp.fit(X, y)
    probe = X.copy()
    probe.loc[:, :] = X.to_numpy() * 1.15  # off-centre sites so p-values vary
    pred = pd.concat([fp.predict(model, X), fp.predict(model, probe)])
    for _, r in pred.iterrows():
        rival = max(r[f"p_{b}"] for b in model.batches if b != r["assigned"])
        assert r["confidence"] == pytest.approx(1.0 - rival)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_incomplete_fingerprint_is_refused(features, bad):
    """#3: a site with non-finite features is refused, not scored on the rest."""
    X, y = features
    model = fp.fit(X, y)
    site = X.iloc[[0]].copy()
    site.loc[:, [c for c in X.columns if c.startswith(("gx_", "gz_"))]] = bad
    with pytest.raises(ValueError, match="non-finite"):
        fp.predict(model, site)
    with pytest.raises(ValueError, match="non-finite"):
        fp.conformal_p(model, site)
    Xbad = X.copy()
    Xbad.iloc[0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        fp.fit(Xbad, y)


def test_labels_align_by_index_not_position(features):
    """#4: reordering y must not change the fit; mismatched keys raise."""
    X, y = features
    m1, m2 = fp.fit(X, y), fp.fit(X, y.iloc[::-1])
    pd.testing.assert_frame_equal(m1.batch_center, m2.batch_center)
    assert fp.permutation_test(X, y.iloc[::-1], n_perm=5, seed=0) == \
        fp.permutation_test(X, y, n_perm=5, seed=0)
    y_bad = y.copy()
    y_bad.index = y_bad.index.set_levels(
        y_bad.index.levels[1].str.replace("s0", "zz"), level=1)
    with pytest.raises(ValueError, match="indexed by exactly"):
        fp.fit(X, y_bad)


def test_loo_evaluate_aligns_labels(features):
    """#4: loo_evaluate gives identical predictions for a reordered y."""
    X, y = features
    p1, m1 = fp.loo_evaluate(X, y)
    p2, m2 = fp.loo_evaluate(X, y.iloc[::-1])
    pd.testing.assert_frame_equal(p1, p2)
    assert m1 == m2


def test_runner_removes_stale_heldout_outputs(tmp_path, monkeypatch):
    """#5: a run without --heldout-dir deletes held-out files of an earlier run."""
    from scripts import run_fingerprint
    curves, tiles = make_tables()
    kdir, out = tmp_path / "kpis", tmp_path / "out"
    kdir.mkdir()
    out.mkdir()
    curves.to_csv(kdir / "curves.csv", index=False)
    tiles.to_csv(kdir / "tile_kpis.csv", index=False)
    for name in run_fingerprint.HELDOUT_OUTPUTS:
        (out / name).write_text("stale\n")
    monkeypatch.setattr("sys.argv", ["run_fingerprint.py", "--kpis-dir", str(kdir),
                                     "--out-dir", str(out), "--n-perm", "3"])
    assert run_fingerprint.main() == 0
    assert (out / "evaluation.json").exists()
    for name in run_fingerprint.HELDOUT_OUTPUTS:
        assert not (out / name).exists()


def test_card_ranks_by_score_contribution():
    """#6: ranking uses dev + log(scale), not dev alone. On dev alone f2 wins
    (gap 1.5 vs 1.0); with the log-scale term f1 wins (1.0 vs ~0.11)."""
    from scripts.plot_fingerprint import decisive_features
    ex = pd.DataFrame({
        "feature": ["f1", "f2"],
        "dev_Batch_1": [0.0, 0.0], "scale_Batch_1": [1.0, 2.0],
        "dev_Batch_2": [1.0, 1.5], "scale_Batch_2": [1.0, 0.5],
    })
    assert decisive_features(ex, "Batch_1", ["Batch_1", "Batch_2"], 1) == ["f1"]


def test_predict_empty_batch(features):
    X, y = features
    model = fp.fit(X, y)
    pred = fp.predict(model, X.iloc[:0])
    assert len(pred) == 0
    assert {"assigned", "credibility", "confidence", "ood"} <= set(pred.columns)
    assert fp.conformal_p(model, X.iloc[:0]).shape == (0, len(model.batches))
