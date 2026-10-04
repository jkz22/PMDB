"""Synthetic tests for pmdb.patch_mil / pmdb.patch_embed (no real data required)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pmdb import patch_mil as pm

FP_CONF = {
    "Batch_1": {"Batch_1": 5, "Batch_2": 0, "Batch_3": 2},
    "Batch_2": {"Batch_1": 3, "Batch_2": 3, "Batch_3": 1},
    "Batch_3": {"Batch_1": 2, "Batch_2": 2, "Batch_3": 13},
}


def make_D(labels, n_patches, seed, separable):
    labels = np.asarray(labels)
    n_sites = len(labels)
    rng = np.random.default_rng(seed)
    patch_site = np.repeat(np.arange(n_sites), n_patches)
    N = len(patch_site)
    if separable:
        D = rng.uniform(0.4, 0.6, size=(N, n_sites))
        same = labels[patch_site][:, None] == labels[None, :]
        D = D - 0.2 * same
    else:
        D = rng.uniform(size=(N, n_sites))
    D[np.arange(N), patch_site] = np.inf
    return D, patch_site


def test_grid_coords():
    c = pm.grid_coords(1000, 3496)
    assert c.shape == (60, 4)
    assert c[:, 0].min() == 0 and c[:, 0].max() == 3
    assert c[:, 1].min() == 0 and c[:, 1].max() == 14
    assert c[:, 2].min() == 52 and c[:, 3].min() == 68
    assert (c[:, 2] + 224 <= 1000).all() and (c[:, 3] + 224 <= 3496).all()


def test_combine_channels_unit_norm():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 1536)), rng.normal(size=(5, 1536))
    f = pm.combine_channels(a, b)
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(f[:, :1536], axis=1), 1 / np.sqrt(2), atol=1e-5)
    assert np.allclose(np.linalg.norm(f[:, 1536:], axis=1), 1 / np.sqrt(2), atol=1e-5)


def test_site_distance_matrix():
    e = np.eye(3, dtype=np.float32)
    bank0 = e[[0, 1]]            # site 0: two patches
    bank1 = e[[2]]               # site 1: one patch
    q = np.stack([e[0], e[2]])
    D = pm.site_distance_matrix(q, np.array([0, 1]), [bank0, bank1], k=1)
    assert np.isinf(D[0, 0]) and np.isinf(D[1, 1])
    assert D[0, 1] == pytest.approx(1.0)
    assert D[1, 0] == pytest.approx(1.0)
    D3 = pm.site_distance_matrix(q, np.array([-1, -1]), [bank0, bank1], k=3)
    # bank0 has 2 patches: mean of both distances for q0 = (0 + 1) / 2
    assert D3[0, 0] == pytest.approx(0.5)
    assert D3[1, 0] == pytest.approx(1.0)


def test_reference_excludes_held_out_site():
    labels = np.array([0] * 3 + [1] * 3 + [2] * 6)
    D, ps = make_D(labels, 20, 0, False)
    mask = np.arange(len(labels)) != 0
    before = [pm.reference_distribution(D, ps, np.flatnonzero(mask & (labels == b))) for b in range(3)]
    rng = np.random.default_rng(99)
    D2 = D.copy()
    D2[ps == 0, :] = rng.uniform(size=(20, len(labels)))
    D2[:, 0] = rng.uniform(size=D.shape[0])
    after = [pm.reference_distribution(D2, ps, np.flatnonzero(mask & (labels == b))) for b in range(3)]
    for a, b in zip(before, after):
        assert np.array_equal(a, b)


def test_loo_fold_ignores_own_column():
    labels = np.array([0] * 3 + [1] * 3 + [2] * 6)
    D, ps = make_D(labels, 20, 1, False)
    base = pm.loo_scores(D, ps, labels, 3)
    for i in (0, 8):
        D2 = D.copy()
        D2[:, i] = 0.0
        new = pm.loo_scores(D2, ps, labels, 3)
        for key in ("top10", "mean"):
            assert np.allclose(new[i].pooled[key], base[i].pooled[key])


def test_reference_requires_two_sites():
    D, ps = make_D(np.array([0, 1]), 5, 0, False)
    with pytest.raises(ValueError):
        pm.reference_distribution(D, ps, np.array([0]))


def test_separable_loo_and_permutation():
    labels = np.array([0] * 4 + [1] * 4 + [2] * 8)
    D, ps = make_D(labels, 30, 0, True)
    sc = pm.loo_scores(D, ps, labels, 3)
    pred = np.array([np.argmin(s.pooled["top10"]) for s in sc])
    assert (pred == labels).mean() == 1.0
    perm = pm.permutation_test(D, ps, labels, 3, n_perm=50, seed=0)
    assert set(perm) == {"observed_accuracy", "null_mean", "null_p95", "p_value", "n_perm", "seed"}
    assert perm["n_perm"] == 50
    assert perm["p_value"] < 0.05


def test_null_not_dominated_by_largest_batch():
    labels = np.array([0] * 7 + [1] * 7 + [2] * 17)
    preds = []
    for seed in (0, 1, 2):
        D, ps = make_D(labels, 60, seed, False)
        sc = pm.loo_scores(D, ps, labels, 3)
        preds += [int(np.argmin(s.pooled["top10"])) for s in sc]
    share = np.bincount(preds, minlength=3) / len(preds)
    assert len(preds) == 93
    assert ((share >= 0.15) & (share <= 0.60)).all(), share


def test_metrics_and_auroc():
    m = pm.metrics_from_confusion(FP_CONF)
    assert m["accuracy"] == pytest.approx(21 / 31)
    assert m["balanced_accuracy"] == pytest.approx(0.635854, abs=1e-5)
    assert m["macro_f1"] == pytest.approx(0.625371, abs=1e-5)
    assert pm.auroc(np.array([3, 4]), np.array([1, 2])) == 1.0
    assert pm.auroc(np.array([1]), np.array([1])) == 0.5


def test_build_outputs_tables():
    labels = np.array([0] * 4 + [1] * 4 + [2] * 8)
    D, ps = make_D(labels, 30, 0, True)
    n = len(labels)
    sites = pd.DataFrame({"batch": [pm.BATCHES[b] for b in labels], "site": [f"s{i}" for i in range(n)]})
    c1 = pm.grid_coords(448, 15 * 224)
    assert len(c1) == 30
    coords = np.tile(c1, (n, 1))
    rng = np.random.default_rng(5)
    Dh = rng.uniform(0.4, 0.6, size=(60, n))
    Dh[:, labels == 0] -= 0.2
    psh = np.repeat(np.arange(2), 30)
    out = pm.build_outputs(D, ps, labels, sites, coords, rng.uniform(size=n), Dh, psh, ["h0", "h1"],
                           np.tile(c1, (2, 1)), rng.uniform(size=2), n_perm=10)
    lp, hp, pt, ev = out["loo_predictions"], out["heldout_predictions"], out["patch_scores"], out["evaluation"]
    assert len(lp) == 16
    assert list(lp.columns) == [
        "batch", "site", "true", "assigned", "confidence", "p_Batch_1", "p_Batch_2", "p_Batch_3",
        "score_Batch_1", "score_Batch_2", "score_Batch_3", "assigned_mean_pool",
        "score_mean_Batch_1", "score_mean_Batch_2", "score_mean_Batch_3",
        "anomaly_vs_B3", "frac_patches_anomalous", "n_patches", "bse_hf"]
    assert len(hp) == 2 and all("Batch 3" in e for e in hp["explanation"])
    assert len(pt) == 16 * 30 + 2 * 30
    for k in ("n_sites", "n_patches", "patches_per_site", "majority_baseline", "loo", "loo_mean_pooling",
              "permutation", "auroc_vs_B3", "loo_anomaly_vs_B3", "shortcut_diagnostic"):
        assert k in ev
    json.dumps(ev)


def test_fingerprint_comparison():
    fp = {"loo": {"confusion": FP_CONF}, "permutation": {"p_value": 0.002}}
    ours = {"loo": pm.metrics_from_confusion(
        {"Batch_1": {"Batch_1": 6, "Batch_2": 1, "Batch_3": 0},
         "Batch_2": {"Batch_1": 1, "Batch_2": 5, "Batch_3": 1},
         "Batch_3": {"Batch_1": 0, "Batch_2": 2, "Batch_3": 15}}),
        "permutation": {"p_value": 0.01}}
    cmp = pm.fingerprint_comparison(fp, ours)
    assert cmp["delta_accuracy"] == pytest.approx(ours["loo"]["accuracy"] - 21 / 31)


def test_render_heldout_figure_png():
    rng = np.random.default_rng(0)
    coords = pm.grid_coords(500, 1000)
    png = pm.render_heldout_figure(rng.uniform(size=(500, 1000)), rng.uniform(size=(500, 1000)), coords,
                                   rng.uniform(size=len(coords)), "t")
    assert png.startswith(b"\x89PNG")


def test_patch_features_random_backbone():
    pytest.importorskip("torchvision")
    from pmdb.patch_embed import build_backbone, patch_features

    import torch

    torch.set_num_threads(1)  # avoids an OpenMP deadlock after fork-based pools in earlier tests
    model = build_backbone(None, "cpu")
    rng = np.random.default_rng(0)
    img = rng.uniform(size=(300, 500)).astype(np.float32)
    f = patch_features(model, img, pm.grid_coords(300, 500), "cpu")
    assert f.shape == (2, 1536) and np.isfinite(f).all()


LAB12 = [0] * 4 + [1] * 4 + [2] * 4


def test_groups_equal_sites_reproduces_loso():
    D, ps = make_D(LAB12, 7, seed=3, separable=False)
    lab = np.array(LAB12)
    a = pm.lopo_scores(D, ps, lab, np.arange(12), 3)
    b = pm.loo_scores(D, ps, lab, 3)
    for x, y in zip(a, b):
        assert np.allclose(x.u, y.u, atol=1e-6)
        assert np.allclose(x.pooled["top10"], y.pooled["top10"], atol=1e-6)
        assert np.allclose(x.pooled["mean"], y.pooled["mean"], atol=1e-6)


def test_grouped_fold_ignores_own_group():
    D, ps = make_D(LAB12, 7, seed=3, separable=False)
    lab = np.array(LAB12)
    groups = np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8])
    base = pm.lopo_scores(D, ps, lab, groups, 3)[0]
    D2 = D.copy()
    D2[:, [0, 1]] = -5.0
    D2[ps == 0, 0] = np.inf
    r = pm.lopo_scores(D2, ps, lab, groups, 3)[0]
    assert np.allclose(base.u, r.u) and np.allclose(base.pooled["top10"], r.pooled["top10"])
    D3 = D.copy()
    D3[ps == 1, :] = -5.0
    r3 = pm.lopo_scores(D3, ps, lab, groups, 3)[0]
    assert np.allclose(base.u, r3.u) and np.allclose(base.pooled["top10"], r3.pooled["top10"])


def test_grouped_reference_requires_two_groups():
    D, ps = make_D(LAB12, 7, seed=3, separable=False)
    with pytest.raises(ValueError):
        pm.reference_distribution(D, ps, np.array([0, 1, 2]), np.zeros(12, dtype=int))


def test_heldout_excludes_sibling_group():
    D, ps = make_D(LAB12, 7, seed=3, separable=False)
    lab = np.array(LAB12)
    groups = np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8])
    rng = np.random.default_rng(1)
    Dh = rng.uniform(size=(5, 12))
    psh = np.zeros(5, dtype=int)
    base = pm.heldout_scores(Dh, psh, 1, D, ps, lab, 3, groups, np.array([0]))[0]
    Dh2 = Dh.copy()
    Dh2[:, [0, 1]] = -5.0
    r = pm.heldout_scores(Dh2, psh, 1, D, ps, lab, 3, groups, np.array([0]))[0]
    assert np.allclose(base.u, r.u) and np.allclose(base.pooled["top10"], r.pooled["top10"])
