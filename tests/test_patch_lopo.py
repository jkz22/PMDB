"""Synthetic tests for pmdb.patch_lopo (no real data required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmdb import fingerprint as fp
from pmdb import patch_lopo as pl


def test_normalise_and_ensemble():
    p = pl.normalise_p(np.array([[1.0, 1.0, 2.0], [3.0, 0.0, 1.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(pl.ensemble(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])), [0.5, 0, 0.5])


def test_stratum_flags():
    out = pl.stratum_flags(np.array([1, 1, 0, 1, 0, 0]), np.array([1, 1, 1, 0, 0, 0], bool), np.arange(6))
    assert out.tolist() == [False, False, True, False, False, False]


def test_expected_rubric():
    assert pl.expected_rubric(np.array([1, 0, 1, 0], bool), np.array([1, 1, 0, 0], bool)) == (2 + 0 + 1 + 1) / 4


def test_cv_feasible():
    assert not pl.cv_feasible(np.array([0, 0, 1, 1, 2, 2]), np.arange(6))
    assert pl.cv_feasible(np.repeat([0, 1, 2], 3), np.arange(9))


def _synth(seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 2], 4)
    # per batch 3 groups of sizes 2,1,1 (a fold must leave >= 2 groups per batch)
    groups = np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8])
    keys = [(pl.BATCHES[b], f"s{i}") for i, b in enumerate(labels)]
    names = ["si_depth_slope", "gx_0.5_2.0", "k15_contact_tilestd"]
    X = pd.DataFrame(rng.normal(size=(12, 3)) + labels[:, None] * 2.0, columns=names,
                     index=pd.MultiIndex.from_tuples(keys, names=["batch", "site"]))
    return labels, groups, keys, X


def make_D(labels, n_patches, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)
    ps = np.repeat(np.arange(n), n_patches)
    D = rng.uniform(0.4, 0.6, size=(len(ps), n)) - 0.2 * (labels[ps][:, None] == labels[None, :])
    D[np.arange(len(ps)), ps] = np.inf
    return D, ps


def test_fingerprint_cv_no_leakage(monkeypatch):
    labels, groups, keys, X = _synth()
    y = pd.Series([pl.BATCHES[c] for c in labels], index=X.index)
    seen = []
    orig = fp.fit

    def spy(Xt, yt, features=None):
        seen.append(set(Xt.index))
        return orig(Xt, yt, features)

    monkeypatch.setattr(fp, "fit", spy)
    out = pl.fingerprint_cv(X, y, groups)
    assert len(out) == 12 and len(seen) == len(np.unique(groups))
    for g, tr in zip(np.unique(groups), seen):
        test_rows = {keys[i] for i in np.flatnonzero(groups == g)}
        assert not (tr & test_rows)


def test_build_lopo_outputs_synthetic():
    labels, groups, keys, X = _synth()
    D, ps = make_D(labels, 6, 0)
    sites = pd.DataFrame(keys, columns=["batch", "site"])
    parents = pd.DataFrame({"batch": sites["batch"], "site": sites["site"],
                            "parent_id": [f"p{g}" for g in groups]})
    parents = pd.concat([parents, pd.DataFrame({"batch": ["Batch_heldout"], "site": ["hh"],
                                                "parent_id": ["p0"]})], ignore_index=True)
    H = pd.DataFrame(np.random.default_rng(5).normal(size=(1, 3)) + 2.0, columns=X.columns,
                     index=pd.MultiIndex.from_tuples([("Batch_heldout", "hh")], names=["batch", "site"]))
    Dh = np.random.default_rng(2).uniform(0.4, 0.6, size=(9, 12))
    psh = np.zeros(9, dtype=int)
    coords_h = np.array([[r, c, 0, 0] for r in range(3) for c in range(3)])
    res = pl.build_lopo_outputs(D, ps, labels, sites, parents, X, H, Dh, psh, ["hh"], coords_h,
                                n_perm=5, seed=0)
    f = res["final_heldout"]
    assert list(f.columns) == ["site", "assigned", "confidence_flag", "p_ens_Batch_1", "p_ens_Batch_2",
                               "p_ens_Batch_3", "patch_call", "fingerprint_call", "explanation"]
    assert f["confidence_flag"].iloc[0] in ("high", "low")
    assert np.isclose(f[["p_ens_Batch_1", "p_ens_Batch_2", "p_ens_Batch_3"]].sum(axis=1).iloc[0], 1.0)
    ex = f["explanation"].iloc[0].lower()
    assert "batch 3" in ex
    for w in pl.FORBIDDEN:
        assert w not in ex, w
    assert res["evaluation"]["parents_used_as_evidence"] is False
    assert "fp_confidence_diagnostic" in res["evaluation"]["confidence"]


def test_phrases():
    assert pl.feature_phrase("gz_2.0_4.0") == "graphite clustering through the depth at 2.0-4.0 um spacing"
    assert pl.image_third(np.array([0, 0, 5]), 6) == "top"
