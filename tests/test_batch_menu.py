"""Tests for pmdb.batch_menu (labels/parents need committed CSVs only; the rest is synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmdb import batch_menu as bm
from pmdb import fingerprint as fp
from pmdb import patch_lopo as plo
from pmdb.parents import parent_groups


def test_labelled_sites():
    lab = bm.labelled_sites()
    assert len(lab) == 34 and not lab.duplicated(["batch", "site"]).any()
    assert set(lab["label"]) <= set(bm.BATCHES)
    assert (lab["label"].iloc[:31] == lab["batch"].iloc[:31]).all()
    h = lab.iloc[31:].set_index("site")
    assert (h["batch"] == "Batch_heldout").all()
    assert h["label"].to_dict() == {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}
    assert list(h.index) == ["3e122cbj", "fn0mhxef", "xrv9xvzb"]


def test_parent_groups_without_test_dir_unchanged():
    assert len(parent_groups(include_test=False)) == 34


def test_fp_prob_matches_call():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 5, size=(50, 3))
    p = bm.fp_prob(s)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p.argmax(axis=1) == s.argmin(axis=1)).all()
    assert bm.fp_prob(np.array([1.0, 2.0, 3.0])).argmax() == 0


def test_confidence_sentence_branches():
    items = [("A", 0), ("B", 0), ("C", 1), ("D", 0)]
    hi = bm.confidence_sentence(0, True, items)
    assert hi.startswith("Confidence is high: A, B and D point to Batch 1, while C looks like Batch 2.")
    assert "consistently" not in hi
    lo = bm.confidence_sentence(0, False, items)
    assert lo.startswith("Confidence is low because the evidence is mixed: A, B and D point to Batch 1, but C")
    allk = [("A", 2)] * 4
    assert "consistently" in bm.confidence_sentence(2, True, allk)
    low_all = bm.confidence_sentence(2, False, allk)
    assert "small relative to normal variation" in low_all and "consistently" not in low_all


def test_centre_by_parent():
    F = pd.DataFrame({"a": [1.0, 3.0, 2.0, 6.0, 5.0], "b": [0.0, 2.0, 1.0, 1.0, 9.0]})
    par = pd.Series(list("aabbc"))
    Xc, sing = bm.centre_by_parent(F, par)
    assert np.allclose(Xc.groupby(par.values).mean().to_numpy(), 0)
    assert sing.tolist() == [False, False, False, False, True]
    assert (Xc.iloc[4] == 0).all()
    F2 = pd.concat([F, pd.DataFrame({"a": [100.0], "b": [100.0]})], ignore_index=True)
    Xc2, _ = bm.centre_by_parent(F2, pd.Series(list("aabbca")))
    assert not np.allclose(Xc2.iloc[:2].to_numpy(), Xc.iloc[:2].to_numpy())


def _synth(seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 2], 5)  # 4 sites in parent pairs + 1 singleton per batch -> 15 sites
    groups = np.array([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8])
    keys = [(plo.BATCHES[b], f"s{i}") for i, b in enumerate(labels)]
    names = ["si_depth_slope", "gx_0.5_2.0", "k15_contact_tilestd"]
    X = pd.DataFrame(rng.normal(size=(15, 3)) + labels[:, None] * 2.0, columns=names,
                     index=pd.MultiIndex.from_tuples(keys, names=["batch", "site"]))
    return labels, groups, keys, X


def test_fingerprint_centred_cv_singletons(monkeypatch):
    labels, groups, keys, X = _synth()
    y = pd.Series([plo.BATCHES[c] for c in labels], index=X.index)
    sizes = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    singleton = sizes == 1
    Xc = X - X.groupby(groups).transform("mean")
    fallback = bm._fp_cv(X, y, groups)
    seen = []
    orig = fp.fit

    def spy(Xt, yt, features=None):
        seen.append(set(Xt.index))
        return orig(Xt, yt, features)

    monkeypatch.setattr(fp, "fit", spy)
    out = bm.fingerprint_centred_cv(Xc, y, groups, singleton, fallback)
    sing_keys = {keys[i] for i in np.flatnonzero(singleton)}
    assert singleton.sum() == 3 and len(seen) == len(np.unique(groups[~singleton]))
    for tr in seen:
        assert not (tr & sing_keys)
    for i in np.flatnonzero(singleton):
        assert out.iloc[i]["fp_call"] == fallback.iloc[i]["fp_call"]
    assert len(out) == 15 and list(out.index) == list(X.index)


def test_global_flags():
    out = bm.global_flags(np.array([1, 1, 0, 0, 1]), np.array([0, 0, 1, 2, 3]))
    assert out.tolist() == [False, False, True, True, False]


def _summ(rub, acc):
    return {o: {"rubric": rub[i], "accuracy": acc[i]} for i, o in enumerate(bm.OPTIONS)}


def test_select_option():
    a = bm.select_option(_summ([1.2, 1.3, 1.1], [.5] * 3))
    assert a["option"] == "fingerprint_centred" and a["confidence_mode"] == "rule"
    b = bm.select_option(_summ([1.0, 0.9, 1.0], [.5, .45, .55]))
    assert b["option"] == "patch" and b["confidence_mode"] == "all_low" and b["rubric"] == 1.0
    c = bm.select_option(_summ([1.3, 1.3, 1.0], [.5] * 3))
    assert c["option"] == "fingerprint"


def test_predict_test_synthetic():
    labels, groups, keys, X = _synth()
    y = pd.Series([plo.BATCHES[c] for c in labels], index=X.index)
    sizes = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    H = pd.DataFrame(np.random.default_rng(5).normal(size=(2, 3)) + 2.0, columns=X.columns,
                     index=pd.MultiIndex.from_tuples([("Batch_test", "t0"), ("Batch_test", "t1")],
                                                     names=["batch", "site"]))
    test_codes = np.array([0, -1])
    pool_par = np.concatenate([groups, test_codes])
    pool = pd.concat([X, H])
    Xc_all, sing_all = bm.centre_by_parent(pool, pd.Series(pool_par))
    Xc, Hc = Xc_all.iloc[:15], Xc_all.iloc[15:]
    # test row with code -1 is alone in its pool group -> singleton; labelled singletons stay singletons
    singleton = sing_all.to_numpy()[:15]
    singleton_h = sing_all.to_numpy()[15:]
    assert singleton_h.tolist() == [False, True] and (singleton | (sizes > 1)).all()
    rng = np.random.default_rng(1)
    n = 15
    D = rng.uniform(0.4, 0.6, size=(n * 6, n))
    ps = np.repeat(np.arange(n), 6)
    D = D - 0.2 * (labels[ps][:, None] == labels[None, :])
    D[np.arange(len(ps)), ps] = np.inf
    menu_df, sc = bm.menu_cv(D, ps, labels, groups, X, Xc, y, singleton)
    summ = bm.menu_summary(menu_df, labels)
    assert set(bm.OPTIONS) <= set(summ) and summ["all_low"] == 1.0
    Dh = rng.uniform(0.4, 0.6, size=(18, n))
    psh = np.repeat([0, 1], 9)
    coords_h = np.array([[r, c, 0, 0] for r in range(3) for c in range(3)] * 2)
    for opt in bm.OPTIONS:
        sel = {"option": opt, "confidence_mode": "rule"}
        f = bm.predict_test(D, ps, labels, groups, X, Xc, y, singleton, menu_df, sc, Dh, psh,
                            [("Batch_test", "t0"), ("Batch_test", "t1")], test_codes, H, Hc, singleton_h,
                            coords_h, sel, ["pa", "pb"])
        assert list(f.columns) == ["site", "assigned", "confidence", "option", "p_Batch_1", "p_Batch_2",
                                   "p_Batch_3", "patch_call", "fingerprint_call", "fingerprint_centred_call",
                                   *[f"p_{o}_{b}" for o in bm.OPTIONS for b in plo.BATCHES],
                                   *[f"{o}_confidence" for o in bm.OPTIONS],
                                   "n_evidence_for_call", "parent_id", "explanation"]
        assert f["option"].eq(opt).all()
        assert set(f["confidence"]) <= {"high", "low"}
        assert np.allclose(f[["p_Batch_1", "p_Batch_2", "p_Batch_3"]].sum(axis=1), 1.0)
        for ex in f["explanation"]:
            for w in plo.FORBIDDEN:
                assert w not in ex.lower(), (w, ex)
    f = bm.predict_test(D, ps, labels, groups, X, Xc, y, singleton, menu_df, sc, Dh, psh,
                        [("Batch_test", "t0"), ("Batch_test", "t1")], test_codes, H, Hc, singleton_h,
                        coords_h, {"option": "patch", "confidence_mode": "all_low"})
    assert (f["confidence"] == "low").all()
    # selected model outside the menu (fem_a1): the menu's own pick comes from menu_option
    f = bm.predict_test(D, ps, labels, groups, X, Xc, y, singleton, menu_df, sc, Dh, psh,
                        [("Batch_test", "t0"), ("Batch_test", "t1")], test_codes, H, Hc, singleton_h,
                        coords_h, {"option": "fem_a1", "menu_option": "fingerprint", "confidence_mode": "rule"})
    assert f["option"].eq("fingerprint").all() and (f["assigned"] == f["fingerprint_call"]).all()
