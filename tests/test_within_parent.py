"""Synthetic tests for pmdb.within_parent."""

from __future__ import annotations

import pandas as pd

from pmdb import patch_lopo as plo
from pmdb import within_parent as wp


def _data():
    F = pd.DataFrame({"f1": [1.0, 0.0, 3.0, 1.0, 0.0, 5.0, 5.0], "f2": [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 3.0], "f3": range(7)},
                     index=[f"s{i}" for i in range(7)])
    label = pd.Series(["Batch_1", "Batch_2", "Batch_1", "Batch_2", "Batch_3", "Batch_3", "Batch_3"], index=F.index)
    parent = pd.Series(["p1", "p1", "p2", "p2", "p2", "p3", "p3"], index=F.index)
    return F, label, parent


def test_parent_contrasts_counts():
    C = wp.parent_contrasts(*_data())
    n = C.groupby("pair")["parent_id"].nunique().to_dict()
    assert n == {"Batch_1-Batch_2": 2, "Batch_1-Batch_3": 1, "Batch_2-Batch_3": 1}
    assert "p3" not in set(C["parent_id"])
    r = C[(C.parent_id == "p1") & (C.feature == "f1") & (C.pair == "Batch_1-Batch_2")].iloc[0]
    assert r["diff"] == 1.0


def test_sign_test_p():
    assert wp.sign_test_p(5, 0) == 0.0625
    assert wp.sign_test_p(2, 2) == 1.0
    assert pd.isna(wp.sign_test_p(0, 0))


def test_summary_and_sentence():
    F, label, parent = _data()
    S = wp.summarise_contrasts(wp.parent_contrasts(F, label, parent), F.std(ddof=1))
    assert len(S) == 9
    assert not S["is_signal"].any()  # at most 2 parents
    assert wp.signal_sentence(S) == ""
    sig = pd.DataFrame([{"feature": "si_depth_slope", "pair": "Batch_1-Batch_2", "n_parents": 5, "n_pos": 5,
                         "n_neg": 0, "p_sign": 0.0625, "median_diff": 0.3, "median_diff_sd": 1.1,
                         "consistency": 1.0, "is_signal": True}])
    s = wp.signal_sentence(sig)
    assert "Batch 1" in s and "higher" in s and "5 of 5" in s
    for w in plo.FORBIDDEN:
        assert w not in s.lower(), w
