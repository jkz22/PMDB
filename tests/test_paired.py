"""Unit tests for pmdb.paired (within-parent contrast selection)."""
import numpy as np
import pandas as pd
import pytest

from pmdb import paired


def test_grouping_covers_labelled_and_heldout():
    sites = pd.read_csv("outputs/fingerprint/features.csv", dtype=str)["site"]
    assert len(paired.parent_of(sites, require=set(sites))) == 31
    assert len(paired.PARENT_SITES) == 13
    assert all(s in paired.SITE_PARENT for s in paired.HELDOUT_SITES)
    with pytest.raises(AssertionError):
        paired.parent_of(sites[:-1], require=set(sites[:-1]))


def test_select_finds_planted_within_parent_feature():
    rng = np.random.default_rng(0)
    parents = np.repeat(["P0", "P1", "P2", "P3"], 6)
    codes = np.tile([0, 0, 1, 1, 2, 2], 4)
    x = rng.normal(size=(24, 10))
    x[:, 3] += 3.0 * codes  # designed feature: monotone in batch inside every parent
    x[:, 7] += 5.0 * (parents == "P1")  # parent (imaging) offset only: no batch contrast
    chosen = paired.select(x, codes, parents, 3, k=2)
    assert chosen[0] == 3 and 7 not in chosen
    d = paired.within_parent_diffs(x, codes, parents, 3)
    assert d[(0, 1)][0] == ["P0", "P1", "P2", "P3"]


def test_cv_assign_nested_selection_shapes():
    rng = np.random.default_rng(1)
    parents = np.repeat(["P0", "P1", "P2", "P3"], 6)
    codes = np.tile([0, 0, 1, 1, 2, 2], 4)
    x = rng.normal(size=(24, 8))
    x[:, 0] += 3.0 * codes
    a, proba, sel = paired.cv_assign(x, codes, parents, parents, 3, "paired", k=3)
    assert len(sel) == 4 and all(len(f) == 3 for f in sel)
    assert paired.balanced_accuracy(codes, a, 3) > 0.6


def test_rubric_and_proba():
    assert paired.rubric_score(np.array([1, 1, 0, 0], bool),
                               np.array([.9, .4, .4, .9])).tolist() == [2, 1, 1, 0]
    p = paired.softmax_proba(np.array([[1.0, 2.0, 3.0]]), 2.0)
    assert np.isclose(p.sum(), 1) and p.argmax() == 0
