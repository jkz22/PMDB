"""Synthetic tests for scripts/batch_classifier.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from batch_classifier import feature_sets, load_combined


def _keys():
    return pd.DataFrame(
        {
            "batch": ["Batch_1"] * 4 + ["Batch_2"] * 4 + ["Batch_3"] * 4,
            "site": [f"s{i}" for i in range(12)],
        }
    )


def _write(tmp_path, anna_sites=None):
    rng = np.random.default_rng(0)
    k = _keys()
    kev = k.assign(
        se_detector="ETD",
        segmenter_version="v0",
        K01_si_frac_adm=rng.random(12),
        K08_pcf_rpeak_x_um=[-1.0] + list(rng.random(11) + 1),
        K08_pcf_rpeak_z_um=rng.random(12) + 1,
        D01_graphite_frac_mean=rng.random(12),
        A01_large_void_frac=rng.random(12),
    )
    leo = k.assign(mat_porosity=rng.random(12), mat_bright_fraction=rng.random(12))
    anna = k.assign(
        se_detector="ETD",
        n_tiles=3,
        si_thr=0.5,
        porosity=rng.random(12),
        porosity__err=rng.random(12),
        si_frac=rng.random(12),
    )
    if anna_sites is not None:
        anna = anna.iloc[:anna_sites]
    paths = []
    for name, d in [("k", kev), ("l", leo), ("a", anna)]:
        p = tmp_path / f"{name}.csv"
        d.to_csv(p, index=False)
        paths.append(p)
    return paths


def test_load_combined(tmp_path):
    df = load_combined(*_write(tmp_path))
    assert len(df) == 12
    for c in ["se_detector", "segmenter_version", "ab_si_thr", "ab_porosity__err", "ab_n_tiles"]:
        assert c not in df.columns
    assert "ab_porosity" in df.columns and "ab_si_frac" in df.columns
    assert "mat_porosity" in df.columns
    assert df["K08_pcf_rpeak_x_um"].isna().sum() == 1


def test_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        load_combined(*_write(tmp_path, anna_sites=10))


def test_feature_sets(tmp_path):
    fs = feature_sets(load_combined(*_write(tmp_path)))
    k, l, a = set(fs["kevin"]), set(fs["leo"]), set(fs["anna"])
    assert not (k & l) and not (k & a) and not (l & a)
    assert set(fs["combined"]) == k | l | a
    for c in ["A01_large_void_frac", "ab_porosity", "mat_porosity"]:
        assert c in fs["combined"]
        assert c not in fs["combined_no_artefact"]
