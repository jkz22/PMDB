"""Synthetic known-answer tests for the leave-one-site-out outlier test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmdb.outliers import DIAGNOSTIC_FAMILIES, VERDICT_FAMILIES, loso_scores, robust_z, run_pipeline


def _synthetic(seed: int = 0, shift_family: str | None = None, shift_site: int = 5, shift_l: float = 6.0):
    rng = np.random.default_rng(seed)
    data = {}
    for fam, (_, signs) in VERDICT_FAMILIES.items():
        lat = rng.normal(size=31)
        if fam == shift_family:
            lat[shift_site] = shift_l
        for c, sg in signs.items():
            data[c] = (sg if sg != 0 else 1) * 0.8 * lat + 0.6 * rng.normal(size=31)
    for cols in DIAGNOSTIC_FAMILIES.values():
        for c in cols:
            data[c] = rng.normal(size=31)
    df = pd.DataFrame(data)
    df["batch"] = ["B1"] * 10 + ["B2"] * 10 + ["B3"] * 11
    df["site"] = [f"s{i}" for i in range(31)]
    df["se_detector"] = "ETD"
    return df


@pytest.mark.xfail(strict=True, reason="block-PCA F-test too conservative: injected outlier gets q_bh 0.15; method parked, see docs/reports/preliminary-site-outliers.md")
def test_injected_outlier_flagged():
    df = _synthetic(shift_family="clustering")
    sc, _, _, _ = run_pipeline(df, n_boot=20)
    row = sc.iloc[0]
    assert row["site"] == "s5" and row["rank"] == 1
    assert row["p_conformal"] == pytest.approx(1 / 31)
    assert row["q_bh"] < 0.05
    assert row["comp_clustering"] > 3


def test_null_calibration():
    any_flag, pooled = [], []
    for seed in range(100, 120):
        sc = loso_scores(_synthetic(seed))
        any_flag.append((sc["q_bh"] < 0.05).any())
        pooled.append(sc["p_param"].to_numpy())
    assert np.mean(any_flag) <= 0.15
    assert 0.4 <= np.concatenate(pooled).mean() <= 0.8


def test_robust_z_zero_scale_raises():
    ref = np.random.default_rng(0).normal(size=(20, 3))
    ref[:, 1] = 2.0
    with pytest.raises(ValueError):
        robust_z(ref, ref)
