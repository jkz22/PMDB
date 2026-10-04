"""Feature-level ComBat (scripts/run_combat_features.py): unseen-session policy and the nested fold."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_combat_features", REPO / "scripts" / "run_combat_features.py")
M = importlib.util.module_from_spec(spec)
sys.modules["run_combat_features"] = M
spec.loader.exec_module(M)
pytest.importorskip("neuroHarmonize")


def _frame(rng, sites, batches, shift=0.0, n=6):
    X = rng.normal(0, 1, (len(sites), n)) + shift
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n)])
    df.insert(0, "site", sites)
    df.insert(0, "batch", batches)
    return df


def test_unseen_session_rows_are_left_uncorrected():
    rng = np.random.default_rng(0)
    clean = [f"c{i}" for i in range(10)]
    mild = list(M.MILD[:5])
    train = _frame(rng, clean + mild, ["Batch_1"] * 4 + ["Batch_2"] * 3 + ["Batch_3"] * 3 + ["Batch_2"] * 2 + ["Batch_3"] * 3)
    held = _frame(rng, list(M.STRONG[:2]) + ["c99"], ["Batch_3"] * 3, shift=2.0)
    cols = [c for c in train.columns if c.startswith("f")]
    _, hout, c = M.combat_fit_apply(train, held, cols)
    assert hout["combat_applied"].tolist() == [False, False, True]
    np.testing.assert_array_equal(hout.loc[:1, c].to_numpy(), held.loc[:1, c].to_numpy())  # strong: untouched
    assert np.isfinite(hout[c].to_numpy(float)).all()
    assert not np.allclose(hout.loc[2, c].to_numpy(float), held.loc[2, c].to_numpy(float))  # clean: corrected


@pytest.mark.data
def test_nested_fold_holding_out_the_strong_parent_completes():
    p = REPO / "outputs/fingerprint/features.csv"
    if not p.exists():
        pytest.skip("fingerprint features not built")
    train = pd.read_csv(p)
    cols = [c for c in train.columns if c not in ("batch", "site") and pd.api.types.is_numeric_dtype(train[c])]
    assert (M._parents(train)[np.isin(train.site, M.STRONG)] == "h2060_ETD_s1").all()
    res = M._nested_lopo(train, cols)
    assert res["nested_rows_uncorrected"] == 4
    assert all(0 <= res[k] <= 1 for k in res if k.endswith("_nested"))
