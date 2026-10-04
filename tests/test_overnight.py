"""Non-data tests for the overnight scripts (tiny synthetic inputs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import overnight_features as of
from scripts import overnight_permutation as op
from scripts import overnight_stability as ost


def _res(batch, site, v):
    return {"curves": [{"batch": batch, "site": site, "curve": "c", "x": 0.0, "value": v}],
            "scalars": {"batch": batch, "site": site, "acl": v},
            "tiles": [{"batch": batch, "site": site, "n_tiles": 16, "tile": 0}]}


def test_write_tables_union_and_replace(tmp_path):
    order = [("B1", "a"), ("B1", "b")]
    results = {"B1/a": _res("B1", "a", 1.0), "B1/b": _res("B1", "b", 2.0)}
    of.write_tables(results, order, tmp_path)
    # subset update replaces only site a; b is retained
    results["B1/a"] = _res("B1", "a", 9.0)
    of.write_tables(results, order, tmp_path)
    df = pd.read_csv(tmp_path / "curves_rich.csv")
    assert dict(zip(df["site"], df["value"])) == {"a": 9.0, "b": 2.0}
    assert len(pd.read_csv(tmp_path / "site_scalars.csv")) == 2


def test_load_existing_rejects_mismatched_columns(tmp_path, capsys):
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "curves_rich.csv", index=False)
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "site_scalars.csv", index=False)
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "tile_kpis_rich.csv", index=False)
    assert of.load_existing(tmp_path) == {}
    assert "NOTE" in capsys.readouterr().err
    assert of.load_existing(tmp_path / "missing") == {}


def test_n_drop3_cap():
    ost.check_n_drop3(0, 5)
    ost.check_n_drop3(10, 5)
    with pytest.raises(SystemExit):
        ost.check_n_drop3(11, 5)
    with pytest.raises(SystemExit):
        ost.check_n_drop3(-1, 5)


def test_resume_rejects_mismatched_metadata(tmp_path):
    meta = {"seed": 0, "n_perm_target": 10, "chunk": 5, "inputs_sha": "x"}
    (tmp_path / "permutation_10k.json").write_text(json.dumps({**meta, "n_perm_done": 5}))
    pd.DataFrame({"perm": range(5), "accuracy": np.linspace(0, 1, 5)}).to_csv(
        tmp_path / "null_accuracies.csv", index=False)
    assert len(op.resume_state(tmp_path, meta, 10, 5)) == 5
    assert op.resume_state(tmp_path, {**meta, "seed": 1}, 10, 5) is None
    assert op.resume_state(tmp_path, {**meta, "inputs_sha": "y"}, 10, 5) is None
    assert op.resume_state(tmp_path, {**meta, "n_perm_target": 20}, 10, 5) is None
