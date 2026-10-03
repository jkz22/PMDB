"""Unit tests for the KPI runner / coverage scripts (synthetic inputs, no real data)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pmdb.kpis import N_TILES, catalogue_columns
from scripts import check_kpi_coverage, run_kpis


def _strict_loads(text: str):
    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token}")
    return json.loads(text, parse_constant=_reject)


def test_kpi_parameters_are_strict_json():
    params = run_kpis.kpi_parameters()
    assert not any(k.endswith(".NAN") for k in params)
    assert "common.N_NULL_SIMS" in params and "pointpattern.K06_CLUSTER_CUT" in params
    assert _strict_loads(run_kpis.dump_run_log({"parameters": params}))["parameters"] == _strict_loads(json.dumps(params))


def test_dump_run_log_rejects_nan():
    with pytest.raises(ValueError):
        run_kpis.dump_run_log({"x": float("nan")})


SITES = [("Batch_1", "aaa"), ("Batch_2", "bbb")]


def _default_frames():
    site_cols, tile_cols = catalogue_columns()
    site_df = pd.DataFrame([{"batch": b, "site": s, **{c: 1.0 for cs in site_cols.values() for c in cs}}
                            for b, s in SITES])
    tile_df = pd.DataFrame([{"batch": b, "site": s, "tile": t,
                             **{c: 1.0 for cs in tile_cols.values() for c in cs}, "nan_reason": ""}
                            for b, s in SITES for t in range(N_TILES)])
    return site_df, tile_df


def _write_kpi_dir(tmp_path, site_df=None, tile_df=None):
    default_site, default_tile = _default_frames()
    site_df = default_site if site_df is None else site_df
    tile_df = default_tile if tile_df is None else tile_df
    pd.DataFrame(SITES, columns=["batch", "site"]).to_csv(tmp_path / "manifest.csv", index=False)
    site_df.to_csv(tmp_path / "site_kpis.csv", index=False)
    tile_df.to_csv(tmp_path / "tile_kpis.csv", index=False)
    return ["--kpi-dir", str(tmp_path), "--manifest", str(tmp_path / "manifest.csv")]


def test_coverage_complete_passes(tmp_path, capsys):
    assert check_kpi_coverage.main(_write_kpi_dir(tmp_path)) == 0


def test_coverage_rejects_duplicated_site_tiles(tmp_path, capsys):
    _, tile_df = _default_frames()
    a = tile_df[tile_df["site"] == "aaa"]
    tile_df = pd.concat([a, a.assign(batch="Batch_1", site="aaa")], ignore_index=True)
    assert len(tile_df) == 2 * N_TILES
    rc = check_kpi_coverage.main(_write_kpi_dir(tmp_path, tile_df=tile_df))
    out = capsys.readouterr().out
    assert rc == 1
    assert "duplicated in tile_kpis.csv: Batch_1/aaa/0 (2 rows)" in out
    assert "missing from tile_kpis.csv: Batch_2/bbb/0" in out


def test_coverage_rejects_extra_tile(tmp_path, capsys):
    _, tile_df = _default_frames()
    extra = tile_df[tile_df["site"] == "aaa"].iloc[[0]].assign(tile=N_TILES)
    tile_df = pd.concat([tile_df, extra], ignore_index=True)
    rc = check_kpi_coverage.main(_write_kpi_dir(tmp_path, tile_df=tile_df))
    out = capsys.readouterr().out
    assert rc == 1
    assert f"unexpected in tile_kpis.csv: Batch_1/aaa/{N_TILES}" in out


def test_coverage_rejects_duplicated_site_row(tmp_path, capsys):
    site_df, _ = _default_frames()
    site_df = pd.concat([site_df, site_df.iloc[[0]]], ignore_index=True)
    rc = check_kpi_coverage.main(_write_kpi_dir(tmp_path, site_df=site_df))
    out = capsys.readouterr().out
    assert rc == 1
    assert "duplicated in site_kpis.csv: Batch_1/aaa (2 rows)" in out


ORDER = [("Batch_1", "a"), ("Batch_1", "b"), ("Batch_2", "c")]


def _rows(site_v):
    return pd.DataFrame([{"batch": "Batch_1" if s != "c" else "Batch_2", "site": s, "v": v} for s, v in site_v])


def test_merge_replaces_site_rows_in_manifest_order():
    out = run_kpis.merge_table(_rows([("a", 1), ("b", 2), ("c", 3)]), _rows([("b", 20)]), ORDER)
    assert list(zip(out["site"], out["v"])) == [("a", 1), ("b", 20), ("c", 3)]


def test_merge_inserts_new_site_by_manifest_rank():
    out = run_kpis.merge_table(_rows([("a", 1), ("c", 3)]), _rows([("b", 2)]), ORDER)
    assert list(out["site"]) == ["a", "b", "c"]


def test_merge_tile_table_replaces_all_site_tiles_and_sorts_by_tile():
    def tiles(rows):
        return pd.DataFrame([{"batch": "Batch_1", "site": s, "tile": t, "v": v} for s, t, v in rows])
    existing = tiles([("a", 0, 1), ("a", 1, 1), ("b", 0, 2), ("b", 1, 2)])
    new = tiles([("b", 1, 21), ("b", 0, 20)])
    out = run_kpis.merge_table(existing, new, ORDER, extra_sort=("tile",))
    assert len(out) == 4
    assert list(zip(out["site"], out["tile"], out["v"])) == [("a", 0, 1), ("a", 1, 1), ("b", 0, 20), ("b", 1, 21)]


def test_merge_keeps_within_site_order_without_extra_sort():
    def curves(rows):
        return pd.DataFrame([{"batch": "Batch_1", "site": s, "x": x} for s, x in rows])
    out = run_kpis.merge_table(curves([("a", 1), ("a", 2)]), curves([("b", 3), ("b", 1)]), ORDER)
    assert list(zip(out["site"], out["x"])) == [("a", 1), ("a", 2), ("b", 3), ("b", 1)]


def test_merge_without_existing_returns_new():
    new = _rows([("b", 2)])
    pd.testing.assert_frame_equal(run_kpis.merge_table(None, new, ORDER), new)


def test_merge_rejects_column_mismatch():
    existing = _rows([("a", 1)]).assign(w=0)
    with pytest.raises(ValueError, match="column mismatch"):
        run_kpis.merge_table(existing, _rows([("b", 2)]), ORDER)
