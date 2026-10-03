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


def _fake_result(batch, site, value):
    site_cols, tile_cols = catalogue_columns()
    meta = {"batch": batch, "site": site, "se_detector": "ETD", "segmenter_version": "t"}
    return {
        "site_row": {**meta, **{c: value for cs in site_cols.values() for c in cs}},
        "tiles": [{**meta, "tile": t, **{c: value for cs in tile_cols.values() for c in cs}, "nan_reason": ""}
                  for t in range(N_TILES)],
        "curves": [{"batch": batch, "site": site, "kpi_id": "K", "curve": "c", "x": 0.0, "value": value}],
        "sweep": [{"batch": batch, "site": site, "param": 1.0, "val": value}],
        "seconds": 0.0, "segmentation": {}, "n_si_objects": 0,
    }


def test_run_kpis_main_failure_and_subset(tmp_path, monkeypatch):
    good, bad = SITES
    monkeypatch.setattr(run_kpis, "all_sites", lambda: list(SITES))

    def worker(args):
        b, s, _ = args
        if (b, s) == bad:
            return b, s, None, "RuntimeError: boom\ntraceback"
        return b, s, _fake_result(b, s, 2.0), None

    base = {f"{b}/{s}": _fake_result(b, s, 1.0) for b, s in SITES}
    for name, df in run_kpis.build_tables(base, list(base)).items():
        df.to_csv(tmp_path / name, index=False)
    before = {n: (tmp_path / n).read_bytes() for n in run_kpis.TABLES}
    common_args = ["--out-dir", str(tmp_path), "--no-overlays"]
    monkeypatch.setattr(run_kpis, "_worker", worker)

    # (a) full run with one failing site: exit 1, tables untouched, attempt logged separately
    assert run_kpis.main(common_args) == 1
    assert {n: (tmp_path / n).read_bytes() for n in run_kpis.TABLES} == before
    assert not (tmp_path / run_kpis.RUN_LOG).exists()
    log = json.loads((tmp_path / run_kpis.ATTEMPT_LOG).read_text())
    assert log["tables_written"] is False and log["failed"] == [f"{bad[0]}/{bad[1]}"]

    # (b) subset of the good site: exit 0, only that site's values replaced
    assert run_kpis.main([*common_args, "--sites", f"{good[0]}/{good[1]}"]) == 0
    site_df = pd.read_csv(tmp_path / "site_kpis.csv", dtype={"batch": str, "site": str})
    col = next(iter(catalogue_columns()[0].values()))[0]
    assert site_df.loc[site_df["site"] == good[1], col].tolist() == [2.0]
    assert site_df.loc[site_df["site"] == bad[1], col].tolist() == [1.0]
    tile_df = pd.read_csv(tmp_path / "tile_kpis.csv", dtype={"batch": str, "site": str})
    assert len(tile_df) == 2 * N_TILES
    log = json.loads((tmp_path / run_kpis.RUN_LOG).read_text())
    assert log["sites"][f"{good[0]}/{good[1]}"]["run"] == log["latest_run"]
    assert log["sites"][f"{bad[0]}/{bad[1]}"]["run"] == run_kpis.UNKNOWN_RUN
    assert not (tmp_path / run_kpis.ATTEMPT_LOG).exists()

    # (c) unknown site: exit 2
    assert run_kpis.main([*common_args, "--sites", "Batch_9/nope"]) == 2


GOOD_KEY, BAD_KEY = (f"{b}/{s}" for b, s in SITES)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """run_kpis.main on SITES with fake workers; set ``state['fail']`` / ``state['commit']`` per call."""
    state = {"fail": set(), "commit": "a" * 40, "value": 1.0}
    monkeypatch.setattr(run_kpis, "all_sites", lambda: list(SITES))
    monkeypatch.setattr(run_kpis, "git_commit", lambda: state["commit"])

    def worker(args):
        b, s, _ = args
        if f"{b}/{s}" in state["fail"]:
            return b, s, None, "RuntimeError: boom\ntraceback"
        return b, s, _fake_result(b, s, state["value"]), None

    monkeypatch.setattr(run_kpis, "_worker", worker)

    def run(*extra):
        return run_kpis.main(["--out-dir", str(tmp_path), "--no-overlays", *extra])

    return run, state, tmp_path


def _log(path):
    return _strict_loads((path / run_kpis.RUN_LOG).read_text())


def test_subset_run_keeps_retained_site_provenance(runner):
    run, state, out = runner
    assert run() == 0
    full = _log(out)
    full_run = full["latest_run"]
    assert {v["run"] for v in full["sites"].values()} == {full_run}
    assert full["runs"][full_run]["git_commit"] == "a" * 40

    state.update(commit="b" * 40, value=2.0)
    assert run("--sites", GOOD_KEY) == 0
    log = _log(out)
    sub_run = log["latest_run"]
    assert sub_run != full_run
    assert log["sites"][GOOD_KEY]["run"] == sub_run
    assert log["sites"][BAD_KEY] == full["sites"][BAD_KEY]
    assert log["runs"][full_run] == full["runs"][full_run]
    assert log["runs"][sub_run]["git_commit"] == "b" * 40
    assert log["runs"][sub_run]["mode"] == "subset" and "parameters" in log["runs"][sub_run]

    state.update(commit="c" * 40)
    assert run("--sites", BAD_KEY) == 0
    log = _log(out)
    assert set(log["runs"]) == {sub_run, log["latest_run"]}  # unreferenced full run pruned


def test_failed_subset_leaves_successful_log_untouched(runner):
    run, state, out = runner
    assert run() == 0
    before = {n: (out / n).read_bytes() for n in (*run_kpis.TABLES, run_kpis.RUN_LOG)}

    state.update(commit="b" * 40, fail={GOOD_KEY})
    assert run("--sites", GOOD_KEY) == 1
    assert {n: (out / n).read_bytes() for n in before} == before
    attempt = _strict_loads((out / run_kpis.ATTEMPT_LOG).read_text())
    assert attempt["failed"] == [GOOD_KEY] and attempt["git_commit"] == "b" * 40

    state.update(fail=set())
    assert run("--sites", GOOD_KEY) == 0
    assert not (out / run_kpis.ATTEMPT_LOG).exists()


def test_merge_error_leaves_successful_log_untouched(runner):
    run, state, out = runner
    assert run() == 0
    site_path = out / "site_kpis.csv"
    pd.read_csv(site_path).assign(extra=0).to_csv(site_path, index=False)
    before = {n: (out / n).read_bytes() for n in (*run_kpis.TABLES, run_kpis.RUN_LOG)}
    assert run("--sites", GOOD_KEY) == 1
    assert {n: (out / n).read_bytes() for n in before} == before
    assert "column mismatch" in _strict_loads((out / run_kpis.ATTEMPT_LOG).read_text())["table_error"]


def test_subset_run_converts_legacy_log(runner):
    run, state, out = runner
    assert run() == 0
    legacy = {"git_commit": "f" * 40, "command": "run_kpis.py --jobs 8", "started_utc": "2026-01-01T00:00:00Z",
              "mode": "full", "tables_written": True, "parameters": {"p": 1},
              "sites": {k: {"status": "ok", "seconds": 1.0} for k in (GOOD_KEY, BAD_KEY)}}
    (out / run_kpis.RUN_LOG).write_text(json.dumps(legacy))
    state.update(commit="b" * 40)
    assert run("--sites", GOOD_KEY) == 0
    log = _log(out)
    old = log["runs"][log["sites"][BAD_KEY]["run"]]
    assert old["git_commit"] == "f" * 40 and old["parameters"] == {"p": 1}
    assert log["sites"][BAD_KEY]["seconds"] == 1.0
    assert log["runs"][log["sites"][GOOD_KEY]["run"]]["git_commit"] == "b" * 40


@pytest.mark.parametrize("legacy", [None, "not json", {"tables_written": False, "sites": {}}])
def test_subset_run_marks_unknown_provenance(runner, legacy):
    run, state, out = runner
    assert run() == 0
    path = out / run_kpis.RUN_LOG
    if legacy is None:
        path.unlink()
    else:
        path.write_text(legacy if isinstance(legacy, str) else json.dumps(legacy))
    assert run("--sites", GOOD_KEY) == 0
    log = _log(out)
    assert log["sites"][BAD_KEY]["run"] == run_kpis.UNKNOWN_RUN
    assert log["runs"][run_kpis.UNKNOWN_RUN]["reason"]
    assert log["sites"][GOOD_KEY]["run"] == log["latest_run"]
