"""Regression tests for the analytical_benchmarks scripts (PR #14 review fixes). Non-data, synthetic inputs only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AB = Path(__file__).resolve().parents[1] / "analytical_benchmarks"
sys.path.insert(0, str(AB))

import compare  # noqa: E402


def test_compare_ignores_nan(monkeypatch):
    monkeypatch.setattr(compare, "NPERM", 500)
    r = compare.compare(np.array([1.0, 2.0, np.nan, 1.5]), np.array([1.2, 1.8, 1.4, 1.6, 2.0]), 1.0)
    assert r["p"] > 0.05
    assert np.isfinite(r["diff"])
    assert "insufficient" not in r


def test_compare_insufficient():
    r = compare.compare(np.array([np.nan, 1.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 1.0)
    assert r["insufficient"] and r["p"] == 1.0
    r["p_holm"] = 1.0
    assert compare.verdict_of(r) == "consistent"


def test_site_flags_nan_does_not_disable_flags():
    df = pd.DataFrame({"batch": ["B1", "B2"] * 6, "site": [f"s{i}" for i in range(12)]})
    for k in compare.KPIS:
        df[k] = np.linspace(1.0, 1.1, 12)
    df.loc[0, "si_d50_um"] = 10.0
    df.loc[1, "si_d50_um"] = np.nan
    flags = {(f["site"], f["kpi"]) for f in compare.site_flags(df)}
    assert ("s0", "si_d50_um") in flags
    assert ("s1", "si_d50_um") not in flags


def test_batch_colours():
    c = compare.batch_colours(["Batch_1", "Batch_2", "Batch_3"])
    assert list(c.values()) == ["#2f6fdf", "#e08a1e", "#c23b3b"]
    names = [f"B{i}" for i in range(10)]
    c = compare.batch_colours(names)
    assert set(c) == set(names) and c["B8"] == c["B0"]


def test_n_cv_splits():
    assert compare.n_cv_splits(["A"] * 7 + ["B"] * 9) == 5
    assert compare.n_cv_splits(["A"] * 7 + ["B"] * 3) == 3
    assert compare.n_cv_splits(["A"] * 7 + ["B"]) == 1


def test_hotspot_sites():
    defaults = [("B1", "d1", "si_frac")]
    res = {"site_flags": [
        dict(batch="B1", site="d1", kpi="porosity", robust_z=5),
        dict(batch="B4", site="new", kpi="si_frac", robust_z=7),
        dict(batch="B4", site="new", kpi="porosity", robust_z=4),
        dict(batch="B2", site="chance", kpi="si_d10_um", robust_z=4),
        dict(batch="B3", site="low", kpi="porosity", robust_z=-5)],
        "loo": {"B2": {"unusual_sites": ["odd"]}}}
    out = compare.hotspot_sites(res, defaults, ["porosity", "si_frac", "interface_um_per_um2", "inlens_texture"])
    assert out == [("B1", "d1", "si_frac"), ("B2", "odd", "si_frac"), ("B4", "new", "si_frac")]


def test_particles_sparse_schema_matches_dense():
    import kpis

    def run(n):
        lab = np.ones((300, 300), np.uint8)
        for i in range(n):
            r, c = 20 + 70 * (i // 4), 20 + 70 * (i % 4)
            lab[r:r + 25, c:c + 25] = 2
        return kpis.particles(lab, 0.05)[0]
    sparse, dense, none = run(2), run(4), run(0)
    assert set(sparse) == set(dense)
    assert sparse["si_n_cracked"] == 0 and sparse["si_n_anomalous"] == 0
    assert np.isnan(sparse["si_d50_um"])
    assert none["si_n_cracked"] == 0 and none["si_n_anomalous"] == 0


def _row(kpi, **kw):
    r = dict(kpi=kpi, mean_a=1.0, mean_b=1.0, diff=0.0, ci95=[-0.1, 0.1], d=0.0, p=0.5, p_holm=1.0,
             spread_diff=0.0, p_spread=0.5, verdict="consistent")
    r.update(kw)
    return r


def _report(tmp_path, compare_json, site_rows, physics_json, physics_rows):
    import json
    import os
    import subprocess
    (tmp_path / "compare.json").write_text(json.dumps(compare_json))
    pd.DataFrame(site_rows).to_csv(tmp_path / "site_kpis.csv", index=False)
    (tmp_path / "physics.json").write_text(json.dumps(physics_json))
    pd.DataFrame(physics_rows).to_csv(tmp_path / "physics.csv", index=False)
    subprocess.run([sys.executable, str(AB / "report.py")], cwd=tmp_path, check=True,
                   env={**os.environ, "PYTHONPATH": os.pathsep.join([str(AB), str(AB.parent)]), "MPLBACKEND": "Agg"})
    return (tmp_path / "body.html").read_text()


def _report_fixture(tmp_path, swell_one=None, p_holm=None):
    import physics
    loo = {b: dict(n_sites=3, rows=[_row(k) for k in compare.KPIS], verdict="consistent",
                   flagged_sites=[], unusual_sites=[]) for b in ("Batch_1", "Batch_2")}
    loo["Batch_2"]["verdict"] = "investigate"
    loo["Batch_2"]["rows"] = [_row("porosity", p_spread=0.004, spread_diff=0.01, verdict="investigate")
                              if r["kpi"] == "porosity" else r for r in loo["Batch_2"]["rows"]]
    cj = dict(loo=loo, pairs={}, site_flags=[], confounds=[])
    sites = [("Batch_1", f"a{i}") for i in (1, 2, 3)] + [("Batch_2", f"b{i}") for i in (1, 2, 3)]
    rows, prow = [], []
    for b, s in sites:
        r = dict(batch=b, site=s, se_detector="ETD")
        r.update({k: 1.0 for k in compare.KPIS}); r.update(si_n_cracked=1.0, si_n_anomalous=1.0)
        for k in ("porosity", "si_frac", "interface_um_per_um2", "inlens_texture"): r[f"{k}__err"] = 0.1
        for k in ("porosity", "si_frac", "interface_um_per_um2"): r[f"{k}__naive_err"] = 0.05
        rows.append(r)
        q = dict(batch=b, site=s, spec_capacity__siox=500.0)
        for k in physics.KPIS: q[k] = 2.0; q[f"{k}__err"] = 0.1
        prow.append(q)
    if swell_one is not None: prow[0]["swell_to_pore"] = swell_one
    pl = {b: [_row(k) for k in physics.KPIS] for b in ("Batch_1", "Batch_2")}
    if p_holm is not None:
        pl["Batch_2"] = [_row(k, p_holm=p_holm, verdict="outlier") if k == "spec_capacity" else _row(k) for k in physics.KPIS]
    return _report(tmp_path, cj, rows, dict(loo=pl, site_flags=[]), prow)


def test_report_spread_only_and_physics_defaults(tmp_path):
    body = _report_fixture(tmp_path)
    assert "larger site-to-site spread in Porosity (area fraction)" in body
    assert "multiple unusual sites" not in body
    assert "so no batch differs as a whole" in body
    assert "at every site" in body


def test_report_physics_difference_and_partial_swelling(tmp_path):
    body = _report_fixture(tmp_path, swell_one=0.8, p_holm=0.004)
    assert "Batch_2: Specific capacity (mAh/g solids) (adjusted p=0.004)" in body
    assert "no batch differs as a whole" not in body
    assert "at 5 of 6 sites" in body
