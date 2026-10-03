"""Tests for the KPI screening tool (synthetic inputs; one data-marked CLI test)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmdb.screen import (
    ScreenConfig, ScreenInputError, batch_eta2, bootstrap_indices, icc1, main, merge_kpi_tables,
    parse_sentinels, partial_spearman, pivot_raw_intensity_stats, read_table, robustness_rho,
    screen, validate_replicates, TABLE_COLUMNS,
)

FAST = ScreenConfig(n_boot_cov=200, n_boot_icc=200, n_boot_cluster=100)


def _strict_loads(text: str):
    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token}")
    return json.loads(text, parse_constant=_reject)


def _sites(n_per_batch: int = 10) -> pd.DataFrame:
    rows = [{"batch": f"Batch_{b}", "site": f"s{b}{i:02d}"} for b in (1, 2, 3) for i in range(n_per_batch)]
    return pd.DataFrame(rows)


def _frame(**cols) -> pd.DataFrame:
    df = _sites()
    for k, v in cols.items():
        df[k] = v
    return df


def _batch_idx(df: pd.DataFrame) -> np.ndarray:
    return df["batch"].str[-1].astype(int).to_numpy() - 1


def _tiles(df: pd.DataFrame, **kpi_values) -> pd.DataFrame:
    parts = []
    for t in range(4):
        part = df[["batch", "site"]].copy()
        part["tile"] = t
        for k, fn in kpi_values.items():
            part[k] = fn(t)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- step 1


def test_merge_rejects_duplicate_kpi_name():
    a, b = _frame(K1=1.0), _frame(K1=2.0)
    with pytest.raises(ScreenInputError, match="K1"):
        merge_kpi_tables([a, b], ["a.csv", "b.csv"])


def test_merge_rejects_mismatched_sites():
    a, b = _frame(K1=1.0), _frame(K2=2.0).iloc[:-1]
    with pytest.raises(ScreenInputError, match="site"):
        merge_kpi_tables([a, b], ["a.csv", "b.csv"])


def test_merge_ignores_meta_and_text_columns():
    df = _frame(se_detector="ETD", elapsed_s=1.0, note="x", K1=np.arange(30.0))
    out, kpis, ignored = merge_kpi_tables([df], ["a.csv"])
    assert kpis == ["K1"]
    assert ignored == {"a.csv": ["note"]}
    assert list(out.columns) == ["batch", "site", "K1"]


def test_kpi_name_rule():
    df = _sites()
    df["K.1"] = 1.0
    with pytest.raises(ScreenInputError):
        merge_kpi_tables([df], ["a.csv"])


def test_parse_sentinels():
    assert parse_sentinels(["A=-1", "A=-1.0", "B=0"]) == (("A", -1.0), ("B", 0.0))
    with pytest.raises(ScreenInputError):
        parse_sentinels(["A"])


def test_validate_replicates_rejects_unknown_kpi():
    site = _frame(K1=1.0)
    rep = _tiles(site, K9=lambda t: 1.0)
    with pytest.raises(ScreenInputError, match="K9"):
        validate_replicates(rep, site, ["K1"], ScreenConfig())


def test_pivot_raw_intensity_stats():
    raw = pd.DataFrame({
        "batch": ["Batch_1"] * 4, "site": ["a", "a", "b", "b"],
        "channel_slot": ["BSE", "Inlens", "BSE", "Inlens"],
        "p1": [0.0, 1.0, 2.0, 3.0], "p50": [10.0, 11.0, 12.0, 13.0],
    })
    out = pivot_raw_intensity_stats(raw)
    assert list(out.columns) == ["batch", "site", "BSE_p1", "BSE_p50"]
    assert out["BSE_p1"].tolist() == [0.0, 2.0]


# ---------------------------------------------------------------- step 2


def test_icc_high_for_site_constant_low_for_noise():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 4)
    batches = np.repeat(np.arange(30) // 10, 4)
    stable = rng.normal(size=30)[groups] + rng.normal(0, 0.1, 120)
    noise = rng.normal(size=120)
    assert icc1(stable, groups, batches) > 0.9
    assert icc1(noise, groups, batches) < 0.3


def test_batch_eta2_batch_constant_is_one():
    batches = np.repeat(np.arange(3), 10)
    assert batch_eta2(batches.astype(float), batches) == pytest.approx(1.0)


def test_partial_spearman_removes_batch_shift():
    rng = np.random.default_rng(1)
    b = np.repeat(np.arange(3), 10)
    x = 10 * b + rng.normal(size=30)
    y = 10 * b + rng.normal(size=30)
    assert partial_spearman(x, y, np.zeros(30)) > 0.8
    assert abs(partial_spearman(x, y, b)) < 0.5


def test_bootstrap_indices_stratified():
    b = np.repeat(np.arange(3), [7, 7, 17])
    idx = bootstrap_indices(b, 50, 0, 1)
    assert idx.shape == (50, 31)
    assert (b[idx] == b[None, :]).all()
    assert np.array_equal(idx, bootstrap_indices(b, 50, 0, 1))
    assert not np.array_equal(idx, bootstrap_indices(b, 50, 0, 2))


def test_robustness_rho_detects_rank_reversal():
    base = _sites()
    v = np.random.default_rng(2).normal(size=30)
    sens = pd.concat([base.assign(p=1, K=v), base.assign(p=2, K=-v)], ignore_index=True)
    rho, n, nan_pair = robustness_rho(sens, "K", ["p"])
    assert rho == pytest.approx(-1.0) and n == 2 and not nan_pair


# ---------------------------------------------------------------- step 3


def _noise(seed: int, n: int = 30) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n)


def test_degeneracy_drops():
    peak = _noise(4)
    peak[:10] = -1.0
    zmad = np.zeros(30)
    zmad[:5] = [1, 2, 3, 4, 5]
    miss = _noise(5)
    miss[:8] = np.nan
    df = _frame(const=1.0, zmad=zmad, miss=miss, peak=peak, ok=_noise(6))
    kpis = ["const", "zmad", "miss", "peak", "ok"]
    t = screen(df, kpis, FAST).table.set_index("kpi")
    assert t.at["const", "deciding_gate"] == "degeneracy:few_unique"
    assert t.at["zmad", "deciding_gate"] == "degeneracy:zero_mad"
    assert t.at["miss", "deciding_gate"] == "degeneracy:missing"
    assert not t.at["peak", "deciding_gate"].startswith("degeneracy")
    t2 = screen(df, kpis, replace(FAST, sentinels=(("peak", -1.0),))).table.set_index("kpi")
    assert t2.at["peak", "deciding_gate"] == "degeneracy:missing"


def test_covariate_tracking_kpi_dropped_as_artefact():
    cov = _noise(7)
    df = _frame(art=cov + _noise(8) * 0.1, ok=_noise(9))
    covdf = _sites().assign(cov=cov)
    t = screen(df, ["art", "ok"], FAST, covariates=covdf).table.set_index("kpi")
    assert t.at["art", "deciding_gate"] == "artefact:cov"
    assert t.at["ok", "decision"] == "keep"
    t0 = screen(df, ["art", "ok"], FAST).table.set_index("kpi")
    assert "artefact" in t0.at["ok", "untested_gates"]


def test_batch_aligned_covariate_does_not_drop_batch_kpi():
    b = _batch_idx(_sites())
    covdf = _sites().assign(cov=b + _noise(10) * 0.01)
    df = _frame(bk=10 * b + _noise(11))
    t = screen(df, ["bk"], FAST, covariates=covdf).table.set_index("kpi")
    assert t.at["bk", "decision"] == "keep"
    assert "high_batch_eta2" in t.at["bk", "flags"]
    assert t.at["bk", "covariate_rho_marginal"] > 0.8


def test_unreliable_kpi_dropped_and_absent_is_untested():
    u = _noise(12)
    df = _frame(stable=u, noisy=_noise(13), norep=_noise(14))
    rng = np.random.default_rng(15)
    rep = _tiles(df, stable=lambda t: u + rng.normal(0, 0.1, 30), noisy=lambda t: rng.normal(size=30) + rng.normal(size=30))
    t = screen(df, ["stable", "noisy", "norep"], FAST, replicates=rep).table.set_index("kpi")
    assert t.at["noisy", "deciding_gate"] == "reliability"
    assert t.at["stable", "decision"] == "keep" and t.at["stable", "icc"] > 0.9
    assert t.at["norep", "decision"] == "keep"
    assert "reliability" in t.at["norep", "untested_gates"]


def test_near_duplicates_collapse_to_higher_icc_rep():
    u = _noise(16)
    df = _frame(a=u + _noise(17) * 0.01, b=u + _noise(18) * 0.01, c=_noise(19))
    rng = np.random.default_rng(20)
    rep = _tiles(df, a=lambda t: u + rng.normal(0, 0.1, 30), b=lambda t: u + rng.normal(0, 0.7, 30))
    t = screen(df, ["a", "b", "c"], FAST, replicates=rep).table.set_index("kpi")
    assert t.at["a", "cluster_id"] == t.at["b", "cluster_id"]
    assert t.at["a", "cluster_rep"] == "a" and t.at["b", "cluster_rep"] == "a"
    assert t.at["b", "deciding_gate"] == "redundant_with:a"
    assert t.at["a", "decision"] == "keep" and t.at["c", "decision"] == "keep"


def test_robustness_drop():
    r, s = _noise(21), _noise(22)
    df = _frame(r=r, s=s)
    base = _sites()
    sens = pd.concat([base.assign(p=1, r=r, s=s),
                      base.assign(p=2, r=-r, s=s + _noise(23) * 1e-3)], ignore_index=True)
    cfg = replace(FAST, sensitivity_params=("p",))
    t = screen(df, ["r", "s"], cfg, sensitivity=sens).table.set_index("kpi")
    assert t.at["r", "deciding_gate"] == "robustness"
    assert t.at["s", "decision"] == "keep"


def test_icc_undefined_is_untested_not_dropped():
    df = _frame(K=_noise(40))
    rng = np.random.default_rng(41)
    first = {f"s{b}00" for b in (1, 2, 3)}
    rows = []
    for _, r in df[["batch", "site"]].iterrows():
        if r["site"] in first:
            rows += [{**r, "tile": t, "K": rng.normal()} for t in range(4)]
        else:
            rows.append({**r, "tile": 0, "K": np.nan})
    rep = pd.DataFrame(rows)
    t = screen(df, ["K"], FAST, replicates=rep).table.set_index("kpi")
    assert t.at["K", "decision"] == "keep" and t.at["K", "deciding_gate"] == ""
    assert "reliability" in t.at["K", "untested_gates"]
    assert "icc_undefined" in t.at["K", "flags"]


def test_constant_within_batch_covariate_is_untested():
    df = _frame(K=_noise(42))
    covdf = _sites().assign(cov=_batch_idx(_sites()).astype(float))
    t = screen(df, ["K"], FAST, covariates=covdf).table.set_index("kpi")
    assert "artefact" in t.at["K", "untested_gates"]


def test_non_numeric_replicate_kpi_column_is_error():
    df = _frame(K=_noise(43))
    rep = _tiles(df, K=lambda t: _noise(44 + t)).astype({"K": object})
    rep.loc[0, "K"] = "n/a"
    with pytest.raises(ScreenInputError, match="not numeric"):
        validate_replicates(rep, df, ["K"], FAST)


def test_first_failing_gate_decides():
    cov = _noise(50)
    x, y = cov + _noise(51) * 0.1, _noise(52)
    df = _frame(x=x, y=y)
    covdf = _sites().assign(cov=cov)
    rng = np.random.default_rng(53)
    rep = _tiles(df, x=lambda t: rng.normal(size=30), y=lambda t: rng.normal(size=30))
    base = _sites()
    sens = pd.concat([base.assign(p=1, x=x, y=y), base.assign(p=2, x=-x, y=-y)], ignore_index=True)
    cfg = replace(FAST, sensitivity_params=("p",))
    t = screen(df, ["x", "y"], cfg, replicates=rep, sensitivity=sens, covariates=covdf).table.set_index("kpi")
    assert t.at["x", "deciding_gate"] == "artefact:cov"
    assert t.at["y", "deciding_gate"] == "reliability"


def test_filtered_columns_equal_keep_set():
    df = _frame(const=1.0, a=_noise(24), b=_noise(25), c=_noise(26))
    kpis = ["const", "a", "b", "c"]
    res = screen(df, kpis, FAST)
    kept = [k for k in kpis if k in set(res.table.loc[res.table["decision"] == "keep", "kpi"])]
    assert list(res.filtered.columns) == ["batch", "site", *kept]
    assert len(res.filtered) == len(df)
    assert list(res.table.columns) == TABLE_COLUMNS


# ---------------------------------------------------------------- step 4

SMALL = ["--n-boot-cov", "100", "--n-boot-icc", "100", "--n-boot-cluster", "50"]


def test_main_writes_outputs(tmp_path):
    u = _noise(30)
    df = _frame(const=1.0, a=u, b=_noise(31), c=_noise(32))
    rng = np.random.default_rng(33)
    rep = _tiles(df, a=lambda t: u + rng.normal(0, 0.1, 30), b=lambda t: rng.normal(size=30))
    cov = _sites().assign(cov=_noise(34))
    for name, d in (("site", df), ("rep", rep), ("cov", cov)):
        d.to_csv(tmp_path / f"{name}.csv", index=False)
    out = tmp_path / "out"
    rc = main(["--kpis", str(tmp_path / "site.csv"), "--replicates", str(tmp_path / "rep.csv"),
               "--covariates", str(tmp_path / "cov.csv"), "--out-dir", str(out), *SMALL])
    assert rc == 0
    for f in ("kpi_screen.csv", "site_kpis_filtered.csv", "screen_config.json"):
        assert (out / f).exists()
    _strict_loads((out / "screen_config.json").read_text())
    table = pd.read_csv(out / "kpi_screen.csv", keep_default_na=False)
    kept = [k for k in ["const", "a", "b", "c"] if k in set(table.loc[table["decision"] == "keep", "kpi"])]
    assert list(pd.read_csv(out / "site_kpis_filtered.csv").columns) == ["batch", "site", *kept]


def test_main_input_error_returns_2(tmp_path, capsys):
    a, b = _frame(K1=_noise(40)), _frame(K2=_noise(41)).iloc[:-1]
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)
    rc = main(["--kpis", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), "--out-dir", str(tmp_path / "o")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.data
def test_screen_cli_on_real_tables(tmp_path):
    root = Path(__file__).resolve().parents[1]
    kp = root / "outputs" / "kpis"
    if not (kp / "site_kpis.csv").exists():
        pytest.skip("outputs/kpis/site_kpis.csv absent")
    cov = tmp_path / "cov.csv"
    pivot_raw_intensity_stats(read_table(root / "outputs" / "raw_intensity_stats.csv")).to_csv(cov, index=False)
    out = tmp_path / "out"
    rc = main(["--kpis", str(kp / "site_kpis.csv"), "--replicates", str(kp / "tile_kpis.csv"),
               "--sensitivity", str(kp / "sensitivity.csv"), "--sensitivity-params", "d_um", "d_star_um",
               "--covariates", str(cov),
               "--sentinel", "K08_pcf_rpeak_x_um=-1", "K08_pcf_rpeak_z_um=-1", *SMALL, "--out-dir", str(out)])
    assert rc == 0
    t = pd.read_csv(out / "kpi_screen.csv", keep_default_na=False, na_values=[""])
    t["deciding_gate"] = t["deciding_gate"].fillna("")
    assert len(t) == 43 and list(t.columns) == TABLE_COLUMNS
    assert set(t["decision"]) <= {"keep", "drop"}
    g = t.set_index("kpi")
    assert g.at["K16_si_graphite_dist_median_um", "deciding_gate"] == "degeneracy:few_unique"
    assert g.at["K08_pcf_rpeak_x_um", "deciding_gate"] == "degeneracy:missing"
    assert g.at["K08_pcf_rpeak_z_um", "deciding_gate"] == "degeneracy:missing"
    assert g.at["K10_cv_w10", "untested_gates"] == "reliability;robustness"
    assert np.isfinite(g.at["K04_agglom_frac", "robustness_rho"])
    assert g.at["K04_agglom_frac", "deciding_gate"] == "reliability"
    f = pd.read_csv(out / "site_kpis_filtered.csv")
    assert len(f) == 31
    assert list(f.columns) == ["batch", "site", *t.loc[t["decision"] == "keep", "kpi"].pipe(
        lambda s: [k for k in pd.read_csv(kp / "site_kpis.csv", nrows=0).columns if k in set(s)])]
