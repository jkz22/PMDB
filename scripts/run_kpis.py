"""Run segmentation + all v1 KPIs on every site (spec 002).

Writes outputs/kpis/{site_kpis,tile_kpis,curves,sensitivity}.csv, run_log.json and
outputs/overlays/<batch>__<site>.png. A full run writes fresh tables. `--sites` merges the
recomputed sites into the existing tables (all rows of each recomputed site are replaced, and
rows stay in manifest order).

run_log.json describes the published tables: `sites` maps every site to the entry in `runs`
(commit, command, parameters, package versions) that produced its rows, so rows retained by a
`--sites` merge keep their original provenance. If any site fails, or the merge fails, no tables
and no run_log.json are written; the attempt is logged to run_attempt.json instead and the exit
status is non-zero (D-018: no fill values). A successful run removes a stale run_attempt.json.

    python scripts/run_kpis.py --jobs 8
    python scripts/run_kpis.py --sites Batch_1/4ih2ggld Batch_3/kbdh4tri
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import segment as segment_mod  # noqa: E402
from pmdb.io import load_site  # noqa: E402
from pmdb.kpis import (N_TILES, catalogue_columns, check_registry, common, compute_site_kpis,  # noqa: E402
                       compute_tile_kpis, crossphase, diagnostic, fields, objects, pointpattern)
from pmdb.overlay import render_overlay  # noqa: E402

MANIFEST = ROOT / "cache" / "half" / "manifest.csv"
PACKAGES = ("numpy", "scipy", "scikit-image", "pandas", "Pillow", "matplotlib", "torch")


def all_sites() -> list[tuple[str, str]]:
    man = pd.read_csv(MANIFEST)
    return sorted(zip(man["batch"].astype(str), man["site"].astype(str)))


def _json_finite(v) -> bool:
    """False if v is, or contains, a non-finite float (NaN/inf); such values are not valid JSON."""
    if isinstance(v, float):  # includes np.float64
        return math.isfinite(v)
    if isinstance(v, np.ndarray):
        return not np.issubdtype(v.dtype, np.floating) or bool(np.isfinite(v).all())
    if isinstance(v, (tuple, list)):
        return all(_json_finite(x) for x in v)
    return True


def kpi_parameters() -> dict[str, object]:
    params: dict[str, object] = {"segmenter": dict(segment_mod.V0_PARAMS), "N_TILES": N_TILES}
    shared = vars(common)
    for mod in (common, objects, pointpattern, fields, crossphase, diagnostic):
        for k, v in vars(mod).items():
            if not (k.isupper() and isinstance(v, (int, float, str, tuple, list, np.ndarray))):
                continue
            if mod is not common and shared.get(k) is v:
                continue  # imported from common (e.g. NAN); dropped by the non-finite filter below
            if not _json_finite(v):
                continue  # sentinels such as NAN are not parameters and are not valid JSON
            params[f"{mod.__name__.split('.')[-1]}.{k}"] = v.tolist() if isinstance(v, np.ndarray) else v
    return params


def dump_run_log(run_log: dict) -> str:
    """Strict JSON (no NaN/Infinity tokens); raises ValueError on a non-finite float."""
    return json.dumps(run_log, indent=2, default=str, allow_nan=False)


TABLES = ("site_kpis.csv", "tile_kpis.csv", "curves.csv", "sensitivity.csv")
RUN_LOG = "run_log.json"
ATTEMPT_LOG = "run_attempt.json"
LOG_SCHEMA = 2
UNKNOWN_RUN = "unknown"
EXTRA_SORT = {"tile_kpis.csv": ("tile",)}


def build_tables(results: dict, ok: list[str]) -> dict[str, pd.DataFrame]:
    """The four deliverable frames for the successful sites ``ok`` (keys 'batch/site', in output order)."""
    site_cols, tile_cols = catalogue_columns()
    meta = ["batch", "site", "se_detector", "segmenter_version"]
    site_order = meta + [c for cols in site_cols.values() for c in cols]
    tile_order = meta + ["tile"] + [c for cols in tile_cols.values() for c in cols] + ["nan_reason"]
    return {
        "site_kpis.csv": pd.DataFrame([results[k]["site_row"] for k in ok], columns=site_order),
        "tile_kpis.csv": pd.DataFrame([r for k in ok for r in results[k]["tiles"]], columns=tile_order),
        "curves.csv": pd.DataFrame([r for k in ok for r in results[k]["curves"]],
                                   columns=["batch", "site", "kpi_id", "curve", "x", "value"]),
        "sensitivity.csv": pd.DataFrame([r for k in ok for r in results[k]["sweep"]]),
    }


def merge_table(existing: pd.DataFrame | None, new: pd.DataFrame, order: list[tuple[str, str]],
                extra_sort: tuple[str, ...] = ()) -> pd.DataFrame:
    """Replace every row of each (batch, site) present in ``new``; keep all other existing rows.

    Rows are stable-sorted by the site's rank in ``order`` (unknown sites last, existing order kept),
    then by ``extra_sort``. Raises ValueError if the column sets differ.
    """
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    if new.empty:
        return existing.reset_index(drop=True)
    if set(existing.columns) != set(new.columns):
        raise ValueError(f"column mismatch: existing-only {sorted(set(existing.columns) - set(new.columns))}, "
                         f"new-only {sorted(set(new.columns) - set(existing.columns))}")
    replaced = set(zip(new["batch"].astype(str), new["site"].astype(str)))
    old_keys = list(zip(existing["batch"].astype(str), existing["site"].astype(str)))
    keep = existing.loc[[k not in replaced for k in old_keys], list(new.columns)]
    out = pd.concat([keep, new], ignore_index=True)
    rank = {k: i for i, k in enumerate(order)}
    out["_rank"] = [rank.get(k, len(order)) for k in zip(out["batch"].astype(str), out["site"].astype(str))]
    out = out.sort_values(["_rank", *extra_sort], kind="stable").drop(columns="_rank")
    return out.reset_index(drop=True)


def run_id(record: dict) -> str:
    return f"{record.get('started_utc', 'unknown-time')}_{str(record.get('git_commit', 'unknown'))[:12]}"


def read_provenance(path: Path) -> tuple[dict[str, dict], dict[str, dict], str | None]:
    """(runs, sites, problem) from an existing run log; legacy single-run logs are converted.

    ``problem`` explains why no provenance could be read (missing, invalid or failed-run log).
    """
    if not path.exists():
        return {}, {}, f"{path.name} not found"
    try:
        log = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        return {}, {}, f"{path.name} unreadable: {e}"
    if not isinstance(log, dict) or not isinstance(log.get("sites"), dict):
        return {}, {}, f"{path.name} has no per-site entries"
    if "runs" in log:
        if not isinstance(log["runs"], dict):
            return {}, {}, f"{path.name} has malformed runs"
        return dict(log["runs"]), dict(log["sites"]), None
    if log.get("tables_written") is False:
        return {}, {}, f"{path.name} records a failed run that wrote no tables"
    record = {k: v for k, v in log.items() if k != "sites"}
    rid = run_id(record)
    sites = {k: {"run": rid, **v} for k, v in log["sites"].items() if v.get("status") == "ok"}
    return {rid: record}, sites, None


def merge_provenance(runs: dict[str, dict], sites: dict[str, dict], new_run_id: str, new_run: dict,
                     new_sites: dict[str, dict], table_keys: list[str],
                     problem: str | None = None) -> tuple[dict[str, dict], dict[str, dict]]:
    """Provenance for the merged tables: recomputed sites point at the new run, retained sites keep
    their old entry, and retained sites without one point at an explicit 'unknown' run."""
    merged: dict[str, dict] = {}
    new_unknown = False
    for key in table_keys:
        if key in new_sites:
            merged[key] = {"run": new_run_id, **new_sites[key]}
        elif key in sites and sites[key].get("run") in runs:
            merged[key] = sites[key]
        else:
            merged[key] = {"run": UNKNOWN_RUN, "status": "retained"}
            new_unknown = True
    used = {v["run"] for v in merged.values()}
    out_runs = {rid: rec for rid, rec in runs.items() if rid in used and rid != new_run_id}
    out_runs[new_run_id] = new_run
    if new_unknown:  # keep the reason recorded for sites that were already unknown
        reason = problem or "site missing from previous run log"
        old = out_runs.get(UNKNOWN_RUN, {}).get("reason")
        out_runs[UNKNOWN_RUN] = {"note": "rows retained from tables without provenance",
                                 "reason": reason if old in (None, reason) else f"{old}; {reason}"}
    return out_runs, merged


def _read_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"batch": str, "site": str}, float_precision="round_trip")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def process_site(batch: str, site: str, overlay_dir: str | None) -> dict:
    t0 = time.time()
    raw = load_site(batch, site, resolution="half", normalise="none")
    norm = load_site(batch, site, resolution="half")
    masks = segment_mod.segment(raw)
    bse_norm = norm.image[..., 0]
    key = f"{batch}/{site}"
    if overlay_dir:
        render_overlay(bse_norm, masks, raw.nm_per_px, key, Path(overlay_dir) / f"{batch}__{site}.png")
    meta = {"batch": batch, "site": site, "se_detector": raw.se_detector, "segmenter_version": masks.version}
    values, curves, ctx = compute_site_kpis(masks, bse_norm, raw.nm_per_px, key)
    tiles = [{**meta, **row} for row in compute_tile_kpis(masks, bse_norm, raw.nm_per_px, key)]
    sweep = [{"batch": batch, "site": site, **row} for row in objects.k04_sweep(ctx)]
    curve_rows = [{"batch": batch, "site": site, "kpi_id": k, "curve": c, "x": x, "value": v}
                  for k, c, x, v in curves]
    seg = {k: v for k, v in masks.params.items() if k in ("T_pore", "T_si", "p1", "p50", "p99")}
    return {"site_row": {**meta, **values}, "tiles": tiles, "curves": curve_rows, "sweep": sweep,
            "seconds": time.time() - t0, "segmentation": seg, "n_si_objects": ctx.n_objects}


def _worker(args):
    batch, site, overlay_dir = args
    try:
        return batch, site, process_site(batch, site, overlay_dir), None
    except Exception as e:  # reported, never filled (D-018)
        return batch, site, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*", help="batch/site identifiers (default: all sites in the cache manifest)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "kpis"))
    ap.add_argument("--overlay-dir", default=str(ROOT / "outputs" / "overlays"))
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args(argv)

    check_registry()
    order = all_sites()
    subset = bool(args.sites)
    if subset:
        known, sites, bad = set(order), [], []
        for s in dict.fromkeys(args.sites):  # de-duplicate, keep first occurrence
            pair = tuple(s.split("/", 1))
            if len(pair) == 2 and pair in known:
                sites.append(pair)
            else:
                bad.append(s)
        if bad:
            print(f"unknown site(s), expected batch/site from {MANIFEST}: {', '.join(bad)}", file=sys.stderr)
            return 2
    else:
        sites = order
    overlay_dir = None if args.no_overlays else args.overlay_dir
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    results, log = {}, {}
    jobs = [(b, s, overlay_dir) for b, s in sites]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(_worker, j) for j in jobs]
            done_iter = (f.result() for f in as_completed(futures))
            for i, (b, s, res, err) in enumerate(done_iter, 1):
                _record(i, len(jobs), b, s, res, err, results, log)
    else:
        for i, j in enumerate(jobs, 1):
            _record(i, len(jobs), *_worker(j), results, log)
    total = time.time() - t_start

    ok = [f"{b}/{s}" for b, s in order if f"{b}/{s}" in results]
    failed = [k for k, v in log.items() if v["status"] != "ok"]
    frames: dict[str, pd.DataFrame] = {}
    table_error = None
    if not failed:
        frames = build_tables(results, ok)
        if subset:
            try:
                frames = {name: merge_table(_read_existing(out / name), df, order, EXTRA_SORT.get(name, ()))
                          for name, df in frames.items()}
            except ValueError as e:
                table_error = f"cannot merge into existing tables in {out}: {e}"
                frames = {}

    record = {
        "git_commit": git_commit(),
        "command": " ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "total_seconds": round(total, 1),
        "jobs": args.jobs,
        "mode": "subset" if subset else "full",
        "n_sites": len(sites),
        "n_ok": len(ok),
        "failed": failed,
        "segmenter_version": segment_mod.SEGMENTER_VERSION,
        "parameters": kpi_parameters(),
        "packages": {p: _version(p) for p in PACKAGES} | {"python": platform.python_version()},
    }
    print(f"done: {len(ok)}/{len(sites)} sites ok in {total / 60:.1f} min -> {out}")
    if failed or table_error:
        attempt = {**record, "tables_written": False, "sites": {k: log[k] for k in sorted(log)}}
        if table_error:
            attempt["table_error"] = table_error
        _atomic_write_text(out / ATTEMPT_LOG, dump_run_log(attempt))
        if failed:
            print("FAILED: no tables written (existing files untouched); failed sites:", ", ".join(failed))
        else:
            print("ERROR:", table_error, "- no tables written", file=sys.stderr)
        print(f"attempt logged to {out / ATTEMPT_LOG}; {RUN_LOG} untouched")
        return 1

    rid = run_id(record)
    site_df = frames["site_kpis.csv"]
    table_keys = [f"{b}/{s}" for b, s in zip(site_df["batch"].astype(str), site_df["site"].astype(str))]
    old_runs, old_sites, problem = read_provenance(out / RUN_LOG) if subset else ({}, {}, None)
    runs, prov = merge_provenance(old_runs, old_sites, rid, record, log, table_keys, problem)
    text = dump_run_log({"schema": LOG_SCHEMA, "latest_run": rid, "runs": runs, "sites": prov})
    for name in TABLES:
        _atomic_write_csv(frames[name], out / name)
    _atomic_write_text(out / RUN_LOG, text)
    (out / ATTEMPT_LOG).unlink(missing_ok=True)
    if any(v["run"] == UNKNOWN_RUN for v in prov.values()):
        print(f"WARNING: retained rows without provenance ({problem or 'missing from previous log'})",
              file=sys.stderr)
    if subset:
        print(f"merged {len(ok)} site(s) into existing tables in {out}")
    return 0


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def _record(i, n, batch, site, res, err, results, log):
    key = f"{batch}/{site}"
    if err is None:
        results[key] = res
        log[key] = {"status": "ok", "seconds": round(res["seconds"], 1), "n_si_objects": res["n_si_objects"],
                    "segmentation": res["segmentation"]}
        r = res["site_row"]
        print(f"[{i:2d}/{n}] {key:22s} ok  {res['seconds']:6.1f}s  K01={r['K01_si_frac_adm']:.3f}  "
              f"n_si={res['n_si_objects']}", flush=True)
    else:
        log[key] = {"status": "error", "error": err}
        print(f"[{i:2d}/{n}] {key:22s} ERROR {err.splitlines()[0]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
