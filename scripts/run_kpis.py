"""Run segmentation + all v1 KPIs on every site (spec 002).

Writes outputs/kpis/{site_kpis,tile_kpis,curves,sensitivity}.csv, run_log.json and
outputs/overlays/<batch>__<site>.png. Exits non-zero if any site fails (D-018: no fill values).

    python scripts/run_kpis.py --jobs 8
    python scripts/run_kpis.py --sites Batch_1/4ih2ggld Batch_3/kbdh4tri
"""

from __future__ import annotations

import argparse
import json
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


def kpi_parameters() -> dict[str, object]:
    params: dict[str, object] = {"segmenter": dict(segment_mod.V0_PARAMS), "N_TILES": N_TILES}
    for mod in (common, objects, pointpattern, fields, crossphase, diagnostic):
        for k, v in vars(mod).items():
            if k.isupper() and isinstance(v, (int, float, str, tuple, list, np.ndarray)):
                params[f"{mod.__name__.split('.')[-1]}.{k}"] = v.tolist() if isinstance(v, np.ndarray) else v
    return params


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
    seg = {k: v for k, v in masks.params.items() if k in ("T_pore", "T_si", "p1", "p50", "median_solid", "mad_solid")}
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
    sites = [tuple(s.split("/", 1)) for s in args.sites] if args.sites else all_sites()
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

    ok = [k for k in sorted(results)]
    site_cols, tile_cols = catalogue_columns()
    meta = ["batch", "site", "se_detector", "segmenter_version"]
    site_order = meta + [c for cols in site_cols.values() for c in cols]
    tile_order = meta + ["tile"] + [c for cols in tile_cols.values() for c in cols] + ["nan_reason"]
    pd.DataFrame([results[k]["site_row"] for k in ok], columns=site_order).to_csv(out / "site_kpis.csv", index=False)
    pd.DataFrame([r for k in ok for r in results[k]["tiles"]], columns=tile_order).to_csv(out / "tile_kpis.csv", index=False)
    pd.DataFrame([r for k in ok for r in results[k]["curves"]],
                 columns=["batch", "site", "kpi_id", "curve", "x", "value"]).to_csv(out / "curves.csv", index=False)
    pd.DataFrame([r for k in ok for r in results[k]["sweep"]]).to_csv(out / "sensitivity.csv", index=False)

    failed = [k for k, v in log.items() if v["status"] != "ok"]
    run_log = {
        "git_commit": git_commit(),
        "command": " ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "total_seconds": round(total, 1),
        "jobs": args.jobs,
        "n_sites": len(sites),
        "n_ok": len(ok),
        "failed": failed,
        "segmenter_version": segment_mod.SEGMENTER_VERSION,
        "parameters": kpi_parameters(),
        "packages": {p: _version(p) for p in PACKAGES} | {"python": platform.python_version()},
        "sites": {k: log[k] for k in sorted(log)},
    }
    (out / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))
    print(f"done: {len(ok)}/{len(sites)} sites ok in {total / 60:.1f} min -> {out}")
    if failed:
        print("FAILED (no values written for these sites):", ", ".join(failed))
        return 1
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
