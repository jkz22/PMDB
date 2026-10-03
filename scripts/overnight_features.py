"""Overnight run 1: richer spatial features for all 34 sites (31 labelled + 3 held-out).

Recomputes, from the existing v0 segmentation (pmdb.segment.segment, unchanged):
  - 15-band depth profiles for Si, graphite and porosity (band15_<phase>_frac curves),
  - finer lateral profiles at 2 and 5 um bins (lat<w>um_<phase>_frac curves),
  - autocorrelation lengths of the 1-px depth and lateral fraction profiles per phase,
  - per-tile KPIs (the full catalogue tile set) at 16 and 32 tiles per site, for
    within-site spread of every KPI family.

Pure data production for tomorrow's single evaluation decision; nothing here touches
the merged fingerprint model or its inputs in outputs/kpis/.

    .venv/Scripts/python scripts/overnight_features.py --jobs 4

Writes to outputs/overnight/features/:
    curves_rich.csv     batch, site, curve, x, value   (depth bands + lateral profiles)
    tile_kpis_rich.csv  batch, site, n_tiles, tile, <catalogue tile columns>, nan_reason
    site_scalars.csv    batch, site, acl_depth/lateral per phase + image geometry
    run_log.json        provenance, per-site status and timing

Tables are rewritten atomically after every completed site and merged by
(batch, site) into whatever already exists in --out-dir, so a crash loses at
most the site in flight, re-running resumes without discarding earlier results,
and a --sites subset run updates only the selected sites.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import segment as segment_mod  # noqa: E402
from pmdb.io import load_site  # noqa: E402
from pmdb.kpis import REGISTRY, catalogue_columns, tile_slices  # noqa: E402
from pmdb.kpis.common import KpiContext, TooFewObjects  # noqa: E402
from pmdb.kpis.fields import band_profile  # noqa: E402

N_DEPTH_BANDS = 15
LATERAL_BINS_UM = (2.0, 5.0)
TILE_COUNTS = (16, 32)
PHASES = ("si", "graphite", "pore")

HELDOUT_BATCH = "Batch_heldout"
HELDOUT_DATA = ROOT / "data_heldout"
HELDOUT_CACHE = ROOT / "cache_heldout"


def all_sites() -> list[tuple[str, str]]:
    """31 labelled sites + 3 held-out sites, in manifest order."""
    sites = []
    for manifest in (ROOT / "cache" / "half" / "manifest.csv",
                     HELDOUT_CACHE / "half" / "manifest.csv"):
        man = pd.read_csv(manifest)
        sites.extend(sorted(zip(man["batch"].astype(str), man["site"].astype(str))))
    return sites


def load_raw_and_norm(batch: str, site: str):
    kw = {}
    if batch == HELDOUT_BATCH:
        kw = {"data_root": HELDOUT_DATA, "cache_root": HELDOUT_CACHE}
    raw = load_site(batch, site, resolution="half", normalise="none", **kw)
    norm = load_site(batch, site, resolution="half", **kw)
    return raw, norm


def phase_arrays(masks) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(numerator, denominator) per phase, matching the v1 KPI conventions
    (K12 for Si, D01 for graphite, D04 for porosity)."""
    return {
        "si": (masks.si, masks.fraction_space),
        "graphite": (masks.graphite, np.ones_like(masks.graphite)),
        "pore": (masks.pore, ~masks.artefact),
    }


def depth_band_curves(masks) -> list[tuple[str, float, float]]:
    rows = []
    for phase, (num, den) in phase_arrays(masks).items():
        vals = band_profile(num, den, n_bands=N_DEPTH_BANDS)
        rows.extend((f"band{N_DEPTH_BANDS}_{phase}_frac", float(i), float(v))
                    for i, v in enumerate(vals))
    return rows


def lateral_curves(masks, px_um: float) -> list[tuple[str, float, float]]:
    rows = []
    H, W = masks.shape
    for w_um in LATERAL_BINS_UM:
        wp = max(1, int(round(w_um / px_um)))
        nb = W // wp
        for phase, (num, den) in phase_arrays(masks).items():
            s = num[:, : nb * wp].reshape(H, nb, wp).sum(axis=(0, 2), dtype=np.int64)
            a = den[:, : nb * wp].reshape(H, nb, wp).sum(axis=(0, 2), dtype=np.int64)
            ok = a > 0
            frac = np.full(nb, np.nan)
            frac[ok] = s[ok] / a[ok]
            centres = (np.arange(nb) + 0.5) * wp * px_um
            rows.extend((f"lat{w_um:g}um_{phase}_frac", float(x), float(v))
                        for x, v in zip(centres, frac))
    return rows


def autocorr_length_um(profile: np.ndarray, px_um: float) -> float:
    """First lag (um) where the normalised autocorrelation of the mean-subtracted
    profile drops below 1/e; NaN if it never does or the profile is degenerate."""
    v = np.asarray(profile, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 8:
        return float("nan")
    v = v - v.mean()
    var = float(np.dot(v, v))
    if var <= 0:
        return float("nan")
    acf = np.correlate(v, v, mode="full")[v.size - 1:] / var
    below = np.flatnonzero(acf < 1.0 / np.e)
    if below.size == 0:
        return float("nan")
    return float(below[0] * px_um)


def site_scalars(masks, px_um: float) -> dict[str, float]:
    out: dict[str, float] = {
        "height_um": masks.shape[0] * px_um,
        "width_um": masks.shape[1] * px_um,
    }
    for phase, (num, den) in phase_arrays(masks).items():
        num_f = num.astype(np.float64)
        den_f = den.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = num_f.sum(axis=1) / den_f.sum(axis=1)
            lateral = num_f.sum(axis=0) / den_f.sum(axis=0)
        out[f"acl_depth_{phase}_um"] = autocorr_length_um(depth, px_um)
        out[f"acl_lateral_{phase}_um"] = autocorr_length_um(lateral, px_um)
    return out


def tile_rows(masks, bse_norm, nm_per_px: float, seed_prefix: str) -> list[dict]:
    """compute_tile_kpis logic, parameterised over the tile count."""
    _, tile_cols = catalogue_columns()
    rows = []
    for n_tiles in TILE_COUNTS:
        for t, cols_slice in enumerate(tile_slices(masks.shape[1], n_tiles)):
            tm = masks.crop(slice(None), cols_slice)
            ctx = KpiContext(masks=tm, nm_per_px=nm_per_px,
                             seed_key=f"{seed_prefix}/tiles{n_tiles}/tile{t}",
                             bse=None if bse_norm is None else bse_norm[:, cols_slice])
            row: dict[str, object] = {"n_tiles": n_tiles, "tile": t}
            reasons = []
            for kpi_id, cols in tile_cols.items():
                try:
                    out = REGISTRY[kpi_id].tile_fn(ctx).values
                    for c in cols:
                        row[c] = float(out[c])
                        if not np.isfinite(row[c]):
                            reasons.append(f"{c}: not computable")
                except TooFewObjects as e:
                    for c in cols:
                        row[c] = float("nan")
                    reasons.append(f"{kpi_id}: {e}")
            row["nan_reason"] = "; ".join(dict.fromkeys(reasons))
            rows.append(row)
    return rows


def process_site(batch: str, site: str) -> dict:
    t0 = time.time()
    raw, norm = load_raw_and_norm(batch, site)
    masks = segment_mod.segment(raw)
    px_um = raw.nm_per_px / 1000.0
    meta = {"batch": batch, "site": site}
    curves = [{**meta, "curve": c, "x": x, "value": v}
              for c, x, v in depth_band_curves(masks) + lateral_curves(masks, px_um)]
    scalars = {**meta, **site_scalars(masks, px_um)}
    tiles = [{**meta, **row}
             for row in tile_rows(masks, norm.image[..., 0], raw.nm_per_px, f"{batch}/{site}")]
    return {"curves": curves, "scalars": scalars, "tiles": tiles,
            "seconds": time.time() - t0}


def _worker(args):
    batch, site = args
    try:
        return batch, site, process_site(batch, site), None
    except Exception as e:
        return batch, site, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _read_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"batch": str, "site": str})


def merge_rows(existing: pd.DataFrame | None, new: pd.DataFrame,
               recomputed: set[tuple[str, str]]) -> pd.DataFrame:
    """Replace every existing row of a recomputed (batch, site); keep all other rows.

    Row order within each site is preserved; sites are ordered by the full
    manifest order (unknown sites last). Raises ValueError on a column mismatch
    so a stale table from an incompatible run is never silently merged.
    """
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    if set(existing.columns) != set(new.columns):
        raise ValueError(
            f"column mismatch with existing table: existing-only "
            f"{sorted(set(existing.columns) - set(new.columns))}, "
            f"new-only {sorted(set(new.columns) - set(existing.columns))}")
    old_keys = list(zip(existing["batch"].astype(str), existing["site"].astype(str)))
    keep = existing.loc[[k not in recomputed for k in old_keys], list(new.columns)]
    out = pd.concat([keep, new], ignore_index=True)
    rank = {k: i for i, k in enumerate(all_sites())}
    out["_rank"] = [rank.get(k, len(rank))
                    for k in zip(out["batch"].astype(str), out["site"].astype(str))]
    out = out.sort_values("_rank", kind="stable").drop(columns="_rank")
    return out.reset_index(drop=True)


def write_tables(results: dict, order: list[tuple[str, str]], out: Path,
                 existing: dict[str, pd.DataFrame | None]) -> None:
    done = [(b, s) for b, s in order if f"{b}/{s}" in results]
    recomputed = set(done)
    new = {
        "curves_rich.csv": pd.DataFrame(
            [r for b, s in done for r in results[f"{b}/{s}"]["curves"]]),
        "site_scalars.csv": pd.DataFrame(
            [results[f"{b}/{s}"]["scalars"] for b, s in done]),
        "tile_kpis_rich.csv": pd.DataFrame(
            [r for b, s in done for r in results[f"{b}/{s}"]["tiles"]]),
    }
    for name, df in new.items():
        _atomic_write_csv(merge_rows(existing[name], df, recomputed), out / name)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sites", nargs="*", help="batch/site subset (default: all 34)")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "overnight" / "features"))
    args = ap.parse_args(argv)

    order = all_sites()
    if args.sites:
        wanted = {tuple(s.split("/", 1)) for s in args.sites}
        unknown = wanted - set(order)
        if unknown:
            print(f"unknown site(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        order = [p for p in order if p in wanted]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = {name: _read_existing(out / name)
                for name in ("curves_rich.csv", "site_scalars.csv", "tile_kpis_rich.csv")}

    t_start = time.time()
    results: dict[str, dict] = {}
    log: dict[str, dict] = {}

    def record(i, batch, site, res, err):
        key = f"{batch}/{site}"
        if err is None:
            results[key] = res
            log[key] = {"status": "ok", "seconds": round(res["seconds"], 1)}
            write_tables(results, order, out, existing)
            print(f"[{i:2d}/{len(order)}] {key:28s} ok  {res['seconds']:6.1f}s", flush=True)
        else:
            log[key] = {"status": "error", "error": err}
            print(f"[{i:2d}/{len(order)}] {key:28s} ERROR {err.splitlines()[0]}", flush=True)

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(_worker, p) for p in order]
            for i, f in enumerate(as_completed(futures), 1):
                record(i, *f.result())
    else:
        for i, p in enumerate(order, 1):
            record(i, *_worker(p))

    run_log = {
        "git_commit": git_commit(),
        "command": " ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "total_seconds": round(time.time() - t_start, 1),
        "n_depth_bands": N_DEPTH_BANDS,
        "lateral_bins_um": list(LATERAL_BINS_UM),
        "tile_counts": list(TILE_COUNTS),
        "segmenter_version": segment_mod.SEGMENTER_VERSION,
        "sites": log,
    }
    (out / "run_log.json").write_text(json.dumps(run_log, indent=2))
    failed = [k for k, v in log.items() if v["status"] != "ok"]
    print(f"done: {len(results)}/{len(order)} sites ok in {(time.time() - t_start) / 60:.1f} min -> {out}")
    if failed:
        print("FAILED sites:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
