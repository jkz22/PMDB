"""Run the pipeline over all batches and write results/kpis.parquet
(columns: image, batch, kpi, value, tier) plus per-image diagnostics in
results/diagnostics.parquet. Everything downstream reads only these.

KPI modules plug in via the shared interface
compute(masks, instances, instances_border, pixel_size_um) -> dict[str, float];
modules not present yet (colleague's geometry, physics on Modal) are
skipped with a notice. KPIs with a _um suffix are in micrometres.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.instances import graphite_instances
from src.io import ROOT, list_batches, load, load_config, normalise
from src.segment import segment

TIERS = {"mat": "materials", "phys": "physics", "geo": "geometry"}
MODULES = ["src.kpis_materials", "src.kpis_physics", "src.kpis_geometry"]


def kpi_modules(quiet: bool = False) -> list:
    mods = []
    for name in MODULES:
        try:
            mods.append(importlib.import_module(name))
        except ImportError:
            if not quiet:
                print(f"note: {name} not available, skipping")
    return mods


def process_image(path: Path, batch: str, cfg: dict) -> dict:
    """Segment one image and compute its KPIs. Runs in a worker process;
    returns small dicts only (no arrays cross the process boundary)."""
    np.random.seed(cfg["seed"])
    ncfg = cfg["normalise"]
    norm, lo, hi = normalise(load(path), ncfg["p_low"], ncfg["p_high"])
    masks = segment(norm, cfg)
    instances, border = graphite_instances(masks, cfg)

    rows = []
    try:
        for mod in kpi_modules(quiet=True):
            for kpi, value in mod.compute(masks, instances, border,
                                          cfg["pixel_size_um"]).items():
                tier = TIERS.get(kpi.split("_")[0])
                if tier is None:
                    print(f"warning: KPI {kpi!r} has no mat_/phys_/geo_ "
                          "prefix, tier set to 'other'")
                    tier = "other"
                rows.append({"image": path.stem, "batch": batch,
                             "kpi": kpi, "value": value, "tier": tier})
    except ValueError as e:
        # unmeasurable image (e.g. no graphite segmented): flag it and keep
        # going rather than losing the whole run
        return {"failed": {"image": path.stem, "batch": batch, "error": str(e)}}
    n_inst = len(np.unique(instances)) - 1
    diag = {"image": path.stem, "batch": batch,
            "norm_p05": lo, "norm_p995": hi,
            "n_graphite_instances": n_inst,
            "frac_border_instances": len(border) / max(n_inst, 1)}
    return {"rows": rows, "diag": diag}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int,
                        default=min(8, max(1, (os.cpu_count() or 2) - 1)),
                        help="worker processes (default: cpu_count-1, max 8)")
    args = parser.parse_args()

    cfg = load_config()
    kpi_modules()  # print the skipped-module notice once, up front

    tasks = [(path, batch)
             for batch, paths in list_batches(cfg).items()
             for path in paths]
    rows, diags, failed = [], [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(process_image, path, batch, cfg): (path, batch)
                   for path, batch in tasks}
        for n, fut in enumerate(as_completed(futures), 1):
            path, batch = futures[fut]
            res = fut.result()
            if "failed" in res:
                failed.append(res["failed"])
                print(f"[{n}/{len(tasks)}] UNMEASURABLE {batch} {path.stem}: "
                      f"{res['failed']['error']}")
                continue
            rows.extend(res["rows"])
            diags.append(res["diag"])
            print(f"[{n}/{len(tasks)}] {batch} {path.stem}: done")

    # deterministic output order regardless of completion order
    rows.sort(key=lambda r: (r["batch"], r["image"], r["kpi"]))
    diags.sort(key=lambda d: (d["batch"], d["image"]))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "kpis.parquet", index=False)
    pd.DataFrame(diags).to_parquet(out / "diagnostics.parquet", index=False)
    print(f"\nwrote {len(rows)} KPI rows for "
          f"{len(diags)} images -> results/kpis.parquet")
    if failed:
        print(f"UNMEASURABLE images ({len(failed)}):")
        for f in failed:
            print(f"  {f['batch']} {f['image']}: {f['error']}")


if __name__ == "__main__":
    main()
