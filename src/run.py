"""Run the pipeline over all batches and write results/kpis.parquet
(columns: image, batch, kpi, value, tier) plus per-image diagnostics in
results/diagnostics.parquet. Everything downstream reads only these.

KPI modules plug in via the shared interface
compute(masks, instances, instances_border) -> dict[str, float];
modules not present yet (colleague's geometry, physics on Modal) are
skipped with a notice.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.instances import graphite_instances
from src.io import ROOT, list_batches, load, load_config, normalise
from src.segment import segment

TIERS = {"mat": "materials", "phys": "physics", "geo": "geometry"}
MODULES = ["src.kpis_materials", "src.kpis_physics", "src.kpis_geometry"]


def kpi_modules() -> list:
    mods = []
    for name in MODULES:
        try:
            mods.append(importlib.import_module(name))
        except ImportError:
            print(f"note: {name} not available, skipping")
    return mods


def main() -> None:
    cfg = load_config()
    np.random.seed(cfg["seed"])
    ncfg = cfg["normalise"]
    mods = kpi_modules()

    rows, diags, failed = [], [], []
    for batch, paths in list_batches(cfg).items():
        for path in paths:
            norm, lo, hi = normalise(load(path), ncfg["p_low"], ncfg["p_high"])
            masks = segment(norm, cfg)
            instances, border = graphite_instances(masks, cfg)

            try:
                for mod in mods:
                    for kpi, value in mod.compute(masks, instances, border).items():
                        tier = TIERS.get(kpi.split("_")[0])
                        if tier is None:
                            print(f"warning: KPI {kpi!r} has no mat_/phys_/geo_ "
                                  "prefix, tier set to 'other'")
                            tier = "other"
                        rows.append({"image": path.stem, "batch": batch,
                                     "kpi": kpi, "value": value, "tier": tier})
            except ValueError as e:
                # unmeasurable image (e.g. no graphite segmented): flag it
                # and keep going rather than losing the whole run
                failed.append({"image": path.stem, "batch": batch, "error": str(e)})
                print(f"UNMEASURABLE {batch} {path.stem}: {e}")
                continue
            n_inst = len(np.unique(instances)) - 1
            diags.append({"image": path.stem, "batch": batch,
                          "norm_p05": lo, "norm_p995": hi,
                          "n_graphite_instances": n_inst,
                          "frac_border_instances": len(border) / max(n_inst, 1)})
            print(f"{batch} {path.stem}: done")

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
