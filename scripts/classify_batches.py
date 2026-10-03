"""Two-stage random-forest batch classifier vs Batch_3 (D18): KPI / FEM / KPI+FEM ablation.

    python scripts/classify_batches.py kpi-tiles [--jobs 8]
    python scripts/classify_batches.py run [--fem-tiles outputs/fem/tile_curves.csv] [--require-fem]
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pmdb.classify.features import kpi_tile_columns  # noqa: E402
from pmdb.classify.kpi_tiles import site_kpi_tiles6  # noqa: E402

DEFAULT_OUT = ROOT / "outputs" / "classifier"
DEFAULT_DOCS = ROOT / "docs" / "classifier" / "results.md"
DEFAULT_FEM = ROOT / "outputs" / "fem" / "tile_curves.csv"
KPI_TILES = DEFAULT_OUT / "kpi_tiles6.csv"


def all_sites() -> list[tuple[str, str, str | None]]:
    lab = pd.read_csv(ROOT / "cache" / "half" / "manifest.csv")
    held = pd.read_csv(ROOT / "cache_heldout" / "half" / "manifest.csv")
    sites = [(str(b), str(s), None) for b, s in zip(lab["batch"], lab["site"])]
    sites += [(str(b), str(s), str(ROOT / "cache_heldout")) for b, s in zip(held["batch"], held["site"])]
    return sorted(sites, key=lambda x: (x[0], x[1]))


def _worker(args):
    batch, site, cache_root = args
    t0 = time.time()
    try:
        return batch, site, site_kpi_tiles6(batch, site, cache_root), None, time.time() - t0
    except Exception as e:  # reported, never filled
        return batch, site, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", time.time() - t0


def cmd_kpi_tiles(args) -> int:
    sites = all_sites()
    rows, failed = [], False
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for batch, site, res, err, sec in ex.map(_worker, sites):
            if err:
                print(f"FAILED {batch}/{site}: {err}", file=sys.stderr)
                failed = True
            else:
                print(f"{batch}/{site}: {sec:.0f}s")
                rows.extend(res)
    if failed:
        return 1
    cols = ["batch", "site", "heldout", "tile", "tile_x0_um", "tile_x1_um", *kpi_tile_columns(), "nan_reason"]
    df = pd.DataFrame(rows)[cols].sort_values(["batch", "site", "tile"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.10g")
    print(f"wrote {out}: {len(df)} rows")
    nan = df[kpi_tile_columns()].isna().sum()
    print("per-column NaN counts:")
    print(nan.to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("kpi-tiles")
    p1.add_argument("--jobs", type=int, default=8)
    p1.add_argument("--out", default=str(KPI_TILES))
    args = ap.parse_args()
    if args.cmd == "kpi-tiles":
        return cmd_kpi_tiles(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
