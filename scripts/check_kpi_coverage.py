"""Check that every v1 KPI column is logged for every site (spec 002, Acceptance 3).

Exits non-zero on any gap: a missing column, a missing/duplicated/unexpected site or (site, tile) row,
a site-level NaN, or a tile-level NaN without a ``nan_reason``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb.kpis import N_TILES, catalogue_columns  # noqa: E402


def key_gaps(df: pd.DataFrame, key_cols: list[str], expected: set[tuple[str, ...]], table: str) -> list[str]:
    """Duplicated, missing and unexpected row keys of ``table``; key parts compared as strings."""
    absent = [c for c in key_cols if c not in df]
    if absent:
        return [f"{table}: key column(s) missing: {', '.join(absent)}"]
    counts = Counter(zip(*(df[c].astype(str) for c in key_cols)))
    gaps = [f"duplicated in {table}: {'/'.join(k)} ({n} rows)" for k, n in sorted(counts.items()) if n > 1]
    gaps += [f"missing from {table}: {'/'.join(k)}" for k in sorted(expected - counts.keys())]
    gaps += [f"unexpected in {table}: {'/'.join(k)}" for k in sorted(counts.keys() - expected)]
    return gaps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kpi-dir", default=str(ROOT / "outputs" / "kpis"))
    ap.add_argument("--manifest", default=str(ROOT / "cache" / "half" / "manifest.csv"))
    args = ap.parse_args(argv)

    man = pd.read_csv(args.manifest)
    expected = set(zip(man["batch"].astype(str), man["site"].astype(str)))
    expected_tiles = {(b, s, str(t)) for b, s in expected for t in range(N_TILES)}
    n_sites = len(expected)
    site_cols, tile_cols = catalogue_columns()
    kdir = Path(args.kpi_dir)
    gaps: list[str] = []

    rd = {"dtype": {"batch": str, "site": str}}
    site_df = pd.read_csv(kdir / "site_kpis.csv", **rd) if (kdir / "site_kpis.csv").exists() else pd.DataFrame()
    tile_df = pd.read_csv(kdir / "tile_kpis.csv", **rd) if (kdir / "tile_kpis.csv").exists() else pd.DataFrame()
    if site_df.empty:
        gaps.append("site_kpis.csv missing or empty")
    else:
        gaps += key_gaps(site_df, ["batch", "site"], expected, "site_kpis.csv")
    if tile_df.empty:
        gaps.append("tile_kpis.csv missing or empty")
    else:
        gaps += key_gaps(tile_df, ["batch", "site", "tile"], expected_tiles, "tile_kpis.csv")

    reason = tile_df["nan_reason"].fillna("").astype(str) if "nan_reason" in tile_df else None
    rows = []
    for kpi_id, cols in site_cols.items():
        n_ok = n_sites
        for c in cols:
            if c not in site_df:
                gaps.append(f"{kpi_id}: column {c} missing from site_kpis.csv")
                n_ok = 0
                continue
            vals = pd.to_numeric(site_df[c], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(vals)
            n_ok = min(n_ok, int(finite.sum()))
            for i in np.flatnonzero(~finite):
                gaps.append(f"{kpi_id}: {c} is NaN at {site_df['batch'].iloc[i]}/{site_df['site'].iloc[i]}")
        tile_status = "-"
        if kpi_id in tile_cols:
            bad = 0
            n_nan = 0
            for c in tile_cols[kpi_id]:
                if c not in tile_df:
                    gaps.append(f"{kpi_id}: column {c} missing from tile_kpis.csv")
                    bad += 1
                    continue
                nan = ~np.isfinite(pd.to_numeric(tile_df[c], errors="coerce").to_numpy(dtype=float))
                n_nan += int(nan.sum())
                unexplained = nan & (reason == "").to_numpy() if reason is not None else nan
                if unexplained.any():
                    gaps.append(f"{kpi_id}: {c} has {int(unexplained.sum())} tile NaN(s) without nan_reason")
                    bad += 1
            tile_status = "missing" if bad else (f"ok ({n_nan} NaN w/ reason)" if n_nan else "ok")
        status = "OK" if n_ok == n_sites and tile_status in ("-",) or (n_ok == n_sites and tile_status.startswith("ok")) else "GAP"
        rows.append((kpi_id, len(cols), f"{n_ok}/{n_sites}", tile_status, status))

    w = max(len(r[3]) for r in rows)
    print(f"| KPI | columns | sites present | tiles{' ' * (w - 5)} | status |")
    print(f"|-----|---------|---------------|{'-' * (w + 2)}|--------|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:7d} | {r[2]:>13s} | {r[3]:{w}s} | {r[4]:6s} |")
    n_cols = sum(r[1] for r in rows)
    n_ok_kpis = sum(r[4] == "OK" for r in rows)
    print(f"\n{n_ok_kpis}/{len(rows)} v1 KPIs ({n_cols} site columns) fully covered for {n_sites} sites")
    if gaps:
        print(f"\n{len(gaps)} gap(s):")
        for g in gaps[:200]:
            print("  -", g)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
