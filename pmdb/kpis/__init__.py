"""Per-slice and per-tile Si dispersion KPIs (spec 002).

``REGISTRY`` maps every v1 catalogue KPI ID to the function producing its site-level
columns and, where the catalogue says ``per_tile``, the tile-level function.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pmdb.kpis import crossphase, diagnostic, fields, objects, pointpattern, stretch
from pmdb.kpis.common import NAN, KpiContext, KpiOutput, TooFewObjects, stable_seed  # noqa: F401
from pmdb.segment import Masks

CATALOGUE_PATH = Path(__file__).resolve().parents[2] / "docs" / "kpis" / "kpi_catalogue.csv"
N_TILES = 4

KpiFn = Callable[[KpiContext], KpiOutput]


@dataclass(frozen=True)
class KpiSpec:
    kpi_id: str
    site_fn: KpiFn
    tile_fn: KpiFn | None = None


REGISTRY: dict[str, KpiSpec] = {s.kpi_id: s for s in [
    KpiSpec("K01", objects.k01_si_fraction, objects.k01_si_fraction),
    KpiSpec("K02", objects.k02_si_density, objects.k02_si_density),
    KpiSpec("K03", objects.k03_si_size, objects.k03_si_size),
    KpiSpec("K04", objects.k04_agglomerates, objects.k04_agglomerates),
    KpiSpec("K05", pointpattern.k05_voronoi, pointpattern.k05_voronoi_tile),
    KpiSpec("K06", pointpattern.k06_voronoi_regions),
    KpiSpec("K07", pointpattern.k07_nearest_neighbour, pointpattern.k07_nearest_neighbour),
    KpiSpec("K08", pointpattern.k08_pair_correlation),
    KpiSpec("K09", pointpattern.k09_mst, pointpattern.k09_mst),
    KpiSpec("K10", fields.k10_scale_of_segregation),
    KpiSpec("K11", fields.k11_lacey),
    KpiSpec("K12", fields.k12_depth_profile),
    KpiSpec("K13", fields.k13_lateral),
    KpiSpec("K14", fields.k14_empty_space, fields.k14_empty_space),
    KpiSpec("K15", crossphase.k15_contact, crossphase.k15_contact),
    KpiSpec("K16", crossphase.k16_distance, crossphase.k16_distance),
    KpiSpec("D01", diagnostic.d01_graphite_depth),
    KpiSpec("D02", diagnostic.d02_graphite_orientation),
    KpiSpec("D03", diagnostic.d03_graphite_size),
    KpiSpec("D04", diagnostic.d04_porosity_depth),
    KpiSpec("D05", diagnostic.d05_pore_size),
    KpiSpec("D06", diagnostic.d06_pore_connectivity),
    KpiSpec("A01", diagnostic.a01_large_voids),
    KpiSpec("A02", diagnostic.a02_curtaining),
    KpiSpec("A03", diagnostic.a03_height),
]}

# Catalogue v2 stretch KPIs (site-level only). Kept out of REGISTRY so the locked v1 tables are
# unchanged; produced by scripts/run_stretch.py into outputs/stretch/.
STRETCH_REGISTRY: dict[str, KpiSpec] = {s.kpi_id: s for s in [
    KpiSpec("S01", stretch.s01_two_point_cluster),
    KpiSpec("S02", stretch.s02_euler_merge_radius),
    KpiSpec("S03", stretch.s03_persistence),
    KpiSpec("S04", stretch.s04_minkowski),
]}


def load_catalogue(path: Path = CATALOGUE_PATH) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def catalogue_columns(version: str = "v1", path: Path = CATALOGUE_PATH) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(site columns per KPI, tile columns per KPI) for the given catalogue version."""
    site, tile = {}, {}
    for row in load_catalogue(path):
        if row["version"] != version:
            continue
        cols = [c for c in row["output_columns"].split(";") if c]
        site[row["id"]] = cols
        pt = row["per_tile"].strip()
        if pt.startswith("yes"):
            tile[row["id"]] = [c for c in cols if c.endswith("_sigma")] if "sigma only" in pt else cols
    return site, tile


def tile_slices(width: int, n_tiles: int = N_TILES) -> list[slice]:
    """Equal-width tiles along x, full height (D-012)."""
    edges = np.linspace(0, width, n_tiles + 1).round().astype(int)
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def compute_site_kpis(masks: Masks, bse_norm: np.ndarray, nm_per_px: float, seed_key: str
                      ) -> tuple[dict[str, float], list[tuple[str, str, float, float]], KpiContext]:
    """All v1 site-level columns. Raises if any value cannot be produced (D-018)."""
    ctx = KpiContext(masks=masks, nm_per_px=nm_per_px, seed_key=seed_key, bse=bse_norm)
    ctx.require_objects()
    site_cols, _ = catalogue_columns()
    values: dict[str, float] = {}
    curves: list[tuple[str, str, float, float]] = []
    for kpi_id, cols in site_cols.items():
        out = REGISTRY[kpi_id].site_fn(ctx)
        missing = set(cols) - set(out.values)
        if missing:
            raise KeyError(f"{kpi_id} did not produce {sorted(missing)}")
        for c in cols:
            v = out.values[c]
            if v is None or not np.isfinite(v):
                raise ValueError(f"{seed_key}: site-level {c} is not finite ({v})")
            values[c] = float(v)
        curves.extend((kpi_id, name, x, y) for name, x, y in out.curves)
    return values, curves, ctx


def compute_tile_kpis(masks: Masks, bse_norm: np.ndarray, nm_per_px: float, seed_prefix: str
                      ) -> list[dict[str, object]]:
    """Per-tile columns (D-012). NaN only with a ``nan_reason`` (D-018)."""
    _, tile_cols = catalogue_columns()
    rows = []
    for t, cols_slice in enumerate(tile_slices(masks.shape[1])):
        tm = masks.crop(slice(None), cols_slice)
        ctx = KpiContext(masks=tm, nm_per_px=nm_per_px, seed_key=f"{seed_prefix}/tile{t}",
                         bse=None if bse_norm is None else bse_norm[:, cols_slice])
        row: dict[str, object] = {"tile": t}
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
                    row[c] = NAN
                reasons.append(f"{kpi_id}: {e}")
        row["nan_reason"] = "; ".join(dict.fromkeys(reasons))
        rows.append(row)
    return rows


def check_registry() -> None:
    site_cols, tile_cols = catalogue_columns()
    missing = set(site_cols) - set(REGISTRY)
    if missing:
        raise KeyError(f"catalogue v1 KPIs without a registered function: {sorted(missing)}")
    no_tile = [k for k in tile_cols if REGISTRY[k].tile_fn is None]
    if no_tile:
        raise KeyError(f"per-tile KPIs without a tile function: {no_tile}")
