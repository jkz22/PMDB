"""Per-tile KPIs on an arbitrary tile grid (classifier plan C6).

``compute_tile_kpis_on_grid`` is a copy of the loop body of ``pmdb.kpis.compute_tile_kpis`` with the
tile slices passed in, so the existing KPI pipeline and its committed outputs stay untouched.
"""

from __future__ import annotations

import numpy as np

from pmdb import segment as segment_mod
from pmdb.classify.features import HELDOUT_BATCH, fem_grid_slices
from pmdb.io import load_site
from pmdb.kpis import REGISTRY, KpiContext, TooFewObjects, catalogue_columns
from pmdb.kpis.common import NAN
from pmdb.segment import Masks


def compute_tile_kpis_on_grid(masks: Masks, bse_norm: np.ndarray | None, nm_per_px: float,
                              seed_prefix: str, slices: list[slice]) -> list[dict[str, object]]:
    _, tile_cols = catalogue_columns()
    rows = []
    for t, cols_slice in enumerate(slices):
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


def site_kpi_tiles6(batch: str, site: str, cache_root: str | None) -> list[dict[str, object]]:
    raw = load_site(batch, site, resolution="half", normalise="none", cache_root=cache_root)
    norm = load_site(batch, site, resolution="half", cache_root=cache_root)
    masks = segment_mod.segment(raw)
    bse_norm = norm.image[..., 0]
    slices = fem_grid_slices(masks.shape[1])
    rows = compute_tile_kpis_on_grid(masks, bse_norm, raw.nm_per_px, f"{batch}/{site}/g6", slices)
    out = []
    for r in rows:
        sl = slices[int(r["tile"])]
        meta = {"batch": batch, "site": site, "heldout": batch == HELDOUT_BATCH, "tile": r["tile"],
                "tile_x0_um": sl.start * raw.nm_per_px / 1000.0, "tile_x1_um": sl.stop * raw.nm_per_px / 1000.0}
        out.append({**meta, **{k: v for k, v in r.items() if k != "tile"}})
    return out
