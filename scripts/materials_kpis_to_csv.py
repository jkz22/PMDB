"""Convert the colleague materials KPIs (long parquet) to a wide per-site CSV.

`results/kpis.parquet` (columns image, batch, kpi, value, tier) is pivoted to one row per
site so `pmdb.screen` can read it. `site` is the `<id>` of `img_<id>_BSE`. The
`mat_graphite_d10/d50/d90` pixel lengths are multiplied by `pixel_size_um` from `config.yaml`
and renamed `*_um`; every other KPI passes through unchanged. Rows follow `site_kpis.csv`.

    python scripts/materials_kpis_to_csv.py
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_RE = re.compile(r"^img_([A-Za-z0-9]+)_BSE$")
PX_LENGTH_KPIS = ("mat_graphite_d10", "mat_graphite_d50", "mat_graphite_d90")
REQUIRED_COLS = ("image", "batch", "kpi", "value")


class ConversionError(ValueError):
    """Raised when the long table cannot be converted faithfully."""


def read_pixel_size_um(config_path: str | Path) -> float:
    """Return `pixel_size_um` from a YAML config; raise ConversionError if absent or invalid."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    v = cfg.get("pixel_size_um") if isinstance(cfg, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise ConversionError(f"pixel_size_um missing or not a positive finite number in {config_path}: {v!r}")
    return float(v)


def to_wide(long: pd.DataFrame, pixel_size_um: float) -> pd.DataFrame:
    """Pivot the long KPI table to batch, site, one column per KPI (lengths converted to um)."""
    missing = [c for c in REQUIRED_COLS if c not in long.columns]
    if missing:
        raise ConversionError(f"missing columns: {missing}")
    site = long["image"].astype(str).str.extract(IMAGE_RE)[0]
    bad = long.loc[site.isna(), "image"].unique()[:5]
    if len(bad):
        raise ConversionError(f"image names not of form img_<site>_BSE: {list(bad)}")
    df = long.assign(site=site, batch=long["batch"].astype(str))
    dup = df.duplicated(["batch", "site", "kpi"], keep=False)
    if dup.any():
        ex = [tuple(r) for r in df.loc[dup, ["batch", "site", "kpi"]].drop_duplicates().head(5).to_numpy()]
        raise ConversionError(f"duplicate (batch, site, kpi) rows: {ex}")
    order = list(pd.unique(df["kpi"]))
    wide = df.pivot(index=["batch", "site"], columns="kpi", values="value")[order].reset_index()
    wide.columns.name = None
    for k in PX_LENGTH_KPIS:
        if k in wide.columns:
            wide[k] = wide[k] * pixel_size_um
    wide = wide.rename(columns={k: f"{k}_um" for k in PX_LENGTH_KPIS})
    return wide


def align_to_reference(wide: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Return `wide` rows in reference row order; the (batch, site) sets must be equal."""
    ref_keys = reference[["batch", "site"]].astype(str)
    a = set(map(tuple, ref_keys.to_numpy()))
    b = set(map(tuple, wide[["batch", "site"]].astype(str).to_numpy()))
    if a != b:
        raise ConversionError(
            f"(batch, site) sets differ: reference has {len(a)} sites, parquet has {len(b)}; "
            f"only in reference: {sorted(a - b)[:5]}; only in parquet: {sorted(b - a)[:5]}")
    return ref_keys.merge(wide, on=["batch", "site"], how="left", validate="one_to_one")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", default="results/kpis.parquet")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--reference", default="outputs/kpis/site_kpis.csv")
    ap.add_argument("--out", default="outputs/kpis/materials_site_kpis.csv")
    a = ap.parse_args(argv)

    def res(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else ROOT / p

    try:
        px = read_pixel_size_um(res(a.config))
        long = pd.read_parquet(res(a.parquet))
        ref = pd.read_csv(res(a.reference), dtype={"batch": str, "site": str})
        wide = align_to_reference(to_wide(long, px), ref)
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = res(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out, index=False)
    print(f"wrote {out}: {len(wide)} sites x {wide.shape[1] - 2} KPIs (lengths px -> um at {px} um/px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
