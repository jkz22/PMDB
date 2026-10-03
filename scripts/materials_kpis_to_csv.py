"""Convert the colleague materials KPIs (long parquet) to a wide per-site CSV.

`results/kpis.parquet` (columns image, batch, kpi, value, tier) is pivoted to one row per
site so `pmdb.screen` can read it. `site` is the `<id>` of `img_<id>_BSE`. The
KPIs pass through unchanged: `src/kpis_materials.py` already reports lengths in micrometres
(`*_um`). Columns follow `CANONICAL_ORDER` (the `compute()` emission order), then any other KPI
alphabetically; a parquet with legacy pixel-unit `mat_graphite_d*` names is rejected.
Rows follow `site_kpis.csv`.

    python scripts/materials_kpis_to_csv.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_RE = re.compile(r"^img_([A-Za-z0-9]+)_BSE$")
CANONICAL_ORDER = (
    "mat_porosity", "mat_bright_fraction", "mat_active_fraction",
    "mat_graphite_d10_um", "mat_graphite_d50_um", "mat_graphite_d90_um",
    "mat_crack_fraction", "mat_rim_coverage", "mat_orientation_anisotropy",
)
LEGACY_PX_KPIS = ("mat_graphite_d10", "mat_graphite_d50", "mat_graphite_d90")
REQUIRED_COLS = ("image", "batch", "kpi", "value")


class ConversionError(ValueError):
    """Raised when the long table cannot be converted faithfully."""


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long KPI table to batch, site, one column per KPI in canonical order."""
    missing = [c for c in REQUIRED_COLS if c not in long.columns]
    if missing:
        raise ConversionError(f"missing columns: {missing}")
    site = long["image"].astype(str).str.extract(IMAGE_RE)[0]
    bad = long.loc[site.isna(), "image"].unique()[:5]
    if len(bad):
        raise ConversionError(f"image names not of form img_<site>_BSE: {list(bad)}")
    df = long.assign(site=site, batch=long["batch"].astype(str))
    legacy = sorted(set(long["kpi"]) & set(LEGACY_PX_KPIS))
    if legacy:
        raise ConversionError(
            f"pixel-unit KPI(s) {legacy} found: parquet predates um reporting; "
            "regenerate results/kpis.parquet with src/run.py")
    dup = df.duplicated(["batch", "site", "kpi"], keep=False)
    if dup.any():
        ex = [tuple(r) for r in df.loc[dup, ["batch", "site", "kpi"]].drop_duplicates().head(5).to_numpy()]
        raise ConversionError(f"duplicate (batch, site, kpi) rows: {ex}")
    present = set(df["kpi"])
    order = [k for k in CANONICAL_ORDER if k in present] + sorted(present - set(CANONICAL_ORDER))
    wide = df.pivot(index=["batch", "site"], columns="kpi", values="value")[order].reset_index()
    wide.columns.name = None
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
    ap.add_argument("--reference", default="outputs/kpis/site_kpis.csv")
    ap.add_argument("--out", default="outputs/kpis/materials_site_kpis.csv")
    a = ap.parse_args(argv)

    def res(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else ROOT / p

    try:
        long = pd.read_parquet(res(a.parquet))
        ref = pd.read_csv(res(a.reference), dtype={"batch": str, "site": str})
        wide = align_to_reference(to_wide(long), ref)
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = res(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out, index=False)
    print(f"wrote {out}: {len(wide)} sites x {wide.shape[1] - 2} KPIs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
