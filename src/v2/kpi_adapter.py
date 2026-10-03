"""Thin wrapper around the teammate KPI code (pmdb.segment + pmdb.kpis, spec 002).

The KPI code is imported from a pinned git worktree so every run records the exact
commit. Nothing here re-implements a KPI: phase fractions are reductions of the
teammate masks, everything else calls ``pmdb.kpis.REGISTRY``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

KPI_COMMIT = os.environ.get("PMDB_KPI_COMMIT", "d23a116")
KPI_ROOT = Path(os.environ.get("PMDB_KPI_ROOT", Path(__file__).resolve().parents[3] / "kpi_wt" / KPI_COMMIT))
os.environ.setdefault("PMDB_DATA", str(Path(__file__).resolve().parents[2] / "data"))
os.environ.setdefault("PMDB_CACHE", str(Path(__file__).resolve().parents[2] / "cache"))
if str(KPI_ROOT) not in sys.path:
    sys.path.insert(0, str(KPI_ROOT))
_pm = sys.modules.get("pmdb")
if _pm is not None and not str(getattr(_pm, "__file__", "")).startswith(str(KPI_ROOT)):
    # main-branch pmdb (no pmdb.kpis) was imported first: swap in the pinned KPI branch package
    for _k in [k for k in sys.modules if k == "pmdb" or k.startswith("pmdb.")]:
        del sys.modules[_k]

from pmdb.kpis import REGISTRY, TooFewObjects  # noqa: E402
from pmdb.kpis.common import KpiContext  # noqa: E402
from pmdb.segment import Masks, segment_bse  # noqa: E402

CROP_KPIS = ("K01", "K02", "K03", "K04")
CROP_COLS = ("frac_si", "frac_graphite", "frac_pore", "frac_artefact", "K01_si_frac_adm",
             "K02_si_density_per_1000um2", "K03_ecd_d50_um", "K03_ecd_d90_um", "K03_ecd_max_um",
             "K04_agglom_frac", "K04_n_clusters_per_1000um2")


# KPIs that pass gates G1-G3 at d23a116; K02/K03 and K04 cluster density excluded (user decision).
GATED_COLS = ("frac_si", "frac_graphite", "frac_pore", "K01_si_frac_adm", "K04_agglom_frac")


def kpi_commit_hash() -> str:
    f = KPI_ROOT / ".kpi_commit"  # shipped alongside the code where there is no .git (Modal image)
    if f.exists():
        return f.read_text().strip()
    return subprocess.check_output(["git", "-C", str(KPI_ROOT), "rev-parse", "HEAD"], text=True).strip()


def segment(bse_raw: np.ndarray, nm_per_px: float) -> Masks:
    return segment_bse(np.asarray(bse_raw, dtype=np.float64), nm_per_px)


def mask_fractions(m: Masks) -> dict[str, float]:
    return {"frac_si": float(m.si.mean()), "frac_graphite": float(m.graphite.mean()),
            "frac_pore": float(m.pore.mean()), "frac_artefact": float(m.artefact.mean())}


def kpis_from_masks(m: Masks, nm_per_px: float, seed_key: str, ids=CROP_KPIS) -> dict[str, float]:
    """Mask fractions + selected registry KPIs; NaN where the KPI needs >=10 objects."""
    ctx = KpiContext(masks=m, nm_per_px=nm_per_px, seed_key=seed_key)
    out = mask_fractions(m)
    for k in ids:
        try:
            out.update(REGISTRY[k].site_fn(ctx).values)
        except TooFewObjects:
            pass
    return {c: float(out.get(c, np.nan)) for c in CROP_COLS}


def crop_kpis(m: Masks, nm_per_px: float, seed_prefix: str, coords, size: int) -> list[dict]:
    rows = []
    for (y, x) in coords:
        sub = m.crop(slice(y, y + size), slice(x, x + size))
        rows.append({"y": y, "x": x, **kpis_from_masks(sub, nm_per_px, f"{seed_prefix}/{y}_{x}")})
    return rows
