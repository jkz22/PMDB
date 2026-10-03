"""Cross-lane consistency check (docs/kpis/ownership.md).

The materials lane (results/kpis.parquet) and the geometric lane's
diagnostics (outputs/kpis/site_kpis.csv) measure several of the same
physical quantities through independent segmentations. This script
joins them per site and reports Spearman rank correlation and the
median ratio for each overlapping pair. Sustained disagreement is a
segmentation-drift alarm.

Exit status: non-zero when any pair's |Spearman r| falls below its
floor. Floors are provisional until enough runs exist to calibrate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import ROOT

# (mat KPI, geometric KPI, expected sign of correlation, |r| floor)
# Pairs follow docs/kpis/ownership.md. mat_active_fraction has no pair:
# it is nearly constant by construction, so rank agreement is noise.
PAIRS = [
    ("mat_porosity", "D04_porosity_mean", +1, 0.5),
    ("mat_graphite_d50_um", "D03_graphite_ecd_d50_um", +1, 0.3),
    ("mat_orientation_anisotropy", "D02_graphite_orient_circsd_deg", -1, 0.3),
    ("mat_bright_fraction", "K01_si_frac_adm", +1, 0.5),
]


def load_joined() -> pd.DataFrame:
    mat = pd.read_parquet(ROOT / "results" / "kpis.parquet")
    mat = mat.pivot(index=["batch", "image"], columns="kpi", values="value")
    # materials image stems are img_<site>_<detector>; geometric sites are <site>
    mat = mat.reset_index()
    mat["site"] = mat["image"].str.extract(r"^img_(.+)_[A-Za-z]+$")
    geo = pd.read_csv(ROOT / "outputs" / "kpis" / "site_kpis.csv")
    joined = mat.merge(geo, on=["batch", "site"], how="inner",
                       suffixes=("", "_geo"))
    if joined.empty:
        sys.exit("cross-lane check: no sites in common between "
                 "results/kpis.parquet and outputs/kpis/site_kpis.csv")
    return joined


def main() -> None:
    joined = load_joined()
    print(f"cross-lane check: {len(joined)} sites joined\n")
    print(f"{'materials KPI':<28} {'geometric KPI':<28} "
          f"{'r_s':>6} {'expect':>6} {'floor':>5}  status")
    failed = []
    for mat_kpi, geo_kpi, sign, floor in PAIRS:
        if mat_kpi not in joined or geo_kpi not in joined:
            print(f"{mat_kpi:<28} {geo_kpi:<28}   SKIP (column missing)")
            continue
        sub = joined[[mat_kpi, geo_kpi]].dropna()
        r, _ = spearmanr(sub[mat_kpi], sub[geo_kpi])
        ok = (r * sign > 0) and (abs(r) >= floor)
        status = "ok" if ok else "DISAGREE"
        if not ok:
            failed.append((mat_kpi, geo_kpi, r))
        print(f"{mat_kpi:<28} {geo_kpi:<28} {r:>6.2f} "
              f"{'+' if sign > 0 else '-':>6} {floor:>5.2f}  {status}")

    if failed:
        print("\ncross-lane disagreement — possible segmentation drift in "
              "one lane; inspect overlays (qc/ and outputs/overlays/) "
              "before trusting either lane's verdict:")
        for mat_kpi, geo_kpi, r in failed:
            print(f"  {mat_kpi} vs {geo_kpi}: r_s={r:.2f}")
        sys.exit(1)
    print("\nall pairs consistent")


if __name__ == "__main__":
    main()
