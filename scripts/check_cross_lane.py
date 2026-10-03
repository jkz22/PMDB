"""Cross-lane consistency check (docs/kpis/ownership.md).

The materials lane (results/kpis.parquet) and the geometric lane
(outputs/kpis/site_kpis.csv) measure several of the same physical
quantities through independent segmentations. This script joins them
per site and reports Spearman rank correlation for each overlapping
pair. Sustained disagreement is a segmentation-drift alarm.

Exit status: non-zero when sites are unmatched between the lanes, a
required pair's column is missing, or a required pair's pooled
|Spearman r| falls below its floor.

Per-batch correlations are reported but not enforced: within one batch
the true spread can sit near measurement noise (e.g. graphite d50 in
Batch_3, CV 0.04), where rank agreement is uninformative. Pooled floors
are provisional until enough runs exist to calibrate per-batch ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import ROOT

# (mat KPI, geometric KPI, expected sign, |r| floor, required)
# Pairs follow docs/kpis/ownership.md. The orientation pair is pending
# until the geometry module provides instance_orientations; flip it to
# required when mat_orientation_anisotropy lands in the parquet.
PAIRS = [
    ("mat_porosity", "D04_porosity_mean", +1, 0.5, True),
    ("mat_graphite_d50_um", "D03_graphite_ecd_d50_um", +1, 0.3, True),
    ("mat_orientation_anisotropy", "D02_graphite_orient_circsd_deg", -1, 0.3, False),
    ("mat_bright_fraction", "K01_si_frac_adm", +1, 0.5, True),
]

MIN_BATCH_N = 5  # below this, a per-batch correlation is not even reported


def load_joined() -> pd.DataFrame:
    mat = pd.read_parquet(ROOT / "results" / "kpis.parquet")
    mat = mat.pivot(index=["batch", "image"], columns="kpi", values="value")
    # materials image stems are img_<site>_<detector>; geometric sites are <site>
    mat = mat.reset_index()
    mat["site"] = mat["image"].str.extract(r"^img_(.+)_[A-Za-z]+$")
    geo = pd.read_csv(ROOT / "outputs" / "kpis" / "site_kpis.csv")

    problems = []
    for name, df in [("materials", mat), ("geometric", geo)]:
        dup = df.duplicated(subset=["batch", "site"])
        if dup.any():
            problems.append(f"{name} lane has duplicate (batch, site) keys: "
                            f"{df.loc[dup, ['batch', 'site']].to_records(index=False)}")
    mat_keys = set(map(tuple, mat[["batch", "site"]].itertuples(index=False)))
    geo_keys = set(map(tuple, geo[["batch", "site"]].itertuples(index=False)))
    for missing_from, keys in [("geometric", mat_keys - geo_keys),
                               ("materials", geo_keys - mat_keys)]:
        if keys:
            problems.append(f"sites missing from the {missing_from} lane, "
                            f"so they escape the check: {sorted(keys)}")
    if problems:
        sys.exit("cross-lane check: site mismatch between lanes\n  "
                 + "\n  ".join(problems))

    return mat.merge(geo, on=["batch", "site"], how="inner")


def main() -> None:
    joined = load_joined()
    print(f"cross-lane check: {len(joined)} sites joined, "
          f"all sites matched in both lanes\n")
    print(f"{'materials KPI':<28} {'geometric KPI':<28} "
          f"{'r_s':>6} {'expect':>6} {'floor':>5}  status")
    failed = []
    for mat_kpi, geo_kpi, sign, floor, required in PAIRS:
        if mat_kpi not in joined or geo_kpi not in joined:
            if required:
                failed.append((mat_kpi, geo_kpi, None,
                               "required column missing"))
                status = "MISSING (required)"
            else:
                status = "pending (column missing)"
            print(f"{mat_kpi:<28} {geo_kpi:<28} {'-':>6} "
                  f"{'+' if sign > 0 else '-':>6} {floor:>5.2f}  {status}")
            continue
        sub = joined[["batch", mat_kpi, geo_kpi]].dropna()
        r, _ = spearmanr(sub[mat_kpi], sub[geo_kpi])
        ok = (r * sign > 0) and (abs(r) >= floor)
        if not ok:
            failed.append((mat_kpi, geo_kpi, r, "below floor"))
        print(f"{mat_kpi:<28} {geo_kpi:<28} {r:>6.2f} "
              f"{'+' if sign > 0 else '-':>6} {floor:>5.2f}  "
              f"{'ok' if ok else 'DISAGREE'}")
        # informational: per-batch agreement (not enforced; see module
        # docstring — low within-batch spread makes ranks noise-dominated)
        for batch, bsub in sub.groupby("batch"):
            if len(bsub) < MIN_BATCH_N:
                continue
            rb, _ = spearmanr(bsub[mat_kpi], bsub[geo_kpi])
            cv = bsub[mat_kpi].std() / abs(bsub[mat_kpi].mean())
            print(f"{'':>8}{batch}: n={len(bsub)} r_s={rb:+.2f} "
                  f"(within-batch CV {cv:.3f}; informational)")

    if failed:
        print("\ncross-lane check FAILED — possible segmentation drift or "
              "incomplete lane; inspect overlays (qc/ and outputs/overlays/) "
              "before trusting either lane's verdict:")
        for mat_kpi, geo_kpi, r, why in failed:
            r_txt = f"r_s={r:.2f}" if r is not None else "no value"
            print(f"  {mat_kpi} vs {geo_kpi}: {why} ({r_txt})")
        sys.exit(1)
    print("\nall required pairs present and consistent")


if __name__ == "__main__":
    main()
