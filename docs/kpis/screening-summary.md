# KPI screening: one-page summary

Full contract and details: [screening.md](screening.md)

## Run it

```bash
python -c "from pmdb.screen import pivot_raw_intensity_stats, read_table; from pathlib import Path; Path('outputs/kpis/screen').mkdir(parents=True, exist_ok=True); pivot_raw_intensity_stats(read_table('outputs/raw_intensity_stats.csv')).to_csv('outputs/kpis/screen/covariates_bse.csv', index=False)"
python -m pmdb.screen --kpis outputs/kpis/site_kpis.csv --replicates outputs/kpis/tile_kpis.csv \
  --sensitivity outputs/kpis/sensitivity.csv --sensitivity-params d_um d_star_um \
  --covariates outputs/kpis/screen/covariates_bse.csv \
  --sentinel K08_pcf_rpeak_x_um=-1 K08_pcf_rpeak_z_um=-1
```

## The 5 gates, in order (first failure decides)

1. **Degeneracy**: drops KPIs that barely vary or are mostly missing.
   Threshold: >20% missing (NaN or declared sentinel), or <5 distinct values, or MAD = 0.
   Why: a KPI with no spread cannot rank sites.
2. **Artefact**: drops KPIs that track an imaging nuisance within batches.
   Threshold: |within-batch Spearman rho| >= 0.5 with any covariate AND 95% bootstrap CI excludes 0.
   Why: it measures the microscope, not the electrode. Batch eta-squared is only flagged (>0.5), never dropped.
3. **Reliability**: drops KPIs that disagree between tiles of the same site.
   Threshold: ICC(1) 95% CI upper bound < 0.5 (batch-residualised, site bootstrap).
   Why: if repeat measurements disagree, site differences are noise.
4. **Robustness**: drops KPIs whose site ranking flips when parameters change.
   Threshold: min pairwise Spearman rho across settings < 0.8.
   Why: a KPI that depends on an arbitrary parameter choice is not a stable measurement.
5. **Redundancy**: drops duplicates of a KPI already kept.
   Threshold: co-clustered (|rho| >= 0.8, average linkage) in >= 80% of 500 site bootstraps.
   Why: duplicates add no information; the most reliable member is kept.

## How to read the verdict

- `decision` = keep/drop; `deciding_gate` = the first gate that failed.
- `untested_gates` = gates that could not be evaluated for that KPI (no data or undefined statistic): untested is NOT passed.
- Kept KPIs are ranked by ICC (highest first); `site_kpis_filtered.csv` holds only them.

> **Limits**
> - n = 31 sites in 3 batches: CIs are wide, borderline calls are fragile.
> - Untested gates are not passes; supply tiles/sweeps to get tested.
> - This screens measurement quality only, not predictive value (no outcome label exists).
