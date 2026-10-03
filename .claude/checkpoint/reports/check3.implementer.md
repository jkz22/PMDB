# check3 implementer report

## Changes made
- `scripts/pooling_compare.py` (new): imports `load_tiles`, `make_pipeline`, `md_table`, constants from `scripts/tile_signal_checks.py`; four pooling rules, three tasks, LOSO, bootstrap CI.
- Outputs (new, uncommitted): `outputs/pooling_checks/check3_{metrics,site_predictions,calibration}.csv`, `check3_results.md`.

## Verification
- Ran end-to-end twice (~2 min each); md5 identical across runs:
  - check3_calibration.csv 4a4cff3749ab70c292a388eb35b60eac
  - check3_metrics.csv d1b8296e3cb66449b12d4883c68bb97f
  - check3_results.md 61bfc608a5cbc0101a0fb24b7c4ae8bf
  - check3_site_predictions.csv 7ebbef93ac8d26e36c2ae40f008a7e58
- Rows excluding header: metrics 12, site predictions 316 (wc 13 / 317).
- Probabilities sum to 1 within 1e-9: enforced by an in-script check (raises otherwise); passed.
- `pytest -q -m "not data"`: 98 passed, 12 deselected (unchanged).

## Notes
- `git status` shows `.claude/plans/fem-build.md` modified; not by me (tree was clean at start). Not touched.
- Binary tasks: p for the absent class is 0 so p_* sum to 1; log-loss and Brier use the two task classes only. Calibration bin 0.33-0.50 is empty for binary tasks by construction. Bins are half-open except the last (inclusive of 1.0).
- Bootstrap CI: percentile, resampling sites; resamples missing a class are scored on present classes only.
- R4 uses 42 numeric catalogue site columns after dropping constants (R1 15, R2 15, R3 45).
- R1 and R4 tie at 0.493 on multiclass balanced accuracy; "best" in the md picks R1 on tie (first).

## check3_results.md (verbatim)

# Check 3: site-level pooling rules (D10)

- Inputs: `/Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/outputs/kpis/tile_kpis.csv`, `/Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/outputs/kpis/site_kpis.csv`
- Tile features (15); dropped constant: ['K16_si_graphite_dist_median_um']
- Feature counts: R1 15; R2 15; R3 45; R4 42
- Model: median impute, StandardScaler, LogisticRegression(class_weight=balanced, C=1.0, max_iter=5000, random_state=0), refitted per leave-one-site-out fold; no tuning.
- Rules: R1 mean of tile probabilities; R2 per-feature tile mean; R3 per-feature [mean, max, std ddof=0]; R4 site-level KPIs (KPIs-alone baseline).
- Bootstrap: 1000 site resamples, seed 0, percentile 95% CI of balanced accuracy.
- Binary tasks: log-loss and Brier use the two task classes only.

## Metrics

| rule | task | n_sites | site_acc | balanced_acc | macro_f1 | log_loss | brier | bacc_ci_lo | bacc_ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| R1_mean_prob | multiclass | 31 | 0.581 | 0.493 | 0.490 | 0.980 | 0.190 | 0.316 | 0.691 |
| R2_feat_mean | multiclass | 31 | 0.452 | 0.387 | 0.384 | 1.356 | 0.237 | 0.221 | 0.563 |
| R3_feat_mean_max_std | multiclass | 31 | 0.484 | 0.406 | 0.399 | 1.750 | 0.258 | 0.234 | 0.584 |
| R4_site_kpis | multiclass | 31 | 0.581 | 0.493 | 0.495 | 1.064 | 0.203 | 0.309 | 0.683 |
| R1_mean_prob | B1_vs_B3 | 24 | 0.750 | 0.655 | 0.667 | 0.591 | 0.202 | 0.444 | 0.875 |
| R2_feat_mean | B1_vs_B3 | 24 | 0.625 | 0.567 | 0.564 | 0.644 | 0.210 | 0.355 | 0.800 |
| R3_feat_mean_max_std | B1_vs_B3 | 24 | 0.542 | 0.466 | 0.467 | 1.124 | 0.319 | 0.277 | 0.681 |
| R4_site_kpis | B1_vs_B3 | 24 | 0.792 | 0.685 | 0.705 | 0.621 | 0.210 | 0.500 | 0.889 |
| R1_mean_prob | B2_vs_B3 | 24 | 0.667 | 0.681 | 0.644 | 0.577 | 0.197 | 0.455 | 0.861 |
| R2_feat_mean | B2_vs_B3 | 24 | 0.667 | 0.639 | 0.625 | 0.608 | 0.209 | 0.414 | 0.833 |
| R3_feat_mean_max_std | B2_vs_B3 | 24 | 0.625 | 0.567 | 0.564 | 0.801 | 0.257 | 0.350 | 0.781 |
| R4_site_kpis | B2_vs_B3 | 24 | 0.708 | 0.668 | 0.661 | 0.577 | 0.201 | 0.437 | 0.874 |

## Calibration (predicted max-prob bin -> n, accuracy)

| rule | task | bin | n | accuracy |
|---|---|---|---|---|
| R1_mean_prob | multiclass | 0.33-0.50 | 19 | 0.474 |
| R1_mean_prob | multiclass | 0.50-0.70 | 8 | 0.625 |
| R1_mean_prob | multiclass | 0.70-0.90 | 3 | 1.000 |
| R1_mean_prob | multiclass | 0.90-1.00 | 1 | 1.000 |
| R2_feat_mean | multiclass | 0.33-0.50 | 10 | 0.200 |
| R2_feat_mean | multiclass | 0.50-0.70 | 9 | 0.667 |
| R2_feat_mean | multiclass | 0.70-0.90 | 9 | 0.556 |
| R2_feat_mean | multiclass | 0.90-1.00 | 3 | 0.333 |
| R3_feat_mean_max_std | multiclass | 0.33-0.50 | 2 | 0.500 |
| R3_feat_mean_max_std | multiclass | 0.50-0.70 | 13 | 0.385 |
| R3_feat_mean_max_std | multiclass | 0.70-0.90 | 8 | 0.625 |
| R3_feat_mean_max_std | multiclass | 0.90-1.00 | 8 | 0.500 |
| R4_site_kpis | multiclass | 0.33-0.50 | 2 | 0.000 |
| R4_site_kpis | multiclass | 0.50-0.70 | 10 | 0.600 |
| R4_site_kpis | multiclass | 0.70-0.90 | 14 | 0.500 |
| R4_site_kpis | multiclass | 0.90-1.00 | 5 | 1.000 |
| R1_mean_prob | B1_vs_B3 | 0.33-0.50 | 0 |  |
| R1_mean_prob | B1_vs_B3 | 0.50-0.70 | 14 | 0.714 |
| R1_mean_prob | B1_vs_B3 | 0.70-0.90 | 9 | 0.778 |
| R1_mean_prob | B1_vs_B3 | 0.90-1.00 | 1 | 1.000 |
| R2_feat_mean | B1_vs_B3 | 0.33-0.50 | 0 |  |
| R2_feat_mean | B1_vs_B3 | 0.50-0.70 | 12 | 0.417 |
| R2_feat_mean | B1_vs_B3 | 0.70-0.90 | 8 | 0.875 |
| R2_feat_mean | B1_vs_B3 | 0.90-1.00 | 4 | 0.750 |
| R3_feat_mean_max_std | B1_vs_B3 | 0.33-0.50 | 0 |  |
| R3_feat_mean_max_std | B1_vs_B3 | 0.50-0.70 | 10 | 0.500 |
| R3_feat_mean_max_std | B1_vs_B3 | 0.70-0.90 | 4 | 0.500 |
| R3_feat_mean_max_std | B1_vs_B3 | 0.90-1.00 | 10 | 0.600 |
| R4_site_kpis | B1_vs_B3 | 0.33-0.50 | 0 |  |
| R4_site_kpis | B1_vs_B3 | 0.50-0.70 | 8 | 0.875 |
| R4_site_kpis | B1_vs_B3 | 0.70-0.90 | 12 | 0.750 |
| R4_site_kpis | B1_vs_B3 | 0.90-1.00 | 4 | 0.750 |
| R1_mean_prob | B2_vs_B3 | 0.33-0.50 | 0 |  |
| R1_mean_prob | B2_vs_B3 | 0.50-0.70 | 15 | 0.600 |
| R1_mean_prob | B2_vs_B3 | 0.70-0.90 | 8 | 0.750 |
| R1_mean_prob | B2_vs_B3 | 0.90-1.00 | 1 | 1.000 |
| R2_feat_mean | B2_vs_B3 | 0.33-0.50 | 0 |  |
| R2_feat_mean | B2_vs_B3 | 0.50-0.70 | 10 | 0.500 |
| R2_feat_mean | B2_vs_B3 | 0.70-0.90 | 11 | 0.727 |
| R2_feat_mean | B2_vs_B3 | 0.90-1.00 | 3 | 1.000 |
| R3_feat_mean_max_std | B2_vs_B3 | 0.33-0.50 | 0 |  |
| R3_feat_mean_max_std | B2_vs_B3 | 0.50-0.70 | 6 | 0.500 |
| R3_feat_mean_max_std | B2_vs_B3 | 0.70-0.90 | 7 | 0.429 |
| R3_feat_mean_max_std | B2_vs_B3 | 0.90-1.00 | 11 | 0.818 |
| R4_site_kpis | B2_vs_B3 | 0.33-0.50 | 0 |  |
| R4_site_kpis | B2_vs_B3 | 0.50-0.70 | 3 | 0.667 |
| R4_site_kpis | B2_vs_B3 | 0.70-0.90 | 12 | 0.583 |
| R4_site_kpis | B2_vs_B3 | 0.90-1.00 | 9 | 0.889 |

## Plain reading

- multiclass: best by balanced accuracy is R1_mean_prob (0.493, 95% CI 0.316-0.691); range across rules 0.387-0.493.
- B1_vs_B3: best by balanced accuracy is R4_site_kpis (0.685, 95% CI 0.500-0.889); range across rules 0.466-0.685.
- B2_vs_B3: best by balanced accuracy is R1_mean_prob (0.681, 95% CI 0.455-0.861); range across rules 0.567-0.681.

Caveat: with 31 labelled sites, differences below about 0.1 in balanced accuracy are within noise (see the bootstrap CIs).
