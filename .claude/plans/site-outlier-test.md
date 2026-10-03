# Plan: leave-one-site-out outlier test (spec 003 preliminary)

Branch `site-outlier-test` (worktree). Input: `outputs/kpis/site_kpis.csv` (31 sites, committed). No batch is assumed to be the baseline: every site is tested against the other 30.

## Files to create

1. `pmdb/outliers.py` — library code (pure functions, no I/O).
2. `scripts/site_outlier_test.py` — CLI: reads `outputs/kpis/site_kpis.csv`, writes `outputs/outliers/*`.
3. `tests/test_site_outlier.py` — synthetic known-answer tests, no `data` marker.

Match the style of `scripts/kpi_overview.py` (module docstring, `from __future__ import annotations`, `ROOT` + `sys.path` insert, `matplotlib.use("Agg")`, argparse with `--out` default). No try/except fallbacks: fail loud.

## Locked decisions

### D1. Feature families (verdict KPIs only enter the distance)

Families and sign for the composite (+1 = direction the catalogue's failure_signature calls worse/more clustered/more segregated; 0 = in PCA but not in the composite):

- `loading` (k=1 PC): K01_si_frac_adm(+1)
- `clustering` (k=3 PCs): K02_si_density_per_1000um2(-1), K03_ecd_d50_um(0), K03_ecd_d90_um(+1), K03_ecd_max_um(+1), K04_agglom_frac(+1), K04_n_clusters_per_1000um2(0), K05_voronoi_sigma(+1), K05_voronoi_sigma_z(+1), K05_local_af_cv(+1), K06_cluster_region_frac(+1), K06_void_region_frac(+1), K07_R_rl(-1), K07_R_csr(-1), K08_pcf_excess_max_x(+1), K08_pcf_rpeak_x_um(0), K08_pcf_excess_max_z(+1), K08_pcf_rpeak_z_um(0), K09_mst_m_norm(-1), K09_mst_sigma_norm(+1)
- `localisation` (k=2 PCs): K10_cv_w10(+1), K10_cv_slope(+1), K11_lacey_w10(-1), K12_depth_maxdev(+1), K12_depth_absslope(+1), K13_lateral_cv(+1), K14_empty_p50_um(+1), K14_empty_p95_um(+1)
- `contact` (k=1 PC): K15_si_graphite_contact_frac(-1)

Excluded: K05_voronoi_sigma_null_mean (a null reference, not a KPI), K16_si_graphite_dist_median_um (identically 0 at all 31 sites — log this in run_log.json).

Diagnostic families (NOT in the distance; reported as unsigned indices for attribution only):
- `process`: all D01–D06 columns present in site_kpis.csv
- `artefact`: A01_large_void_frac, A02_curtaining_index, A03_height_um

Put these as module-level constants `VERDICT_FAMILIES: dict[str, tuple[int, dict[str, int]]]` (family -> (k, {column: sign})) and `DIAGNOSTIC_FAMILIES: dict[str, list[str]]` in `pmdb/outliers.py`.

### D2. Robust standardisation (fitted on the reference only)

`robust_z(ref: ndarray, x: ndarray) -> ndarray`: per column, med = median(ref), scale = 1.4826 * MAD(ref); where scale == 0 use std(ref, ddof=1); if that is also 0 raise ValueError naming the column index. Return clip((x - med)/scale, -5, 5).

### D3. Fold model `fit_model(ref_df) -> Model` and `score(model, x_df) -> float`

Within a fold (reference = all sites except the held-out one):
1. For each verdict family: robust_z of the family columns (fit on ref), then PCA (numpy SVD on the centred ref z-matrix; centre = ref mean of z) keeping k components; project ref and x. Fix each PC's sign so that its loading vector has a non-negative dot product with the family sign vector (for `loading`/`contact` with a single column this makes the PC equal to ±z aligned with the sign). If a family has fewer columns than k, raise.
2. Concatenate PCs → 7-dim feature (1+3+2+1).
3. Classical covariance on the ref 7-dim features: mean m, S = np.cov(ddof=1). If S is singular (np.linalg.cond > 1e12) raise ValueError. T² = (x−m)ᵀ S⁻¹ (x−m); distance d = sqrt(T²). No MinCovDet anywhere (removed in rev 1).
4. Hotelling prediction F statistic for a new point with n = len(ref), p = 7: F_stat = n(n−p) / (p(n+1)(n−1)) · T², which is exactly F(p, n−p) under Gaussian features. `score` returns (d, F_stat, n).

### D4. Leave-one-site-out scores and conformal p

For each site i (n = 31):
- ref = all others; fit model; s_i = score.
- Calibration scores: for each j in ref, fit on ref \ {j} and score j → c_{i,j} (30 values).
- conformal p_i = (1 + #{j: c_{i,j} >= s_i}) / (len(ref) + 1). Minimum attainable = 1/31.

### D5. Parametric p (Hotelling prediction F-test, rev 1)

p_param_i = scipy.stats.f.sf(F_stat_i, 7, n_ref − 7) with n_ref = 30.
Calibration check (write to run_log.json): for all 31×30 calibration scores, PIT u = f.cdf(F_stat, 7, n_fit − 7) with n_fit = 29; report the KS statistic and p-value vs Uniform(0,1) and the empirical fraction of u > 0.95 and u > 0.99.
BH q-values over the 31 p_param (implement BH directly, monotone, capped at 1).

### D6. Distance uncertainty

Bootstrap the reference: for b in 0..499 (seed numpy default_rng(0)), resample the 30 reference sites with replacement, fit model, score site i. Report the 5th and 95th percentile of the bootstrap distances as `d_lo`, `d_hi` (90% interval). If the covariance is singular on a replicate, let it raise (do not catch).

### D7. Attribution (per held-out site, using its own fold's standardisation)

- Composite per verdict family: mean over columns with sign != 0 of sign × z (z from the fold's robust_z). Columns `comp_loading`, `comp_clustering`, `comp_localisation`, `comp_contact`.
- Diagnostic indices: RMS of robust z over the family columns (fit on fold ref): `idx_process`, `idx_artefact`.
- Top 3 individual verdict KPIs by |z|: written to `kpi_contributions.csv` (long format: site, kpi, z) for all verdict KPIs, plus a `top_kpis` string column in site_scores.csv like `K01_si_frac_adm:+5.0; K09_mst_m_norm:-5.0; ...`.

### D8. Batch level (3 original batches only — the outlier split is post hoc and is not tested here)

Statistic per batch: T_b = mean over its sites of -log(p_param). Null: permute batch labels across sites 2000 times (rng seed 1), same group sizes. Report T_b and permutation p per batch.

### D9. Outputs in `outputs/outliers/`

- `site_scores.csv`: batch, site, se_detector, d, d_lo, d_hi, p_conformal, p_param, q_bh, rank (1 = largest d), comp_loading, comp_clustering, comp_localisation, comp_contact, idx_process, idx_artefact, top_kpis. Sorted by d descending. Floats rounded to 4 dp.
- `kpi_contributions.csv` (long).
- `batch_scores.csv`: batch, n_sites, T, p_perm.
- `run_log.json`: input path + sha256, n_sites, families + k, excluded columns with reason, n_boot, seeds, calibration check numbers, runtime.
- `outlier_summary.png` (one figure, 2 panels, 12×6 in, dpi 150):
  - Left: horizontal dot plot of d per site (sorted), with d_lo–d_hi error bars, coloured by batch (tab10 sorted batch order like kpi_overview.batch_colours), label = `batch short (B1/B2/B3) site`. Mark sites with q_bh < 0.05 with a filled star at the right of the error bar. Vertical dashed line at the distance where p_param = 0.05 from the F test with n=30, p=7: sqrt(f.ppf(0.95,7,23) × 7×31×29 / (30×23)) (the T² at which p_param = 0.05). x label "Distance from the other 30 sites (Mahalanobis, block-PCA KPIs)".
  - Right: heatmap (same row order) of the 6 columns comp_loading, comp_clustering, comp_localisation, comp_contact, idx_process, idx_artefact, diverging colormap `RdBu_r` clipped at ±5, annotate cell values with 1 dp. Column titles: "Si loading", "Si clustering", "Si localisation", "Si–graphite contact (−)", "Process diag. (|z|)", "Artefact (|z|)".

### D10. Tests (`tests/test_site_outlier.py`, no data marker)

Build a synthetic DataFrame with all verdict + diagnostic columns, 31 rows, i.i.d. N(0,1) values (seed 0) plus `batch`, `site`, `se_detector` columns.
Synthetic design (rev 2): within each verdict family, columns share one latent factor: value = sign_c × 0.8·L_family + 0.6·e_c, with L_family and e_c iid N(0,1) per site (seed 0), sign_c the D1 sign (0 → +1). Diagnostic columns iid N(0,1).
1. `test_injected_outlier_flagged`: for one site set L_clustering = 6 (i.e. add 0.8×6 along the signed factor to every clustering column); run_pipeline with n_boot=20; assert that site has rank 1, p_conformal == 1/31, q_bh < 0.05, comp_clustering > 3.
2. `test_null_calibration`: 20 null datasets (rev-2 design, seeds 100..119), `loso_scores` only. Assert (i) fraction of datasets with any q_bh < 0.05 is <= 0.15, and (ii) pooled p_param mean >= 0.4 (the test may be conservative, must not be anti-conservative). Also assert pooled p_param mean <= 0.8 as a sanity bound.
3. `test_robust_z_zero_scale_raises`: constant column in ref → ValueError.

Expose `run_pipeline(df, n_boot=500) -> tuple[site_scores_df, contributions_df, batch_df, log_dict]` in `pmdb/outliers.py` so tests and the script share one code path. Plotting lives in the script only.

Runtime: keep n_boot=500 for the real run; tests use n_boot=20 for test 1.

## Steps

1. Write `pmdb/outliers.py` per D1–D8.
2. Write `tests/test_site_outlier.py` per D10; run `pytest -q tests/test_site_outlier.py` until green.
3. Write `scripts/site_outlier_test.py` per D9; run `python scripts/site_outlier_test.py`.
4. Run `pytest -q -m "not data"` (full non-data suite) and `pytest -q -m data`; record pass/fail counts.
5. Report: paste site_scores.csv top 8 rows (all columns), batch_scores.csv, the calibration-check block of run_log.json, test counts, runtime. Do not commit.

## Revision log

### rev 1 (dispatcher, after implementer DEVIATION REPORT at step 2)
- MCD + chi2×median-scale calibration was anti-conservative: on iid noise 3/31 sites got q_bh=0.015; injected outlier got p_conformal 2/31 (a heavy-tailed MCD calibration score exceeded it); runtime 3.5 min at n_boot=50.
- Change: D3 classical covariance + Hotelling prediction F statistic; D5 p_param from F(7, n−7) (exact under normality, no scale estimation); D6 drops MinCovDet; D10 null test becomes a 40-replicate calibration test instead of a single seed.
- Accepted risk: classical covariance can be inflated by outliers left in a reference fold (masking). Disclosed in the report; the observed outliers are ~10× the between-batch distance, so masking cannot hide them.

### rev 2 (dispatcher, after implementer-r2 DEVIATION REPORT at step 2)
- Null p_param mean 0.667: the F test is conservative because PCA is fitted on the reference (in-sample PC variance inflated, held-out T² smaller). Accepted: conservative is the safe direction for QC; disclosed in the report as "confidence is a lower bound". Test 2 now asserts not-anti-conservative (mean >= 0.4, flag rate <= 0.15), 20 replicates.
- Injected outlier undetectable under iid columns (shift spread over a random 3-D PC subspace). Unrealistic: real KPIs are strongly correlated. Synthetic design now uses a shared latent factor per family.
- Method (D1–D9) unchanged.
