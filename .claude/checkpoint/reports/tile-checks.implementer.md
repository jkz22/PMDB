## Changes made
- `scripts/tile_signal_checks.py` (new) - steps 1-5 (loading/validation, LOSO check 1, summaries, check 2, figures, results.md).
- `outputs/pooling_checks/` (new): 9 files (6 CSV, 2 PNG, results.md).
- Nothing committed.

## Verification
- Step 1: --help exit 0; load_tiles prints `124 15 ['K16_si_graphite_dist_median_um']` (as expected).
- Row counts (excl. header): tile_predictions 316, site_tile_spread 79, batch_summary 7, pooling_side_note 3, variance_by_feature 15, site_feature_spread 465. n_correct in 0..4. p sums to 1 within 3.3e-16. ss sums to 1 within 6.7e-16. Flags only in {tail_driven, diffuse_shift, none}. Verdict non-empty in exactly 2 rows. `grep -c "Overall reading"` = 1.
- Determinism: md5 of all CSVs + results.md identical across two runs.
- pytest -q -m "not data": baseline (pre-change) 98 passed; after 98 passed, 12 deselected, 0 failures.
- No ConvergenceWarning (ran once with UserWarning as error).

## Key results (from outputs/pooling_checks/results.md)
Overall reading:
B1 vs B3: mixed/inconclusive
B2 vs B3: localised

Batch summary (batch_summary.csv):
variant,batch,n_sites,n_correct_0..4,median_frac_correct,diffuse_correct,witness,mixed,miss,fa_rate,verdict
multiclass,Batch_1,7,3,2,0,0,2,0.25,2,0,2,3
multiclass,Batch_2,7,1,2,4,0,0,0.5,0,1,5,1
multiclass,Batch_3,17,2,1,4,7,3,0.75,10,4,1,2
B1_vs_B3,Batch_1,7,1,1,1,2,2,0.75,4,0,2,1,fa 0.294,diffuse
B1_vs_B3,Batch_3,17,1,0,2,10,4,0.75,14,1,1,1
B2_vs_B3,Batch_2,7,0,1,2,3,1,0.75,4,2,1,0,fa 0.412,diffuse
B2_vs_B3,Batch_3,17,2,2,5,2,6,0.5,8,5,2,2

Binary verdicts: B1 diffuse, B2 diffuse (neither localised). Note B2 overall reading is "localised" only through the >=3 tail_driven rule (3 flags), not the classifier verdict.

Pooling side note:
multiclass n=31 tile_bal_acc 0.444 site_acc mean 0.581 max 0.548
B1_vs_B3 n=24 tile_bal_acc 0.671 mean 0.750 max 0.417
B2_vs_B3 n=24 tile_bal_acc 0.630 mean 0.667 max 0.542

Check 2 flags: B1: tail_driven 1 (K04_agglom_frac), diffuse_shift 1 (K15 contact frac). B2: tail_driven 3 (K04_agglom_frac, K14_empty_p50_um, K15_si_graphite_contact_frac).
icc_all (>=0.5 site-level): K01 0.549, K02 0.928, K03_max 0.518, K04_n_clusters 0.500, K14_p95 0.520, K15 0.617; others tile-dominated (K07_R_csr -0.102, K07_R_rl 0.038, K04_agglom 0.165). ss_batch max 0.184 (K15), 0.159 (K02), 0.139 (K01).
Features dropped: ['K16_si_graphite_dist_median_um'] (only one).

## Notes
- No deviations. Mechanical: `catalogue_columns()[1]` is a dict (id -> cols); flattened its values in catalogue order. `fig_tile_probs` takes no site_df arg (unused).
- git status also shows untracked `.claude/plans/fem-build.md`, not created by me.
