# Explorer report: Leo's KPI batch classifier (relayed by dispatcher from the explorer's hand-back message)

## Where
- origin/main, merged PR #19 "Batch fingerprint: conformal batch assignment + held-out site calls" (author leoparada; review fixes #20, #21). Follow-ups: #23, #26 overnight runs; #31 pre-registered extended-feature eval (rejected; merged model stands); #34 pitch deck.
- Code: pmdb/fingerprint.py, scripts/run_fingerprint.py, scripts/plot_fingerprint.py, scripts/demo_confound.py, scripts/demo_reject.py, tests/test_fingerprint.py. Docs: docs/fingerprint.md, docs/eval-plan-oct4.md, docs/presentation-strategy.md. Outputs: outputs/fingerprint/{evaluation.json, features.csv, heldout_features.csv, heldout_predictions.csv, heldout_explain.csv, loo_predictions.csv, figures/}; outputs/overnight/{features,stability,permutation,eval_extended}/.
- Earlier Leo KPI work: PR #3 kpis-leo (src/kpis_materials.py, src/verdict.py rule-based verdicts), #9, #11. docs/kpis/batch-classifier.md (Kevin, PR #15) uses Leo's `mat_` KPI lane; results "not yet run".

## Features (origin/main:pmdb/fingerprint.py:98-134, build_features) — site-level, 16 features, from run_kpis curves.csv + tile_kpis.csv
- si_depth_rel_band0..4: band_si_frac curve (5 depth bands) divided by its site mean; si_depth_slope = rel[4]-rel[0]; si_depth_mid_dip = rel[2] - (rel[0]+rel[4])/2.
- gx_{0.5_2.0,2.0_4.0,4.0_7.0,7.0_10.0}, gz_ same bins: pair-correlation g_obs in x and z averaged over distance bins G_OBS_BINS = ((0.5,2),(2,4),(4,7),(7,10)) µm.
- k15_contact_tilestd: std across tiles of K15_si_graphite_contact_frac.
- docs/fingerprint.md:19-23: "None of the 51 scalar site KPIs separates the batches: the best reaches Kruskal–Wallis p = 0.065 (uncorrected, across 51 tests), and leave-one-site-out classification on the screened KPI set stays below the 55% majority-class baseline."
- docs/fingerprint.md:38-42: BSE black level deliberately excluded (acquisition artefact); segmentation-derived geometry only; no harmonise argument in code.
- Feature families chosen by exploring labelled data (acknowledged, docs/fingerprint.md:84-86).

## Model (pmdb/fingerprint.py:64-91, 202-203, 368; docs/fingerprint.md:44-57)
- Robust per-batch naive Bayes: per batch per feature a median and robust scale (MAD×1.4826, shrunk 50% toward pooled, SCALE_SHRINK=0.5, MIN_MAD_TO_SD_RATIO=0.1); pooled median/robust standardisation winsorised at MAX_Z=10; score = mean Laplace negative log-likelihood; assign argmin score. Mondrian full-conformal p-values; credibility = p of assigned batch; confidence = 1 − max other p; OOD if all p < 0.1 or ≤ 1/(n_b+1). numpy/pandas only, no fitted weights. Assigning by max-p instead drops LOO acc to 0.355.
- Framing: single-stage 3-class + OOD flag (not hierarchical).

## Evaluation and metrics
- LOO over 31 sites, refit per fold (pmdb/fingerprint.py:402-419); label-permutation test of the whole LOO (440-473).
- outputs/fingerprint/evaluation.json: majority_baseline 0.548; LOO accuracy 0.677 (21/31); recall B1 0.714, B2 0.429, B3 0.765; confusion B1→(5,0,2), B2→(3,3,1), B3→(2,2,13); permutation p 0.002 (500 perms). 10k perms: p 0.0025, null mean 0.373, null max 0.742.
- No balanced accuracy / macro-F1 / Brier / bootstrap CI reported (balanced acc from recalls ≈ 0.636).
- Pre-registered extended 26-feature set rejected: LOO 0.613, p 0.0185 (docs/eval-plan-oct4.md:52-83).
- Stability: jackknife drop-1..3 held-out shares (3e122cbj B1 0.869; fn0mhxef B3 1.0; xrv9xvzb B2 1.0); hyperparameter grid LOO 0.452-0.677 (fine6 bins 0.452).

## Held-out predictions (outputs/fingerprint/heldout_predictions.csv)
- 3e122cbj → Batch_1, credibility 0.875, confidence 0.0 (p 0.875/0.875/1.0) — "could be anyone".
- fn0mhxef → Batch_3, credibility 0.444, confidence 0.125.
- xrv9xvzb → Batch_2, credibility 1.0, confidence 0.5 (has elevated BSE black level p1 = 6 but material fingerprint says B2).

## Ours (origin/fem-sim) for comparison
- Two-stage RF (500 trees, sqrt, min_samples_leaf 3, balanced), tile-level (6 × ~22 µm), mean tile prob pooling; features 15 tile scalar KPIs + 16 curated FEM; LOSO end-to-end acc KPI 0.581 (bacc 0.437), FEM 0.516 (0.370), KPI+FEM 0.484 (0.350). Held-out (KPI arm): 3e122cbj B1 0.855, fn0mhxef B3 0.562, xrv9xvzb B3 0.586; FEM arms: xrv9xvzb B2.

## Not checked
Branches site-outlier-test, overnight-script-fixes, devin/* contents; outputs/clean/eval/fingerprint.csv on main; pmdb/classify/features.py in full.
