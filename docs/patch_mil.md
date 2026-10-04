# Patch-level MicroNet batch classifier ("patch MIL")

## Purpose

An image-texture route to batch assignment that is independent of the fingerprint classifier: non-overlapping 224 px patches of harmonised BSE and Inlens images are embedded with a frozen microscopy-pretrained ResNet50, scored by calibrated nearest-neighbour distance to each batch's patch bank, and pooled to a site score. It yields strict leave-one-site-out (LOO) metrics on the 31 labelled sites, predictions for the three held-back sites with a confidence and a comparison to Batch 3, and per-site figures.

## Method

- **Backbone**: torchvision `resnet50`, NASA MicroNet v1.1 weights (frozen, `eval()`, fp32), `https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/resnet50_pretrained_microscopynet_v1.1.pth.tar` (downloaded at Modal image build, size asserted).
- **Patching**: 224 x 224 non-overlapping grid centred in the image, at 50 nm/px (half resolution), 60 patches per site on average (3-5 rows x 15 columns). Features are PatchCore-style: layer2 and bilinearly upsampled layer3 outputs concatenated (1536 ch), 3x3 average pooling, global mean per patch.
- **Channels**: BSE and Inlens only (each L2-normalised, concatenated, unit norm overall). SE_type is dropped because 3 of the 4 SE-detector sites are Batch_3, so detector identity leaks the batch.
- **Harmonisation**: `load_site(..., normalise="fixed", harmonise="affine2")`. `hybrid` is not used because histmatch on its Inlens channel changes noise texture differently per batch (docs/harmonisation.md:186-188). Harmonisation does NOT fix noise, sharpness, BSE zero-clipping or re-quantisation (docs/harmonisation.md:176-188). The physical route `load_clean(..., kind="harm")` (docs/clean.md) is the stronger alternative; it is not used here because the clean arrays are not built.
- **Patch-to-site distance**: cosine distance; `D[p, s]` is the mean of the 3 smallest distances from patch p to the patches of site s (own site = +inf). No PCA, no fitted transform.

## LOO / leakage design

For held-out site i, `train_mask = sites != i`: site i is excluded from every bank and every calibration reference, and its own column in `D` is +inf. Held-back sites are never in any bank. No transform is fitted on any data.

## Calibration and pooling

- A test patch is compared with each batch bank with each bank site dropped in turn (`min` over the remaining m-1 sites), because the calibration reference (every training patch against the other sites of its batch) can only use m-1 sites. Without this matching, banks of different size are not comparable: in a planning simulation on i.i.d. data with 7/7/17 sites, the uncalibrated min-distance assigned 100% of sites to Batch_3, an unmatched z-score assigned 11%, and the matched ECDF scheme about 30/25/40%.
- Calibrated score `u` = ECDF of the reference distances (above its maximum, continued linearly in units of 1.4826 MAD). Higher u = more atypical for that batch.
- Headline site score: mean over dropped sites of the top-10% mean of u over patches; assign the batch with the lowest score. Secondary: plain mean pooling (not permutation-tested).
- Confidence = max of `softmax(-S / 0.1)`. This is a heuristic, not a calibrated probability.
- `anomaly_vs_B3` = top-10% pooled u against the Batch 3 bank; a patch is anomalous if `u_Batch_3 > 0.95`.

## Shortcut diagnostic

Spearman correlation of LOO `anomaly_vs_B3` with `bse_hf` = std of the Laplacian of the affine2-harmonised BSE image over the 31 labelled sites; flagged if |rho| > 0.6. No mitigation is applied.

## How to run and cost

```
modal volume put pmdb-data cache/harmonised/affine2/luts.npz /harmonised/affine2/luts.npz
modal volume put pmdb-data cache_heldout/harmonised/affine2/luts.npz /heldout/harmonised/affine2/luts.npz
modal run modal_patch_mil.py --smoke   # 3 sites/batch + 1 held-out, 20 permutations -> outputs/modal/patch_mil_smoke/ (gitignored)
modal run modal_patch_mil.py           # full -> outputs/patch_mil/
```

Observed: smoke run 2m27s wall and full run 2m39s wall (all three stages, image already built; one T4 embedding call, CPU distance and evaluation calls; evaluation incl. 1000 permutations about 45 s remote). Estimated cost well under $1.

## Results

LOO on 31 labelled sites (B1 7, B2 7, B3 17). Fingerprint permutation p uses 500 permutations, patch MIL 1000.

| Method | LOO acc | Balanced acc | Macro-F1 | Recall B1 / B2 / B3 | Perm p (n) |
|---|---|---|---|---|---|
| Fingerprint NB (classifier of record) | 0.677 | 0.636 | 0.625 | 5/7, 3/7, 13/17 | 0.002 (500) |
| Patch MIL, top-10% pooling (headline) | 0.710 | 0.655 | 0.644 | 3/7, 5/7, 14/17 | 0.001 (1000) |
| Patch MIL, mean pooling (secondary) | 0.645 | 0.616 | 0.621 | 4/7, 4/7, 12/17 | - |
| Majority baseline | 0.548 | 0.333 | - | - | - |

Headline LOO accuracy (0.710) is above 0.677 and p < 0.05.

Confusion (true -> assigned, headline): B1 {B1 3, B2 4, B3 0}; B2 {B1 1, B2 5, B3 1}; B3 {B1 1, B2 2, B3 14}.

Anomaly vs Batch 3, site-level AUROC: Batch_1 vs Batch_3 0.689; Batch_2 vs Batch_3 0.496.

Shortcut diagnostic: Spearman rho(anomaly_vs_B3, BSE Laplacian std) = 0.229, p = 0.214 (flag false).

Held-out predictions:

| Site | Assigned | Confidence | anomaly_vs_B3 | Frac. patches anomalous | Figure |
|---|---|---|---|---|---|
| 3e122cbj | Batch_1 | 0.65 | 1.60 | 40% | [figure](../outputs/patch_mil/figures/heldout_3e122cbj.png) |
| fn0mhxef | Batch_2 | 0.39 | 0.92 | 2% | [figure](../outputs/patch_mil/figures/heldout_fn0mhxef.png) |
| xrv9xvzb | Batch_3 | 0.41 | 0.90 | 2% | [figure](../outputs/patch_mil/figures/heldout_xrv9xvzb.png) |

- 3e122cbj: assigned Batch_1 (confidence 0.65; top-10% scores B1 1.52, B2 1.76, B3 1.60, lower = more typical). Versus Batch 3 (supplier baseline): anomaly score 1.60, above the LOO range of Batch 3 sites (0.77-1.16, median 0.94; Batch 1/2 sites median 0.95); 40% of 75 patches exceed the Batch 3 95th-percentile self-distance (LOO Batch 3 sites median 4%). Most Batch-3-atypical patches lie in image rows [0, 1, 2, 3, 4] of 0-4; see figures/heldout_3e122cbj.png.
- fn0mhxef: assigned Batch_2 (confidence 0.39; top-10% scores B1 0.96, B2 0.91, B3 0.92, lower = more typical). Versus Batch 3 (supplier baseline): anomaly score 0.92, inside the LOO range of Batch 3 sites (0.77-1.16, median 0.94; Batch 1/2 sites median 0.95); 2% of 60 patches exceed the Batch 3 95th-percentile self-distance (LOO Batch 3 sites median 4%). Most Batch-3-atypical patches lie in image rows [0, 2, 3] of 0-3; see figures/heldout_fn0mhxef.png.
- xrv9xvzb: assigned Batch_3 (confidence 0.41; top-10% scores B1 0.95, B2 0.92, B3 0.90, lower = more typical). Versus Batch 3 (supplier baseline): anomaly score 0.90, inside the LOO range of Batch 3 sites (0.77-1.16, median 0.94; Batch 1/2 sites median 0.95); 2% of 60 patches exceed the Batch 3 95th-percentile self-distance (LOO Batch 3 sites median 4%). Most Batch-3-atypical patches lie in image rows [0, 3] of 0-3; see figures/heldout_xrv9xvzb.png.

## Parent images (identified confound)

The 31 labelled sites are crops of 13 parent electrode images regrouped into artificial batches. A parent is identified by the shared full-resolution image height, SE detector and BSE grey-level step (`pmdb/parents.py`, table in `outputs/parent_groups.csv`). Leave-one-site-out (LOSO) is optimistic because sibling crops of the held-out site sit in the training bank / training set. Parent membership is used only to exclude siblings from training banks and CV folds; it never contributes evidence to a batch call or confidence.

| Parent | Labelled sites (batch) | Held-out members |
|---|---|---|
| h1612_ETD_s1 | ptg8lmto (3), xgj4xftb (3) | - |
| h1780_ETD_s1 | iv6g2oq0 (1) | - |
| h1880_ETD_s1 | uhdslk0o (1) | - |
| h1904_ETD_s1 | 0grcilhi (3), hawkfj64 (3), mgxahqnk (3) | - |
| h2048_ETD_s2 | 3806gxp0 (2), avn74qx1 (2) | fn0mhxef |
| h2060_ETD_s1 | 71vgq3fw (3), kbdh4tri (3), tuy3zymq (3), x7u69zsw (3) | - |
| h2068_SE_s1 | rxax5ozo (2), utfgcjfa (3), vc2whyaq (3), x77cy643 (3) | - |
| h2080_ETD_s1 | ffwubibz (1), r17byphk (2), cfe5vt7s (3) | - |
| h2088_ETD_s3 | 9luzk4jm (3), hzumfsms (3), ufdvpb81 (3) | xrv9xvzb |
| h2148_ETD_s1 | f1vzngrs (1), epqdaau9 (2) | - |
| h2156_ETD_s2 | fzrt2k6r (1), b3esycq1 (2) | - |
| h2272_ETD_s2 | i9jiqjwl (2), pl8uabbv (3) | - |
| h2316_ETD_s2 | 4ih2ggld (1), 5n1q8atc (1) | 3e122cbj |

## LOSO vs LOPO

| Model | LOSO acc | LOPO acc | LOPO balanced acc | LOPO macro-F1 | LOPO permutation p |
|---|---|---|---|---|---|
| patch | 0.710 | 0.645 | 0.588 | 0.573 | 0.0040 |
| fingerprint | 0.677 | 0.677 | 0.636 | 0.625 | 0.0020 |
| ensemble | 0.677 | 0.677 | 0.664 | 0.649 | 0.0020 |

LOPO ensemble confusion (true -> assigned): Batch_1 {1 4, 2 1, 3 2}; Batch_2 {1 1, 2 5, 3 1}; Batch_3 {1 2, 2 3, 3 12}.

Permutation scheme: site labels permuted, parent groups fixed, infeasible draws (a batch with <2 training parents in some fold) redrawn; 1000 permutations, seed 0, 0 rejected. Ensemble = average of the patch softmax and the fingerprint conformal p-values normalised to sum 1 (a heuristic, not a posterior).

## Confidence flag and rubric

Rule: high iff ensemble LOPO accuracy over the same agreement stratum (patch call == fingerprint call) of labelled sites of other parents > 0.5; forced low if the fingerprint marks the site OOD.

Strata (LOPO ensemble): agree n=20 acc=0.750; disagree n=11 acc=0.545.

| Strategy | Expected rubric score |
|---|---|
| ensemble_flag | 1.194 |
| ensemble_all_high | 1.355 |
| all_low | 1.000 |
| patch_flag | 1.387 |
| patch_all_high | 1.290 |
| fingerprint_flag | 1.355 |
| fingerprint_all_high | 1.355 |
| fp_conf_flag | 1.258 |

All-low scores 1.0 by construction. Fingerprint-confidence diagnostic (not used in the flag): see `confidence.fp_confidence_diagnostic` in `outputs/patch_mil/lopo_evaluation.json`.

## Final held-out calls

| Site | Assigned | Confidence | p(B1) | p(B2) | p(B3) |
|---|---|---|---|---|---|
| 3e122cbj | Batch_3 | high | 0.13 | 0.17 | 0.70 |
| fn0mhxef | Batch_2 | high | 0.22 | 0.45 | 0.34 |
| xrv9xvzb | Batch_2 | high | 0.36 | 0.42 | 0.22 |

- 3e122cbj: Batch 3 (high confidence). Compared with Batch 3 (supplier baseline): (1) graphite clustering along the layer at 2.0-4.0 um spacing is lower than in Batch 3, typical of Batch 1; (2) graphite clustering along the layer at 7.0-10.0 um spacing is lower than in Batch 3, typical of Batch 1; (3) variability of Si-graphite contact across the image is lower than in Batch 3, typical of Batch 3; (4) local microstructure appearance: 27% of the image's 11.2 um areas look unlike any Batch 3 image (Batch 3 images: typically 4%), spread across the whole image; overall it looks most like Batch 3. Confidence is high because the depth-profile, graphite-arrangement and local-appearance evidence consistently point to Batch 3.
- fn0mhxef: Batch 2 (high confidence). Compared with Batch 3 (supplier baseline): (1) Si fraction in depth band 4 of 5 (1 = top) is lower than in Batch 3, typical of Batch 1; (2) graphite clustering along the layer at 0.5-2.0 um spacing is lower than in Batch 3, typical of Batch 3; (3) graphite clustering along the layer at 2.0-4.0 um spacing is higher than in Batch 3, typical of Batch 2; (4) local microstructure appearance: 2% of the image's 11.2 um areas look unlike any Batch 3 image (Batch 3 images: typically 4%), mostly in the middle third of the image (see outputs/patch_mil/figures/heldout_fn0mhxef.png); overall it looks most like Batch 2. Confidence is high because the depth-profile, graphite-arrangement and local-appearance evidence consistently point to Batch 2.
- xrv9xvzb: Batch 2 (high confidence). Compared with Batch 3 (supplier baseline): (1) graphite clustering along the layer at 2.0-4.0 um spacing is lower than in Batch 3, typical of Batch 1; (2) Si fraction in depth band 4 of 5 (1 = top) is lower than in Batch 3, typical of Batch 1; (3) graphite clustering along the layer at 4.0-7.0 um spacing is higher than in Batch 3, typical of Batch 2; (4) local microstructure appearance: 3% of the image's 11.2 um areas look unlike any Batch 3 image (Batch 3 images: typically 4%), spread across the whole image; overall it looks most like Batch 2. Confidence is high because the depth-profile, graphite-arrangement and local-appearance evidence consistently point to Batch 2.

## Scoring new test images

1. Add the raw TIFFs and rebuild the held-out cache and clean summary (`python scripts/build_clean.py --data-root data_heldout --out outputs/clean_heldout --targets outputs/clean/targets.json --workers 3`, see docs/clean.md).
2. Compute held-out KPIs and fingerprint features: `python scripts/run_fingerprint.py --heldout-dir outputs/heldout/kpis` (see docs/fingerprint.md; writes `outputs/fingerprint/heldout_features.csv`).
3. Upload new held-out cache files: `modal volume put pmdb-data cache_heldout/... /heldout/...` (see the commands in `modal_patch_mil.py`).
4. `modal run modal_patch_mil.py --mode heldout` (embeds only missing held-out sites, then distances, eval, LOPO).

## Limitations

- Only 31 labelled sites (7/7/17); per-class recall is noisy.
- One a-priori configuration, no tuning.
- Confidence is a softmax heuristic, not a calibrated probability.
- MicroNet's training pixel scale is undocumented.
- affine2 does not remove noise, sharpness, 0-clipping or grey-level re-quantisation differences (`grey_step`: `9luzk4jm`, `hzumfsms`, `ufdvpb81` and held-out `xrv9xvzb` populate every 3rd DN), which a CNN may detect.

## Feedback round 1 (2026-10-04)

**Submission.** Fingerprint classifier of record (`outputs/fingerprint/heldout_predictions.csv`): 3e122cbj Batch_1, fn0mhxef Batch_3, xrv9xvzb Batch_2. Organiser truths: Batch_2 / Batch_1 / Batch_3 (`outputs/heldout_labels.csv`): 0/3 correct. The majority batch of the labelled siblings would have been right on 1/3, so parent images are deliberately split across batches. Lesson: the designed batch signal is what differs between crops of the same parent. The 3 truths are now labelled sites (34 sites, 13 parents). Their fingerprint features were recomputed on Modal (`outputs/heldout_features_modal.csv`; the original rows came from a newer local numeric stack; per-feature differences up to 0.45 labelled SD, graphite-clustering features).

**Within-parent contrasts** (`pmdb/within_parent.py`, `scripts/within_parent.py`, `outputs/within_parent/`). For each parent and batch pair with both batches present, `diff = mean_a - mean_b` per feature; per feature x pair: sign counts over parents, exact two-sided sign test, median diff in labelled SD. Pre-registered `is_signal`: >= 3 parents and all the same sign. Parents per pair: B1-B2 = 5, B1-B3 = 1, B2-B3 = 3. Min achievable p is 0.0625 (n = 5) / 0.25 (n = 3), so p is reported, not thresholded. Figure: `outputs/within_parent/within_parent.png`.

| feature | pair | k/n | sign p | median diff (SD) |
|---|---|---|---|---|
| Si fraction in depth band 3 of 5 | B1 - B2 | 5/5 higher in B1 | 0.0625 | +1.57 |
| variability of Si-graphite contact across the image | B2 - B3 | 3/3 higher in B2 | 0.25 | +1.94 |
| Si depletion at mid-depth | B2 - B3 | 3/3 lower in B2 | 0.25 | -1.24 |
| Si fraction in depth band 3 of 5 | B2 - B3 | 3/3 lower in B2 | 0.25 | -1.08 |
| Si fraction in depth band 1 of 5 | B2 - B3 | 3/3 higher in B2 | 0.25 | +0.57 |

Plain language: between crops of the same source image, only the mid-depth Si fraction separates Batch 1 from Batch 2 in all 5 source images (Batch 1 higher), and four depth-profile / contact-variability features separate Batch 2 from Batch 3 in all 3 source images. None is statistically strong with so few source images, and the graphite-clustering features show no consistent within-source-image difference. B1-B3 has a single source image, so no conclusion.

**Parent-centred fingerprint.** `Xc = X - mean of the parent's rows in the pool` (labels never read; a test site's own row joins its parent's mean). Singleton rule: a site that is the only member of its parent in the pool is excluded from centred-model training and predicted with the uncentred fingerprint (its centred row is identically zero and carries no information).

**Model menu (LOPO, 34 sites, `outputs/menu/`).** Ensembles use `fp_prob = softmax(-score / 0.1)` of the fingerprint likelihood scores (same tau as the patch softmax), so the fingerprint probability agrees with its call (the lopo run normalised conformal p-values, which could contradict the call).

| option | accuracy | balanced acc | rubric (flag rule) | rubric SE | rubric all-high | n high |
|---|---|---|---|---|---|---|
| fingerprint | 0.471 | 0.412 | 0.912 | 0.049 | 0.941 | 3 |
| fingerprint_centred | 0.441 | 0.486 | 1.000 | 0.000 | 0.882 | 0 |
| patch | 0.529 | 0.495 | 0.794 | 0.145 | 1.059 | 25 |
| ensemble | 0.618 | 0.597 | 1.206 | 0.168 | 1.235 | 33 |
| ensemble_centred | 0.559 | 0.583 | 0.971 | 0.130 | 1.118 | 19 |
| all-low | - | - | 1.000 | - | - | 0 |

Selection (`outputs/menu/selection.json`, frozen): option `ensemble`, confidence mode `rule`. Caveat: Selection is the maximum of 5 LOPO rubric estimates on 34 sites from 13 parent images; the winning estimate is optimistically biased (winner's curse). Differences smaller than about one SE are not meaningful. The 3 held-out sites enter LOPO as ordinary labelled sites; nothing was tuned on them.

Explanations: the confidence sentence is built from the four evidence items (3 depth/graphite lines + local appearance) that resemble the called batch, not from the flag; `n_evidence_for_call` (0-4) is reported. "Consistently" appears only if all 4 resemble the call.

**Test day runbook.** Copy the organiser TIFFs to `data_test/Batch_test/img_<site>_<BSE|Inlens|ETD|SE>.tif` (gitignored), then:

```
bash scripts/score_test_sites.sh
```

which runs `modal run modal_test_prep.py::main` (uploads only those TIFFs to `pmdb-data:/test/raw`; on Modal: half cache, affine2 LUTs against the shipped labelled reference, clean, KPIs, fingerprint features; pulls back small CSV/JSON to `cache_test/`, `outputs/clean_test/`, `outputs/test/`) and `modal run modal_patch_mil.py --mode test` (GPU embedding, 34-site distances, every menu option's call to `outputs/test/menu_predictions.csv`), then the free-lateral FEM on the test sites (`modal_fem.py` bottom and top, `scripts/fem_collect.py --edge`, Batch_test rows kept as `outputs/fem/free_lateral_test/`) and `python scripts/score_test_all.py`, which writes the final calls. Any failing step aborts the script (`set -e`), so the submission is never silently stale. Compute runs on Modal; the prep image pins the labelled KPI run's package versions. Storage on the volume: `/test/raw`, `/test/half`, `/test/harmonised`, `/test/clean`, `/test/kpis`, `/test/features.csv`. Outputs: `outputs/test/final_predictions.csv`, `outputs/test/submission.md` (written by `score_test_all.py`, not by `--mode test`). The selection is frozen in `outputs/menu/selection.json`; do not re-run `--mode menu` after test images arrive.

**Test day, 2026-10-04 (6 sites, run).** The menu was trimmed to the 3 single models (fingerprint, fingerprint_centred, patch); the two ensembles were removed and `fem_a1` was promoted to the selected model (`outputs/menu/selection.json`, `menu_option` = fingerprint_centred, the best remaining LOPO rubric). `--mode test` now writes every menu option per site to `outputs/test/menu_predictions.csv`. fem_a1 needs free-lateral FEM on the test sites: `modal run modal_fem.py --mode full --orientation {bottom,top} --sites Batch_test/<site>,... --tag full_free_test --solver-overrides '{"lateral":"left"}'`, then `python scripts/fem_collect.py --edge outputs/modal/fem/results/full_free_test <tmp>` and keep the Batch_test rows as `outputs/fem/free_lateral_test/{site,tile}_curves.csv` (12/12 cases ok, USD 0.29). `python scripts/score_test_all.py` builds fem_a1 as registered (34 labelled sites, menu fingerprint features, the test site's parent left out of training as for every menu option; asserts arm A0 reproduces the menu fingerprint call at every test site), combines it with the menu table and writes `outputs/test/{all_models_predictions,all_models_wide,final_predictions}.csv` and `submission.md`. Final confidence: high iff fem_a1 is not out-of-distribution and at least 2 of the 3 menu options agree with it.

| site | parent (seam test) | fem_a1 (final) | fingerprint | fingerprint_centred | patch |
|---|---|---|---|---|---|
| 0eryguqq | G1612 (B3 xgj4xftb, ptg8lmto) | B2 low | B2 low | B1 high | B3 high |
| 4hq27w4c | G2148 (B1 f1vzngrs, B2 epqdaau9) | B1 low | B1 low | B3 high | B2 high |
| fhwrjtet | G1612 | B2 low | B3 low | B3 high | B3 high |
| fspqbkxl | G2148 | B3 high | B3 low | B3 low | B2 high |
| soo2ax3r | G2156 (abuts B2 b3esycq1) | B3 high | B3 low | B3 high | B2 high |
| y59rxmxl | G1880 (B1 uhdslk0o, no seam) | B3 high | B3 low | B3 high | B1 high |

Per-model scores (`score_kind` says what `p_Batch_*` means: `conformal_p` for fem_a1, `probability` for the menu options; fem_a1: credibility, conformal confidence) are in `outputs/test/all_models_predictions.csv`; the stitched parent figure with the test crops is `outputs/stitching/parents_annotated.png`.

Rehearsal (the 3 held-out sites copied to `data_test/`, then removed): half cache exact, affine2 LUT exact, parent key equal for all 3; Modal-vs-old held-out feature differences are informational (max 0.45 labelled SD). Prep ~3.3 min wall incl. image build (clean 127 s, KPIs 22 s for 3 sites); `--mode test` ~3 min wall total with script. Expected for 4-6 sites: ~10 min prep, ~10 min test, well under USD 2.

## Supervised linear probe (LOPO, 34 sites)

**Status: the numbers below are from the previous run (balanced class weights, 200 permutations) and are pending a rerun with the current code (site-balanced weights, `--n-perm 1000` honoured); see `outputs/patch_probe/STALE.md`. Treat them as provisional.**

Method: frozen MicroNet patch embeddings (affine2 BSE+Inlens, 31 labelled + 3 held-out sites with organiser truths, 13 parents). Per leave-one-parent-out fold: StandardScaler, PCA(64) fit on training patches, multinomial `LogisticRegression(C=0.1)`, each patch labelled with its site's batch and site-balanced sample weights (each site sums to 1/(sites in its batch), so every batch carries equal total weight; `class_weight="balanced"` is no longer used). Site probability = softmax of the mean per-patch log-probability. `probe+ensemble` averages the probe with `p_ens_*` of `outputs/menu/menu_predictions.csv` (same folds). One configuration, no tuning. Code: `pmdb/patch_probe.py`, `modal_patch_mil.py --mode probe`; outputs in `outputs/patch_probe/`.

| model | acc | bal acc | macro-F1 | recall B1/B2/B3 | rubric all-high | rubric (high iff max p >= 0.5) |
|---|---|---|---|---|---|---|
| probe | 0.706 | 0.630 | 0.622 | 0.38/0.62/0.89 | 1.412 | 1.412 (28 high) |
| probe+ensemble | 0.706 | 0.630 | 0.627 | 0.62/0.38/0.89 | 1.412 | 1.471 (18 high) |
| ensemble (reference) | 0.618 | 0.597 | 0.596 | 0.62/0.50/0.67 | 1.235 | 1.294 (20 high) |

Held-out sites, LOPO-fold calls (truth in brackets):

| site | probe | probe+ensemble | ensemble |
|---|---|---|---|
| 3e122cbj (B2) | B2 (0.62) | B3 (0.61) | B3 (0.89) |
| fn0mhxef (B1) | B2 (0.57) | B2 (0.44) | B3 (0.52) |
| xrv9xvzb (B3) | B3 (0.70) | B3 (0.43) | B2 (0.59) |

Permutation test (probe, previous run: 200 site-label permutations, parents fixed, seed 0; the current code honours `--n-perm`, rerun planned with 1000): observed accuracy 0.706, null mean 0.337, p = 0.005.

Verdict: probe+ensemble beats the ensemble on LOPO rubric all-high (1.412 vs 1.235) and balanced accuracy (0.630 vs 0.597), so by the pre-set criterion it is adopted as the better candidate. Caveats: 34 sites from 13 parents, one configuration, and the improvement is a few sites; Batch 1 recall remains weak for the probe alone (0.38). Prior evidence: the v2 fine-tune reached only 0.64 accuracy with Batch 2 recall 0.20 under field-grouped CV, so a frozen-feature linear probe is the better-behaved supervised route.

### Explaining the probe in KPI terms

Method (`pmdb/probe_explain.py`): the probe is refitted once on all 34 labelled sites (`fit_full_probe`, same config as a LOPO fold, no parent centring). 7 cheap KPIs are computed with direct numpy/scipy.ndimage on the exact 224 px embedding patches (segmentation on unharmonised raw; no null simulations or point-pattern functions, for speed): Si area fraction, Si particle number density, Si particle size (mean area), Si-graphite contact (fraction of Si boundary pixels touching graphite), porosity, and depth position (control); graphite fraction is cached in `patch_kpis.csv` but not regressed, because Si + graphite + pore fractions sum to about 1 (collinear). KPIs with more than 20 % NaN patches would be dropped (none were); the rest are used complete-case. Each PC score is regressed (OLS) on the z-scored KPIs; a PC counts as explained when R² >= 0.2 and is labelled by its top-2 standardised coefficients. Batch directions push the centred logistic weights through the regression (`logit_per_sd`), and `explainable_share = sum_explained |w|*sd / sum |w|*sd`. A site explanation is the exact decomposition of the log-prob margin between the called batch and the runner-up, `margin = (coef_c - coef_r) . zbar + (b_c - b_r)`; the explained share is the |contribution| fraction carried by explained PCs.

Run: `modal run modal_patch_mil.py --mode explain` (reuses `outputs/probe_explain/patch_kpis.csv` if present; also explains any `Batch_test__*` embeddings on the volume). Outputs in `outputs/probe_explain/`.

Results: all 6 regressed KPIs kept (max NaN 7.2 %), 1921 complete-case patches. Only 4 of 64 PCs reach R² >= 0.2 (PC1 -Si/-porosity, PC2 +Si/+porosity, PC3 +porosity/-Si, PC4 +Si density/-Si fraction).

| KPI | R² from the 64 PCs | kept |
|---|---|---|
| Si area fraction | 0.91 | yes |
| Si particle number density | 0.70 | yes |
| Si particle size | 0.58 | yes |
| Si-graphite contact | 0.25 | yes |
| porosity | 0.84 | yes |
| depth position | 0.41 | yes |

Batch directions (explainable share): Batch_1 0.40 (more Si particle number density, less Si area fraction, more Si particle size); Batch_2 0.12 (more Si particle number density, more Si area fraction, more porosity); Batch_3 0.30 (less Si particle number density, more Si area fraction, more Si-graphite contact).

Held-out (truths 3e122cbj B2, fn0mhxef B1, xrv9xvzb B3; in-sample probe calls Batch 1, Batch 2, Batch 3, so the first two are in-sample misses): 3e122cbj 40 % explainable (higher Si density, lower Si fraction, higher porosity, plus fine texture); fn0mhxef 35 % (mainly fine texture); xrv9xvzb 51 % (lower Si density, higher Si fraction, fine texture). Full sentences in `site_explanations.csv`.

Caveats: in-sample explanations of the final 34-site model (its probabilities are not accuracy; LOPO numbers remain the accuracy reference); OLS on correlated KPIs; segmentation on unharmonised raw; KPIs on an 11 um patch are noisy.

### Parent-centred probe

Same probe, but each parent's mean patch embedding (all patches of all its sites, test site included, labels never read) is subtracted before the per-fold scaler/PCA/LR (`--centre parent`, outputs in `outputs/patch_probe_centred/`, no ensemble).

| | acc | bal acc | macro-F1 | recall B1/B2/B3 | rubric all-high | rubric p_max>=0.5 | perm p |
|---|---|---|---|---|---|---|---|
| probe | 0.706 | 0.630 | 0.622 | 0.38/0.62/0.89 | 1.412 | 1.412 | 0.005 |
| probe_centred | 0.529 | 0.403 | 0.365 | 0.38/0.00/0.83 | 1.059 | 1.000 (0 high) | 0.109 |

Held-out LOPO calls (centred): fn0mhxef B1 correct (p 0.37/0.30/0.32), xrv9xvzb B3 correct (0.35/0.30/0.35), 3e122cbj B2 wrong -> B3 (0.32/0.31/0.37); near-uniform probabilities. Singleton parents iv6g2oq0 and uhdslk0o: 0/2 centred (both B3) vs 1/2 uncentred.

Verdict: parent centring hurts (acc 0.71 -> 0.53, not significant under permutation); the between-parent offset carries batch signal that the probe uses, so it is not adopted.
