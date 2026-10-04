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
