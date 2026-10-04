# Shift (non-intensity) harmonisation – results

Methods compared on the `pmdb.clean`-masked half-res grey (`none` is the raw grey at the same coordinates):

* **spectrum** = radial amplitude-spectrum (MTF/NPS) matching filter to the labelled median, DC kept (`pmdb/harmonise_shift.py`, `docs/harmonisation_shift.md`)
* **fda** = Fourier Domain Adaptation, low-frequency amplitude window (β = 0.01) from the labelled reference (`pmdb/harmonise_shift.py`)
* **hybrid** = in-house LUT route (`pmdb/harmonise.py`), for reference

## Summary (labelled sites)

| metric | none | spectrum | fda | hybrid |
|---|---|---|---|---|
| black_gap_strong_BSE | 22.000 | 11.596 | 11.712 | 1.231 |
| black_gap_strong_Inlens | 1.077 | 1.788 | 36.731 | -5.692 |
| black_gap_strong_SE_type | 19.308 | 8.173 | 15.231 | 0.308 |
| iqr_gap_strong_BSE | -9.327 | -2.058 | -6.558 | 3.942 |
| black_sd_BSE | 8.520 | 5.721 | 5.988 | 4.102 |
| graphite_sd_BSE | 3.638 | 3.612 | 1.340 | 0.180 |
| contrast_rel_change_clean | 0.000 | 0.016 | 0.041 | 0.003 |
| contrast_ratio_sd_all | 0.199 | 0.197 | 0.166 | 0.196 |
| fixed_fsi_gap_strong | 0.016 | 0.030 | 0.002 | 0.026 |
| fixed_fpore_gap_strong | -0.032 | -0.028 | -0.023 | -0.009 |
| fixed_vs_seg_fsi_r | 0.941 | 0.896 | 0.924 | 0.910 |
| seg_fsi_sd_all | 0.026 | 0.025 | 0.024 | 0.025 |
| clip0_BSE | 0.008 | 0.006 | 0.005 | 0.010 |
| clip255_BSE | 0.000 | 0.000 | 0.000 | 0.000 |
| mean_abs_change_clean | 0.000 | 1.167 | 4.616 | 3.512 |
| mean_abs_change_strong | 0.000 | 2.674 | 6.764 | 6.756 |
| shortcut_batch_acc | 0.613 | 0.645 | 0.677 | 0.548 |
| shortcut_batch3_recall | 0.882 | 0.882 | 0.882 | 0.882 |
| shortcut_strong_vs_rest_b3 | 1.000 | 1.000 | 1.000 | 1.000 |
| hf_ratio_gap_strong_BSE | -0.064 | -0.002 | -0.066 | -0.060 |
| hf_ratio_cv_BSE | 0.111 | 0.007 | 0.111 | 0.110 |
| noise_sigma_cv_BSE | 0.153 | 0.030 | 0.153 | 0.125 |
| edge_sigma_sd_BSE | 0.024 | 0.001 | 0.024 | 0.023 |
| hf_ratio_gap_strong_Inlens | -0.409 | -0.024 | -0.415 | -0.162 |
| hf_ratio_cv_Inlens | 0.196 | 0.034 | 0.198 | 0.159 |
| noise_sigma_cv_Inlens | 0.235 | 0.166 | 0.232 | 0.234 |
| texture_shortcut_batch_acc | 0.677 | 0.677 | 0.677 | 0.645 |
| texture_shortcut_strong_vs_rest_all | 1.000 | 0.839 | 1.000 | 0.968 |
| texture_shortcut_batch3_recall | 0.941 | 0.882 | 0.941 | 0.882 |
| all_shortcut_batch_acc | 0.645 | 0.677 | 0.677 | 0.548 |

* `black_gap_strong_*`: mean 1st percentile of the 4 strong Batch-3 sites minus the other Batch-3 sites (grey levels; 0 = artefact removed).
* `iqr_gap_strong_BSE`: p90−p10 width difference strong vs rest (gain artefact; BaSiC cannot correct gain).
* `contrast_rel_change_clean`: relative change of the Si/graphite contrast ratio on Batch 1/2 sites (material preservation; 0 = untouched).
* `fixed_fsi_gap_strong`, `seg_fsi_sd_all`: Si fraction at a fixed threshold (set on each method's own grey scale from the Batch 1/2 anchors) /
  with the segmenter. Nyúl matches 11 landmarks per site, which by construction forces equal percentile positions and so pulls phase
  fractions towards a common value (the known limitation of histogram standardisation).
* `mean_abs_change_*`: |method − raw| per pixel in stored uint8 units; for `nyul` (standard scale) and `basic` (raw − bᵢ + 64) this includes the scale change itself.
* `shortcut_*`: leave-one-out logistic regression on grey statistics only; `_acc`/`_b3` = accuracy (chance 0.45 batch, 0.76 strong-vs-rest), `batch3_recall` = fraction of Batch-3 sites predicted Batch 3.
* `hf_ratio_gap_strong_*`: relative high-frequency (0.35–0.5 c/px) / low-frequency amplitude ratio of the strong sites vs the rest (0 = texture shift removed);
  `hf_ratio_cv_*`, `noise_sigma_cv_*`, `edge_sigma_sd_*`: across-site spread of texture, noise and blur width (lower = more homogeneous).
* `texture_shortcut_*`: the same leave-one-out classifier on the four texture statistics only (is the site still identifiable from blur/noise?); `all_shortcut_batch_acc` uses grey + texture.

## What is cut out ('crops')

No field contains a Cu collector or the coating free surface (`collector_found`/`free_surface_found` are False on all 34 sites), so
nothing is cropped for those reasons; `crops_gallery.png` shows the fields where the mask excludes more than 0.2 % of the interior and why

| batch | site | detector | interior excluded | border |
|---|---|---|---|---|
| Batch_3 | x77cy643 | Inlens | 9.82 % | 1.0 % |
| Batch_3 | xgj4xftb | Inlens | 9.47 % | 1.2 % |
| Batch_3 | hawkfj64 | Inlens | 8.79 % | 1.1 % |
| Batch_3 | mgxahqnk | Inlens | 8.27 % | 1.1 % |
| Batch_1 | uhdslk0o | Inlens | 7.74 % | 1.1 % |
| Batch_3 | 0grcilhi | Inlens | 7.73 % | 1.1 % |
| Batch_3 | vc2whyaq | Inlens | 7.09 % | 1.0 % |
| Batch_3 | pl8uabbv | Inlens | 6.96 % | 0.9 % |
| Batch_1 | 4ih2ggld | SE_type | 6.85 % | 0.9 % |
| Batch_2 | rxax5ozo | Inlens | 6.82 % | 1.0 % |
| Batch_2 | b3esycq1 | Inlens | 6.72 % | 1.0 % |
| Batch_2 | epqdaau9 | Inlens | 6.43 % | 1.0 % |

## Held-out (models fitted on labelled sites only)

* none: BSE p1 = 3e122cbj 0.0, fn0mhxef 0.0, xrv9xvzb 7.0; BSE graphite anchor = 56, 52, 60
* spectrum: BSE p1 = 3e122cbj 4.0, fn0mhxef 0.0, xrv9xvzb 0.0; BSE graphite anchor = 56, 53, 60
* fda: BSE p1 = 3e122cbj 0.0, fn0mhxef 4.0, xrv9xvzb 7.0; BSE graphite anchor = 56, 57, 59

## Figures

`effects_BSE.png`, `effects_Inlens.png`, `effects_SE_type.png` (none | spectrum | fda for every site, mask overlaid),
`crops_gallery.png`, `hist_by_method.png`, `black_level_by_method.png`, `fractions_by_method.png`.
