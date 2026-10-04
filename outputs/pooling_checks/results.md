# Tile-signal checks 1-2 (D10 pre-checks)

- Input: `outputs/kpis/tile_kpis.csv`
- Rows: 124; sites: 31 (4 tiles each); batches: Batch_1=7, Batch_2=7, Batch_3=17
- Features used (15): K01_si_frac_adm, K02_si_density_per_1000um2, K03_ecd_d50_um, K03_ecd_d90_um, K03_ecd_max_um, K04_agglom_frac, K04_n_clusters_per_1000um2, K05_voronoi_sigma, K07_R_rl, K07_R_csr, K09_mst_m_norm, K09_mst_sigma_norm, K14_empty_p50_um, K14_empty_p95_um, K15_si_graphite_contact_frac
- Features dropped (zero variance): ['K16_si_graphite_dist_median_um']
- Seeds: LogisticRegression random_state=0; figure jitter np.random.default_rng(0). Outputs are deterministic.

## Definitions

- Classifier: median impute, standard scale, LogisticRegression(C=1.0, class_weight=balanced, max_iter=5000), refitted per fold; leave-one-site-out (groups = site).
- Site category (DD7): `diffuse_correct` n_correct >= 3; `witness` n_correct in {1,2} and max tile p_true >= 0.7; `mixed` n_correct in {1,2} and max tile p_true < 0.7; `miss` n_correct == 0.
- False-alarm rate FA (DD8, binary variants): fraction of Batch_3 sites with any tile p(non-B3 class) >= 0.7.
- Batch verdict (DD9, binary, n_A = 7; W = #witness, D = #diffuse_correct), first match wins: `localised` if W >= 2 and W/n_A > FA; `diffuse` if D >= 4; `weak` if D + W <= 2; else `inconclusive`.
- ICC (DD11): one-way random-effects ICC(1), unbalanced-safe k0; bands (Koo & Li 2016): >= 0.5 site-level feature, < 0.5 tile-dominated. ss_batch/ss_site/ss_tile are nested sum-of-squares fractions summing to 1. sw_all = sqrt(MSW) pooled within-site SD; sw_ratio = sw_batch / sw_B3.
- Single-tile flag (DD12), per contrast A vs Batch_3: dmed, dmax, dmin = difference over sites of the median of site median / max / min, divided by sw_all; dext = larger-|.| of dmax, dmin. `tail_driven` if |dext| >= 1.0 and |dext| >= 2|dmed|; else `diffuse_shift` if |dmed| >= 1.0; else `none`.
- Overall reading (DD14): Localised = any binary verdict is `localised` OR >= 3 features flagged `tail_driven` for that contrast. Diffuse = verdict is `diffuse` and `diffuse_shift` flags outnumber `tail_driven` flags. Otherwise `mixed/inconclusive`.

## Check 1: LOSO tile classifier

### Batch summary

| variant | batch | n_sites | n_correct_0 | n_correct_1 | n_correct_2 | n_correct_3 | n_correct_4 | median_frac_correct | diffuse_correct | witness | mixed | miss | fa_rate | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multiclass | Batch_1 | 7 | 3 | 2 | 0 | 0 | 2 | 0.250 | 2 | 0 | 2 | 3 |  |  |
| multiclass | Batch_2 | 7 | 1 | 2 | 4 | 0 | 0 | 0.500 | 0 | 1 | 5 | 1 |  |  |
| multiclass | Batch_3 | 17 | 2 | 1 | 4 | 7 | 3 | 0.750 | 10 | 4 | 1 | 2 |  |  |
| B1_vs_B3 | Batch_1 | 7 | 1 | 1 | 1 | 2 | 2 | 0.750 | 4 | 0 | 2 | 1 | 0.294 | diffuse |
| B1_vs_B3 | Batch_3 | 17 | 1 | 0 | 2 | 10 | 4 | 0.750 | 14 | 1 | 1 | 1 |  |  |
| B2_vs_B3 | Batch_2 | 7 | 0 | 1 | 2 | 3 | 1 | 0.750 | 4 | 2 | 1 | 0 | 0.412 | diffuse |
| B2_vs_B3 | Batch_3 | 17 | 2 | 2 | 5 | 2 | 6 | 0.500 | 8 | 5 | 2 | 2 |  |  |

### Per-site spread

| variant | batch | site | n_correct | p_true_min | p_true_median | p_true_max | category |
|---|---|---|---|---|---|---|---|
| B1_vs_B3 | Batch_1 | 4ih2ggld | 4 | 0.929 | 0.964 | 0.981 | diffuse_correct |
| B1_vs_B3 | Batch_1 | 5n1q8atc | 4 | 0.546 | 0.844 | 0.876 | diffuse_correct |
| B1_vs_B3 | Batch_1 | f1vzngrs | 3 | 0.323 | 0.513 | 0.603 | diffuse_correct |
| B1_vs_B3 | Batch_1 | ffwubibz | 0 | 0.048 | 0.110 | 0.356 | miss |
| B1_vs_B3 | Batch_1 | fzrt2k6r | 1 | 0.104 | 0.217 | 0.593 | mixed |
| B1_vs_B3 | Batch_1 | iv6g2oq0 | 2 | 0.289 | 0.398 | 0.581 | mixed |
| B1_vs_B3 | Batch_1 | uhdslk0o | 3 | 0.107 | 0.742 | 0.782 | diffuse_correct |
| B1_vs_B3 | Batch_3 | 0grcilhi | 3 | 0.061 | 0.594 | 0.937 | diffuse_correct |
| B1_vs_B3 | Batch_3 | 71vgq3fw | 3 | 0.362 | 0.670 | 0.776 | diffuse_correct |
| B1_vs_B3 | Batch_3 | 9luzk4jm | 3 | 0.368 | 0.564 | 0.626 | diffuse_correct |
| B1_vs_B3 | Batch_3 | cfe5vt7s | 0 | 0.275 | 0.379 | 0.496 | miss |
| B1_vs_B3 | Batch_3 | hawkfj64 | 4 | 0.555 | 0.755 | 0.868 | diffuse_correct |
| B1_vs_B3 | Batch_3 | hzumfsms | 4 | 0.825 | 0.870 | 0.902 | diffuse_correct |
| B1_vs_B3 | Batch_3 | kbdh4tri | 3 | 0.400 | 0.624 | 0.948 | diffuse_correct |
| B1_vs_B3 | Batch_3 | mgxahqnk | 2 | 0.414 | 0.533 | 0.630 | mixed |
| B1_vs_B3 | Batch_3 | pl8uabbv | 3 | 0.252 | 0.612 | 0.760 | diffuse_correct |
| B1_vs_B3 | Batch_3 | ptg8lmto | 3 | 0.472 | 0.594 | 0.956 | diffuse_correct |
| B1_vs_B3 | Batch_3 | tuy3zymq | 3 | 0.403 | 0.788 | 0.952 | diffuse_correct |
| B1_vs_B3 | Batch_3 | ufdvpb81 | 3 | 0.235 | 0.607 | 0.693 | diffuse_correct |
| B1_vs_B3 | Batch_3 | utfgcjfa | 3 | 0.335 | 0.802 | 0.908 | diffuse_correct |
| B1_vs_B3 | Batch_3 | vc2whyaq | 4 | 0.684 | 0.745 | 0.868 | diffuse_correct |
| B1_vs_B3 | Batch_3 | x77cy643 | 4 | 0.572 | 0.737 | 0.778 | diffuse_correct |
| B1_vs_B3 | Batch_3 | x7u69zsw | 3 | 0.387 | 0.541 | 0.617 | diffuse_correct |
| B1_vs_B3 | Batch_3 | xgj4xftb | 2 | 0.068 | 0.362 | 0.848 | witness |
| B2_vs_B3 | Batch_2 | 3806gxp0 | 3 | 0.488 | 0.678 | 0.750 | diffuse_correct |
| B2_vs_B3 | Batch_2 | avn74qx1 | 4 | 0.587 | 0.706 | 0.817 | diffuse_correct |
| B2_vs_B3 | Batch_2 | b3esycq1 | 2 | 0.419 | 0.636 | 0.885 | witness |
| B2_vs_B3 | Batch_2 | epqdaau9 | 3 | 0.049 | 0.707 | 0.820 | diffuse_correct |
| B2_vs_B3 | Batch_2 | i9jiqjwl | 3 | 0.074 | 0.671 | 0.708 | diffuse_correct |
| B2_vs_B3 | Batch_2 | r17byphk | 2 | 0.221 | 0.421 | 0.709 | witness |
| B2_vs_B3 | Batch_2 | rxax5ozo | 1 | 0.120 | 0.272 | 0.524 | mixed |
| B2_vs_B3 | Batch_3 | 0grcilhi | 2 | 0.071 | 0.509 | 0.897 | witness |
| B2_vs_B3 | Batch_3 | 71vgq3fw | 4 | 0.557 | 0.712 | 0.861 | diffuse_correct |
| B2_vs_B3 | Batch_3 | 9luzk4jm | 2 | 0.366 | 0.516 | 0.597 | mixed |
| B2_vs_B3 | Batch_3 | cfe5vt7s | 0 | 0.118 | 0.306 | 0.452 | miss |
| B2_vs_B3 | Batch_3 | hawkfj64 | 4 | 0.626 | 0.713 | 0.850 | diffuse_correct |
| B2_vs_B3 | Batch_3 | hzumfsms | 4 | 0.654 | 0.823 | 0.899 | diffuse_correct |
| B2_vs_B3 | Batch_3 | kbdh4tri | 4 | 0.526 | 0.858 | 0.975 | diffuse_correct |
| B2_vs_B3 | Batch_3 | mgxahqnk | 0 | 0.276 | 0.309 | 0.426 | miss |
| B2_vs_B3 | Batch_3 | pl8uabbv | 2 | 0.221 | 0.557 | 0.870 | witness |
| B2_vs_B3 | Batch_3 | ptg8lmto | 4 | 0.516 | 0.660 | 0.969 | diffuse_correct |
| B2_vs_B3 | Batch_3 | tuy3zymq | 4 | 0.819 | 0.956 | 0.991 | diffuse_correct |
| B2_vs_B3 | Batch_3 | ufdvpb81 | 1 | 0.060 | 0.384 | 0.938 | witness |
| B2_vs_B3 | Batch_3 | utfgcjfa | 2 | 0.442 | 0.595 | 0.753 | witness |
| B2_vs_B3 | Batch_3 | vc2whyaq | 2 | 0.460 | 0.554 | 0.947 | witness |
| B2_vs_B3 | Batch_3 | x77cy643 | 3 | 0.263 | 0.692 | 0.867 | diffuse_correct |
| B2_vs_B3 | Batch_3 | x7u69zsw | 3 | 0.451 | 0.573 | 0.783 | diffuse_correct |
| B2_vs_B3 | Batch_3 | xgj4xftb | 1 | 0.269 | 0.451 | 0.675 | mixed |
| multiclass | Batch_1 | 4ih2ggld | 4 | 0.888 | 0.929 | 0.945 | diffuse_correct |
| multiclass | Batch_1 | 5n1q8atc | 4 | 0.649 | 0.734 | 0.782 | diffuse_correct |
| multiclass | Batch_1 | f1vzngrs | 0 | 0.225 | 0.290 | 0.312 | miss |
| multiclass | Batch_1 | ffwubibz | 0 | 0.085 | 0.111 | 0.217 | miss |
| multiclass | Batch_1 | fzrt2k6r | 1 | 0.041 | 0.129 | 0.377 | mixed |
| multiclass | Batch_1 | iv6g2oq0 | 0 | 0.072 | 0.110 | 0.144 | miss |
| multiclass | Batch_1 | uhdslk0o | 1 | 0.108 | 0.212 | 0.522 | mixed |
| multiclass | Batch_2 | 3806gxp0 | 2 | 0.370 | 0.436 | 0.738 | witness |
| multiclass | Batch_2 | avn74qx1 | 2 | 0.171 | 0.393 | 0.522 | mixed |
| multiclass | Batch_2 | b3esycq1 | 1 | 0.254 | 0.335 | 0.495 | mixed |
| multiclass | Batch_2 | epqdaau9 | 2 | 0.036 | 0.379 | 0.692 | mixed |
| multiclass | Batch_2 | i9jiqjwl | 2 | 0.049 | 0.398 | 0.471 | mixed |
| multiclass | Batch_2 | r17byphk | 1 | 0.231 | 0.362 | 0.599 | mixed |
| multiclass | Batch_2 | rxax5ozo | 0 | 0.057 | 0.217 | 0.330 | miss |
| multiclass | Batch_3 | 0grcilhi | 2 | 0.032 | 0.367 | 0.822 | witness |
| multiclass | Batch_3 | 71vgq3fw | 3 | 0.251 | 0.497 | 0.673 | diffuse_correct |
| multiclass | Batch_3 | 9luzk4jm | 3 | 0.278 | 0.423 | 0.537 | diffuse_correct |
| multiclass | Batch_3 | cfe5vt7s | 0 | 0.098 | 0.165 | 0.221 | miss |
| multiclass | Batch_3 | hawkfj64 | 4 | 0.471 | 0.622 | 0.768 | diffuse_correct |
| multiclass | Batch_3 | hzumfsms | 4 | 0.606 | 0.733 | 0.799 | diffuse_correct |
| multiclass | Batch_3 | kbdh4tri | 3 | 0.311 | 0.564 | 0.941 | diffuse_correct |
| multiclass | Batch_3 | mgxahqnk | 0 | 0.168 | 0.294 | 0.320 | miss |
| multiclass | Batch_3 | pl8uabbv | 2 | 0.099 | 0.443 | 0.767 | witness |
| multiclass | Batch_3 | ptg8lmto | 3 | 0.311 | 0.431 | 0.955 | diffuse_correct |
| multiclass | Batch_3 | tuy3zymq | 4 | 0.417 | 0.805 | 0.960 | diffuse_correct |
| multiclass | Batch_3 | ufdvpb81 | 1 | 0.044 | 0.292 | 0.736 | witness |
| multiclass | Batch_3 | utfgcjfa | 3 | 0.176 | 0.555 | 0.677 | diffuse_correct |
| multiclass | Batch_3 | vc2whyaq | 2 | 0.341 | 0.471 | 0.893 | witness |
| multiclass | Batch_3 | x77cy643 | 3 | 0.231 | 0.623 | 0.790 | diffuse_correct |
| multiclass | Batch_3 | x7u69zsw | 3 | 0.285 | 0.404 | 0.555 | diffuse_correct |
| multiclass | Batch_3 | xgj4xftb | 2 | 0.058 | 0.208 | 0.535 | mixed |

### Pooling side note (side note; see check 3)

| variant | n_sites | tile_balanced_acc | site_acc_mean_pool | site_acc_max_pool |
|---|---|---|---|---|
| multiclass | 31 | 0.444 | 0.581 | 0.548 |
| B1_vs_B3 | 24 | 0.671 | 0.750 | 0.417 |
| B2_vs_B3 | 24 | 0.630 | 0.667 | 0.542 |

## Check 2: variance decomposition and single-tile flags

| feature | icc_all | icc_B1 | icc_B2 | icc_B3 | ss_batch | ss_site | ss_tile | sw_all | sw_ratio_B1 | sw_ratio_B2 | flag_B1 | flag_B2 | dmed_B1 | dext_B1 | dmed_B2 | dext_B2 | range_sw_med_B1 | range_sw_med_B2 | range_sw_med_B3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K01_si_frac_adm | 0.549 | 0.748 | 0.083 | 0.032 | 0.139 | 0.515 | 0.346 | 0.021 | 1.421 | 1.232 | none | none | 0.794 | 0.360 | -0.021 | -0.331 | 2.153 | 2.152 | 1.634 |
| K02_si_density_per_1000um2 | 0.928 | 0.970 | 0.415 | 0.743 | 0.159 | 0.785 | 0.056 | 5.792 | 1.260 | 0.952 | none | none | -0.096 | 0.522 | -0.221 | -0.179 | 1.748 | 1.183 | 1.817 |
| K03_ecd_d50_um | 0.419 | 0.673 | -0.024 | 0.443 | 0.038 | 0.518 | 0.444 | 0.188 | 0.443 | 0.839 | none | none | -0.682 | -0.564 | -0.438 | -0.143 | 0.855 | 1.686 | 1.454 |
| K03_ecd_d90_um | 0.285 | 0.548 | 0.035 | 0.170 | 0.010 | 0.445 | 0.544 | 0.664 | 1.266 | 1.245 | none | none | 0.363 | 0.186 | 0.447 | -0.186 | 2.116 | 2.237 | 1.794 |
| K03_ecd_max_um | 0.518 | 0.790 | -0.242 | 0.038 | 0.103 | 0.528 | 0.369 | 1.098 | 1.084 | 0.909 | none | none | -0.039 | 0.287 | 0.198 | -0.136 | 1.884 | 1.966 | 2.155 |
| K04_agglom_frac | 0.165 | 0.296 | -0.237 | 0.157 | 0.044 | 0.323 | 0.634 | 0.199 | 1.058 | 0.929 | tail_driven | tail_driven | 0.327 | 1.061 | 0.224 | 1.118 | 2.191 | 1.775 | 2.137 |
| K04_n_clusters_per_1000um2 | 0.500 | 0.028 | 0.134 | 0.663 | 0.043 | 0.574 | 0.383 | 39.381 | 1.233 | 1.128 | none | none | -0.664 | 0.501 | -0.594 | -0.822 | 2.122 | 2.363 | 1.779 |
| K05_voronoi_sigma | 0.420 | 0.583 | 0.237 | 0.402 | 0.035 | 0.522 | 0.443 | 0.201 | 0.984 | 1.083 | none | none | 0.320 | 0.617 | -0.001 | -0.463 | 1.936 | 1.845 | 1.890 |
| K07_R_rl | 0.038 | 0.235 | -0.095 | 0.011 | 0.033 | 0.238 | 0.728 | 0.123 | 0.832 | 1.236 | none | none | 0.755 | 0.427 | 0.139 | -0.318 | 1.720 | 2.307 | 2.062 |
| K07_R_csr | -0.102 | 0.177 | -0.212 | -0.145 | 0.016 | 0.152 | 0.831 | 0.094 | 0.656 | 0.905 | none | none | 0.570 | 0.694 | 0.293 | 0.463 | 1.582 | 1.573 | 2.281 |
| K09_mst_m_norm | 0.405 | 0.816 | 0.189 | -0.007 | 0.089 | 0.456 | 0.455 | 0.214 | 0.677 | 0.808 | none | none | -0.487 | -0.646 | 0.074 | 0.246 | 1.565 | 1.892 | 2.066 |
| K09_mst_sigma_norm | 0.300 | 0.723 | -0.127 | 0.069 | 0.096 | 0.371 | 0.533 | 0.211 | 0.756 | 1.190 | none | none | -0.352 | -0.550 | -0.029 | -0.314 | 1.605 | 1.938 | 1.797 |
| K14_empty_p50_um | 0.185 | 0.148 | 0.391 | -0.025 | 0.029 | 0.351 | 0.619 | 0.359 | 0.821 | 1.285 | none | tail_driven | 0.000 | -0.197 | 0.000 | 1.183 | 0.197 | 1.577 | 0.394 |
| K14_empty_p95_um | 0.520 | 0.808 | 0.244 | 0.420 | 0.095 | 0.537 | 0.368 | 1.462 | 0.680 | 1.168 | none | none | -0.316 | -0.656 | 0.191 | 0.657 | 1.016 | 2.851 | 1.495 |
| K15_si_graphite_contact_frac | 0.617 | 0.857 | 0.189 | 0.338 | 0.184 | 0.522 | 0.294 | 0.037 | 1.000 | 1.656 | diffuse_shift | tail_driven | -1.287 | -1.308 | -0.686 | -1.630 | 1.781 | 2.576 | 1.618 |

## Overall reading

B1 vs B3: mixed/inconclusive

B2 vs B3: localised

## Caveats

- n = 7 sites per non-B3 batch, so the counts are descriptive and not significance tests.
- SE-detector confound: 1 Batch_2 site and 3 Batch_3 sites use SE rather than ETD; not adjusted for.
- The KPIs come from the provisional v0 segmenter (`segmenter_version` v0r1).
