# Feature hunt: can one scalar + two cutpoints reproduce the 34 batch labels?

246 scalar features, 3 batch orderings, 34 sites (31 labelled + 3 released truths), majority-class accuracy 0.529. p is a max-statistic permutation p (1000 label permutations, maximum over all features and orderings each time), so it already pays for the search.
`wp` = within-parent pairs of sites with different batches whose feature ordering agrees with the batch ordering.

## raw

Null (max over features) : median 0.706, 95th pct 0.765, max 0.794.
Best observed: 0.794 (SE_type_D, B1<B2<B3), p = 0.009.

| feature | order | acc | p_maxstat | wp_concordant | wp_pairs | cut1 | cut2 |
|---|---|---|---|---|---|---|---|
| SE_type_D | B1<B2<B3 | 0.794 | 0.009 | 12 | 13 | -8.84 | 0.546 |
| grey_harm_pore_mean | B1<B2<B3 | 0.765 | 0.061 | 9 | 13 | 3.53 | 4.55 |
| BSE_D | B1<B2<B3 | 0.765 | 0.061 | 12 | 13 | 0.936 | 8.01 |
| SE_type_si_graphite | B1<B2<B3 | 0.735 | 0.283 | 10 | 13 | 1.64 | 1.64 |
| raw_Inlens_p1 | B1<B2<B3 | 0.735 | 0.283 | 5 | 13 | 6 | 7 |
| grey_harm_pore_mean | B2<B1<B3 | 0.735 | 0.283 | 6 | 13 | 3.04 | 4.51 |
| BSE_D | B1<B3<B2 | 0.706 | 0.749 | 9 | 13 | 6.71 | 32.7 |
| BSE_noise_beta | B1<B2<B3 | 0.706 | 0.749 | 10 | 13 | 0.00968 | 0.0111 |
| BSE_D_frac_zero | B1<B3<B2 | 0.706 | 0.749 | 4 | 13 | -inf | 0.188 |
| grey_harm_pore_mean | B1<B3<B2 | 0.706 | 0.749 | 8 | 13 | 4.51 | 7.8 |
| grey_harm_pore_iqr | B2<B1<B3 | 0.706 | 0.749 | 2 | 13 | 4 | 7 |
| raw_Inlens_p1 | B2<B1<B3 | 0.706 | 0.749 | 5 | 13 | -inf | 6 |
| grey_depth_raw_b8 | B2<B1<B3 | 0.706 | 0.749 | 8 | 13 | 52.8 | 54.1 |
| BSE_si_graphite | B1<B2<B3 | 0.706 | 0.749 | 9 | 13 | 2.19 | 2.43 |
| SE_type_si_graphite | B1<B3<B2 | 0.706 | 0.749 | 11 | 13 | 1.66 | 1.9 |
| grey_harm_pore_std | B1<B2<B3 | 0.706 | 0.749 | 6 | 13 | 4.85 | 5.17 |
| SE_type_si_graphite | B2<B1<B3 | 0.706 | 0.749 | 3 | 13 | -inf | 1.66 |
| SE_type_D | B1<B3<B2 | 0.706 | 0.749 | 9 | 13 | -0.0924 | 23.8 |
| SE_type_D | B2<B1<B3 | 0.706 | 0.749 | 5 | 13 | -inf | -0.0924 |
| grey_raw_p1 | B2<B1<B3 | 0.706 | 0.749 | 0 | 13 | -inf | 0 |

### raw: ranked by within-parent concordance (the split that was actually designed)

| feature | order | acc | p_maxstat | wp_concordant | wp_pairs | cut1 | cut2 |
|---|---|---|---|---|---|---|---|
| raw_Inlens_frac_255 | B1<B2<B3 | 0.559 | 1.000 | 13 | 13 | 0.0186 | 0.0327 |
| Inlens_mask_clip_high | B1<B2<B3 | 0.559 | 1.000 | 13 | 13 | 0.0186 | 0.0327 |
| SE_type_D | B1<B2<B3 | 0.794 | 0.009 | 12 | 13 | -8.84 | 0.546 |
| BSE_D | B1<B2<B3 | 0.765 | 0.061 | 12 | 13 | 0.936 | 8.01 |
| overlay_depth_si_b10 | B1<B2<B3 | 0.618 | 1.000 | 12 | 13 | 0.415 | 0.415 |
| SE_type_si_graphite | B1<B3<B2 | 0.706 | 0.749 | 11 | 13 | 1.66 | 1.9 |
| BSE_noise_sigma_g | B1<B2<B3 | 0.706 | 0.749 | 11 | 13 | 0.215 | 0.217 |
| porosity | B1<B2<B3 | 0.676 | 0.993 | 11 | 13 | 0.0363 | 0.0456 |
| si_depth_mid_dip | B2<B1<B3 | 0.676 | 0.993 | 11 | 13 | -0.557 | -0.489 |
| overlay_depth_si_b7 | B2<B1<B3 | 0.618 | 1.000 | 11 | 13 | 0.424 | 0.424 |
| overlay_depth_si_b0 | B1<B3<B2 | 0.618 | 1.000 | 11 | 13 | 0.463 | 2.02 |
| F02_soc050_pore_loss | B2<B1<B3 | 0.559 | 1.000 | 11 | 13 | 0.0359 | 0.0359 |
| gx_4.0_7.0 | B1<B3<B2 | 0.559 | 1.000 | 11 | 13 | 0.75 | 1.34 |
| raw_Inlens_std | B1<B2<B3 | 0.559 | 1.000 | 11 | 13 | 49.5 | 52.9 |
| overlay_depth_graphite_b8 | B1<B2<B3 | 0.559 | 1.000 | 11 | 13 | 0.929 | 0.929 |
| SE_type_noise_alpha | B1<B3<B2 | 0.529 | 1.000 | 11 | 13 | -inf | 0.0125 |
| BSE_noise_alpha | B1<B3<B2 | 0.529 | 1.000 | 11 | 13 | -inf | 0.0456 |
| Inlens_G_ptp_rel | B1<B2<B3 | 0.529 | 1.000 | 11 | 13 | -inf | -inf |
| SE_type_noise_sigma_g | B1<B3<B2 | 0.529 | 1.000 | 11 | 13 | -inf | 0.112 |
| SE_type_si_graphite | B1<B2<B3 | 0.735 | 0.283 | 10 | 13 | 1.64 | 1.64 |

## parent_centred

Null (max over features) : median 0.706, 95th pct 0.765, max 0.824.
Best observed: 0.706 (BSE_noise_alpha, B1<B3<B2), p = 0.774.

| feature | order | acc | p_maxstat | wp_concordant | wp_pairs | cut1 | cut2 |
|---|---|---|---|---|---|---|---|
| BSE_noise_alpha | B1<B3<B2 | 0.706 | 0.774 | 11 | 13 | -0.00168 | 0.00111 |
| overlay_binder | B1<B3<B2 | 0.706 | 0.774 | 7 | 13 | -0.00181 | 0.00103 |
| F02_soc025_pore_frac | B1<B3<B2 | 0.676 | 0.986 | 7 | 13 | -0.00354 | 0.00309 |
| D04_porosity_mean | B1<B3<B2 | 0.676 | 0.986 | 7 | 13 | -0.00341 | 0.00288 |
| F01_pore_frac | B1<B3<B2 | 0.676 | 0.986 | 7 | 13 | -0.00341 | 0.00288 |
| overlay_pore | B1<B3<B2 | 0.676 | 0.986 | 7 | 13 | -0.00341 | 0.00288 |
| grey_depth_harm_b10 | B1<B3<B2 | 0.647 | 1.000 | 10 | 13 | -2.58 | 0.225 |
| si_depth_rel_band3 | B1<B3<B2 | 0.647 | 1.000 | 7 | 13 | -0.317 | 0.353 |
| si_depth_rel_band3 | B2<B1<B3 | 0.647 | 1.000 | 7 | 13 | -0.518 | -0.317 |
| overlay_depth_graphite_b2 | B1<B3<B2 | 0.647 | 1.000 | 6 | 13 | -0.0367 | 0.0489 |
| overlay_depth_graphite_b2 | B2<B1<B3 | 0.647 | 1.000 | 8 | 13 | -0.0422 | -0.0367 |
| overlay_pore | B2<B1<B3 | 0.647 | 1.000 | 7 | 13 | -inf | -0.00341 |
| F02_soc025_pore_frac | B1<B2<B3 | 0.647 | 1.000 | 10 | 13 | -0.00354 | -0.00354 |
| SE_type_D_se | B2<B1<B3 | 0.647 | 1.000 | 8 | 13 | -0.0028 | -0.00147 |
| F02_soc100_pore_frac | B1<B2<B3 | 0.647 | 1.000 | 9 | 13 | -0.00344 | -0.00165 |
| F02_soc025_pore_frac | B2<B1<B3 | 0.647 | 1.000 | 7 | 13 | -inf | -0.00354 |
| overlay_pore | B1<B2<B3 | 0.647 | 1.000 | 10 | 13 | -0.00341 | -0.00341 |
| D04_porosity_mean | B1<B2<B3 | 0.647 | 1.000 | 10 | 13 | -0.00341 | -0.00341 |
| SE_type_mask_clip_low | B1<B2<B3 | 0.647 | 1.000 | 5 | 13 | -0.0095 | -0.000965 |
| D01_graphite_band_maxdev | B1<B3<B2 | 0.647 | 1.000 | 9 | 13 | -0.0303 | 0.0174 |

### parent_centred: ranked by within-parent concordance (the split that was actually designed)

| feature | order | acc | p_maxstat | wp_concordant | wp_pairs | cut1 | cut2 |
|---|---|---|---|---|---|---|---|
| raw_Inlens_frac_255 | B1<B2<B3 | 0.588 | 1.000 | 13 | 13 | -0.00137 | -0.00137 |
| Inlens_mask_clip_high | B1<B2<B3 | 0.588 | 1.000 | 13 | 13 | -0.00136 | -0.00136 |
| SE_type_D | B1<B2<B3 | 0.618 | 1.000 | 12 | 13 | -0.824 | -0.824 |
| overlay_depth_si_b10 | B1<B2<B3 | 0.618 | 1.000 | 12 | 13 | -0.616 | -0.602 |
| BSE_D | B1<B2<B3 | 0.588 | 1.000 | 12 | 13 | -1.32 | -1.32 |
| BSE_noise_alpha | B1<B3<B2 | 0.706 | 0.774 | 11 | 13 | -0.00168 | 0.00111 |
| overlay_depth_graphite_b8 | B1<B2<B3 | 0.618 | 1.000 | 11 | 13 | -0.0353 | -0.0353 |
| overlay_depth_si_b7 | B2<B1<B3 | 0.618 | 1.000 | 11 | 13 | -0.142 | 0.0104 |
| SE_type_noise_sigma_g | B1<B3<B2 | 0.618 | 1.000 | 11 | 13 | -0.00499 | 0.00218 |
| SE_type_noise_alpha | B1<B3<B2 | 0.588 | 1.000 | 11 | 13 | -0.00107 | 0.000973 |
| SE_type_si_graphite | B1<B3<B2 | 0.588 | 1.000 | 11 | 13 | -0.0561 | 0.0163 |
| Inlens_G_ptp_rel | B1<B2<B3 | 0.588 | 1.000 | 11 | 13 | -0.513 | -0.242 |
| BSE_noise_sigma_g | B1<B2<B3 | 0.588 | 1.000 | 11 | 13 | -0.00787 | -0.00787 |
| raw_Inlens_std | B1<B2<B3 | 0.559 | 1.000 | 11 | 13 | -7.46 | -7.46 |
| si_depth_mid_dip | B2<B1<B3 | 0.559 | 1.000 | 11 | 13 | -0.289 | -0.289 |
| overlay_depth_si_b0 | B1<B3<B2 | 0.559 | 1.000 | 11 | 13 | -inf | 0.507 |
| porosity | B1<B2<B3 | 0.559 | 1.000 | 11 | 13 | -0.00953 | -0.00702 |
| F02_soc050_pore_loss | B2<B1<B3 | 0.559 | 1.000 | 11 | 13 | -0.0279 | -0.0279 |
| gx_4.0_7.0 | B1<B3<B2 | 0.529 | 1.000 | 11 | 13 | -inf | 0.192 |
| grey_depth_harm_b10 | B1<B3<B2 | 0.647 | 1.000 | 10 | 13 | -2.58 | 0.225 |

## within-parent concordance (the designed split: labels permuted within parents)

13 within-parent pairs with different batches. Statistic = fraction of those pairs whose feature ordering matches the batch ordering; null = labels shuffled within each parent, max over all features and orderings.
Null: median 0.923, 95th pct 1.000, max 1.000. Best observed: 1.000 (Inlens_mask_clip_high, B1<B2<B3), p = 0.232.

| feature | order | wp_frac | p_maxstat |
|---|---|---|---|
| Inlens_mask_clip_high | B1<B2<B3 | 1.000 | 0.232 |
| raw_Inlens_frac_255 | B1<B2<B3 | 1.000 | 0.232 |
| BSE_D | B1<B2<B3 | 0.923 | 0.909 |
| SE_type_D | B1<B2<B3 | 0.923 | 0.909 |
| overlay_depth_si_b10 | B1<B2<B3 | 0.923 | 0.909 |
| gx_4.0_7.0 | B1<B3<B2 | 0.846 | 1.000 |
| BSE_noise_alpha | B1<B3<B2 | 0.846 | 1.000 |
| si_depth_mid_dip | B2<B1<B3 | 0.846 | 1.000 |
| overlay_depth_graphite_b8 | B1<B2<B3 | 0.846 | 1.000 |
| BSE_noise_sigma_g | B1<B2<B3 | 0.846 | 1.000 |
| Inlens_G_ptp_rel | B1<B2<B3 | 0.846 | 1.000 |
| raw_Inlens_std | B1<B2<B3 | 0.846 | 1.000 |
| F02_soc050_pore_loss | B2<B1<B3 | 0.846 | 1.000 |
| overlay_depth_si_b0 | B1<B3<B2 | 0.846 | 1.000 |
| porosity | B1<B2<B3 | 0.846 | 1.000 |
| overlay_depth_si_b7 | B2<B1<B3 | 0.846 | 1.000 |
| SE_type_si_graphite | B1<B3<B2 | 0.846 | 1.000 |
| SE_type_noise_alpha | B1<B3<B2 | 0.846 | 1.000 |
| SE_type_noise_sigma_g | B1<B3<B2 | 0.846 | 1.000 |
| Inlens_grey_levels | B2<B1<B3 | 0.808 | 1.000 |
| F02_soc050_into_graphite | B1<B2<B3 | 0.769 | 1.000 |
| F01_pore_clusters_per_1000um2 | B1<B2<B3 | 0.769 | 1.000 |
| raw_BSE_p99 | B2<B1<B3 | 0.769 | 1.000 |
| overlay_si | B2<B1<B3 | 0.769 | 1.000 |
| largest_pore_um2 | B1<B2<B3 | 0.769 | 1.000 |

