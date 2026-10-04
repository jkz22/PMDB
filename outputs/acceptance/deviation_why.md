# Why is this field off? (vs Batch 3 baseline, noise-aware)

Flag `off` = |robust z| > 2 vs Batch 3 *and* deviation > 2 × the SD of a 16-tile site mean (`outputs/reliability/icc.csv`); `within_sampling_noise` = |z| > 2 but not resolvable at this field size; `off_noise_unknown` = no tile-level noise estimate for that KPI. Batch 3 fields are scored leave-one-out.

| field | batch | n off | n unresolved | off columns (z) |
|---|---|---|---|---|
| Batch_1/4ih2ggld | Batch_1 | 9 | 2 | K01_si_frac_adm (+11.7), K02_si_density_per_1000um2 (+9.2), K03_ecd_max_um (+7.0), K09_mst_sigma_norm (-6.1), K15_si_graphite_contact_frac (-5.8) |
| Batch_1/5n1q8atc | Batch_1 | 10 | 0 | K01_si_frac_adm (+14.0), K03_ecd_max_um (+10.8), K02_si_density_per_1000um2 (+7.8), K09_mst_m_norm (-5.8), K09_mst_sigma_norm (-5.6) |
| Batch_1/f1vzngrs | Batch_1 | 2 | 1 | K07_R_rl (+3.8), K03_ecd_d90_um (+2.3) |
| Batch_1/ffwubibz | Batch_1 | 0 | 1 |  |
| Batch_1/fzrt2k6r | Batch_1 | 1 | 0 | K03_ecd_max_um (+2.1) |
| Batch_1/iv6g2oq0 | Batch_1 | 0 | 1 |  |
| Batch_1/uhdslk0o | Batch_1 | 0 | 1 |  |
| Batch_2/3806gxp0 | Batch_2 | 1 | 0 | K03_ecd_d90_um (+2.7) |
| Batch_2/avn74qx1 | Batch_2 | 1 | 1 | K07_R_rl (-3.2) |
| Batch_2/b3esycq1 | Batch_2 | 1 | 0 | K01_si_frac_adm (+3.2) |
| Batch_2/epqdaau9 | Batch_2 | 0 | 0 |  |
| Batch_2/i9jiqjwl | Batch_2 | 2 | 1 | K15_si_graphite_contact_frac (-2.3), K03_ecd_max_um (+2.1) |
| Batch_2/r17byphk | Batch_2 | 0 | 0 |  |
| Batch_2/rxax5ozo | Batch_2 | 0 | 0 |  |
| Batch_3/0grcilhi | Batch_3 | 0 | 0 |  |
| Batch_3/71vgq3fw | Batch_3 | 0 | 0 |  |
| Batch_3/9luzk4jm | Batch_3 | 0 | 1 |  |
| Batch_3/cfe5vt7s | Batch_3 | 0 | 0 |  |
| Batch_3/hawkfj64 | Batch_3 | 1 | 2 | K04_agglom_frac (-2.2) |
| Batch_3/hzumfsms | Batch_3 | 1 | 1 | K07_R_rl (-5.0) |
| Batch_3/kbdh4tri | Batch_3 | 3 | 0 | K02_si_density_per_1000um2 (+3.4), K04_n_clusters_per_1000um2 (+2.4), K05_voronoi_sigma (+2.2) |
| Batch_3/mgxahqnk | Batch_3 | 2 | 0 | K04_agglom_frac (+3.6), K03_ecd_max_um (+2.9) |
| Batch_3/pl8uabbv | Batch_3 | 0 | 0 |  |
| Batch_3/ptg8lmto | Batch_3 | 0 | 1 |  |
| Batch_3/tuy3zymq | Batch_3 | 3 | 0 | K02_si_density_per_1000um2 (+4.0), K04_n_clusters_per_1000um2 (+3.3), K03_ecd_d90_um (-3.0) |
| Batch_3/ufdvpb81 | Batch_3 | 0 | 1 |  |
| Batch_3/utfgcjfa | Batch_3 | 1 | 0 | K04_agglom_frac (+2.0) |
| Batch_3/vc2whyaq | Batch_3 | 2 | 1 | K03_ecd_d50_um (+2.3), K03_ecd_d90_um (+2.1) |
| Batch_3/x77cy643 | Batch_3 | 2 | 0 | K03_ecd_d50_um (+3.3), K03_ecd_d90_um (+2.1) |
| Batch_3/x7u69zsw | Batch_3 | 3 | 0 | K01_si_frac_adm (+4.2), K03_ecd_max_um (+3.0), K02_si_density_per_1000um2 (+2.2) |
| Batch_3/xgj4xftb | Batch_3 | 1 | 1 | K05_voronoi_sigma (+2.9) |
| Batch_heldout/3e122cbj | Batch_heldout | 8 | 2 | K01_si_frac_adm (+10.3), K02_si_density_per_1000um2 (+9.3), K09_mst_sigma_norm (-6.1), K09_mst_m_norm (-5.3), K15_si_graphite_contact_frac (-5.3) |
| Batch_heldout/fn0mhxef | Batch_heldout | 0 | 0 |  |
| Batch_heldout/xrv9xvzb | Batch_heldout | 0 | 1 |  |

## Per batch

| batch | n_off median | n_off max | n_unresolved median | n_unresolved max | n_off_unknown median | n_off_unknown max |
|---|---|---|---|---|---|---|
| Batch_1 | 1 | 10 | 1 | 2 | 2 | 10 |
| Batch_2 | 1 | 2 | 0 | 1 | 1 | 3 |
| Batch_3 | 1 | 3 | 0 | 2 | 1 | 3 |
| Batch_heldout | 0 | 8 | 1 | 2 | 2 | 6 |

## Per column: how many fields are `off` / `within_sampling_noise`

|  | inside | off | off_noise_unknown | undefined | within_sampling_noise |
|---|---|---|---|---|---|
| K01_si_frac_adm | 27 | 5 | 0 | 0 | 2 |
| K02_si_density_per_1000um2 | 28 | 6 | 0 | 0 | 0 |
| K03_ecd_d50_um | 32 | 2 | 0 | 0 | 0 |
| K03_ecd_d90_um | 26 | 8 | 0 | 0 | 0 |
| K03_ecd_max_um | 27 | 7 | 0 | 0 | 0 |
| K04_agglom_frac | 29 | 5 | 0 | 0 | 0 |
| K04_n_clusters_per_1000um2 | 32 | 2 | 0 | 0 | 0 |
| K05_voronoi_sigma | 31 | 3 | 0 | 0 | 0 |
| K05_voronoi_sigma_null_mean | 29 | 0 | 5 | 0 | 0 |
| K05_voronoi_sigma_z | 30 | 0 | 4 | 0 | 0 |
| K05_local_af_cv | 22 | 0 | 12 | 0 | 0 |
| K06_cluster_region_frac | 33 | 0 | 1 | 0 | 0 |
| K06_void_region_frac | 32 | 0 | 2 | 0 | 0 |
| K07_R_rl | 24 | 3 | 0 | 0 | 7 |
| K07_R_csr | 29 | 0 | 0 | 0 | 5 |
| K08_pcf_excess_max_x | 0 | 0 | 0 | 34 | 0 |
| K08_pcf_rpeak_x_um | 0 | 0 | 0 | 34 | 0 |
| K08_pcf_excess_max_z | 0 | 0 | 0 | 34 | 0 |
| K08_pcf_rpeak_z_um | 0 | 0 | 0 | 34 | 0 |
| K09_mst_m_norm | 31 | 3 | 0 | 0 | 0 |
| K09_mst_sigma_norm | 26 | 3 | 0 | 0 | 5 |
| K10_cv_w10 | 28 | 0 | 6 | 0 | 0 |
| K10_cv_slope | 31 | 0 | 3 | 0 | 0 |
| K11_lacey_w10 | 30 | 0 | 4 | 0 | 0 |
| K12_depth_maxdev | 29 | 0 | 5 | 0 | 0 |
| K12_depth_absslope | 22 | 0 | 12 | 0 | 0 |
| K13_lateral_cv | 34 | 0 | 0 | 0 | 0 |
| K14_empty_p50_um | 0 | 0 | 0 | 34 | 0 |
| K14_empty_p95_um | 31 | 3 | 0 | 0 | 0 |
| K15_si_graphite_contact_frac | 30 | 4 | 0 | 0 | 0 |
| F02_soc100_pore_loss | 31 | 0 | 3 | 0 | 0 |
| F02_soc100_into_graphite | 31 | 0 | 3 | 0 | 0 |
