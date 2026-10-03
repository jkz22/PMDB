# Auto-generated tables (`python -m src.v2.report`)

## Compute

- `allkpi`: {'done': 24}, run-time sum 6.47 GPU-h (8 concurrent), wall 0.88 h, $5.16 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `baseline3`: {'done': 8}, run-time sum 3.13 GPU-h (8 concurrent), wall 0.45 h, $2.61 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `cls_a`: {'done': 60}, run-time sum 5.18 GPU-h (8 concurrent), wall 0.66 h, $3.88 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `cls_b`: {'done': 60}, run-time sum 5.17 GPU-h (8 concurrent), wall 0.66 h, $3.88 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `cls_c`: {'done': 55}, run-time sum 4.80 GPU-h (8 concurrent), wall 0.61 h, $3.59 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `cls_d`: {'done': 26, 'failed': 24}, run-time sum 3.26 GPU-h (8 concurrent), wall 0.53 h, $3.07 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `cls_e`: {'done': 25}, run-time sum 2.13 GPU-h (8 concurrent), wall 0.31 h, $1.81 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `cls_f`: {'done': 5}, run-time sum 0.27 GPU-h (8 concurrent), wall 0.08 h, $0.47 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `round2a`: {'done': 12}, run-time sum 5.84 GPU-h (8 concurrent), wall 0.82 h, $4.82 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `round2b`: {'done': 4}, run-time sum 0.90 GPU-h (8 concurrent), wall 0.23 h, $1.34 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `round2c`: {'done': 4}, run-time sum 1.42 GPU-h (8 concurrent), wall 0.36 h, $2.10 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `round2d`: {'done': 8}, run-time sum 3.11 GPU-h (8 concurrent), wall 0.39 h, $2.31 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `round3a`: {'running': 7}, run-time sum 0.00 GPU-h (8 concurrent), wall 0.40 h, $2.34 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `round3b`: {'failed': 6, 'done': 6}, run-time sum 1.78 GPU-h (8 concurrent), wall 0.35 h, $2.03 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `round3c`: {'done': 6}, run-time sum 1.80 GPU-h (8 concurrent), wall 0.35 h, $2.04 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `round3l`: {'done': 6}, run-time sum 1.65 GPU-h (8 concurrent), wall 0.28 h, $1.62 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `round3o`: {'done': 9}, run-time sum 0.06 GPU-h (8 concurrent), wall 0.02 h, $0.13 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `stage_ab`: {'queued': 29, 'done': 16, 'running': 8, 'failed': 2}, run-time sum 8.19 GPU-h (8 concurrent), wall 1.21 h, $7.08 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `stage_ab2`: {'done': 20}, run-time sum 5.16 GPU-h (8 concurrent), wall 0.73 h, $4.27 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `stage_ab_tail`: {'done': 13}, run-time sum 3.51 GPU-h (8 concurrent), wall 0.51 h, $2.96 (H100 + 24 CPU + 96 GiB at $5.85/h)
- `stage_c`: {'done': 20}, run-time sum 6.62 GPU-h (8 concurrent), wall 0.92 h, $5.38 (H100 + 24 CPU + 96 GiB at $5.85/h)

## Leaderboard (Stage OTS/A/B/R2, top 20)

| selection_rank | stage | factor | family | view | input | train_set | aug | vae_mask | mae_mask | harmonise | kpi_r2 | img_r2 | image_id_ratio | lift_shift | recon_kpi_err | psnr | ssim | knn_batch_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | R2 | r2_norm | vae_c | stack | norm | baseline | aug1 | 0.0 | 0.75 | False | 0.812 | 0.005 | 0.85 | 0.281 | 0.311 | 18.128 | 0.372 | 0.434 |
| 2 | B | train_set | vae_c | stack | raw | baseline | aug1 | 0.0 | 0.75 | False | 0.824 | -0.066 | 1.05 | 0.261 | 0.409 | 20.745 | 0.49 | 0.176 |
| 3 | R2 | r2_aug2 | vae_c | stack | raw | baseline | aug2 | 0.0 | 0.75 | False | 0.781 | -0.083 | 0.9 | 0.278 | 0.429 | 20.277 | 0.483 | 0.19 |
| 4 | B | aug | vae_c | stack | raw | all | aug0 | 0.0 | 0.75 | False | 0.814 | -0.06 | 1.55 | 0.261 | 0.354 | 20.618 | 0.489 | 0.154 |
| 5 | R3 | r3_hybrid | vae_c | stack | raw | baseline | aug1 | 0.0 | 0.75 | hybrid | 0.844 | -0.03 | 0.65 | 0.714 | 0.382 | 20.497 | 0.446 | 0.167 |
| 6 | R3 | r3_hybrid_aug2 | vae_c | stack | raw | baseline | aug2 | 0.0 | 0.75 | hybrid | 0.819 | -0.04 | 0.8 | 0.383 | 0.415 | 20.168 | 0.439 | 0.204 |
| 7 | R2 | r2_harm_aug2 | vae_c | stack | raw | baseline | aug2 | 0.0 | 0.75 | True | 0.806 | -0.051 | 1.05 | 0.307 | 0.403 | 20.62 | 0.49 | 0.525 |
| 8 | B | aug | vae_c | stack | raw | all | aug2 | 0.0 | 0.75 | False | 0.834 | -0.095 | 1.2 | 0.327 | 0.454 | 20.355 | 0.483 | 0.262 |
| 9 | A |  | vae_c | stack | raw | all | aug1 | 0.0 | 0.75 | False | 0.857 | -0.091 | 1.25 | 0.346 | 0.439 | 20.607 | 0.489 | 0.262 |
| 10 | B | input | vae_c | stack | norm | all | aug1 | 0.0 | 0.75 | False | 0.811 | 0.049 | 1.1 | 0.334 | 0.319 | 18.069 | 0.37 | 0.217 |
| 11 | R3 | r3_histmatch | vae_c | stack | raw | baseline | aug1 | 0.0 | 0.75 | histmatch | 0.814 | -0.001 | 0.8 | 0.711 | 0.412 | 20.473 | 0.423 | 0.267 |
| 12 | B | train_set | vae_c | stack | raw | baseline | aug1 | 0.0 | 0.75 | False | 0.801 | -0.025 | 0.95 | 0.435 | 0.422 | 20.674 | 0.491 | 0.222 |
| 13 | R3 | r3_affine2 | vae_c | stack | raw | baseline | aug1 | 0.0 | 0.75 | affine2 | 0.818 | -0.069 | 1.7 | 0.444 | 0.367 | 21.037 | 0.496 | 0.208 |
| 14 | R2 | r2_harmonise | vae_c | stack | raw | baseline | aug1 | 0.0 | 0.75 | True | 0.796 | -0.044 | 1.25 | 0.477 | 0.412 | 21.061 | 0.499 | 0.208 |
| 15 | B | view | vae_c | BSE | raw | all | aug1 | 0.0 | 0.75 | False | 0.83 | 0.248 | 0.95 | 0.983 | 0.395 | 25.278 | 0.546 | 0.376 |
| 16 | B | view | vae_c | SE_type | raw | all | aug1 | 0.0 | 0.75 | False | 0.809 | 0.124 | 1.1 | 0.945 |  | 21.237 | 0.529 | 0.443 |
| 17 | B | view | mae_scratch | BSE | raw | all | aug1 | 0.0 | 0.75 | False | 0.757 | 0.196 | 1.75 | 1.413 |  |  |  | 0.321 |
| 18 | R2 | r2_norm | mae_adapted | BSE | norm | all | aug1 | 0.0 | 0.75 | False | 0.672 | 0.454 | 1.95 | 0.628 |  |  |  | 0.538 |
| 19 | B | view | mae_adapted | BSE | raw | all | aug1 | 0.0 | 0.75 | False | 0.692 | 0.637 | 1.75 | 2.358 |  |  |  | 0.538 |
| 20 | R2 | r2_aug2 | mae_adapted | BSE | raw | all | aug2 | 0.0 | 0.75 | False | 0.716 | 0.623 | 1.85 | 2.428 |  |  |  | 0.498 |

## Best configuration per family

| selection_rank | stage | factor | family | view | input | train_set | aug | vae_mask | mae_mask | harmonise | kpi_r2 | img_r2 | image_id_ratio | lift_shift | recon_kpi_err | psnr | ssim | knn_batch_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | R2 | r2_norm | vae_c | stack | norm | baseline | aug1 | 0.0 | 0.75 | False | 0.812 | 0.005 | 0.85 | 0.281 | 0.311 | 18.128 | 0.372 | 0.434 |
| 17 | B | view | mae_scratch | BSE | raw | all | aug1 | 0.0 | 0.75 | False | 0.757 | 0.196 | 1.75 | 1.413 |  |  |  | 0.321 |
| 18 | R2 | r2_norm | mae_adapted | BSE | norm | all | aug1 | 0.0 | 0.75 | False | 0.672 | 0.454 | 1.95 | 0.628 |  |  |  | 0.538 |
| 25 | R2 | r2_aug2 | vae_b | Inlens | raw | all | aug2 | 0.0 | 0.75 | False | 0.604 | -0.291 | 0.9 | 0.16 |  | 17.65 | 0.417 | 0.176 |
| 27 | R2 | r2_norm | vae_a | stack | norm | baseline | aug1 | 0.0 | 0.75 | False | 0.177 | -0.189 | 0.85 | 0.203 | 0.298 | 18.483 | 0.381 | 0.136 |
| 50 | R2 | r2_harm_aug2 | dino_ft | stack | raw | all | aug2 | 0.0 | 0.75 | True | 0.535 | 0.048 | 2.1 | 0.032 |  |  |  | 0.416 |
| 83 | OTS |  | ots_dinov2 | stack | raw | all | aug1 | 0.0 | 0.75 | False | 0.424 | 0.05 | 2.0 | 0.112 |  |  |  | 0.421 |
| 89 | R3 | r3_affine2 | ots_micronet | stack | raw | all | aug1 | 0.0 | 0.75 | affine2 | 0.236 | -0.034 | 1.95 | 0.16 |  |  |  | 0.394 |
| 100 | R3 | r3_affine2 | ots_vitmae | stack | raw | all | aug1 | 0.0 | 0.75 | affine2 | 0.432 | 0.051 | 2.25 | 0.255 |  |  |  | 0.357 |

## Stage B: change vs the family's Stage A default (B minus A)

| family | factor | level | d_kpi_r2 | d_img_r2 | d_image_id_ratio | d_knn_batch_acc | d_lift_shift | rank | rank_default |
|---|---|---|---|---|---|---|---|---|---|
| mae_adapted | aug | aug0 | -0.069 | 0.033 | 0.1 | -0.059 | -0.093 | 114 | 22 |
| mae_scratch | aug | aug0 | 0.031 | 0.044 | -0.35 | 0.0 | 0.004 | 82 | 93 |
| vae_a | aug | aug0 | -0.053 | -0.005 | -0.1 | 0.036 | -0.145 | 30 | 32 |
| vae_b | aug | aug0 | -0.123 | -0.033 | 0.15 | 0.027 | -0.244 | 29 | 38 |
| vae_c | aug | aug0 | -0.042 | 0.031 | 0.3 | -0.109 | -0.086 | 4 | 9 |
| dino_ft | aug | aug2 | 0.018 | 0.011 | 0.1 | -0.036 | -0.012 | 61 | 62 |
| mae_adapted | aug | aug2 | -0.014 | -0.02 | -0.05 | -0.018 | -0.049 | 21 | 22 |
| mae_scratch | aug | aug2 | -0.094 | -0.058 | -0.2 | -0.005 | -0.073 | 81 | 93 |
| vae_a | aug | aug2 | -0.06 | -0.004 | 0.0 | -0.005 | -0.008 | 41 | 32 |
| vae_b | aug | aug2 | 0.212 | -0.01 | 0.15 | -0.045 | -0.076 | 35 | 38 |
| vae_c | aug | aug2 | -0.023 | -0.005 | -0.05 | 0.0 | -0.019 | 8 | 9 |
| dino_ft | input | norm | -0.003 | -0.001 | 0.2 | 0.027 | -0.006 | 65 | 62 |
| mae_adapted | input | norm | -0.042 | -0.144 | 0.05 | 0.081 | -0.296 | 103 | 22 |
| mae_scratch | input | norm | -0.156 | -0.505 | 0.05 | 0.176 | -0.083 | 45 | 93 |
| vae_a | input | norm | -0.004 | 0.01 | 0.25 | 0.127 | -0.057 | 33 | 32 |
| vae_b | input | norm | -0.032 | 0.031 | 0.25 | 0.005 | -0.074 | 43 | 38 |
| vae_c | input | norm | -0.045 | 0.139 | -0.15 | -0.045 | -0.012 | 10 | 9 |
| mae_adapted | mask | 0.5 | 0.005 | 0.011 | 0.05 | -0.014 | 0.053 | 24 | 22 |
| mae_scratch | mask | 0.5 | -0.02 | 0.056 | -0.25 | 0.0 | -0.016 | 90 | 93 |
| vae_a | mask | 0.5 | -0.055 | -0.039 | 0.6 | 0.176 | 0.021 | 66 | 32 |
| vae_b | mask | 0.5 | -0.079 | -0.029 | 0.4 | -0.154 | -0.018 | 76 | 38 |
| vae_c | mask | 0.5 | -0.236 | -0.052 | -0.1 | -0.045 | 0.195 | 47 | 9 |
| dino_ft | train_set | baseline | 0.017 | 0.01 | 0.05 | -0.018 | -0.005 | 63 | 62 |
| dino_ft | train_set | baseline | -0.035 | 0.007 | 0.0 | 0.068 | 0.01 | 71 | 62 |
| mae_adapted | train_set | baseline | -0.03 | -0.053 | -0.1 | -0.1 | -0.069 | 108 | 22 |
| mae_adapted | train_set | baseline | -0.054 | -0.034 | 0.0 | -0.068 | -0.084 | 110 | 22 |
| mae_scratch | train_set | baseline | 0.015 | 0.025 | 0.0 | 0.018 | -0.032 | 87 | 93 |
| mae_scratch | train_set | baseline | -0.061 | 0.031 | -0.25 | 0.009 | -0.004 | 92 | 93 |
| vae_a | train_set | baseline | -0.041 | -0.004 | -0.15 | -0.027 | -0.065 | 28 | 32 |
| vae_a | train_set | baseline | -0.047 | 0.015 | 0.35 | 0.118 | -0.009 | 56 | 32 |
| vae_b | train_set | baseline | 0.017 | 0.007 | 0.0 | -0.009 | -0.037 | 34 | 38 |
| vae_b | train_set | baseline | -0.034 | 0.022 | 0.25 | 0.172 | -0.011 | 59 | 38 |
| vae_c | train_set | baseline | -0.033 | 0.024 | -0.2 | -0.086 | -0.085 | 2 | 9 |
| vae_c | train_set | baseline | -0.055 | 0.065 | -0.3 | -0.041 | 0.089 | 12 | 9 |
| dino_ft | view | BSE | 0.078 | 0.254 | -0.3 | -0.014 | 0.088 | 79 | 62 |
| mae_adapted | view | BSE | 0.033 | 0.27 | -0.8 | 0.063 | 1.71 | 19 | 22 |
| mae_scratch | view | BSE | 0.28 | 0.152 | -0.6 | -0.054 | 1.291 | 17 | 93 |
| vae_a | view | BSE | 0.389 | 0.349 | 0.15 | 0.131 | 0.539 | 76 | 32 |
| vae_b | view | BSE | 0.277 | 0.204 | 0.25 | 0.113 | 0.66 | 78 | 38 |
| vae_c | view | BSE | -0.026 | 0.339 | -0.3 | 0.113 | 0.637 | 15 | 9 |
| dino_ft | view | Inlens | -0.107 | -0.01 | 0.15 | -0.014 | -0.015 | 71 | 62 |
| mae_adapted | view | Inlens | -0.274 | -0.012 | -0.4 | -0.104 | -0.357 | 109 | 22 |
| mae_scratch | view | Inlens | -0.617 | -0.044 | -0.65 | 0.005 | -0.02 | 95 | 93 |
| vae_a | view | Inlens | -0.096 | -0.047 | 0.1 | -0.054 | -0.11 | 37 | 32 |
| vae_b | view | Inlens | 0.386 | -0.107 | 0.4 | -0.009 | -0.177 | 26 | 38 |
| vae_c | view | Inlens | -0.461 | -0.035 | -0.2 | -0.199 | -0.092 | 53 | 9 |
| dino_ft | view | SE_type | -0.082 | -0.019 | -0.1 | 0.032 | 0.029 | 71 | 62 |
| mae_adapted | view | SE_type | 0.033 | -0.024 | 0.1 | -0.109 | 0.198 | 23 | 22 |
| mae_scratch | view | SE_type | 0.074 | 0.052 | -0.6 | -0.023 | 0.258 | 98 | 93 |
| vae_a | view | SE_type | 0.115 | 0.052 | 0.45 | 0.127 | 0.135 | 96 | 32 |
| vae_b | view | SE_type | -0.11 | 0.115 | 0.4 | 0.1 | 0.008 | 99 | 38 |
| vae_c | view | SE_type | -0.048 | 0.215 | -0.15 | 0.181 | 0.599 | 16 | 9 |

### Median effect over families: `view`

| level | d_kpi_r2 | d_img_r2 | d_image_id_ratio | d_knn_batch_acc | d_lift_shift |
|---|---|---|---|---|---|
| BSE | 0.177 | 0.262 | -0.3 | 0.088 | 0.648 |
| Inlens | -0.19 | -0.039 | -0.05 | -0.034 | -0.101 |
| SE_type | -0.007 | 0.052 | 0.0 | 0.066 | 0.167 |

### Median effect over families: `train_set`

| level | d_kpi_r2 | d_img_r2 | d_image_id_ratio | d_knn_batch_acc | d_lift_shift |
|---|---|---|---|---|---|
| baseline | -0.034 | 0.013 | 0.0 | -0.014 | -0.021 |

### Median effect over families: `aug`

| level | d_kpi_r2 | d_img_r2 | d_image_id_ratio | d_knn_batch_acc | d_lift_shift |
|---|---|---|---|---|---|
| aug0 | -0.053 | 0.031 | 0.1 | 0.0 | -0.093 |
| aug2 | -0.018 | -0.007 | -0.025 | -0.011 | -0.034 |

### Median effect over families: `input`

| level | d_kpi_r2 | d_img_r2 | d_image_id_ratio | d_knn_batch_acc | d_lift_shift |
|---|---|---|---|---|---|
| norm | -0.037 | 0.004 | 0.125 | 0.054 | -0.065 |

### Median effect over families: `mask`

| level | d_kpi_r2 | d_img_r2 | d_image_id_ratio | d_knn_batch_acc | d_lift_shift |
|---|---|---|---|---|---|
| 0.5 | -0.055 | -0.029 | 0.05 | -0.014 | 0.021 |

## Stage C: grouped 5-fold cross-fitting (mean, sd over folds)

| index | kpi_r2_mean | kpi_r2_std | img_r2_mean | img_r2_std | image_id_ratio_mean | image_id_ratio_std | lift_shift_mean | lift_shift_std | recon_kpi_err_mean | recon_kpi_err_std | psnr_mean | psnr_std | ssim_mean | ssim_std | knn_batch_acc_mean | knn_batch_acc_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vae_b|Inlens|raw|all|aug1|0.0|0.75|False | 0.126 | 0.235 | -0.274 | 0.021 | 1.02 | 0.115 | 0.181 | 0.006 |  |  | 17.935 | 0.287 | 0.424 | 0.007 | 0.176 | 0.057 |
| vae_c|stack|raw|all|aug0|0.0|0.75|False | 0.883 | 0.029 | -0.026 | 0.011 | 1.03 | 0.189 | 0.277 | 0.027 | 0.352 | 0.014 | 21.522 | 0.239 | 0.507 | 0.006 | 0.103 | 0.023 |
| vae_c|stack|raw|baseline|aug1|0.0|0.75|False | 0.837 | 0.015 | -0.051 | 0.031 | 1.06 | 0.204 | 0.304 | 0.078 | 0.4 | 0.011 | 20.769 | 0.076 | 0.491 | 0.002 | 0.211 | 0.061 |

## Stage C: extra seeds of the best configuration

| seed | kpi_r2 | img_r2 | image_id_ratio | lift_shift | recon_kpi_err | psnr | ssim | knn_batch_acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.809 | -0.066 | 1.3 | 0.271 | 0.42 | 20.769 | 0.49 | 0.131 |
| 2 | 0.833 | -0.029 | 1.05 | 0.295 | 0.391 | 20.73 | 0.49 | 0.195 |

## Stage C: leave-one-batch-out retraining (`noop`: baseline-only parent never saw that batch, so the row equals the parent)

| lobo | train_set | harmonise | noop | kpi_r2 | img_r2 | image_id_ratio | lift_shift | recon_kpi_err | psnr | ssim | knn_batch_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Batch_1 | all | hybrid | False | 0.83 | -0.002 | 0.6 | 0.561 | 0.366 | 20.131 | 0.438 | 0.176 |
| Batch_2 | all | hybrid | False | 0.805 | -0.022 | 0.75 | 0.523 | 0.367 | 19.991 | 0.435 | 0.172 |
| Batch_3 | all | hybrid | False | 0.755 | -0.024 | 0.8 | 0.426 | 0.396 | 19.429 | 0.423 | 0.24 |
| Batch_1 | all | none | False | 0.802 | -0.041 | 1.1 | 0.287 | 0.376 | 20.555 | 0.486 | 0.149 |
| Batch_2 | all | none | False | 0.771 | -0.044 | 1.15 | 0.247 | 0.382 | 20.488 | 0.486 | 0.163 |
| Batch_3 | all | none | False | 0.759 | -0.067 | 1.5 | 0.315 | 0.374 | 19.886 | 0.472 | 0.231 |
| Batch_1 | baseline | False | True | 0.824 | -0.066 | 1.05 | 0.261 | 0.409 | 20.745 | 0.49 | 0.176 |
| Batch_2 | baseline | False | True | 0.824 | -0.066 | 1.05 | 0.261 | 0.409 | 20.745 | 0.49 | 0.176 |

## Phase 6 latent audit (best configuration per family, all 31 fields)

| hash | family | view | img_r2_max | img_r2_mean | batch_probe_acc | image_id_ratio | knn_batch_acc | active_units | round2_trigger | kpi_r2.frac_si | kpi_r2.frac_graphite | kpi_r2.frac_pore | kpi_r2.K01_si_frac_adm | kpi_r2.K04_agglom_frac | img_r2.p1_c0 | img_r2.p99_c0 | img_r2.dyn_range_c0 | img_r2.noise_sigma_c0 | img_r2.sharpness_c0 | img_r2.gmm_mu0_c0 | img_r2.gmm_mu1_c0 | img_r2.gmm_mu2_c0 | img_r2.p1_c1 | img_r2.p99_c1 | img_r2.dyn_range_c1 | img_r2.noise_sigma_c1 | img_r2.sharpness_c1 | img_r2.gmm_mu0_c1 | img_r2.gmm_mu1_c1 | img_r2.gmm_mu2_c1 | img_r2.p1_c2 | img_r2.p99_c2 | img_r2.dyn_range_c2 | img_r2.noise_sigma_c2 | img_r2.sharpness_c2 | img_r2.gmm_mu0_c2 | img_r2.gmm_mu1_c2 | img_r2.gmm_mu2_c2 | knn_batch_by_batch.Batch_1 | knn_batch_by_batch.Batch_2 | knn_batch_by_batch.Batch_3 | field_inference.Batch_1.dist | field_inference.Batch_1.p_perm | field_inference.Batch_1.ci95 | field_inference.Batch_1.n_fields | field_inference.Batch_2.dist | field_inference.Batch_2.p_perm | field_inference.Batch_2.ci95 | field_inference.Batch_2.n_fields | conformal_p.Batch_1/4ih2ggld | conformal_p.Batch_1/5n1q8atc | conformal_p.Batch_1/f1vzngrs | conformal_p.Batch_1/ffwubibz | conformal_p.Batch_1/fzrt2k6r | conformal_p.Batch_1/iv6g2oq0 | conformal_p.Batch_1/uhdslk0o | conformal_p.Batch_2/3806gxp0 | conformal_p.Batch_2/avn74qx1 | conformal_p.Batch_2/b3esycq1 | conformal_p.Batch_2/epqdaau9 | conformal_p.Batch_2/i9jiqjwl | conformal_p.Batch_2/r17byphk | conformal_p.Batch_2/rxax5ozo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0a162269e402 | vae_a | stack | 0.587 | 0.16 | 0.442 | 1.649 | 0.446 | 128.0 | True | 0.03 | 0.009 | 0.013 | 0.029 | 0.018 | 0.291 | 0.091 | 0.258 | 0.177 | -0.006 | -0.023 | -0.05 | 0.05 | -0.018 | 0.355 | 0.186 | 0.522 | 0.24 | 0.174 | 0.587 | 0.234 | 0.319 | 0.009 | 0.101 | 0.12 | -0.037 | -0.025 | 0.122 | 0.163 | 0.237 | 0.184 | 0.65 | 1.171 | 0.126 | [0.9891979694366455, 2.1022746562957764] | 7 | 1.307 | 0.073 | [1.0052430629730225, 2.3594226837158203] | 7 | 1.0 | 1.0 | 0.444 | 1.0 | 0.111 | 0.778 | 0.333 | 0.778 | 1.0 | 0.056 | 0.5 | 0.056 | 0.944 | 1.0 |
| 12585cbeed11 | vae_c | stack | 0.733 | 0.398 | 0.448 | 2.195 | 0.385 | 128.0 | True | 0.94 | 0.905 | 0.864 | 0.94 | 0.728 | 0.58 | 0.733 | 0.714 | 0.203 | 0.344 | 0.627 | 0.259 | 0.719 | 0.301 | 0.557 | 0.431 | 0.52 | 0.335 | 0.324 | 0.651 | 0.495 | 0.61 | 0.032 | 0.173 | 0.114 | 0.015 | 0.299 | 0.249 | 0.255 | 0.308 | 0.165 | 0.514 | 1.34 | 0.135 | [1.1717498302459717, 2.4322116374969482] | 7 | 1.455 | 0.075 | [1.0962754487991333, 2.6729931831359863] | 7 | 0.333 | 0.056 | 0.333 | 1.0 | 0.056 | 0.333 | 0.333 | 0.889 | 0.556 | 0.056 | 0.333 | 0.056 | 0.722 | 0.722 |
| 63cdec8d0715 | dino_ft | stack | 0.786 | 0.591 | 0.6 | 5.506 | 0.607 |  | True | 0.816 | 0.795 | 0.687 | 0.816 | 0.383 | 0.777 | 0.713 | 0.731 | 0.639 | 0.604 | 0.65 | 0.453 | 0.667 | 0.568 | 0.652 | 0.59 | 0.738 | 0.589 | 0.37 | 0.624 | 0.587 | 0.786 | 0.553 | 0.609 | 0.597 | 0.496 | 0.428 | 0.43 | 0.339 | 0.438 | 0.299 | 0.813 | 13.266 | 0.004 | [11.211848258972168, 19.23870849609375] | 7 | 11.832 | 0.006 | [8.5978364944458, 16.91506004333496] | 7 | 0.056 | 0.056 | 0.056 | 0.111 | 0.111 | 0.167 | 0.611 | 0.444 | 0.278 | 0.111 | 0.278 | 0.111 | 0.833 | 0.5 |
| 8340f10a4151 | mae_scratch | stack | 0.819 | 0.627 | 0.659 | 6.27 | 0.663 |  | True | 0.903 | 0.878 | 0.667 | 0.904 | 0.451 | 0.758 | 0.819 | 0.778 | 0.642 | 0.715 | 0.587 | 0.673 | 0.776 | 0.48 | 0.728 | 0.649 | 0.801 | 0.521 | 0.403 | 0.719 | 0.64 | 0.753 | 0.568 | 0.595 | 0.644 | 0.589 | 0.396 | 0.486 | 0.333 | 0.524 | 0.33 | 0.869 | 8.424 | 0.146 | [6.057879447937012, 18.5849609375] | 7 | 11.27 | 0.012 | [6.965776443481445, 19.08089828491211] | 7 | 0.111 | 0.111 | 0.111 | 0.056 | 0.111 | 0.111 | 0.111 | 0.111 | 0.111 | 0.111 | 0.111 | 0.056 | 0.833 | 0.667 |
| c347e7a018c3 | mae_adapted | stack | 0.938 | 0.724 | 0.694 | 7.921 | 0.656 |  | True | 0.94 | 0.927 | 0.798 | 0.941 | 0.498 | 0.834 | 0.802 | 0.808 | 0.863 | 0.725 | 0.718 | 0.74 | 0.733 | 0.626 | 0.726 | 0.695 | 0.938 | 0.748 | 0.429 | 0.737 | 0.724 | 0.788 | 0.797 | 0.783 | 0.886 | 0.727 | 0.575 | 0.551 | 0.414 | 0.405 | 0.346 | 0.897 | 18.574 | 0.001 | [16.504335403442383, 25.174108505249023] | 7 | 14.652 | 0.001 | [11.1264066696167, 20.692171096801758] | 7 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.111 | 0.111 | 0.056 | 0.111 | 0.056 | 0.056 | 1.0 | 0.444 |
| fe90c70648ab | vae_b | Inlens | 0.647 | 0.397 | 0.463 | 1.869 | 0.49 | 128.0 | True | 0.944 | 0.918 | 0.716 | 0.945 | 0.834 |  |  |  |  |  |  |  |  | 0.202 | 0.436 | 0.346 | 0.511 | 0.293 | 0.301 | 0.647 | 0.441 |  |  |  |  |  |  |  |  | 0.178 | 0.143 | 0.773 | 1.359 | 0.147 | [1.0147806406021118, 2.5971596240997314] | 7 | 1.583 | 0.073 | [1.098110318183899, 2.896864175796509] | 7 | 0.889 | 0.944 | 0.5 | 0.944 | 0.111 | 1.0 | 0.278 | 1.0 | 1.0 | 0.056 | 0.5 | 0.056 | 0.944 | 0.889 |

## Round 2 trigger (imaging-probe R^2 > 0.5 or image-ID ratio > 2x chance)

| family | hash | img_r2_max | image_id_ratio | rerun |
|---|---|---|---|---|
| vae_c | 12585cbeed11 | 0.628 | 1.05 | True |
| vae_b | fe90c70648ab | -0.034 | 1.15 | False |
| vae_a | 0a162269e402 | 0.005 | 0.65 | False |
| mae_scratch | 8340f10a4151 | 0.386 | 2.4 | True |
| dino_ft | 63cdec8d0715 | 0.606 | 1.9 | True |
| mae_adapted | c347e7a018c3 | 0.614 | 2.6 | True |

### Round 2 reruns

| family | factor | parent_rank | selection_rank | kpi_r2 | img_r2 | image_id_ratio | lift_shift | recon_kpi_err | psnr | ssim | knn_batch_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vae_c | r2_norm | 2 | 1 | 0.812 | 0.005 | 0.85 | 0.281 | 0.311 | 18.128 | 0.372 | 0.434 |
| vae_c | r2_aug2 | 2 | 3 | 0.781 | -0.083 | 0.9 | 0.278 | 0.429 | 20.277 | 0.483 | 0.19 |
| vae_c | r2_harm_aug2 | 2 | 7 | 0.806 | -0.051 | 1.05 | 0.307 | 0.403 | 20.62 | 0.49 | 0.525 |
| vae_c | r2_harmonise | 2 | 14 | 0.796 | -0.044 | 1.25 | 0.477 | 0.412 | 21.061 | 0.499 | 0.208 |
| mae_adapted | r2_norm | 19 | 18 | 0.672 | 0.454 | 1.95 | 0.628 |  |  |  | 0.538 |
| mae_adapted | r2_aug2 | 19 | 20 | 0.716 | 0.623 | 1.85 | 2.428 |  |  |  | 0.498 |
| vae_b | r2_aug2 | 26 | 25 | 0.604 | -0.291 | 0.9 | 0.16 |  | 17.65 | 0.417 | 0.176 |
| vae_a | r2_norm | 28 | 27 | 0.177 | -0.189 | 0.85 | 0.203 | 0.298 | 18.483 | 0.381 | 0.136 |
| vae_b | r2_harm_aug2 | 26 | 31 | 0.344 | -0.256 | 1.15 | 0.159 |  | 18.38 | 0.433 | 0.276 |
| vae_a | r2_harmonise | 28 | 36 | 0.127 | -0.214 | 1.0 | 0.264 | 0.354 | 21.648 | 0.51 | 0.335 |
| mae_scratch | r2_aug2 | 45 | 39 | 0.227 | -0.418 | 2.25 | 0.03 |  |  |  | 0.561 |
| vae_a | r2_aug2 | 28 | 40 | 0.033 | -0.217 | 0.85 | 0.25 | 0.458 | 20.784 | 0.491 | 0.213 |
| vae_a | r2_harm_aug2 | 28 | 42 | 0.178 | -0.205 | 1.0 | 0.271 | 0.399 | 21.293 | 0.501 | 0.19 |
| mae_scratch | r2_norm | 45 | 50 | 0.273 | -0.384 | 2.45 | 0.037 |  |  |  | 0.557 |
| dino_ft | r2_harm_aug2 | 62 | 50 | 0.535 | 0.048 | 2.1 | 0.032 |  |  |  | 0.416 |
| vae_b | r2_norm | 26 | 52 | -0.115 | -0.245 | 0.9 | 0.214 |  | 17.715 | 0.416 | 0.222 |
| dino_ft | r2_aug2 | 62 | 58 | 0.525 | 0.096 | 1.85 | 0.035 |  |  |  | 0.403 |
| vae_b | r2_harmonise | 26 | 64 | -0.02 | -0.252 | 1.4 | 0.16 |  | 18.701 | 0.441 | 0.267 |
| dino_ft | r2_harmonise | 62 | 67 | 0.462 | 0.012 | 2.15 | 0.038 |  |  |  | 0.398 |
| mae_scratch | r2_harm_aug2 | 45 | 69 | 0.323 | -0.15 | 2.45 | 0.027 |  |  |  | 0.538 |
| dino_ft | r2_norm | 62 | 80 | 0.5 | 0.092 | 2.15 | 0.04 |  |  |  | 0.443 |
| mae_scratch | r2_harmonise | 45 | 86 | 0.318 | -0.073 | 2.45 | 0.045 |  |  |  | 0.538 |
| mae_adapted | r2_harm_aug2 | 103 | 104 | 0.552 | 0.188 | 2.8 | 0.322 |  |  |  | 0.575 |
| mae_adapted | r2_harmonise | 103 | 104 | 0.552 | 0.218 | 2.8 | 0.319 |  |  |  | 0.579 |
| mae_adapted | r2_aug2 | 103 | 106 | 0.597 | 0.231 | 2.7 | 0.348 |  |  |  | 0.538 |
| mae_adapted | r2_norm | 103 | 107 | 0.583 | 0.237 | 2.65 | 0.35 |  |  |  | 0.557 |
| mae_adapted | r2_harmonise | 19 | 116 | 0.566 | 0.238 | 2.9 | 1.446 |  |  |  | 0.181 |
| mae_adapted | r2_harm_aug2 | 19 | 117 | 0.599 | 0.361 | 2.9 | 1.562 |  |  |  | 0.24 |

## All-KPI track (Kevin's 10 crop KPIs incl. ungated K02/K03/K04-density; top 15)

Ungated KPIs fail gates G2/G3 (size/count not scale-stable); shown for comparison only.

| selection_rank | family | view | input | train_set | aug | vae_mask | mae_mask | harmonise | kpi_set | kpi_r2 | kpi_r2_all | kpi_r2_ungated | img_r2 | image_id_ratio | knn_batch_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | vae_c | stack | norm | baseline | aug1 | 0.0 | 0.75 | False | gated | 0.812 | 0.643 | 0.222 | 0.005 | 0.85 | 0.434 |
| 2 | vae_c | stack | raw | baseline | aug2 | 0.0 | 0.75 | False | gated | 0.781 | 0.631 | 0.254 | -0.083 | 0.9 | 0.19 |
| 3 | vae_c | stack | raw | baseline | aug1 | 0.0 | 0.75 | False | gated | 0.824 | 0.654 | 0.228 | -0.066 | 1.05 | 0.176 |
| 4 | vae_c | stack | raw | all | aug1 | 0.0 | 0.75 | False | all | 0.833 | 0.734 | 0.487 | -0.03 | 0.7 | 0.317 |
| 5 | vae_c | stack | raw | baseline | aug1 | 0.0 | 0.75 | hybrid | gated | 0.844 | 0.673 | 0.247 | -0.03 | 0.65 | 0.167 |
| 6 | vae_c | stack | raw | baseline | aug2 | 0.0 | 0.75 | True | gated | 0.806 | 0.65 | 0.261 | -0.051 | 1.05 | 0.525 |
| 7 | vae_c | stack | raw | all | aug0 | 0.0 | 0.75 | False | gated | 0.814 | 0.625 | 0.154 | -0.06 | 1.55 | 0.154 |
| 8 | vae_c | stack | raw | baseline | aug1 | 0.0 | 0.75 | False | all | 0.804 | 0.697 | 0.43 | -0.019 | 0.95 | 0.249 |
| 9 | vae_c | stack | raw | baseline | aug2 | 0.0 | 0.75 | hybrid | gated | 0.819 | 0.65 | 0.229 | -0.04 | 0.8 | 0.204 |
| 10 | vae_c | stack | raw | all | aug2 | 0.0 | 0.75 | False | all | 0.78 | 0.686 | 0.451 | -0.057 | 1.15 | 0.303 |
| 10 | vae_c | stack | raw | all | aug0 | 0.0 | 0.75 | False | all | 0.772 | 0.659 | 0.376 | -0.057 | 1.3 | 0.131 |
| 12 | vae_c | stack | raw | all | aug1 | 0.0 | 0.75 | False | gated | 0.857 | 0.682 | 0.247 | -0.091 | 1.25 | 0.262 |
| 13 | vae_c | stack | raw | all | aug2 | 0.0 | 0.75 | False | gated | 0.834 | 0.668 | 0.253 | -0.095 | 1.2 | 0.262 |
| 14 | vae_c | stack | norm | all | aug1 | 0.0 | 0.75 | False | gated | 0.811 | 0.627 | 0.168 | 0.049 | 1.1 | 0.217 |
| 15 | vae_c | stack | norm | all | aug1 | 0.0 | 0.75 | False | all | 0.821 | 0.675 | 0.31 | 0.09 | 1.35 | 0.226 |

## Baseline-only training: Batch_1 (6 fields, original assumption) vs Batch_3 (14 fields, corrected baseline)

| family | kpi_set | baseline | kpi_r2 | img_r2 | image_id_ratio | lift_shift | recon_kpi_err | psnr | ssim | knn_batch_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| dino_ft | gated | Batch_1 | 0.485 | 0.091 | 1.9 | 0.055 |  |  |  | 0.489 |
| dino_ft | gated | Batch_3 | 0.537 | 0.095 | 1.95 | 0.041 |  |  |  | 0.403 |
| mae_adapted | gated | Batch_1 | 0.629 | 0.314 | 2.45 | 0.579 |  |  |  | 0.376 |
| mae_adapted | gated | Batch_3 | 0.605 | 0.333 | 2.55 | 0.564 |  |  |  | 0.407 |
| mae_scratch | gated | Batch_1 | 0.492 | 0.069 | 2.35 | 0.09 |  |  |  | 0.394 |
| mae_scratch | gated | Batch_3 | 0.416 | 0.075 | 2.1 | 0.117 |  |  |  | 0.385 |
| vae_a | gated | Batch_1 | 0.064 | -0.202 | 1.15 | 0.27 | 0.392 | 21.164 | 0.503 | 0.271 |
| vae_a | gated | Batch_3 | 0.177 | -0.189 | 0.85 | 0.203 | 0.298 | 18.483 | 0.381 | 0.136 |
| vae_a | gated | Batch_3 | 0.07 | -0.222 | 0.65 | 0.215 | 0.366 | 21.347 | 0.502 | 0.127 |
| vae_a | gated | Batch_3 | 0.127 | -0.214 | 1.0 | 0.264 | 0.354 | 21.648 | 0.51 | 0.335 |
| vae_a | gated | Batch_3 | 0.033 | -0.217 | 0.85 | 0.25 | 0.458 | 20.784 | 0.491 | 0.213 |
| vae_a | gated | Batch_3 | 0.178 | -0.205 | 1.0 | 0.271 | 0.399 | 21.293 | 0.501 | 0.19 |
| vae_a | gated | Batch_3 | 0.171 | -0.172 | 0.8 | 0.438 | 0.341 | 21.064 | 0.457 | 0.235 |
| vae_a | gated | Batch_3 | 0.158 | -0.175 | 0.85 | 0.375 | 0.345 | 21.056 | 0.434 | 0.226 |
| vae_a | gated | Batch_3 | -0.036 | -0.212 | 1.2 | 0.214 | 0.393 | 21.421 | 0.505 | 0.235 |
| vae_b | gated | Batch_1 | 0.07 | -0.174 | 1.0 | 0.348 | 0.35 | 21.228 | 0.503 | 0.353 |
| vae_b | gated | Batch_3 | 0.121 | -0.189 | 0.75 | 0.322 | 0.341 | 21.359 | 0.502 | 0.172 |
| vae_c | gated | Batch_1 | 0.801 | -0.025 | 0.95 | 0.435 | 0.422 | 20.674 | 0.491 | 0.222 |
| vae_c | gated | Batch_3 | 0.812 | 0.005 | 0.85 | 0.281 | 0.311 | 18.128 | 0.372 | 0.434 |
| vae_c | gated | Batch_3 | 0.824 | -0.066 | 1.05 | 0.261 | 0.409 | 20.745 | 0.49 | 0.176 |
| vae_c | gated | Batch_3 | 0.781 | -0.083 | 0.9 | 0.278 | 0.429 | 20.277 | 0.483 | 0.19 |
| vae_c | gated | Batch_3 | 0.844 | -0.03 | 0.65 | 0.714 | 0.382 | 20.497 | 0.446 | 0.167 |
| vae_c | gated | Batch_3 | 0.819 | -0.04 | 0.8 | 0.383 | 0.415 | 20.168 | 0.439 | 0.204 |
| vae_c | gated | Batch_3 | 0.806 | -0.051 | 1.05 | 0.307 | 0.403 | 20.62 | 0.49 | 0.525 |
| vae_c | gated | Batch_3 | 0.814 | -0.001 | 0.8 | 0.711 | 0.412 | 20.473 | 0.423 | 0.267 |
| vae_c | gated | Batch_3 | 0.818 | -0.069 | 1.7 | 0.444 | 0.367 | 21.037 | 0.496 | 0.208 |
| vae_c | gated | Batch_3 | 0.796 | -0.044 | 1.25 | 0.477 | 0.412 | 21.061 | 0.499 | 0.208 |

## Supervised 3-class batch classification (stratified grouped 5-fold by field; mean over folds)

| arch | view | input | harmonise | aug | n_folds_done | crop_acc | field_acc | field_acc_sd | field_f1 | field_acc_Batch_1 | field_acc_Batch_2 | field_acc_Batch_3 | raw_minus_harm_field_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet18_imnet | SE_type | raw | gmm | aug1 | 5 | 0.626 | 0.722 | 0.186 | 0.588 | 0.571 | 0.286 | 0.941 | -0.237 |
| resnet18_imnet | stack | norm | none | aug1 | 5 | 0.676 | 0.717 | 0.094 | 0.642 | 0.714 | 0.429 | 0.824 | -0.033 |
| dinov2_ft | BSE | raw | none | aug1 | 5 | 0.618 | 0.677 | 0.093 | 0.495 | 0.429 | 0.286 | 0.941 | 0.0 |
| dinov2_ft | BSE | raw | gmm | aug1 | 5 | 0.617 | 0.677 | 0.093 | 0.495 | 0.429 | 0.286 | 0.941 | 0.0 |
| resnet18_imnet | BSE | raw | none | aug1 | 5 | 0.604 | 0.644 | 0.051 | 0.48 | 0.429 | 0.286 | 0.882 | 0.126 |
| dinov2_ft | SE_type | raw | none | aug1 | 5 | 0.585 | 0.644 | 0.051 | 0.463 | 0.286 | 0.429 | 0.882 | 0.04 |
| dinov2_ft | BSE | raw | hybrid | aug1 | 5 | 0.616 | 0.644 | 0.051 | 0.45 | 0.286 | 0.286 | 0.941 | 0.0 |
| resnet18_imnet | stack | raw | gmm | aug1 | 5 | 0.651 | 0.637 | 0.146 | 0.482 | 0.429 | 0.286 | 0.882 | -0.033 |
| dinov2_ft | stack | raw | affine2 | aug1 | 5 | 0.605 | 0.637 | 0.146 | 0.471 | 0.429 | 0.286 | 0.882 | -0.005 |
| resnet18_imnet | Inlens | raw | none | aug1 | 5 | 0.57 | 0.637 | 0.146 | 0.482 | 0.429 | 0.286 | 0.882 | 0.0 |
| effb4_imnet | stack | raw | gmm | aug1 | 5 | 0.638 | 0.637 | 0.146 | 0.482 | 0.429 | 0.286 | 0.882 | -0.029 |
| resnet18_imnet | Inlens | raw | gmm | aug1 | 5 | 0.632 | 0.637 | 0.146 | 0.482 | 0.429 | 0.286 | 0.882 | 0.0 |
| dinov2_ft | stack | raw | gmm | aug1 | 5 | 0.622 | 0.637 | 0.146 | 0.522 | 0.571 | 0.286 | 0.824 | -0.005 |
| dinov2_linear | stack | raw | histmatch | aug1 | 5 | 0.579 | 0.632 | 0.149 | 0.526 | 0.571 | 0.286 | 0.824 | 0.035 |
| dinov2_ft | stack | raw | none | aug1 | 5 | 0.608 | 0.632 | 0.149 | 0.484 | 0.429 | 0.286 | 0.882 | -0.005 |
| effb4_imnet | stack | raw | hybrid | aug1 | 5 | 0.603 | 0.632 | 0.149 | 0.528 | 0.571 | 0.286 | 0.824 | -0.029 |
| dinov2_ft | stack | raw | histmatch | aug1 | 5 | 0.599 | 0.632 | 0.149 | 0.528 | 0.571 | 0.286 | 0.824 | -0.005 |
| effb4_imnet | stack | raw | histmatch | aug1 | 5 | 0.586 | 0.632 | 0.149 | 0.528 | 0.571 | 0.286 | 0.824 | -0.029 |
| dinov2_linear | stack | raw | hybrid | aug1 | 5 | 0.565 | 0.632 | 0.149 | 0.526 | 0.571 | 0.286 | 0.824 | 0.035 |
| resnet18_scratch | stack | raw | gmm | aug1 | 5 | 0.598 | 0.609 | 0.168 | 0.433 | 0.286 | 0.286 | 0.882 | -0.011 |
| effb4_imnet | stack | raw | none | aug1 | 5 | 0.582 | 0.609 | 0.142 | 0.464 | 0.429 | 0.286 | 0.824 | -0.029 |
| effb4_imnet | stack | raw | affine2 | aug1 | 5 | 0.587 | 0.604 | 0.112 | 0.47 | 0.429 | 0.286 | 0.824 | -0.029 |
| dinov2_ft | Inlens | raw | gmm | aug1 | 5 | 0.585 | 0.604 | 0.112 | 0.463 | 0.429 | 0.286 | 0.824 | 0.0 |
| dinov2_ft | SE_type | raw | gmm | aug1 | 5 | 0.584 | 0.604 | 0.112 | 0.421 | 0.286 | 0.286 | 0.882 | 0.04 |
| dinov2_linear | stack | raw | none | aug1 | 5 | 0.581 | 0.604 | 0.112 | 0.47 | 0.429 | 0.286 | 0.824 | 0.035 |
| dinov2_ft | Inlens | raw | none | aug1 | 5 | 0.581 | 0.604 | 0.112 | 0.463 | 0.429 | 0.286 | 0.824 | 0.0 |
| dinov2_linear | stack | raw | affine2 | aug1 | 5 | 0.581 | 0.604 | 0.112 | 0.47 | 0.429 | 0.286 | 0.824 | 0.035 |
| dinov2_ft | SE_type | raw | hybrid | aug1 | 5 | 0.579 | 0.604 | 0.112 | 0.421 | 0.286 | 0.286 | 0.882 | 0.04 |
| resnet18_imnet | stack | raw | none | aug1 | 5 | 0.577 | 0.604 | 0.112 | 0.437 | 0.286 | 0.286 | 0.882 | -0.033 |
| dinov2_ft | stack | raw | hybrid | aug1 | 5 | 0.606 | 0.599 | 0.156 | 0.474 | 0.429 | 0.286 | 0.824 | -0.005 |
| resnet18_imnet | BSE | raw | hybrid | aug1 | 5 | 0.597 | 0.599 | 0.156 | 0.482 | 0.429 | 0.286 | 0.824 | 0.126 |
| resnet18_imnet | stack | raw | histmatch | aug1 | 5 | 0.576 | 0.599 | 0.156 | 0.483 | 0.429 | 0.286 | 0.824 | -0.033 |
| resnet18_scratch | stack | raw | hybrid | aug1 | 5 | 0.569 | 0.599 | 0.156 | 0.483 | 0.429 | 0.286 | 0.824 | -0.011 |
| resnet18_scratch | stack | raw | none | aug1 | 5 | 0.576 | 0.597 | 0.216 | 0.46 | 0.429 | 0.286 | 0.824 | -0.011 |
| resnet18_imnet | stack | raw | none | aug2 | 5 | 0.572 | 0.597 | 0.216 | 0.462 | 0.429 | 0.286 | 0.824 | -0.033 |
| resnet18_imnet | stack | raw | affine2 | aug1 | 5 | 0.577 | 0.592 | 0.241 | 0.504 | 0.571 | 0.286 | 0.765 | -0.033 |
| resnet18_scratch | stack | raw | histmatch | aug1 | 5 | 0.56 | 0.575 | 0.133 | 0.481 | 0.571 | 0.143 | 0.765 | -0.011 |
| dinov2_ft | Inlens | raw | hybrid | aug1 | 5 | 0.577 | 0.57 | 0.113 | 0.419 | 0.286 | 0.286 | 0.824 | 0.0 |
| dinov2_linear | stack | raw | gmm | aug1 | 5 | 0.595 | 0.569 | 0.208 | 0.445 | 0.429 | 0.286 | 0.765 | 0.035 |
| resnet18_imnet | Inlens | raw | hybrid | aug1 | 5 | 0.572 | 0.566 | 0.156 | 0.463 | 0.429 | 0.286 | 0.765 | 0.0 |
| resnet18_imnet | stack | raw | hybrid | aug1 | 5 | 0.586 | 0.566 | 0.156 | 0.47 | 0.429 | 0.286 | 0.765 | -0.033 |
| resnet18_scratch | stack | raw | affine2 | aug1 | 5 | 0.544 | 0.559 | 0.216 | 0.495 | 0.571 | 0.286 | 0.706 | -0.011 |
| resnet18_imnet | SE_type | raw | hybrid | aug1 | 5 | 0.56 | 0.542 | 0.089 | 0.41 | 0.286 | 0.286 | 0.765 | -0.237 |
| resnet18_imnet | BSE | raw | gmm | aug1 | 5 | 0.507 | 0.518 | 0.152 | 0.326 | 0.286 | 0.0 | 0.824 | 0.126 |
| resnet18_imnet | SE_type | raw | none | aug1 | 5 | 0.545 | 0.485 | 0.132 | 0.372 | 0.286 | 0.286 | 0.647 | -0.237 |

## Round 3: best config per family retrained on PR #16 per-site LUT harmonisation (`none` = raw grey levels; `hybrid` = affine BSE/SE + histmatch Inlens; `affine2`; `histmatch`)

`gmm` is the earlier 3-peak per-image linear map (Round 2). Lower `img_r2`/`knn_batch_acc` = less grey-level shortcut.

| family | harmonise | aug | input | view | train_set | kpi_r2 | img_r2 | img_r2__p1_c0 | image_id_ratio | knn_batch_acc | lift_shift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dino_ft | affine2 | aug1 | raw | stack | all | 0.447 | 0.013 | -0.043 | 2.15 | 0.376 | 0.036 |
| dino_ft | histmatch | aug1 | raw | stack | all | 0.457 | -0.063 | -0.203 | 2.25 | 0.385 | 0.045 |
| dino_ft | hybrid | aug1 | raw | stack | all | 0.473 | -0.074 | -0.076 | 2.15 | 0.367 | 0.043 |
| dino_ft | none | aug1 | raw | stack | all | 0.52 | 0.085 | 0.123 | 1.9 | 0.421 | 0.045 |
| mae_adapted | affine2 | aug1 | norm | stack | all | 0.625 | 0.089 | -0.328 | 2.85 | 0.452 | 0.337 |
| mae_adapted | histmatch | aug1 | norm | stack | all | 0.625 | 0.213 | 0.328 | 2.8 | 0.48 | 0.776 |
| mae_adapted | hybrid | aug1 | norm | stack | all | 0.624 | 0.021 | -0.425 | 2.75 | 0.502 | 0.509 |
| mae_adapted | none | aug1 | norm | stack | all | 0.617 | 0.223 | 0.007 | 2.6 | 0.557 | 0.352 |
| mae_scratch | affine2 | aug1 | norm | stack | all | 0.401 | -0.338 | -0.948 | 2.8 | 0.552 | 0.037 |
| mae_scratch | histmatch | aug1 | norm | stack | all | 0.46 | 0.005 | 0.041 | 1.75 | 0.439 | 0.173 |
| mae_scratch | hybrid | aug1 | norm | stack | all | 0.361 | -0.315 | -0.897 | 2.3 | 0.439 | 0.057 |
| mae_scratch | none | aug1 | norm | stack | all | 0.321 | -0.461 | -1.768 | 2.4 | 0.552 | 0.038 |
| ots_dinov2 | affine2 | aug1 | raw | stack | all | 0.361 | 0.012 | -0.15 | 2.0 | 0.398 | 0.107 |
| ots_dinov2 | histmatch | aug1 | raw | stack | all | 0.336 | -0.042 | -0.104 | 2.15 | 0.38 | 0.111 |
| ots_dinov2 | hybrid | aug1 | raw | stack | all | 0.342 | -0.042 | -0.077 | 2.2 | 0.416 | 0.115 |
| ots_micronet | affine2 | aug1 | raw | stack | all | 0.236 | -0.034 | 0.077 | 1.95 | 0.394 | 0.16 |
| ots_micronet | histmatch | aug1 | raw | stack | all | 0.252 | -0.108 | 0.052 | 2.15 | 0.344 | 0.174 |
| ots_micronet | hybrid | aug1 | raw | stack | all | 0.301 | -0.076 | 0.082 | 2.15 | 0.353 | 0.175 |
| ots_vitmae | affine2 | aug1 | raw | stack | all | 0.432 | 0.051 | 0.081 | 2.25 | 0.357 | 0.255 |
| ots_vitmae | histmatch | aug1 | raw | stack | all | 0.412 | 0.117 | 0.314 | 2.3 | 0.407 | 1.052 |
| ots_vitmae | hybrid | aug1 | raw | stack | all | 0.48 | 0.184 | 0.265 | 2.4 | 0.407 | 0.931 |
| vae_a | affine2 | aug1 | raw | stack | baseline | -0.036 | -0.212 | -0.174 | 1.2 | 0.235 | 0.214 |
| vae_a | histmatch | aug1 | raw | stack | baseline | 0.158 | -0.175 | -0.159 | 0.85 | 0.226 | 0.375 |
| vae_a | hybrid | aug1 | raw | stack | baseline | 0.171 | -0.172 | -0.161 | 0.8 | 0.235 | 0.438 |
| vae_a | none | aug1 | raw | stack | baseline | 0.07 | -0.222 | -0.171 | 0.65 | 0.127 | 0.215 |
| vae_b | affine2 | aug1 | raw | Inlens | all | -0.052 | -0.225 |  | 1.65 | 0.235 | 0.134 |
| vae_b | histmatch | aug1 | raw | Inlens | all | -0.028 | -0.231 |  | 0.95 | 0.204 | 0.242 |
| vae_b | hybrid | aug1 | raw | Inlens | all | -0.04 | -0.232 |  | 0.9 | 0.195 | 0.237 |
| vae_b | none | aug1 | raw | Inlens | all | 0.49 | -0.304 |  | 1.15 | 0.172 | 0.182 |
| vae_c | affine2 | aug1 | raw | stack | baseline | 0.818 | -0.069 | -0.036 | 1.7 | 0.208 | 0.444 |
| vae_c | histmatch | aug1 | raw | stack | baseline | 0.814 | -0.001 | 0.069 | 0.8 | 0.267 | 0.711 |
| vae_c | hybrid | aug1 | raw | stack | baseline | 0.844 | -0.03 | 0.034 | 0.65 | 0.167 | 0.714 |
| vae_c | hybrid | aug2 | raw | stack | baseline | 0.819 | -0.04 | -0.003 | 0.8 | 0.204 | 0.383 |
| vae_c | none | aug1 | raw | stack | baseline | 0.824 | -0.066 | 0.026 | 1.05 | 0.176 | 0.261 |

## Classification: field accuracy by harmonisation method (raw input, aug1; chance = 0.33)

If accuracy survives `hybrid`, the batch signal is not the Batch_3 black-level/gain artefact.

| arch | view | affine2 | gmm | histmatch | hybrid | none |
|---|---|---|---|---|---|---|
| dinov2_ft | BSE |  | 0.677 |  | 0.644 | 0.677 |
| dinov2_ft | Inlens |  | 0.604 |  | 0.57 | 0.604 |
| dinov2_ft | SE_type |  | 0.604 |  | 0.604 | 0.644 |
| dinov2_ft | stack | 0.637 | 0.637 | 0.632 | 0.599 | 0.632 |
| dinov2_linear | stack | 0.604 | 0.569 | 0.632 | 0.632 | 0.604 |
| effb4_imnet | stack | 0.604 | 0.637 | 0.632 | 0.632 | 0.609 |
| resnet18_imnet | BSE |  | 0.518 |  | 0.599 | 0.644 |
| resnet18_imnet | Inlens |  | 0.637 |  | 0.566 | 0.637 |
| resnet18_imnet | SE_type |  | 0.722 |  | 0.542 | 0.485 |
| resnet18_imnet | stack | 0.592 | 0.637 | 0.599 | 0.566 | 0.604 |
| resnet18_scratch | stack | 0.559 | 0.609 | 0.575 | 0.599 | 0.597 |

### Batch_3 field recall by harmonisation method

| arch | view | affine2 | gmm | histmatch | hybrid | none |
|---|---|---|---|---|---|---|
| dinov2_ft | BSE |  | 0.941 |  | 0.941 | 0.941 |
| dinov2_ft | Inlens |  | 0.824 |  | 0.824 | 0.824 |
| dinov2_ft | SE_type |  | 0.882 |  | 0.882 | 0.882 |
| dinov2_ft | stack | 0.882 | 0.824 | 0.824 | 0.824 | 0.882 |
| dinov2_linear | stack | 0.824 | 0.765 | 0.824 | 0.824 | 0.824 |
| effb4_imnet | stack | 0.824 | 0.882 | 0.824 | 0.824 | 0.824 |
| resnet18_imnet | BSE |  | 0.824 |  | 0.824 | 0.882 |
| resnet18_imnet | Inlens |  | 0.882 |  | 0.765 | 0.882 |
| resnet18_imnet | SE_type |  | 0.941 |  | 0.765 | 0.647 |
| resnet18_imnet | stack | 0.765 | 0.882 | 0.824 | 0.765 | 0.882 |
| resnet18_scratch | stack | 0.706 | 0.882 | 0.765 | 0.824 | 0.824 |

## Batch classification from features only (leave-one-field-out)

| name | clf | n_crops | crop_acc | field_acc | field_f1 | field_acc_Batch_1 | field_acc_Batch_2 | field_acc_Batch_3 |
|---|---|---|---|---|---|---|---|---|
| imaging_stats | logreg | 1521 | 0.637 | 0.645 | 0.532 | 0.429 | 0.286 | 0.882 |
| imaging_stats | knn5 | 1521 | 0.609 | 0.613 | 0.486 | 0.286 | 0.286 | 0.882 |
| kpi_all10_imputed | knn5 | 1521 | 0.407 | 0.613 | 0.395 | 0.286 | 0.0 | 1.0 |
| kpi_gated5 | knn5 | 1464 | 0.398 | 0.548 | 0.236 | 0.0 | 0.0 | 1.0 |
| kpi_gated5 | logreg | 1464 | 0.425 | 0.355 | 0.345 | 0.286 | 0.429 | 0.353 |
| kpi_all10_imputed | logreg | 1521 | 0.382 | 0.323 | 0.322 | 0.286 | 0.429 | 0.294 |

## Batch classification from saved embeddings (leave-one-field-out; top 15 by field accuracy)

| name | clf | n_crops | crop_acc | field_acc | field_f1 | field_acc_Batch_1 | field_acc_Batch_2 | field_acc_Batch_3 |
|---|---|---|---|---|---|---|---|---|
| dino_ft|stack|norm|all|aug1|kpi=gated | logreg | 1521 | 0.636 | 0.742 | 0.653 | 0.571 | 0.429 | 0.941 |
| mae_scratch|BSE|raw|all|aug1|kpi=gated | logreg | 1521 | 0.609 | 0.71 | 0.633 | 0.571 | 0.429 | 0.882 |
| dino_ft|stack|raw|all|aug1|kpi=gated | logreg | 1521 | 0.619 | 0.71 | 0.598 | 0.571 | 0.286 | 0.941 |
| dino_ft|stack|norm|all|aug1|kpi=gated | knn5 | 1521 | 0.602 | 0.71 | 0.605 | 0.429 | 0.429 | 0.941 |
| mae_scratch|stack|norm|all|aug1|kpi=gated | logreg | 1521 | 0.652 | 0.71 | 0.626 | 0.714 | 0.286 | 0.882 |
| mae_scratch|Inlens|raw|all|aug1|kpi=gated | logreg | 1521 | 0.566 | 0.677 | 0.603 | 0.571 | 0.429 | 0.824 |
| mae_adapted|stack|raw|baseline|aug1|kpi=gated | logreg | 1521 | 0.615 | 0.677 | 0.586 | 0.429 | 0.429 | 0.882 |
| mae_adapted|stack|raw|all|aug1|kpi=gated | logreg | 1521 | 0.635 | 0.677 | 0.586 | 0.429 | 0.429 | 0.882 |
| mae_adapted|Inlens|raw|all|aug1|kpi=gated | logreg | 1521 | 0.637 | 0.677 | 0.58 | 0.571 | 0.286 | 0.882 |
| mae_scratch|stack|norm|all|aug1|kpi=gated | knn5 | 1521 | 0.663 | 0.677 | 0.542 | 0.571 | 0.143 | 0.941 |
| mae_scratch|BSE|raw|all|aug1|kpi=gated | knn5 | 1521 | 0.525 | 0.645 | 0.515 | 0.286 | 0.286 | 0.941 |
| mae_scratch|SE_type|raw|all|aug1|kpi=gated | logreg | 1521 | 0.529 | 0.645 | 0.614 | 0.714 | 0.429 | 0.706 |
| vae_a|stack|raw|all|aug1|kpi=gated | logreg | 1521 | 0.404 | 0.645 | 0.477 | 1.0 | 0.0 | 0.765 |
| dino_ft|stack|raw|all|aug1|kpi=gated | logreg | 1521 | 0.594 | 0.645 | 0.537 | 0.429 | 0.286 | 0.882 |
| mae_adapted|SE_type|raw|all|aug1|kpi=gated | logreg | 1521 | 0.64 | 0.645 | 0.535 | 0.429 | 0.286 | 0.882 |
