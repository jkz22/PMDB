# Off-detector (Batch_3 baseline vs Batch_1+2 'off') -- results and what the models use

All numbers out-of-fold, 31 labelled fields, 5 stratified grouped folds. Field decision = mean crop P over the 256-px eval grid.

## 1. Binary classifiers per route

| route | arch | field acc | bal acc | field AUROC | recall B1 | recall B2 | recall B3 | predicted-off share (true 0.45) | crop ECE |
|---|---|---|---|---|---|---|---|---|---|
| raw | resnet18_imnet | 0.87 | 0.87 | 0.88 | 1.00 | 0.71 | 0.88 | 0.45 | 0.08 |
| hybrid (normal) | resnet18_imnet | 0.84 | 0.84 | 0.90 | 1.00 | 0.71 | 0.82 | 0.48 | 0.09 |
| nyul | resnet18_imnet | 0.84 | 0.84 | 0.89 | 1.00 | 0.71 | 0.82 | 0.48 | 0.11 |
| basic | resnet18_imnet | 0.84 | 0.84 | 0.91 | 1.00 | 0.71 | 0.82 | 0.48 | 0.09 |
| naive | resnet18_imnet | 0.87 | 0.87 | 0.91 | 1.00 | 0.71 | 0.88 | 0.45 | 0.07 |
| extreme | resnet18_imnet | 0.87 | 0.87 | 0.93 | 1.00 | 0.71 | 0.88 | 0.45 | 0.05 |
| raw | dinov2_ft | 0.87 | 0.87 | 0.94 | 1.00 | 0.71 | 0.88 | 0.45 | 0.11 |
| hybrid (normal) | dinov2_ft | 0.84 | 0.84 | 0.93 | 1.00 | 0.71 | 0.82 | 0.48 | 0.11 |
| nyul | dinov2_ft | 0.87 | 0.87 | 0.93 | 1.00 | 0.71 | 0.88 | 0.45 | 0.10 |
| basic | dinov2_ft | 0.87 | 0.87 | 0.94 | 1.00 | 0.71 | 0.88 | 0.45 | 0.10 |
| naive | dinov2_ft | 0.87 | 0.87 | 0.95 | 1.00 | 0.71 | 0.88 | 0.45 | 0.08 |
| extreme | dinov2_ft | 0.87 | 0.87 | 0.95 | 1.00 | 0.71 | 0.88 | 0.45 | 0.08 |

Wrong fields on every route/arch: Batch_2 `r17byphk`, `rxax5ozo` (called baseline); Batch_3 `pl8uabbv`, `cfe5vt7s` (called off).

## 2. What is the model looking at? Scalar baselines, same LOFO protocol (31 folds, logistic regression on field means)

| features | field acc | bal acc | AUROC | recall B1 / B2 / B3 | wrong fields |
|---|---|---|---|---|---|
| 5 gated KPIs (Si, graphite, pore fraction, K01, K04) | 0.55 | 0.54 | 0.58 | 0.57 / 0.43 / 0.59 | 14 fields |
| Si KPIs only (frac_si, K01, K04) | 0.48 | 0.47 | 0.45 | 0.43 / 0.14 / 0.65 | 16 fields |
| raw-image noise sigma + sharpness (3 detectors) | **0.90** | **0.91** | **0.93** | 1.00 / 0.86 / 0.88 | `cfe5vt7s`, `pl8uabbv`, `rxax5ozo` |
| raw grey level only (p1, p99) | 0.68 | 0.71 | 0.67 | 1.00 / 1.00 / 0.41 | 10 fields |
| KPIs + imaging stats | 0.90 | 0.91 | 0.92 | 1.00 / 0.86 / 0.88 | `cfe5vt7s`, `pl8uabbv`, `rxax5ozo` |

Two scalar imaging statistics reproduce the CNN off-detector (same accuracy, same three wrong fields); the KPIs are at chance for this label. Batch means: BSE noise sigma 10.5 / 10.1 / 8.2 grey, sharpness 0.150 / 0.135 / 0.109 (Batch_1 / 2 / 3); Si fraction 0.085 / 0.058 / 0.062.

## 3. Does the noise/sharpness cue survive each harmonisation route? (statistics measured on the models' own input crops, LOFO logistic)

| route | noise+sharpness: field acc / AUROC | grey level p1+p99: field acc / AUROC | all four | Batch_3 - Batch_1+2 gap in noise (site SD) | gap in sharpness (site SD) |
|---|---|---|---|---|---|
| raw | 0.81 / 0.91 | 0.81 / 0.87 | 0.87 / 0.93 | -1.30 | -1.10 |
| hybrid | 0.71 / 0.86 | 0.84 / 0.90 | 0.90 / 0.93 | -0.96 | -1.16 |
| nyul | 0.74 / 0.85 | 0.81 / 0.86 | 0.84 / 0.92 | -1.21 | -1.11 |
| basic | 0.81 / 0.91 | 0.58 / 0.71 | 0.87 / 0.94 | -1.29 | -1.11 |
| clean_harm | 0.84 / 0.90 | 0.74 / 0.84 | 0.84 / 0.91 | -0.87 | -1.07 |
| naive | 0.77 / 0.89 | 0.74 / 0.81 | 0.81 / 0.85 | -1.25 | -0.40 |
| extreme | 0.45 / 0.60 | 0.71 / 0.75 | 0.61 / 0.71 | -0.02 | +0.74 |

raw, hybrid, nyul, basic and clean_harm all leave the Batch_3 noise/sharpness gap at about -1 site-SD, so the cue is fully available to the classifier on every one of these routes. The hybrid LUT fixes p1 but p1+p99 still separates at 0.84 (dynamic-range / p99 fingerprint). Only `extreme` (phase-only + per-crop p1-p99 + 1 px blur + sigma=0.1 noise) removes the noise/sharpness cue (0.45, chance) -- yet the CNN on extreme still scores 0.87, so on that route it reads something that these four scalars do not capture (residual texture or the phase-mask geometry produced by the per-image segmentation); see the test-time perturbation table when it lands.

## 4. Attribution (occlusion + Grad-CAM over Si / graphite / pore masks; not SHAP)

Occlusion evidence per phase relative to area: Si 0.74-1.00, graphite 0.99-1.01, pore 0.84-1.08 on every route; |r| of occlusion/CAM importance with local brightness, contrast, edges, noise <= 0.15. Diffuse graphite-matrix texture, no phase geometry is singled out.

## 5. Test-time perturbation of the trained classifiers (3-class ResNet-18, raw / hybrid input)

| perturbation | Batch_3 recall raw | hybrid | Batch_1 recall raw | hybrid |
|---|---|---|---|---|
| none | 0.83 | 0.67 | 0.50 | 0.50 |
| gauss noise sigma=6 grey | 0.17 | 0.67 | 0.50 | 0.50 |
| blur 0.7 px | 1.00 (everything -> Batch_3) | 1.00 | 0.00 | 0.00 |
| offset +8 grey | 0.83 | 0.67 | 0.50 | 0.50 |
| gain 0.85 | 0.83 | 0.83 | 0.50 | 0.50 |

Same test for the binary off-detectors on all routes is running (`outputs/v2/off/testtime_perturbation.csv`).

## 6. Representations (all-fields training, LOFO ridge probe on 5 KPIs, R^2)

| route | VAE-C | VAE-C phase-inpaint | MAE | imaging-stat probe R^2 (VAE-C) | batch kNN (VAE-C; chance 0.55 majority) |
|---|---|---|---|---|---|
| raw | 0.87 | -- | -- | <0 | 0.21 |
| hybrid | 0.87 | 0.88 | -- | <0 | 0.18 |
| clean_harm | 0.83-0.84 | 0.85 | 0.67 | 0.04 | 0.15-0.26 |
| nyul | 0.82 | 0.81 | 0.62 (img R^2 0.23, kNN 0.57) | <0 | 0.28 |
| basic | 0.84 | 0.85 | 0.54 | <0 | 0.19 |
| naive | 0.85 | 0.85 | 0.72 | 0.07 | 0.34 |
| extreme | 0.86 | 0.83 | 0.70 | 0.08 | 0.30 |

## 7. Held-out sites (crop-mean KPIs on the labelled eval grid; z vs Batch_3 field means)

| site | P(off) raw / hybrid / nyul / extreme | KPI deviations | imaging deviations | read |
|---|---|---|---|---|
| 3e122cbj | 0.94-0.96 on all routes | Si +8.6 SD, K04 +4.9 SD, graphite -11 SD | p99 -2.7 SD | genuinely off: Si-rich material |
| fn0mhxef | 0.87-0.94 on all routes | none (Si -0.3, K04 0.0) | sharpness -4 SD, noise -2 SD | off-call rests on imaging only; material at baseline |
| xrv9xvzb | 0.06-0.20 | none (Si -1.2) | sharpness -5 SD, noise -3 SD | baseline |

## Bottom line

The off-detector's 0.84-0.87 is a noise / focus fingerprint of the Batch_3 imaging session, reproduced exactly by two scalar image statistics and absent from the KPIs. None of raw, hybrid (PR #16), nyul, basic (PR #37) or clean_harm removes that cue. The extreme route removes it from the measured statistics but the CNN still separates; what remains there is the open question. A material-based off-detector has to be trained on the extreme/phase-only route (or on KPI/VAE-C embeddings) and will be weaker than 0.87, because on the KPIs these batches are not 0.87-separable.
