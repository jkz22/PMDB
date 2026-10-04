# Held-out calls scored against the released labels

Truth (organiser, Modelling-session chat 2026-10-04): **3e122cbj = Batch 2, fn0mhxef = Batch 1, xrv9xvzb = Batch 3.**

| method | kind | 3e122cbj | fn0mhxef | xrv9xvzb | correct | LOSO acc | LOSO recall B1/B2/B3 |
|---|---|---|---|---|---|---|---|
| fingerprint (16 curve features, conformal NB) | arrangement | Batch_1 | Batch_3 | Batch_2 | 0/3 | 0.68 | 0.71/0.43/0.76 |
| fingerprint + stretch S01-S04 | arrangement | Batch_1 | Batch_3 | **Batch_3** ✓ | 1/3 |  |  |
| fingerprint + functional F | arrangement | Batch_1 | Batch_2 | **Batch_3** ✓ | 1/3 |  |  |
| two-stage RF, KPI+FEM arm | arrangement | Batch_1 | Batch_3 | Batch_2 | 0/3 |  |  |
| two-stage RF, FEM arm | arrangement | Batch_1 | Batch_3 | Batch_2 | 0/3 |  |  |
| two-stage RF, KPI arm (final) | scalar KPIs | Batch_1 | Batch_3 | **Batch_3** ✓ | 1/3 |  |  |
| XGBoost reliable_kpis | scalar KPIs | Batch_1 | Batch_3 | **Batch_3** ✓ | 1/3 | 0.45 | 0.29/0.00/0.71 |
| XGBoost fingerprint16 | arrangement | Batch_3 | Batch_3 | Batch_2 | 0/3 | 0.68 | 0.43/0.57/0.82 |
| XGBoost reliable_kpis+functional | arrangement | Batch_1 | Batch_3 | Batch_2 | 0/3 | 0.52 | 0.14/0.14/0.82 |
| XGBoost all_reliable | arrangement | Batch_1 | Batch_3 | Batch_2 | 0/3 | 0.61 | 0.43/0.29/0.82 |
| tile vote all|lr|16 | tiles (scalar + per-tile depth) | Batch_1 | Batch_3 | **Batch_3** ✓ | 1/3 | 0.61 | 0.29/0.00/1.00 |
| nearest mean grey_raw | grey level (acquisition) | Batch_1 | Batch_2 | **Batch_3** ✓ | 1/3 | 0.61 | 0.57/0.43/0.71 |
| nearest mean grey_harm | grey level (acquisition) | Batch_1 | Batch_2 | **Batch_3** ✓ | 1/3 | 0.65 | 0.71/0.29/0.76 |
| nearest mean grey_depth_raw | grey level (acquisition) | Batch_3 | Batch_2 | **Batch_3** ✓ | 1/3 | 0.55 | 0.29/0.86/0.53 |
| nearest mean grey_depth_harm | grey level (acquisition) | Batch_3 | **Batch_1** ✓ | Batch_2 | 1/3 | 0.42 | 0.29/0.14/0.59 |
| nearest mean grey_harm_moments | grey level (acquisition) | Batch_1 | Batch_2 | **Batch_3** ✓ | 1/3 | 0.61 | 0.71/0.43/0.65 |
| nearest mean grey_harm_std_only | grey level (acquisition) | Batch_1 | Batch_2 | Batch_1 | 0/3 | 0.29 | 0.43/0.43/0.18 |
| nearest mean grey_harm_hist | grey level (acquisition) | Batch_1 | Batch_2 | **Batch_3** ✓ | 1/3 | 0.68 | 0.71/0.29/0.82 |
| nearest mean grey_harm_graphite_only | grey level (acquisition) | Batch_1 | Batch_2 | **Batch_3** ✓ | 1/3 | 0.74 | 0.86/0.71/0.71 |
| nearest mean grey_harm_pore_only | grey level (acquisition) | Batch_1 | **Batch_1** ✓ | **Batch_3** ✓ | 2/3 | 0.71 | 0.86/0.43/0.76 |
| nearest mean overlay_frac | scalar KPIs | Batch_1 | Batch_2 | Batch_2 | 0/3 | 0.48 | 0.29/0.43/0.59 |
| nearest mean overlay_depth | scalar KPIs | Batch_3 | Batch_3 | Batch_1 | 0/3 | 0.48 | 0.57/0.43/0.47 |
| nearest mean fingerprint16 | arrangement | Batch_3 | Batch_3 | Batch_2 | 0/3 | 0.65 | 0.71/0.57/0.65 |
| decision card (consensus call) | ensemble | Batch_1 | Batch_3 | Batch_2 | 0/3 |  |  |
| one-class kpi_v1 score (95th pct of Batch 3) | binary off/baseline | **off** ✓ | baseline | **baseline** ✓ | 2/3 |  |  |
| one-class fingerprint score (95th pct of Batch 3) | binary off/baseline | baseline | baseline | **baseline** ✓ | 1/3 |  |  |
| one-class functional score (95th pct of Batch 3) | binary off/baseline | **off** ✓ | baseline | **baseline** ✓ | 2/3 |  |  |
| noise-aware KPI deviation (any KPI off) | binary off/baseline | **off** ✓ | baseline | **baseline** ✓ | 2/3 |  |  |
| Modelling session binary off-detector (CNN, every route) | binary off/baseline | **off** ✓ | **off** ✓ | **baseline** ✓ | 3/3 |  |  |

Three-class methods: 24. Distribution of correct calls (observed vs expected if every method guessed uniformly):

| correct | observed | expected at chance |
|---|---|---|
| 0/3 | 11 | 7.1 |
| 1/3 | 12 | 10.7 |
| 2/3 | 1 | 5.3 |
| 3/3 | 0 | 0.9 |

Mean correct per three-class method: 0.58 (chance 1.00).
