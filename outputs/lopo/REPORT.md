# Leave-one-parent-out (LOPO) vs leave-one-site-out (LOSO)

Script: `scripts/modal_lopo.py` (all fitting and permutations on Modal; `modal run scripts/modal_lopo.py`).
Parent grouping: organiser-confirmed 13 parents covering the 31 labelled sites (`PARENTS` dict in the script, asserted).
LOPO = 13 folds, every labelled crop of one parent held out together. LOSO = 31 folds.
Permutation test: 200 site-level label shuffles (`np.random.default_rng(seed)`, seeds 0..199, same site order and
seeds for every model and protocol), statistic = balanced accuracy, p = (1 + #{null >= obs}) / 201.

## Models (pre-stated before running, no tuning)

- `nb`: baseline robust naive Bayes (`pmdb/fingerprint.py`), 16 fingerprint features.
- `nb_parentavg` (variant a): training crops collapsed to one row per (parent, batch) pair (feature-wise median)
  before standardisation and per-batch params, so big parents do not dominate. Expected to help LOPO.
- `nb_parentctr` (variant b): label-free within-parent centring (crop minus median of its parent's crops), same for
  train and test crops; uses parent membership only, never labels. Single-crop parents become all-zero.
- `rf_kpi`: two-stage random forest, KPI arm (`pmdb/classify/model.py`), re-run with group folds (cheap, ~minutes on Modal).

## Results (31 labelled sites; majority baseline acc 0.548, chance bacc 0.333)

| model | protocol | acc | bacc | recall B1/B2/B3 | perm null mean | perm p (bacc) |
|---|---|---|---|---|---|---|
| nb | LOSO | 0.677 | 0.636 | 0.71/0.43/0.76 | 0.339 | 0.015 |
| nb | LOPO | 0.677 | 0.636 | 0.71/0.43/0.76 | 0.323 | 0.010 |
| nb_parentavg | LOSO | 0.548 | 0.529 | 0.57/0.43/0.59 | 0.317 | 0.045 |
| nb_parentavg | LOPO | 0.484 | 0.462 | 0.43/0.43/0.53 | 0.317 | 0.109 |
| nb_parentctr | LOSO | 0.452 | 0.499 | 0.86/0.29/0.35 | 0.319 | 0.060 |
| nb_parentctr | LOPO | 0.516 | 0.510 | 0.86/0.14/0.53 | 0.322 | 0.060 |
| rf_kpi | LOSO | 0.581 | 0.437 | 0.29/0.14/0.88 | 0.278 | 0.045 |
| rf_kpi | LOPO | 0.290 | 0.204 | 0.00/0.14/0.47 | 0.284 | 0.866 |

Confusions (rows true B1/B2/B3, columns predicted B1/B2/B3):

| model | LOSO | LOPO |
|---|---|---|
| nb | [5 0 2] [3 3 1] [2 2 13] | [5 0 2] [3 3 1] [2 2 13] |
| nb_parentavg | [4 1 2] [2 3 2] [3 4 10] | [3 1 3] [2 3 2] [3 5 9] |
| nb_parentctr | [6 0 1] [1 2 4] [5 6 6] | [6 0 1] [1 1 5] [3 5 9] |
| rf_kpi | [2 0 5] [0 1 6] [0 2 15] | [0 0 7] [0 1 6] [2 7 8] |

Per-site predictions: `predictions.csv`; full metrics: `evaluation.json`.

## Interpretation

- The baseline NB does **not** rely on sibling crops: LOPO gives the same accuracy (0.677) and balanced accuracy
  (0.636) as LOSO (two sites swap: kbdh4tri becomes wrong, utfgcjfa becomes right), perm p = 0.010. Its batch signal
  (Si depth profile, pair correlation, K15 tile spread) carries over to parents it has never seen.
- The random forest on scalar KPIs **collapses under LOPO** (bacc 0.204, below chance, p = 0.87). Its weak LOSO
  score (0.437) came from recognising sibling crops of the same parent, not from batch features.
- Neither group-aware NB variant beats the baseline under either protocol. Parent-averaging throws away within-parent
  batch contrast (6 of 13 parents span two or three batches) and halves the training rows; parent-centring erases the
  between-parent signal and zeroes the two single-crop parents. Both were pre-stated; no further variants were tried.
- Caveat: the permutation null shuffles labels at site level, so it does not preserve the parent structure. A
  parent-preserving null is not well defined here because parents are split across batches. With n = 31 every
  bacc is noisy (one site moves B1/B2 recall by 0.14).

## Organiser rubric (expected score per site, max 2)

`modal run scripts/modal_lopo.py::rubric` -> `rubric_table.md`, `rubric.json`, `predictions_rubric.csv`.
Rule: "high" iff P(predicted) > 0.5; high and correct = 2, high and wrong = 0, low = 1 whether right or wrong (low-correct = 1 assumed).
NB probabilities: `fixed` = softmax(-T * score) with T = number of features (16); since the NB score is the
mean negative Laplace log-likelihood, this is the naive-Bayes posterior with a flat prior. `cal` = T picked from
{0.25, ..., 64} per outer fold by minimum log-loss of an inner same-protocol CV on that fold's training sites only
(chosen T = 4-8 for `nb`). RF: its own two-stage probabilities. Reliability bins pool the out-of-fold
predictions over all folds (per-fold bins would hold 1-4 sites).

| model | probs | protocol | acc | rubric mean (max 2) | n high | acc high | acc low | rel <0.5 n/acc | rel 0.5-0.7 n/acc | rel >0.7 n/acc |
|---|---|---|---|---|---|---|---|---|---|---|
| nb | fixed | LOSO | 0.677 | 1.323 | 30 | 0.67 | 1.00 | 1/1.00 | 10/0.60 | 20/0.70 |
| nb | cal | LOSO | 0.677 | 1.226 | 21 | 0.67 | 0.70 | 10/0.70 | 15/0.73 | 6/0.50 |
| nb | fixed | LOPO | 0.677 | 1.355 | 29 | 0.69 | 0.50 | 2/0.50 | 7/0.57 | 22/0.73 |
| nb | cal | LOPO | 0.677 | 1.258 | 20 | 0.70 | 0.64 | 11/0.64 | 17/0.76 | 3/0.33 |
| nb_parentavg | fixed | LOSO | 0.548 | 1.097 | 29 | 0.55 | 0.50 | 2/0.50 | 3/0.33 | 26/0.58 |
| nb_parentavg | cal | LOSO | 0.548 | 1.032 | 15 | 0.53 | 0.56 | 16/0.56 | 14/0.57 | 1/0.00 |
| nb_parentavg | fixed | LOPO | 0.484 | 1.000 | 28 | 0.50 | 0.33 | 3/0.33 | 6/0.33 | 22/0.55 |
| nb_parentavg | cal | LOPO | 0.484 | 1.032 | 11 | 0.55 | 0.45 | 20/0.45 | 10/0.50 | 1/1.00 |
| nb_parentctr | fixed | LOSO | 0.452 | 0.903 | 29 | 0.45 | 0.50 | 2/0.50 | 9/0.33 | 20/0.50 |
| nb_parentctr | cal | LOSO | 0.452 | 1.065 | 18 | 0.56 | 0.31 | 13/0.31 | 15/0.53 | 3/0.67 |
| nb_parentctr | fixed | LOPO | 0.516 | 1.032 | 31 | 0.52 | - | 0/- | 11/0.55 | 20/0.50 |
| nb_parentctr | cal | LOPO | 0.516 | 0.935 | 16 | 0.44 | 0.60 | 15/0.60 | 13/0.38 | 3/0.67 |
| rf_kpi | rf | LOSO | 0.581 | 1.194 | 28 | 0.61 | 0.33 | 3/0.33 | 28/0.61 | 0/- |
| rf_kpi | rf | LOPO | 0.290 | 0.839 | 21 | 0.38 | 0.10 | 10/0.10 | 19/0.42 | 2/0.00 |

Reading: under this rubric "low" always pays 1, so declaring high is worth it whenever accuracy exceeds 0.5;
answering "low" everywhere scores 1.00, answering "high" everywhere scores 2 x accuracy. The baseline NB with fixed
T = 16 is the best model under both protocols (1.323 LOSO, 1.355 LOPO; it says high on 29-30 of 31 sites, which is
correct behaviour given 0.68 accuracy). Calibrating T inside the folds makes it better calibrated by log-loss but
less often "high", and it scores lower (1.226 / 1.258), because low-confidence sites it demotes are still right
~65-70% of the time. No model loses on accuracy but wins on rubric score: both variants and the RF score at or
below the baseline and the always-low floor of 1.0 under LOPO (RF LOPO 0.839 is below the floor).

## Held-out sites

No variant beats the baseline, so the baseline `nb` held-out assignment stands (unchanged from
`outputs/fingerprint/heldout_predictions.csv`): 3e122cbj -> Batch_1, fn0mhxef -> Batch_3, xrv9xvzb -> Batch_2.
Variant predictions are in `heldout_predictions.csv` for reference only. Note: the parents imply fn0mhxef shares a
parent with two Batch_2 crops (G2048) and xrv9xvzb with three Batch_3 crops (G2088), 3e122cbj with two Batch_1 crops
(G2316); since parents are split across batches by design, parent membership is suggestive, not a label.
