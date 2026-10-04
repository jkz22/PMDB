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

## Held-out sites

No variant beats the baseline, so the baseline `nb` held-out assignment stands (unchanged from
`outputs/fingerprint/heldout_predictions.csv`): 3e122cbj -> Batch_1, fn0mhxef -> Batch_3, xrv9xvzb -> Batch_2.
Variant predictions are in `heldout_predictions.csv` for reference only. Note: the parents imply fn0mhxef shares a
parent with two Batch_2 crops (G2048) and xrv9xvzb with three Batch_3 crops (G2088), 3e122cbj with two Batch_1 crops
(G2316); since parents are split across batches by design, parent membership is suggestive, not a label.
