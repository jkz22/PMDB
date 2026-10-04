## Plan ready
- **Plan file**: `/Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/plans/tile-signal-checks.md`
- **Steps**: 6 steps, complexity LOW-MEDIUM
- **Key decisions**:
  - **Script and outputs**: `scripts/tile_signal_checks.py` writes 9 files to `outputs/pooling_checks/`: 6 CSVs, 2 figures (`fig_tile_probs.png`, `fig_variance.png`) and `results.md`. The script generates `results.md` itself, using a local markdown-table helper because `tabulate` is not installed.
  - **Features**: the 16 per-tile columns from `catalogue_columns()`, minus any column with zero variance. Only `K16_si_graphite_dist_median_um` is dropped, since it is 0 in every tile, leaving 15 features.
  - **Check 1 classifier**: impute missing values with the median, standardise, then a class-weighted L2 logistic regression (C=1, random_state=0), refitted inside each fold. It runs leave-one-site-out in 3 variants: all three batches, B1 vs B3, and B2 vs B3.
  - **Site categories**: each site is labelled by its number of correct tiles and its highest p_true:
    - `diffuse_correct`: at least 3 of 4 tiles correct.
    - `witness`: 1 or 2 tiles correct and the highest tile p_true is at least 0.7.
    - `mixed`: 1 or 2 tiles correct and the highest tile p_true is below 0.7.
    - `miss`: no tiles correct.
  - **Batch 3 false-alarm rate**: the fraction of Batch 3 sites with any tile at p(other batch) of at least 0.7. This is the D10 calibration.
  - **Verdict per comparison against B3** (first match wins):
    1. `localised` if at least 2 witness sites and the witness fraction is above the false-alarm rate.
    2. `diffuse` if at least 4 sites are `diffuse_correct`.
    3. `weak` if `diffuse_correct` plus `witness` is at most 2.
    4. Otherwise `inconclusive`.
  - **Pooling side note**: site accuracy under mean pooling and under max pooling. For the binary variants, max pooling predicts the non-B3 batch if any tile has p(non-B3) of at least 0.5.
  - **Check 2 variance**: ICC(1) per feature over all sites and within each batch, using the formula that stays valid if some tiles are missing. Also the share of variance from batch, from site within batch, and from tile within site, plus within-site SD ratios of B1 and B2 against B3.
  - **Check 2 single-tile flag**: computed for each comparison against B3 and scaled by the pooled within-site SD.
    - `tail_driven` if the batch difference in site max or site min is at least 1.0 and at least 2 times the difference in site median.
    - Otherwise `diffuse_shift` if the median difference is at least 1.0.
    - Otherwise `none`.
  - **Overall reading** (the script applies these rules and writes one line per comparison):
    - "localised" if the comparison's verdict is `localised` or at least 3 features are `tail_driven`.
    - "diffuse" if the verdict is `diffuse` and `diffuse_shift` flags outnumber `tail_driven` flags.
    - Otherwise "mixed/inconclusive".
- **Open questions for dispatcher**: none

Things I checked against the repo:
- `tile_kpis.csv` has 124 rows (31 sites x 4 tiles) and no missing values today. The code still handles them.
- Installed: scikit-learn 1.3.2, scipy 1.14, matplotlib 3.10. Not installed: tabulate.
- One Batch 2 site and three Batch 3 sites use the SE detector rather than ETD. `results.md` lists this as a caveat only; it is not modelled.

Verification in the plan:
- Expected row counts are 316 / 79 / 7 / 3 / 15 / 465.
- Two runs must give byte-identical CSVs and `results.md`, compared by md5.
- `pytest -q -m "not data"` must give the same result as before the change.

Out of scope: checks 3 and 4, anything under `data/` or the held-out sites, edits to `pmdb/`, and commits.
