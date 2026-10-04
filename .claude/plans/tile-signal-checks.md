# Tile-signal checks (D10 pre-checks)

## Plan: tile-signal checks 1-2

### 1. Context

Decision D10 in `.claude/plans/fem-swelling.md` leaves the site-pooling rule open. The risk is that the batch signal
sits in one or two "witness" tiles, and mean pooling would hide it. Checks 1-2 test this on the existing per-tile KPIs
in `outputs/kpis/tile_kpis.csv`: 124 rows = 31 sites x 4 full-height tiles, with B1/B2/B3 = 7/7/17 sites and Batch_3 as
the supplier baseline. Check 1 measures how a LOSO tile classifier's predictions spread across a site's tiles. Check 2
splits each KPI's variance into batch, site and tile parts, and flags features whose batch difference comes from
extreme tiles rather than from a shifted site median. Checks 3-4 (pooling-rule comparison, synthetic injection) are
out of scope.

Facts this plan relies on (checked against the code and data on 2026-10-03):
- `tile_kpis.csv` columns are `batch, site, se_detector, segmenter_version, tile, <16 KPI columns>, nan_reason`.
  It has 0 NaN today, but the code must still handle NaN (D-018 allows it).
- `pmdb.kpis.catalogue_columns()` returns `(site_cols, tile_cols)`. Flattening `tile_cols` gives exactly the 16 KPI
  columns. `K16_si_graphite_dist_median_um` is identically 0 in every tile.
- Installed packages: scikit-learn 1.3.2, scipy 1.14, matplotlib 3.10, pandas. `tabulate` is NOT installed, so
  `DataFrame.to_markdown()` is unavailable.
- Script style to match is `scripts/kpi_overview.py`: module docstring, `ROOT = Path(__file__).resolve().parents[1]`,
  `sys.path.insert`, `matplotlib.use("Agg")`, argparse `main()`.

### 2. Design decisions

- **DD1 Script / outputs**: `scripts/tile_signal_checks.py`, which writes to `outputs/pooling_checks/`. Takes only the
  CLI args `--tile-kpis` (default `outputs/kpis/tile_kpis.csv`) and `--out` (default `outputs/pooling_checks`). Rationale: the
  path the brief suggested, and the same CLI shape as the other `scripts/` tools.
- **DD2 Features**: flatten `catalogue_columns()[1]` in catalogue order, then drop any column whose `nanstd` over all
  124 tiles is 0. `K16` is expected to be the only drop, and every drop is listed in results.md. No log or other
  transform. Rationale: the catalogue is the source of truth for per-tile columns. A constant column has no
  information and would give ICC = 0/0.
- **DD3 Input validation (raise ValueError on failure)**: exactly 4 rows per site, one batch per site, the batch set is
  {Batch_1, Batch_2, Batch_3}, and NaN in a feature column only on rows where `nan_reason` is non-empty. Rationale:
  the analysis assumes the D-012 tiling and the D-018 NaN contract.
- **DD4 Classifier**: `Pipeline([SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0,
  class_weight="balanced", max_iter=5000, random_state=0)])`. Default lbfgs, which is multinomial for 3 classes; do
  NOT pass `multi_class`. A fresh pipeline is fitted per fold, so imputation and scaling are fold-internal. Rationale:
  the classifier the brief specifies, with fixed defaults so nothing is tuned on 31 sites.
- **DD5 LOSO**: `LeaveOneGroupOut` with groups = site. Three variants:
  - `multiclass`: all 31 sites.
  - `B1_vs_B3`: 24 sites.
  - `B2_vs_B3`: 24 sites.

  Each held-out site's 4 tiles get class probabilities from the model fitted on the other sites in that variant.
  Rationale: the site is the unit of independence, and tiles from the same site must never be in training.
- **DD6 Per-tile outcome**:
  - `pred` = argmax class.
  - `p_true` = probability of the site's true batch.
  - `correct` = (pred == true batch).
- **DD7 Per-site category** (same rule in every variant, from the site's own true label):
  - `diffuse_correct`: n_correct >= 3.
  - `witness`: n_correct in {1, 2} and max tile p_true >= 0.7.
  - `mixed`: n_correct in {1, 2} and max tile p_true < 0.7.
  - `miss`: n_correct == 0.

  Rationale: the brief asks for a threshold-based description. 0.7 is clearly above the binary 0.5 cut, so one tile
  barely over the line does not count as a witness.
- **DD8 Baseline false-alarm rate (binary variants only)**: FA = fraction of Batch_3 sites with any tile p(non-B3
  class) >= 0.7. Rationale: D10 says witness thresholds should be calibrated on Batch 3's own tile spread. FA is
  the null rate of witness-like tiles.
- **DD9 Batch verdict** (binary variants only, for the non-B3 batch A with n_A = 7 sites; W = #witness, D =
  #diffuse_correct). Evaluated in this order, first match wins:
  1. `localised` if W >= 2 and W/n_A > FA.
  2. `diffuse` if D >= 4.
  3. `weak` if D + W <= 2.
  4. Otherwise `inconclusive`.

  The multiclass variant gets the category counts only, with no verdict. Rationale: the order deliberately leans
  towards flagging localisation, because that is the risk being checked. Binary-vs-baseline is the comparison the
  track spec cares about.
- **DD10 Pooling side note**:
  - **Mean pooling**: argmax of the tile-mean probability vector.
  - **Max pooling, multiclass**: argmax over classes c of max over tiles of p_c.
  - **Max pooling, binary**: predict A if max over tiles of p_A >= 0.5, else Batch_3. This is the MIL "any tile
    deviates from baseline" rule.

  Report site accuracy for both rules per variant, as one table row and nothing more. Rationale: this is a side note,
  and the proper pooling comparison is check 3.
- **DD11 Variance decomposition (check 2)**: per feature, using only non-NaN tiles.
  - `icc_all`: one-way random-effects ICC(1) over all 31 sites. Uses the unbalanced-safe k0 formula, which equals 4
    when there is no NaN. Sites with fewer than 2 non-NaN tiles are dropped for that feature.
  - `icc_B1`, `icc_B2`, `icc_B3`: the same, using only that batch's sites.
  - Nested sum-of-squares fractions: `ss_batch`, `ss_site` (site within batch), `ss_tile` (tile within site). The
    three sum to 1.
  - `sw_all` = sqrt(MSW over all sites), the pooled within-site SD. `sw_ratio_B1` = sw_B1 / sw_B3, and likewise
    for B2.

  ICC bands (Koo & Li 2016): >= 0.5 means a "site-level feature" (tiles agree); < 0.5 means "tile-dominated".
  Rationale: this is the standard ICC, and the nested fractions separate the batch contribution from the
  between-site contribution in one view.
- **DD12 Single-tile flag** per feature, for each contrast A in {Batch_1, Batch_2} vs Batch_3:
  - Per site: compute tile median, min and max.
  - dmed = median over A sites of the site median − median over B3 sites of the site median. dmax and dmin are
    defined the same way from site max and site min.
  - dext = whichever of dmax and dmin has the larger absolute value.
  - Divide all of these by `sw_all`.
  - Flag:
    1. `tail_driven` if |dext| >= 1.0 and |dext| >= 2·|dmed|.
    2. else `diffuse_shift` if |dmed| >= 1.0.
    3. else `none`.

  Rationale: this makes "site-max differs more than site-median" precise and scale-free. Medians across sites stop
  one outlier site (e.g. Batch_1/4ih2ggld, K02 = 120) from creating a flag on its own.
- **DD13 Within-site spread table**: long format, one row per (site, feature). Columns:
  - `median`, `min`, `max`, `range`
  - `max_abs_dev` = max |x − site median|
  - `range_sw` = range / sw_all

  In results.md, show the median `range_sw` per batch as three extra columns of the check-2 table.
- **DD14 Overall reading** (written into results.md as definitions, applied by the script, not by hand):
  - **Localised**: any binary verdict is `localised`, OR at least 3 features are flagged `tail_driven` for that
    contrast.
  - **Diffuse**: the verdict is `diffuse` and `diffuse_shift` flags outnumber `tail_driven` flags for that contrast.

  The script prints one line per contrast: `B1 vs B3: <localised|diffuse|mixed/inconclusive>`. `mixed/inconclusive`
  covers every other case. Rationale: the dispatcher gets a rule-based answer without hand interpretation.
- **DD15 Seeds / determinism**: `random_state=0` on LogisticRegression; lbfgs is deterministic anyway. Figure jitter
  uses `np.random.default_rng(0)`. Two runs must give byte-identical CSVs.
- **DD16 results.md is fully script-generated**. Use a local helper `md_table(df, floatfmt=".3f") -> str` that writes
  a pipe table (no tabulate). Rationale: no numbers transcribed by hand, and no new dependency.
- **DD17 No new tests.** Verification is a smoke run plus the existing suite. Rationale: this is an analysis script,
  not library code, and the brief says no tests are required.

### 3. Out of scope

- Do not touch `data/`, `data_heldout/`, `cache_heldout/`, or held-out sites in any way. The script reads only
  `tile_kpis.csv`.
- Do not modify `pmdb/`, other scripts, `outputs/kpis/`, existing tests, `requirements*.txt`, or `fem-swelling.md`.
- Checks 3 (pooling-rule comparison) and 4 (synthetic injection, `tests/synthetic_patterns.py`) are out of scope.
- No hyperparameter tuning, no feature selection beyond DD2, no site-level KPIs (`site_kpis.csv`), and no
  se_detector covariate. Detector counts are 1 SE site in B2 and 3 in B3; mention them in results.md as a known
  confound and do nothing else with them.
- Do not commit. The dispatcher owns git.

### 4. Steps

**Step 1: Script skeleton, loading, validation, feature selection.**
Create `scripts/tile_signal_checks.py` in the `scripts/kpi_overview.py` style.
- The docstring lists the outputs.
- `load_tiles(path) -> tuple[pd.DataFrame, list[str], list[str]]` returns `(df, features, dropped)`. It implements
  DD2 and DD3. Treat `nan_reason` NaN or empty string as "no reason".
- `main()` creates the `--out` directory.

Verify: `python scripts/tile_signal_checks.py --help` exits 0. A temporary
`python -c "import sys; sys.path.insert(0,'scripts'); import tile_signal_checks as t; d,f,x=t.load_tiles('outputs/kpis/tile_kpis.csv'); print(len(d), len(f), x)"`
prints `124 15 ['K16_si_graphite_dist_median_um']`.

**Step 2: Check 1, LOSO tile predictions.**
Implement `loso_tile_probs(df, features, variant) -> pd.DataFrame` per DD4–DD6. Sketch:
```python
VARIANTS = {"multiclass": ["Batch_1", "Batch_2", "Batch_3"],
            "B1_vs_B3": ["Batch_1", "Batch_3"], "B2_vs_B3": ["Batch_2", "Batch_3"]}
sub = df[df.batch.isin(VARIANTS[variant])].reset_index(drop=True)
X, y, g = sub[features].to_numpy(float), sub.batch.to_numpy(), sub.site.to_numpy()
for tr, te in LeaveOneGroupOut().split(X, y, g):
    model = make_pipeline()          # DD4, fresh each fold
    model.fit(X[tr], y[tr])
    P = model.predict_proba(X[te])   # columns follow model.classes_
    # write p_Batch_1/p_Batch_2/p_Batch_3 (NaN for classes absent from the variant), pred, p_true, correct
```
Output columns: `variant, batch, site, tile, pred, p_true, correct, p_Batch_1, p_Batch_2, p_Batch_3`. Concatenate all
three variants and write `tile_predictions.csv`, sorted by variant, batch, site, tile.

Verify: run the script. `tile_predictions.csv` has 316 rows (124 + 96 + 96). In each row, the non-NaN `p_*` values
sum to 1 within 1e-9.

**Step 3: Check 1, per-site summary, verdicts, pooling side note.**
Implement `site_summary(tile_preds) -> pd.DataFrame`. One row per (variant, site), with columns:
- `variant, batch, site, n_tiles, n_correct, frac_correct`
- `p_true_min, p_true_median, p_true_max, p_true_range`
- `category` (DD7)
- `max_p_nonbase`: binary variants, B3 sites only, = max tile p of the non-B3 class. NaN everywhere else.
- `mean_pool_pred, max_pool_pred, mean_pool_correct, max_pool_correct` (DD10)

Write it to `site_tile_spread.csv` (79 rows).

Implement `batch_summary(site_df) -> pd.DataFrame`. One row per (variant, batch), with columns:
- `n_sites`
- `n_correct_0` … `n_correct_4`: counts of sites with that many correct tiles
- `median_frac_correct`
- `diffuse_correct, witness, mixed, miss`: category counts
- `fa_rate`: DD8. Filled on the non-B3 batch's row as the B3 rate for that variant; NaN otherwise.
- `verdict`: DD9. Binary non-B3 rows only; empty string otherwise.

Also add one row per variant with tile-level balanced accuracy and site accuracy for mean and max pooling. Write the
two tables to `batch_summary.csv` and `pooling_side_note.csv`.

Verify: rerun. `site_tile_spread.csv` has 79 rows, with `n_correct` in 0..4. `batch_summary.csv` has 7 rows (3
multiclass + 2 + 2), and the `verdict` column is non-empty for exactly 2 rows (Batch_1 in `B1_vs_B3`, Batch_2 in
`B2_vs_B3`).

**Step 4: Check 2, variance decomposition and flags.**
Implement `icc1(values, groups) -> tuple[float, float]`, which returns `(icc, msw)`. Sketch (unbalanced-safe):
```python
# keep groups with >=2 non-NaN values
N, g = total n, number of groups; n_i, m_i group sizes/means; m = grand mean
MSB = sum(n_i*(m_i-m)**2)/(g-1); MSW = sum((x-m_i)**2)/(N-g)
k0 = (N - sum(n_i**2)/N)/(g-1)
icc = (MSB-MSW)/(MSB+(k0-1)*MSW)   # may be negative; report as-is, do not clip
```
Implement `variance_table(df, features) -> pd.DataFrame`. One row per feature, with columns:
- `feature, icc_all, icc_B1, icc_B2, icc_B3`
- `ss_batch, ss_site, ss_tile`
- `sw_all, sw_ratio_B1, sw_ratio_B2`
- `flag_B1, flag_B2`, with the matching `dmed_B1, dext_B1, dmed_B2, dext_B2`, all standardised (DD12)
- `range_sw_med_B1, range_sw_med_B2, range_sw_med_B3`

`range_sw_med_*` is the median over that batch's sites of `range_sw` (DD13). Write it to `variance_by_feature.csv`
(15 rows). Write the DD13 long table to `site_feature_spread.csv` (31 × 15 = 465 rows).

Verify: rerun. `variance_by_feature.csv` has 15 rows, `ss_batch + ss_site + ss_tile` is within 1e-9 of 1 in every
row, and the flags are drawn from {tail_driven, diffuse_shift, none}. `site_feature_spread.csv` has 465 rows.

**Step 5: Figures and results.md.**
- `fig_tile_probs.png`: 3 rows (one per variant). On the x-axis, sites are grouped by batch and sorted within each
  batch by mean p_true. Each site's 4 tile p_true values are drawn as dots in the batch colour (tab10, sorted
  batches as in `kpi_overview.batch_colours`; reimplement locally rather than importing the script). Draw dashed
  horizontal lines at 0.5 and 0.7, plus 1/3 on the multiclass row. Titles carry the variant and verdicts. Save at
  dpi=110.
- `fig_variance.png`: one horizontal stacked bar per feature showing ss_batch / ss_site / ss_tile. Append the
  non-`none` flags to each y-tick label, e.g. `K03_ecd_max_um [B1:tail, B2:shift]`. Mark icc_all as a black dot
  against a twin x-axis from 0 to 1.

`results.md` is generated with `md_table` and contains, in order:
1. A header with the input path, the row/site counts, the features used and dropped, and the seed note.
2. The definitions block: verbatim thresholds from DD7, DD8, DD9, DD11, DD12, DD14.
3. Check 1: the batch summary table, then the per-site table (variant, batch, site, n_correct, p_true min/median/max,
   category), then the pooling side note, labelled "side note; see check 3".
4. Check 2: the variance table, with columns rounded to 3 dp.
5. "Overall reading": the DD14 line per contrast.
6. Caveats, with exactly these three bullets: n = 7 sites per non-B3 batch, so the counts are descriptive and not
   significance tests; the SE-detector confound (1 B2 site, 3 B3 sites); the KPIs come from the provisional v0
   segmenter (`segmenter_version` v0r1).

Verify: rerun. `outputs/pooling_checks/` contains exactly these 9 files:
- `tile_predictions.csv`, `site_tile_spread.csv`, `batch_summary.csv`, `pooling_side_note.csv`
- `variance_by_feature.csv`, `site_feature_spread.csv`
- `fig_tile_probs.png`, `fig_variance.png`, `results.md`

`grep -c "Overall reading" outputs/pooling_checks/results.md` prints 1.

**Step 6: Determinism and regression check.**
Run the script twice, taking `md5 -q outputs/pooling_checks/*.csv outputs/pooling_checks/results.md` after each run.
The hashes must be identical. Then run `pytest -q -m "not data"`; the pass count must be unchanged and there must be
no failures.

### 5. Expected surprises

- **More than one zero-variance column is dropped** (e.g. `K14_empty_p50_um` collapses after a future KPI rerun):
  pre-authorised. Drop it per DD2, list it, and adjust the expected row counts in Steps 1/4 to match the feature
  count. Escalate if more than 3 columns are dropped.
- **lbfgs ConvergenceWarning**: pre-authorised to raise `max_iter` to 20000. Do not change C, the solver or scaling.
- **A per-batch ICC is negative or NaN** (MSB < MSW, or a constant within the batch): report it as-is. Do not clip it
  to 0 and do not drop the feature.
- **sw_B3 = 0 for a feature** (ratio division by zero): write NaN for that ratio.
- **`sw_all` = 0** (impossible after DD2 unless every site is internally constant): stop and escalate.
- **The `nan_reason` column is all-NaN floats**, as in the current file: this is the expected state. Treat it as "no
  reason" (DD3).
- **The binary verdict is `localised` for one batch and `diffuse` for the other**: this is a valid outcome. Report
  it as-is; it is the dispatcher's call.
- **`pytest -q -m "not data"` already fails on this branch before your change**: record the failing test names from
  the pre-change run, confirm the same set fails after the change, and report it. Do not fix it.

### 6. Done criteria

- `python scripts/tile_signal_checks.py` runs end-to-end with exit 0 on the current `tile_kpis.csv`. It writes the 9
  files in `outputs/pooling_checks/` with the row counts from Steps 2–4: 316 / 79 / 7 / 3 / 15 / 465. The
  `pooling_side_note.csv` count of 3 assumes one row per variant.
- `results.md` contains the definitions, both check tables, the pooling side note and an "Overall reading" line for
  B1 vs B3 and B2 vs B3.
- Two runs give byte-identical CSVs and results.md.
- `pytest -q -m "not data"` gives the same result as before.
- `git status` shows changes only under `scripts/tile_signal_checks.py` and `outputs/pooling_checks/`, plus the plan
  and report files.

### 7. Revision log

(empty)
