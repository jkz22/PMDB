# KPI screening and submission contract

**One-page summary:** [screening-summary.md](screening-summary.md)

**Status:** v1.0 (`pmdb/screen.py`, run with `python -m pmdb.screen`). Unsupervised: it checks measurement quality only, because there is no outcome label.

## 1. Why gates, not a weighted score

- Every KPI passes five sequential gates: degeneracy, artefact, reliability, robustness, redundancy.
- The first gate a KPI fails decides its fate (`deciding_gate`). All statistics are still computed for every KPI.
- The gates guard against different failures and cannot compensate for each other. A highly reliable artefact is still an artefact.
- Weights would be arbitrary without an outcome label.
- A gate with no data for a KPI is reported as untested. Untested is not passed.

## 2. KPI submission contract

This section is normative. MUST and SHOULD are used in the RFC sense.

### 2.1 Site CSV

- A site CSV MUST have the columns `batch` and `site`, plus one numeric column per KPI.
- It MUST contain exactly one row per site and cover all 31 sites. `batch`/`site` are read as strings.
- Several `--kpis` files MAY be given; all MUST cover the identical site set.
- KPI names MUST match `^[A-Za-z][A-Za-z0-9_]*$` and SHOULD follow `<ID>_<short_name>_<unit>` as in `kpi_catalogue.csv` (for example `K03_ecd_d50_um`).
- KPI names MUST be unique across all submitted files.
- Non-numeric columns are ignored and the tool warns about them. These names are ignored silently, in every table: `se_detector`, `segmenter_version`, `nan_reason`, `runner`, `elapsed_s`, `error` (add more with `--ignore-cols`).

### 2.2 Missing values

- Missing means empty, NaN or +-inf.
- A sentinel value MUST be declared with `--sentinel COL=VALUE` (repeatable; several values per column allowed). Undeclared sentinels are treated as real values.
- Example: `K08_pcf_rpeak_x_um=-1` and `K08_pcf_rpeak_z_um=-1` (no peak found). A global `-1` would corrupt signed KPIs such as `K10_cv_slope`.
- Declared sentinels are applied in the site, replicate and sensitivity tables. `site_kpis_filtered.csv` keeps original values.
- All statistics use pairwise-complete sites.

### 2.3 Hard errors

| Error | Cause | Fix |
|---|---|---|
| missing `batch`/`site` column | key column absent in a file | add the columns |
| duplicated (batch, site) keys | a site appears twice in one file | one row per site |
| invalid KPI name | name breaks the regex (also a `.1` suffix from duplicated headers) | rename the column |
| KPI appears in two files | same name in more than one `--kpis` file | rename or drop one |
| site sets differ | `--kpis` files cover different sites (message gives counts and up to 5 examples) | cover all 31 sites in every file |
| no numeric KPI columns | a `--kpis` file has none | check the file |
| replicates: site set differs / duplicates / missing replicate column | table does not cover exactly the site set, has duplicate (batch, site, tile) rows or no `tile` column | fix the table, or set `--replicate-col` |
| replicates/sensitivity: unknown KPI name | column not in the site KPIs (naming typo) | use the exact site KPI name |
| sensitivity: params missing, NaN, fewer than 2 settings, incomplete setting, duplicate rows | see 2.4 | fix the sweep |
| `--sensitivity` without `--sensitivity-params` (or the reverse) | params are never inferred | give both |
| covariates: site set differs, duplicates or no numeric column | see 2.4 | fix the table |
| `--sentinel` column is not a site KPI, or malformed item | typo, no `=`, non-numeric value | use `COL=VALUE` with a site KPI |

Hard errors print `error: <message>` and exit with code 2.

### 2.4 Getting the other gates tested

- **Replicates** (reliability): CSV `batch, site, tile, <KPI cols>` with the same KPI names as the site table, all sites, at least 2 replicates per site (4 tiles recommended). Pass `--replicates`; the replicate column name is `--replicate-col` (default `tile`).
- **Sensitivity** (robustness): CSV `batch, site, <param cols>, <KPI cols>` with one row per site per setting, plus `--sensitivity-params <param cols>`. Every setting MUST cover all sites and there MUST be at least 2 settings.
- **Covariates** (artefact): wide CSV `batch, site, <numeric covariates>`, all sites. For the imaging black level and brightness:

```python
from pmdb.screen import pivot_raw_intensity_stats, read_table
cov = pivot_raw_intensity_stats(read_table("outputs/raw_intensity_stats.csv"))  # BSE_p1, BSE_p50
cov.to_csv("outputs/kpis/screen/covariates_bse.csv", index=False)
```

## 3. The gates

### 3.0 Degeneracy

- **Measures:** whether the KPI varies at all on the 31 sites.
- **Statistics:** `n_missing_frac`; `n_unique` (distinct non-missing values); `mad` = `median(|x - median(x)|)`, unscaled.
- **Threshold:** checked in order, first failure decides: `n_missing_frac > 0.2` (`--max-missing-frac`) gives `degeneracy:missing`; `n_unique < 5` (`--min-unique`) gives `degeneracy:few_unique`; MAD 0 or NaN gives `degeneracy:zero_mad`.
- **Why:** a KPI with no spread cannot rank sites, and a point mass breaks rank statistics.
- **On failure:** check for undeclared sentinels, a saturated or constant measurement, or a too-coarse quantisation.

### 3.1 Artefact

- **Measures:** whether the KPI follows an imaging nuisance covariate (for example BSE p1 or p50) rather than the sample.
- **Statistic:** the batch-partial Spearman rho. Rank KPI and covariate separately on pairwise-complete sites, subtract each batch's mean rank from both, and take the Pearson correlation of the residuals. NaN if fewer than 5 complete sites or zero variance.
- **CI:** percentile 95% CI (`--ci-level`) from a batch-stratified site bootstrap, 1000 resamples (`--n-boot-cov`).
- **Threshold:** drop if any covariate has |rho| >= 0.5 (`--covariate-rho`) and a CI that excludes 0; the decision is `artefact:<cov>` for the qualifying covariate with the largest |rho|.
- **Batch eta-squared** (rank-based SS_between / SS_total, `batch_eta2`) is reported only. It adds the flag `high_batch_eta2` when > 0.5 (`--eta2-flag`) and never drops a KPI.
- **Why within-batch:** the documented confound, the Batch_3 BSE p1 offset (0 in Batches 1-2, mean 7.18 in Batch 3), is almost constant within a batch. A marginal rho with it is just batch membership, and would drop exactly the KPIs that separate batches. The within-batch association is the part attributable to the covariate itself. `covariate_rho_marginal` is still reported.
- **Order:** artefact runs before reliability, because batch artefacts are constant within a site and inflate ICC.
- **On failure:** the KPI measures the microscope. Normalise for the covariate or fix the segmentation.

### 3.2 Reliability

- **Measures:** whether replicate measurements (tiles) of one site agree compared to differences between sites.
- **Statistic:** ICC(1) on batch-residualised replicates. Drop NaN rows, drop sites with fewer than 2 values, subtract each batch's mean, then run a one-way random-effects ANOVA: a = number of sites, g = number of batches present, N = number of rows, n_i = replicates of site i.
  - SSB = sum n_i (ybar_i - ybar)^2, SSW = sum sum (y - ybar_i)^2
  - MSB = SSB / (a - g) (g batch means were removed), MSW = SSW / (N - a)
  - n0 = (N - sum(n_i^2) / N) / (a - 1)
  - ICC = (MSB - MSW) / (MSB + (n0 - 1) MSW), not clipped at 0.
- **CI:** batch-stratified site-cluster bootstrap, 1000 resamples (`--n-boot-icc`); each drawn site is its own group.
- **Threshold:** drop if the CI upper bound < 0.5 (`--icc-upper-min`).
- **Interpretation:** ICC(1) is the reliability of a single replicate, so it is a conservative bound for the site value (a mean over tiles).
- **Untested** when the KPI has no replicates, the point estimate is NaN (flag `icc_undefined`) or fewer than half of the resamples are finite (flag `icc_ci_undefined`).
- **On failure:** the site differences are mostly noise; increase the sampled area or stabilise the method.

### 3.3 Robustness

- **Measures:** whether the site ranking survives changes of an analysis parameter.
- **Statistic:** each distinct tuple of `--sensitivity-params` values is a setting. `robustness_rho` is the minimum pairwise-complete Spearman rho over all pairs of settings. A NaN pair (constant setting) counts as 0.0 and adds the flag `robustness_constant_setting`.
- **Threshold:** drop if `robustness_rho < 0.8` (`--robustness-min`).
- **Why:** a KPI that depends on an arbitrary parameter choice is not a stable measurement.
- **On failure:** justify a parameter, or report the KPI at a fixed setting and accept the dependence.

### 3.4 Redundancy

- **Measures:** whether a surviving KPI duplicates another survivor (untested gates count as survived).
- **Statistic:** |Spearman rho| matrix (pairwise-complete, NaN as 0), distance 1 - |rho|, average-linkage clustering cut at distance 0.2 (`--redundancy-rho 0.8`). This is repeated on 500 batch-stratified site resamples (`--n-boot-cluster`).
- **Threshold:** two KPIs are linked when co-clustered in >= 80% of resamples (`--cocluster-frac`). Clusters are the connected components of that graph, so they can chain (A~B and B~C puts A and C together).
- **Representative:** the member with the highest finite ICC; otherwise the member with the highest mean full-sample |rho| to the others; final tiebreak alphabetical. Others get `redundant_with:<rep>`. Singletons are their own representative.
- **On failure:** use the representative, or combine the duplicates deliberately.

### 3.5 Reproducibility

Bootstrap resamples are stratified by batch (every batch stays present). Each gate draws from `default_rng([seed, stream])` with streams covariate = 1, icc = 2, cluster = 3 (`--seed`), and one index matrix per gate is shared by all KPIs. The indices depend only on the seed, stream and site list, so adding a KPI never changes another KPI's CI.

## 4. Reading the output

`kpi_screen.csv` (kept KPIs by rank, then dropped KPIs in input order):

| Column | Meaning |
|---|---|
| `kpi` | KPI name |
| `decision` | `keep` or `drop` |
| `deciding_gate` | first failed gate (vocabulary below); empty if kept |
| `untested_gates` | `;`-joined subset of `artefact;reliability;robustness` lacking data. Untested is not passed |
| `flags` | `;`-joined warnings (vocabulary below) |
| `rank` | 1..n for kept KPIs, empty for dropped |
| `n_missing_frac`, `n_unique`, `mad` | gate 0 statistics |
| `batch_eta2` | rank-based batch eta-squared (reported only) |
| `max_abs_rho_covariate` | \|rho\| of the reported covariate |
| `covariate_of_max` | reported covariate: the qualifying one with largest \|rho\|, else the one with the largest finite \|rho\| |
| `covariate_rho`, `covariate_rho_lo`, `covariate_rho_hi` | within-batch rho and its bootstrap CI |
| `covariate_rho_marginal` | marginal (batch-ignoring) Spearman rho |
| `icc`, `icc_lo`, `icc_hi` | ICC(1) and bootstrap CI |
| `n_rep_sites` | sites with at least 2 non-missing replicates |
| `robustness_rho`, `n_settings` | minimum pairwise rho and number of settings |
| `cluster_id`, `cluster_rep` | gate 4 cluster (1..k by earliest member) and its representative; survivors only |

- `deciding_gate` values: `degeneracy:missing`, `degeneracy:few_unique`, `degeneracy:zero_mad`, `artefact:<covariate>`, `reliability`, `robustness`, `redundant_with:<rep>`, empty for kept.
- `flags` values (fixed order): `high_batch_eta2`, `covariate_suspect:<cov>` (|rho| above threshold but the CI includes 0), `icc_undefined`, `icc_ci_undefined`, `robustness_constant_setting`.
- Ranking: kept KPIs by `icc` descending (NaN last), then `batch_eta2` ascending (NaN last), then name.
- `site_kpis_filtered.csv`: `batch, site` plus the kept KPIs in input order, original values, site-table row order.
- `screen_config.json`: strict JSON with version, UTC timestamp, full config, input paths, seeds, counts and ignored columns.

## 5. Worked example on our data

```python
from pmdb.screen import pivot_raw_intensity_stats, read_table
cov = pivot_raw_intensity_stats(read_table("outputs/raw_intensity_stats.csv"))
cov.to_csv("outputs/kpis/screen/covariates_bse.csv", index=False)
```

```bash
python -m pmdb.screen --kpis outputs/kpis/site_kpis.csv --replicates outputs/kpis/tile_kpis.csv \
  --sensitivity outputs/kpis/sensitivity.csv --sensitivity-params d_um d_star_um \
  --covariates outputs/kpis/screen/covariates_bse.csv \
  --sentinel K08_pcf_rpeak_x_um=-1 K08_pcf_rpeak_z_um=-1
```

Result (outputs in `outputs/kpis/screen/`): `kept 26/43 KPIs; drops by gate: {'reliability': 6, 'robustness': 1, 'redundant_with': 2, 'degeneracy': 8}`. Degeneracy: 3 `few_unique`, 2 `missing`, 3 `zero_mad`; artefact: 0 drops. Both K04 KPIs fail: `K04_agglom_frac` at reliability and `K04_n_clusters_per_1000um2` at robustness (the gate order decides which gate is named; all statistics are reported).

## 6. Limits

- n = 31 sites in 3 batches, unbalanced (7/7/17): CIs are wide and borderline calls are fragile.
- Untested is not passed. Currently only 16 of 43 KPIs have tile replicates and only K04 has a sweep.
- Covariates that are constant within batches cannot be tested.
- The min-pairwise robustness rule is strict across wide sweeps.
- This is not predictive importance. If an outcome label appears later, use leave-one-batch-out permutation importance combined with stability selection, never in-sample importance.
- v0 segmentation caveat: see [README.md](README.md) section 6.
