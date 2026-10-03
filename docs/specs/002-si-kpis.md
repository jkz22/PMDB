# Spec 002: v0 segmentation and per-slice KPI logging

**Status:** ready for dispatch
**Repo:** https://github.com/jkz22/PMDB
**Depends on:** spec 001 **revision 2** (site loader and half-resolution cache, `pmdb/io.py`), which crops a fixed 4-px margin from the left and right edges (D-003). **This dependency is satisfied:** rev 2 is on `main` (commit 5f6bb62) and has been merged into `geometric-kpis` (5143dd6). Read `AGENTS.md` for the current data-loading conventions.
**Revision 2 of this spec:** the dependency is met; the cache is tracked in git; D-011 segments on un-normalised BSE; KPI outputs are committed to the branch.
**Locked decisions touched:** D-001 to D-008 from spec 001 are used and must not be changed. This spec sets D-009 to D-020.
**KPI definitions:** [`docs/kpis/kpi_catalogue.csv`](../kpis/kpi_catalogue.csv) (source of truth) and [`docs/kpis/README.md`](../kpis/README.md) (rationale). Read both before starting.

## Goal

Every site (31 slices) gets every v1 KPI in the catalogue computed and logged to versioned CSV outputs. A coverage script proves that nothing is missing. Segmentation sits behind one replaceable function, so a better segmenter can be dropped in later without touching the KPI code.

**Done means:** `python scripts/run_kpis.py && python scripts/check_kpi_coverage.py` exits 0. The coverage check confirms that every `output_columns` entry of every `version == v1` row in the catalogue is present and non-NaN for all 31 sites in `outputs/kpis/site_kpis.csv`, and the synthetic test suite passes.

## Background a planner cannot infer

- Read the "Background" section of spec 001 for the data facts.
- **The half-resolution cache is already committed** in `cache/half/` (31 `.npz` files plus `manifest.csv`). Do not rebuild or modify it. The `path` column in `manifest.csv` holds an absolute path from the author's machine. `load_site()` does not use it (it builds the path from the cache root), so ignore that column.
- **Percentile normalisation depends on image content** (a site with more bright Si gets a higher p99.5, and values above p99.5 are clipped to 1). So segment on `load_site(..., normalise="none")` (see D-011).
- Work at **half resolution, 50 nm/px** (`load_site(..., resolution="half")`). Si particles are a few µm across, so 40 px or more.
- **Directions:** image columns are **x**, along the coating width (in-plane). Image rows are **z**, through the coating thickness. No Cu foil is in frame, so the sign of z is unknown. Never report a signed depth trend.
- In BSE, Si is a **bright tail**, not a separate histogram mode. A global or Otsu threshold is known to be unreliable here.
- Graphite flakes are large (10–30 µm), elongated and aligned with x. Binder and carbon black look similar to graphite in BSE, so v0 separates them by object size.
- ETD/SE (channel 2) shows vertical curtaining streaks.
- Batch_3 has a brightness offset, which percentile normalisation (D-005) absorbs.
- Which batch is the baseline is **unknown**. Nothing in this spec may use batch identity to set a parameter.

## Decisions

| # | Fork | Resolution | Owner |
|---|---|---|---|
| D-009 | Working resolution | Half (50 nm/px), from the spec-001 cache. All lengths in outputs are in **µm** and all areas in **µm²** | research (locked) |
| D-010 | Segmentation interface | `pmdb/segment.py: segment(site) -> Masks`. `Masks` is a dataclass of boolean arrays `si, graphite, pore, artefact, admissible`, plus `version: str` and `params: dict`. Here `admissible = ~graphite & ~artefact`. The KPI code consumes only `Masks` and the BSE channel | research (locked) |
| D-011 | v0 segmenter (provisional) | All constants are global; no per-batch or per-site tuning. (1) Take the **un-normalised** BSE (`normalise="none"`, cast to float) and smooth it with a Gaussian, σ = 1 px. The rules below are affine-invariant, so they need no normalisation and avoid the p99.5 clipping of the Si tail. (2) **pore** = smoothed BSE below the 1st-percentile-anchored dark level: `T_pore = p1 + 0.25·(p50 − p1)`. (3) **si** = smoothed BSE above `median_solid + 4·MAD_solid`, where solid = non-pore pixels; then binary closing (disk radius 2 px), fill holes, and remove objects under 20 px (0.05 µm²). (4) **graphite** = remaining solid pixels after a binary opening (disk radius 5 px), keeping connected components of at least 4 µm². (5) **artefact** = pore components of at least 25 µm². The implementer may revise the v0 rule **once**, before the full run, if the overlays (D-019) show an obvious failure common to all batches. Document any change in `docs/kpis/segmentation-v0-notes.md` with before/after overlays. Never tune to make batches look more or less alike | research (locked, one revision allowed) |
| D-012 | Tiles | Split each slice into 4 equal-width tiles along x, each spanning the full height. Every KPI with `per_tile == yes` in the catalogue is also computed per tile | research (locked) |
| D-013 | Point-pattern null | **Random labelling in admissible space.** Place n points (n = number of Si centroids) uniformly over admissible pixels, with 99 simulations and `numpy.random.default_rng(seed)`, where seed is a stable hash of `batch/site/tile`. Null-relative KPIs (K05 null mean and z, K07 R_rl, K08 envelope) use these simulations. Because observed and null patterns share the same window, no separate edge correction is applied to the null-relative statistics. K07 R_csr is the classical Clark–Evans R and is logged only for comparison with the literature | research (locked) |
| D-014 | Voronoi | Discrete Voronoi on the pixel grid: each admissible pixel is assigned to the nearest Si centroid (for example, EDT with `return_indices`). Drop cells that touch the image or tile border. The local area fraction for K05 uses the same construction seeded by whole Si objects (SKIZ). K06 cut-offs on normalised cell area: < 0.5 (cluster), > 2.0 (void) | research (locked) |
| D-015 | Clusters (K04) | Dilate the Si mask by d/2; each connected component is one cluster. Cluster size = ECD of the Si area inside the component. v0 uses **d = 0.5 µm, d\* = 5 µm**. Also log the sweep d ∈ {0.25, 0.5, 1.0} × d\* ∈ {3, 5, 8} to `sensitivity.csv`. These values are re-frozen from the baseline in a later spec, so do not change them here | research (locked) |
| D-016 | Pair correlation (K08) | r = 0.25 to 10 µm in 0.25 µm steps. Directional sectors are pairs whose separation angle is within ±30° of x or of z. Excess = max over r of (g_obs − envelope_97.5%), floored at 0; r_peak is the r where the maximum occurs. If the excess is 0, write r_peak as **−1** (not NaN), so that coverage stays strict | research (locked) |
| D-017 | Windows and bands | K10 windows w ∈ {1, 2, 5, 10, 20} µm, non-overlapping, keeping windows with admissible fraction ≥ 0.5. K11 uses w = 10 µm. K12, D01 and D04 use 5 equal depth bands. K13 uses 10 µm column bins. All local Si fractions are Si pixels / admissible pixels | research (locked) |
| D-018 | Missing values | Site-level values must not be NaN. If a site genuinely cannot produce a value (for example, fewer than 10 Si objects), stop and report it; do not invent a fill value. Tile-level NaN is allowed when a tile has fewer than 10 Si objects; record a reason in a `nan_reason` column | research (locked) |
| D-019 | Segmentation review artefacts | For every site, write `outputs/overlays/<batch>__<site>.png`: a BSE crop with Si in red, graphite in blue, pore in yellow, artefact in magenta, at a scale a human can inspect. Show at least one full-height 40 µm-wide crop per site, plus a full-slice thumbnail | research (locked) |
| D-020 | Dependencies | Use the numpy/scipy/scikit-image/pandas stack already in `requirements.txt`. **Do not add porespy, spatstat/R, gudhi or other heavy dependencies**, and do not touch the torch/numpy pins. `scipy.spatial` (cKDTree, Delaunay) and `scipy.sparse.csgraph.minimum_spanning_tree` are sufficient | research (locked) |

## Scope

- **`pmdb/segment.py`**: the `Masks` dataclass and `segment(site)` implementing v0 (D-010, D-011).
- **`pmdb/kpis/`**: a package of pure functions from masks (plus the BSE channel and `nm_per_px`) to dicts of the catalogue's column names. Split it into modules however you like (for example `objects.py`, `pointpattern.py`, `fields.py`, `crossphase.py`, `diagnostic.py`). A registry maps every catalogue KPI ID to the function that produces it.
- **`scripts/run_kpis.py`**: loops over all sites and writes:
  - `outputs/kpis/site_kpis.csv`: 31 rows; columns `batch, site, se_detector, segmenter_version`, then every v1 output column.
  - `outputs/kpis/tile_kpis.csv`: 124 rows (31 × 4); per-tile columns plus `tile, nan_reason`.
  - `outputs/kpis/curves.csv`: long format with `batch, site, kpi_id, curve, x, value`. It holds the K08 g_obs, env_lo and env_hi curves for x and z; the K10 CV vs w curve; the K12 band values; the K13 column-bin profile; and the D01 and D04 band values.
  - `outputs/kpis/sensitivity.csv`: the K04 sweep (D-015).
  - `outputs/kpis/run_log.json`: git commit hash, all parameters, segmenter version, per-site wall time and status, and package versions.
  - `outputs/overlays/*.png` (D-019).
  - It prints one progress line per site, supports `--sites batch/site ...` for partial runs, and supports `--jobs N` for parallelism.
- **`scripts/check_kpi_coverage.py`**: reads the catalogue and checks every v1 output column in `site_kpis.csv` (31 rows, no NaN) and every per-tile column in `tile_kpis.csv` (NaN only where `nan_reason` is set). It prints a coverage table (KPI ID, columns, sites present, status) and exits non-zero on any gap.
- **`scripts/kpi_overview.py`**: a few plain QC plots to `outputs/kpis/figures/`: a strip plot of each verdict KPI by batch, and the K10 curves coloured by batch. These are for inspection only; no statistics.
- **`tests/test_kpis_synthetic.py`**: tests with known answers, needing no real data (see Acceptance 1).
- **`docs/kpis/kpi_status.md`**: generated or updated at the end. It lists each KPI ID, its implementing function, its test, and coverage, plus any deviations from this spec.

## Out of scope

- Baselines, thresholds, p-values, verdicts, conformal calibration, and per-KPI attribution. Those belong to spec 003.
- Choosing which batch is the baseline.
- v2 stretch KPIs S01–S04.
- Learned segmentation (ilastik or random forest) or any neural network.
- Writing anything under `data/`; modifying `cache/` or the spec-001 files in `outputs/`; changing spec 001 behaviour.
- Pushing to `main` or merging the PR. Commit to the `geometric-kpis` branch and open a PR from `geometric-kpis` to `main`.

## What to commit

Commit code, tests, docs, and these outputs: `outputs/kpis/*.csv`, `outputs/kpis/run_log.json`, `outputs/kpis/figures/*.png` and `outputs/overlays/*.png`. Keep each overlay PNG under about 1 MB (downscale the full-slice thumbnail). These outputs are the deliverable, so the reviewer must see them in the PR.

## Acceptance criteria

1. `python -m pytest -q -m "not data"` passes, including these synthetic known-answer tests. Use fixed seeds; patterns are about 2000 × 2000 px with Si discs of radius 4 px unless stated.
   - **Poisson** pattern (about 400 discs, full admissible area): K05 σ within 0.53 ± 0.07; K07 R_csr and R_rl within 1.0 ± 0.1; K08 excess ≈ 0 (≤ 0.2) in x and z.
   - **Hexagonal lattice:** R_csr > 1.8; K05 σ < 0.15.
   - **Thomas cluster process** (cluster σ ≈ 1 µm): R_rl < 0.7; K05 σ > 0.8; K08 excess > 0.5 with r_peak < 3 µm. K04 must take `d` and `d_star` as parameters: with d = 0.5 µm and d\* = 1 µm, the agglomerate fraction is > 0.5 here and < 0.1 for the Poisson pattern. Synthetic discs are too small for the production d\* = 5 µm.
   - **Null-model test (required):** random Si placed *only inside horizontal admissible stripes* covering about 30% of the area (graphite elsewhere). R_rl must be within 1.0 ± 0.15 while R_csr is < 0.9. This proves that the admissible-space null removes the "graphite makes it look clustered" bias.
   - **Localisation:** Si confined to the top 40% of rows gives K12 depth_maxdev > 0.8; a uniform random map gives depth_maxdev < 0.2, K10 slope < −0.6 and K11 Lacey > 0.7.
   - **Cross-phase:** a Si disc fully embedded in graphite gives K15 = 1.0; a Si disc fully inside pore gives K15 = 0.0 and K16 > 0.
   - **Segmentation:** a synthetic BSE image (mid-grey matrix, bright grainy discs, dark holes, plus Gaussian noise) is recovered by `segment()` with Si IoU > 0.85 and pore IoU > 0.85.
2. `python scripts/run_kpis.py` completes on all 31 sites in under 60 minutes on a 12-core laptop with `--jobs 8`.
3. `python scripts/check_kpi_coverage.py` exits 0: every v1 KPI is present for 31/31 sites.
4. `outputs/overlays/` holds 31 PNGs. `docs/kpis/segmentation-v0-notes.md` exists, listing per-batch Si-fraction ranges and any sites where the overlays look wrong. **Report** those sites; do not fix them by per-site tuning.
5. Sanity check on real data: K01 is between 0.1% and 50% at every site. If not, stop and report; do not loosen the bound.
6. `docs/kpis/kpi_status.md` is complete, and the PR description includes the coverage table and the deviations list.

## Stop-and-report rules

Stop and report the situation, rather than working around it, if any of these occur:
- a locked decision cannot be implemented as written;
- a synthetic test fails and the fix would require loosening its tolerance;
- the v0 segmentation is visibly wrong in a way that one global revision cannot fix;
- a site cannot produce a site-level value;
- the runtime exceeds the budget by more than 2×.

Never tune any parameter by looking at between-batch differences in KPI values.
