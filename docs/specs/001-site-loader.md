# Spec 001: Site loader and half-resolution cache

**Status:** ready for dispatch (revision 2: D-003 amended after the first run failed on border columns)
**Repo:** https://github.com/jkz22/PMDB (all changes land here)
**Locked decisions touched:** none yet (the repo has no decision log). This spec sets D-001 to D-008 below.
**Invalidates:** None (nothing downstream exists yet)

## Goal

Every later stage (Si segmentation, spatial KPIs, the frozen-feature anomaly detector) needs the same input: one aligned, multi-detector image per imaging site, with its physical scale and enough metadata to audit imaging confounds. This spec builds that single entry point, plus a half-resolution cache so development iterations take seconds rather than minutes.

## Background a planner cannot infer

- **Data.** `data/Batch_1`, `data/Batch_2` and `data/Batch_3` hold SEM cross-sections of a graphite anode that contains silicon.
  - Files are LZW-compressed TIFFs named `img_<site>_<detector>.tif`, with detector ∈ {BSE, Inlens, ETD, SE}.
  - The data folder also contains a `.DS_Store` file, so iterate over directories matching `Batch_*` only.
- **Expected inventory (verified):** 93 files and 31 sites (Batch_1: 7, Batch_2: 7, Batch_3: 17).
  - Every site has BSE and Inlens.
  - 27 sites have ETD; 4 have SE in its place (1 in Batch_2, 3 in Batch_3). No site has both.
- **Image format (verified):**
  - Files are `uint8` RGB with shape (H, W, 3); H ranges from about 1600 to 2316 px, W from 6960 to 7000 px.
  - R == B in every pixel of all 93 files. G differs from R only in full-height border columns, in 39 files (13 sites × 3 detectors): left-edge columns 0–3 and/or the last two columns (W−2, W−1). No mismatched pixel lies more than 3 px from the left edge or 1 px from the right edge. Once a 4-px margin is cropped from both sides, every image is greyscale. *(Revision 2: an earlier version of this spec wrongly said only column 0 was affected.)*
  - All detectors at a site share the same shape.
- **Scale (verified):** the TIFF `XResolution` tag (unit: inch) gives 25.000 ± 0.001 nm/px for all 93 files. Vendor SEM metadata (kV, working distance and so on) has been stripped.
- **Registration (verified):** all detectors align to BSE within 0.5 px. This was measured by phase cross-correlation on the central 1024×2048 crop, for all 62 pairs. Stack them without any registration step.
- **Known confound (verified):** Batch_3 was imaged with a different brightness offset. Its BSE 1st percentile averages about 7, against 0 in Batches 1 and 2, and its fraction of exactly-zero pixels is about 0.8%, against about 2.5%. The raw-statistics output must make this visible, and normalisation must not erase it from the record.
- **Reading LZW TIFFs requires `imagecodecs`.** `tifffile` records a missing codec at import time, so `imagecodecs` must be installed before the first `import tifffile` in a process.
- **Performance:** full-resolution FFT-heavy work across all sites took more than 10 minutes on a laptop. Downstream development will use the half-resolution cache.
- **Environment:** `requirements.txt` pins `torch==2.2.2` and `numpy<2` for macOS Intel. Do not change these pins.

## Decisions

| # | Fork | Resolution and why not the alternative | Owner |
|---|---|---|---|
| D-001 | Average or stack detectors | **Stack** them as channels in the fixed order `[BSE, Inlens, SE_type]`. Each detector carries different physics: BSE gives Z-contrast between Si and graphite, Inlens shows edges and the carbon-binder web, ETD/SE shows topography and pores. Averaging would dilute the Si contrast | research (locked) |
| D-002 | ETD vs SE | Both go in channel 2 (`SE_type`), and each site records `se_detector ∈ {"ETD","SE"}`. Their edge correlation with BSE is about 0.75 vs 0.74, so they behave alike. Do not drop the 4 SE sites | research (locked) |
| D-003 (rev 2) | Border artefact columns | Crop a **fixed 4-px margin from both the left and right edges** (`arr[:, 4:W-4]`) of every detector image before anything else. Use the same crop for every file, not a crop chosen from each file's contents, so all sites share the same geometry and the margin loses only 100 nm per side. Then assert R == G == B on the cropped array and keep a single channel. If the assertion fails, raise an error naming the file and the offending columns; never average the channels silently. A failure on a future batch is a data-format finding to report, not something to work around | research (locked) |
| D-004 | When to normalise | The cache stores **raw** `uint8` images (downsampled, not normalised). Normalisation happens at load time, so the method can change without rebuilding the cache | research (locked) |
| D-005 | Normalisation method (v1) | Robust percentile scaling per image and per channel: p0.5 maps to 0 and p99.5 maps to 1, clipped to [0, 1] and returned as `float32`. Expose `normalise: Literal["percentile","none"]` with `"percentile"` as the default. Normalisation anchored on the pore and graphite intensity levels is left for a later spec | research (locked) |
| D-006 | Downsampling | 2×2 block mean (`skimage.transform.downscale_local_mean`) after cropping to an even H and W, rounded to `uint8`. Record `nm_per_px = 50.0` | research (locked) |
| D-007 | Cache format and location | `cache/half/<batch>__<site>.npz` with key `image` (uint8, H×W×3), plus `cache/half/manifest.csv`. Add `cache/` to `.gitignore` | research (locked) |
| D-008 | Data root | Read the env var `PMDB_DATA`, defaulting to `<repo>/data`. The data folder is read-only: never write under it | research (locked) |

## Scope

Create a package `pmdb/` at the repo root (no `src/` layout; tests run from the repo root).

- **`pmdb/io.py`**
  - `list_sites(data_root=None) -> pandas.DataFrame`: the manifest, one row per site, with columns `batch, site, se_detector, height, width, nm_per_px, path_bse, path_inlens, path_se_type`. It validates the inventory and raises a clear error on a missing or duplicate detector, a shape mismatch, or a scale outside 25 ± 0.01 nm/px.
  - `load_site(batch, site, resolution="full"|"half", normalise="percentile"|"none", data_root=None, cache_root=None) -> Site`. With `resolution="half"` it reads from the cache. If the cache is missing it raises an error saying how to build it, rather than building it implicitly.
  - `Site` dataclass with fields:
    - `image`: float32 (H, W, 3) when normalised, uint8 otherwise
    - `channels = ("BSE","Inlens","SE_type")`
    - `batch`, `site`, `se_detector`, `nm_per_px`, `resolution`
    - `raw_stats`: a dict per channel, computed on the raw uint8 image at the loaded resolution
- **`pmdb/stats.py`**: `raw_intensity_stats(arr_uint8) -> dict` returning `mean, std, p0_5, p1, p50, p99, p99_5, frac_zero, frac_255`.
- **`scripts/build_cache.py`**: builds the half-resolution cache and writes `outputs/raw_intensity_stats.csv`.
  - The CSV has one row per **file** (93 rows): `batch, site, detector, channel_slot` plus the statistics, computed at full resolution after the 4-px margin crop.
  - The script is idempotent: it skips existing cache files unless `--force` is passed.
  - It prints one progress line per site.
- **`scripts/qc_overview.py`**: writes two figures for auditing imaging confounds.
  - `outputs/qc_contact_sheet.png`: one row per site, grouped by batch, showing thumbnails of the three detectors, labelled `batch/site/se_detector`.
  - `outputs/raw_stats_by_batch.png`: BSE p1, p50 and p99 for each site, coloured by batch.
- **`tests/`**
  - Unit tests that build tiny synthetic LZW TIFFs in `tmp_path` and need no real data. Include one file whose G channel differs in border columns (cropped away, so it passes), one whose G differs in an interior column (must raise), and one site with SE instead of ETD.
  - Integration tests marked `@pytest.mark.data` that run against the real `data/` and skip automatically if it is absent.
  - Register the marker in `pytest.ini`.
- Add `tifffile`, `imagecodecs`, `scikit-image` and `pytest` to `requirements.txt` without changing the existing pins.
- Add a short "Data loading" section to `README.md` with three usage lines.
- Add `cache/` and `outputs/` to `.gitignore`.

## Out of scope

- Segmentation, KPIs, spatial statistics, anomaly detection, or any learned model.
- Image registration or distortion correction.
- Normalisation anchored on pore and graphite intensity levels, denoising, and removal of the vertical milling streaks (curtaining).
- Anything under `data/`: no writes, renames or deletions.
- Changing the torch/numpy pins. Editing `main.py` beyond what is strictly needed; leaving it empty is fine.
- Committing anything in `cache/` or `outputs/`.

## Acceptance criteria

1. `python -m pytest -q -m "not data"` passes with no real data available (e.g. `PMDB_DATA=/nonexistent`).
2. `python -m pytest -q -m data` passes on the real data, asserting:
   - 31 sites, with per-batch counts {Batch_1: 7, Batch_2: 7, Batch_3: 17}
   - `se_detector` counts of {ETD: 27, SE: 4}
   - every `nm_per_px` within 25 ± 0.01
   - R == G == B in all 93 files after cropping the 4-px left and right margins
   - all detectors at each site sharing one shape
   - a negative control: on the *uncropped* arrays, exactly 39 files have R != G, and every mismatched column satisfies `x <= 3` or `x >= W-2`
3. `python scripts/build_cache.py` produces `cache/half/manifest.csv` with 31 rows, plus 31 `.npz` files. A second run finishes in under 10 s and rewrites nothing (file modification times unchanged).
4. For every site, the half-resolution image shape is `(H//2, (W-8)//2, 3)`, where (H, W) is the original TIFF shape.
5. For all 31 sites, `load_site(b, s, "half")` returns float32 with no NaN, minimum ≥ 0 and maximum ≤ 1. With `normalise="none"` it returns uint8 identical to the cached array.
6. `outputs/raw_intensity_stats.csv` has 93 rows. The mean BSE `p1` is above 3 for Batch_3 and below 1 for Batch_1 and Batch_2, reproducing the known offset.
7. `python scripts/qc_overview.py` writes both PNGs, and both open as valid images.
8. `git status --porcelain` shows no changes under `data/` and no tracked or untracked files under `cache/` or `outputs/`.
9. Building the full cache from scratch takes under 10 minutes on a 12-core laptop. Report the measured time.

## Known divergences

- **Percentile normalisation vs anchoring on phase intensities:** percentile scaling partly absorbs the Batch_3 offset, but it is also sensitive to phase fractions; a site with more pore area shifts its p0.5. That is acceptable for v1 because the raw statistics are kept separately.
- **`tifffile` vs Pillow:** use `tifffile` with `imagecodecs` for both pixels and tags. Pillow also reads these files, but do not mix the two.
- **Image width:** if W is odd after the 4-px margin crop, crop the last column before downsampling rather than padding. (All current widths are even.)
- **Where the raw statistics come from:** `raw_intensity_stats.csv` is computed after the 4-px margin crop, not after a column-0 drop.

## Reporting back

- The list of files changed or added. Do not commit; leave the working tree for review.
- Full output of both pytest runs.
- The first 5 rows of `cache/half/manifest.csv`, and the per-batch mean BSE p1 from `outputs/raw_intensity_stats.csv`.
- The time taken to build the cache.
- Paths to the two QC PNGs.
- Any assertion that failed on the real data and what was found, with no workaround. Stop and report rather than loosening a check.
