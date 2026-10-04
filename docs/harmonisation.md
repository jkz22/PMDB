# Grey-level harmonisation of the imaging artefact

**Applies to:** `pmdb/harmonise.py`, `load_site(..., harmonise=..., normalise="fixed")`,
`scripts/build_harmonised.py`, `scripts/eval_harmonisation.py`, `outputs/harmonisation/`.
**Audience:** anyone training a model or computing KPIs across batches.

## TL;DR

* The "Batch 3 black-level" confound is a **per-site, spatially uniform, affine grey-level change
  (offset + gain) on all three detectors**, confined to 9 of the 17 Batch 3 sites. It is an
  imaging-session setting, not a material difference: after removing offset and gain, the
  Si/graphite contrast ratio of the affected sites equals that of Batches 1/2.
* Six per-site look-up-table (LUT) methods were implemented and compared. **Recommended:
  `hybrid`** = material-anchored affine correction on BSE and SE_type (`affine2`), histogram
  matching on Inlens. Load with

  ```python
  site = load_site(batch, site, resolution="half", normalise="fixed", harmonise="hybrid")
  ```

* Nothing under `data/`, `data_heldout/`, `cache/half/` or `cache_heldout/half/` is modified.
  LUTs (~1 KB per site) live in `cache/harmonised/<method>/`, are applied at load time, and
  `Site.raw_stats` still records the un-harmonised statistics so the confound stays documented.

## 1. What the artefact actually is

Per-site anchors were estimated on `cache/half` (50 nm/px) for every channel: `black` = 0.5th
percentile, `pore` / `graphite` / `si` = median grey level inside the BSE phase masks from
`pmdb.segment.segment_bse` (affine-invariant, so the masks are the same before and after
correction). Table: raw anchors, range over the sites in each group (`outputs/harmonisation/groups.csv`).

| group (n) | sites | BSE black | BSE graphite | BSE Si | SE_type black | Inlens black | Inlens graphite |
|---|---|---|---|---|---|---|---|
| clean (22): B1, B2 and 8 × B3 | B3: `0grcilhi cfe5vt7s hawkfj64 mgxahqnk pl8uabbv utfgcjfa vc2whyaq x77cy643` | 0–1 | 50–62 | 90–118 | 0 | 0–10 | 76–150 |
| mild (5) | `9luzk4jm hzumfsms ptg8lmto ufdvpb81 xgj4xftb` | 2–4 | 57–60 | 107–120 | 0 | 8–18 | 61–129 |
| strong (4) | `71vgq3fw kbdh4tri tuy3zymq x7u69zsw` | 22–24 | 62–63 | 104–107 | 19 | 14–18 | 42–44 |

Observations that fix the correction model:

* **Offset and gain, not offset alone.** In the strong group black rises by +22 but Si only by
  −8: the whole scale is compressed. The `affine2` fit gives gain 1/0.69 ≈ 1.46 (BSE) and 1.45
  (SE_type) for all four strong sites against 0.92–1.14 for clean sites. The four strong sites
  are the four 1030-px-tall images, i.e. one imaging session with one brightness/contrast setting.
* **Spatially uniform.** Row- and column-strip 0.5th percentiles vary by < 2 grey levels across
  the field, so a global per-site LUT is the right tool (no background subtraction, no inpainting).
* **Clean sites are clipped at 0** (2–3 % of pixels exactly 0); the affected sites are not. Pore
  interiors are therefore the natural black anchor.
* **Material contrast survives.** BSE Si/graphite ratio `(si − black)/(graphite − black)` is 2.1
  for the strong group and 1.6–2.2 for clean sites, so the microstructure information is intact and
  recoverable by an affine map.
* **Inlens is different.** Inlens (topographic, in-lens SE) in the strong group has graphite at 43
  but Si at 92 (clean: 105 / 175): its response is compressed non-uniformly, not affinely. Any
  anchor-based affine map either leaves the black level (`affine3`) or clips the highlights
  (`affine2`, 31 % of pixels at 255). Inlens carries no compositional contrast that needs
  protecting, so matching its full histogram is acceptable there.
* Held-out `xrv9xvzb` belongs to the mild group (BSE black 4, Inlens black 16); `3e122cbj` and
  `fn0mhxef` are clean.

## 2. Methods

All methods produce one `(3, 256)` uint8 LUT per site and are monotone in grey level. The
reference is the **median anchor over the 31 labelled sites** (BSE black 0 / graphite 57 / Si 112;
SE_type 0 / 76 / 127; Inlens 0 / 104 / 159) and, for histogram matching, the mean per-channel CDF.
Held-out sites are mapped onto this labelled reference and never contribute to it.

| method | map per channel | removes | preserves |
|---|---|---|---|
| `none` | identity | – | everything (baseline) |
| `offset` | `x − black + ref.black` | black level | gain error stays; Si/graphite ratio untouched |
| `affine2` | line through `black→ref.black`, `graphite→ref.graphite` | offset + gain | Si brightness is free ⇒ Si/graphite contrast (material) preserved |
| `affine3` | least-squares line through black, graphite **and** Si anchors | offset + gain, partly Si contrast | compromise; residual black level where the three anchors are not collinear |
| `histmatch` | full CDF → reference CDF | everything, incl. phase fractions | only pixel rank order (aggressive control) |
| **`hybrid`** | `affine2` on BSE + SE_type, `histmatch` on Inlens | offset + gain (BSE/SE), Inlens non-affine distortion | BSE/SE material contrast; Inlens highlights not clipped |
| `percentile` | existing `normalise="percentile"` (per-image p0.5→0, p99.5→1) | offset and gain **plus** whatever the image content does to the percentiles | – (not a harmonisation; shown for comparison) |

## 3. Results (`outputs/harmonisation/summary.csv`)

Lower is better unless stated. "strong" = the 4 strong sites, "rest" = the other 13 Batch 3 sites,
"clean" = Batches 1 + 2.

| metric | none | offset | affine2 | affine3 | histmatch | **hybrid** | percentile |
|---|---|---|---|---|---|---|---|
| black-level gap strong−rest, BSE (grey) | 21.8 | 0.0 | 0.0 | 0.5 | 0.0 | **0.0** | 0.0 |
| black-level gap strong−rest, SE_type | 19.0 | 0.0 | 0.0 | 1.3 | 0.0 | **0.0** | 0.0 |
| black-level gap strong−rest, Inlens | 9.0 | 0.0 | 0.0 | 13.7 | 0.1 | **0.1** | 0.0 |
| graphite anchor SD over sites, BSE | 3.7 | 6.3 | 0.0 | 3.1 | 0.5 | **0.0** | 8.1 |
| graphite anchor SD over sites, Inlens | 30.8 | 35.3 | 0.2 | 16.1 | 1.8 | **2.0** | 33.5 |
| Inlens pixels clipped at 255, strong sites | 0.000 | 0.000 | 0.307 | 0.085 | 0.007 | **0.007** | 0.005 |
| Si/graphite contrast change on clean sites (rel.) | 0 | 0 | 0.003 | 0.006 | 0.055 | **0.003** | 0.003 |
| SD of Si/graphite contrast over all sites (higher = more material info kept) | 0.161 | 0.161 | 0.159 | 0.151 | 0.105 | **0.159** | 0.159 |
| fixed-threshold Si fraction, strong−rest | 0.015 | −0.029 | 0.028 | 0.017 | 0.006 | 0.028 | 0.018 |
| fixed-threshold pore fraction, strong−rest | −0.033 | 0.004 | −0.009 | −0.009 | −0.001 | **−0.009** | −0.005 |
| corr(fixed-threshold pore fraction, per-image segmenter) | 0.75 | 0.83 | 0.92 | 0.91 | 0.33 | **0.92** | 0.84 |
| corr(fixed-threshold Si fraction, per-image segmenter) | 0.94 | 0.75 | 0.91 | **1.00** | 0.92 | 0.91 | 0.50 |
| mean |Δgrey| on clean sites | 0 | 0.1 | 7.9 | 7.3 | 8.6 | 8.0 | 30.9 |
| imaging-stats-only classifier, strong vs rest of B3 (LOO acc; 0.76 = chance) | 1.00 | 1.00 | 0.82 | 0.88 | 0.65 | **0.82** | 0.94 |
| imaging-stats-only classifier, Batch 3 recall (LOO) | 0.84 | 0.71 | 0.71 | 0.77 | 0.52 | 0.74 | 0.84 |

Reading the table:

* **`offset` is not enough.** It zeroes the black level but the 0.69× gain remains, so strong-group
  graphite sits at 39 instead of 57 and a fixed threshold reads 3 % *less* Si than the per-image
  segmenter. An imaging-stats classifier still identifies the strong group perfectly.
* **`affine2`/`hybrid` remove the artefact on BSE and SE_type exactly** (anchor SD 0.0) while
  changing the clean sites' material contrast by 0.3 %. The remaining strong-vs-rest
  separability (0.82 vs chance 0.76) comes from genuine differences in the remaining statistics
  (e.g. the 1030-px sites have slightly brighter Si at p99), not from the black level.
* **`affine3`** is the Modelling session's GMM-style fit; it leaves +13.7 grey levels of Inlens black
  level and +0.5–1.3 on BSE/SE because the three anchors are not collinear on the affected sites.
  It is the best choice *only* if Si must land at the same grey level everywhere (fixed-threshold
  Si correlation 1.00), at the price of flattening the Si-brightness variation that may be material.
* **`histmatch`** halves the spread of the Si/graphite contrast over sites (0.161 → 0.105) and drops the
  correlation of fixed-threshold pore fraction with the segmenter to 0.33: it is removing
  material information, as predicted. Keep as a control only.
* **`percentile`** (status quo) changes clean sites by 31 grey levels on average, inflates Si fraction
  under a fixed threshold to 0.9 (saturation) and only halves the strong-group separability. It should
  not be used as the input to models that are meant to learn material properties.
* The small residual *mild*-group differences are within the clean-group spread after any
  anchor-based method (see `black_level_by_method.png`).

Figures: `black_level_by_method.png` (per-site 0.5th percentile after each method, three channels),
`bse_hist_by_method.png` (31 BSE histograms per method), `visual_by_method.png` (same crop, one
fixed grey scale), `fixed_threshold_fractions.png` (phase fractions with one global threshold).

Held-out sites (`heldout_metrics.csv`): `hybrid` moves `xrv9xvzb` BSE black 4 → 0, Inlens black
16 → 0, and leaves fixed-threshold Si fraction within 0.01 of the per-image segmenter for all three.

## 4. Recommendation for modelling pipelines

1. **Input images:** `load_site(..., normalise="fixed", harmonise="hybrid")`. `normalise="fixed"` is
   grey/255 so every site sits on the same scale; `percentile` on top of a harmonised image would
   re-introduce content-dependent scaling.
2. **Ablation / robustness:** train the same model on `harmonise="affine2"`, `"affine3"` and
   `"none"`; a model whose batch accuracy survives `hybrid` but collapses on `histmatch` is reading
   phase fractions (material), one that collapses already on `hybrid` was reading the grey levels
   (imaging). All LUTs are built, so this is a one-argument change.
3. **KPIs computed from thresholds** (phase fractions, Si detection by grey level): use the
   harmonised uint8 (`normalise="none", harmonise="hybrid"`) with thresholds derived from the
   reference anchors (`cache/harmonised/hybrid/reference.json`). The existing affine-invariant
   segmenter in `pmdb.segment` is unaffected by any method.
4. **Imaging-covariate screening** (`docs/kpis/screening.md`): keep using `Site.raw_stats`, which is
   always computed *before* the LUT, so the confound remains testable.

## 5. Files and reproduction

```
cache/harmonised/anchors.csv                 per-site, per-channel anchors (labelled)
cache/harmonised/<method>/luts.npz           one (3, 256) uint8 LUT per site, key "Batch_x__site"
cache/harmonised/<method>/params.csv         slope / intercept per channel (NaN for histogram matching)
cache/harmonised/<method>/reference.json     reference anchors + CDF used for the fit
cache_heldout/harmonised/...                 same for the 3 held-out sites, fitted to the labelled reference
outputs/harmonisation/{summary,site_metrics,heldout_metrics,groups}.csv + 4 figures
```

```bash
python scripts/build_harmonised.py                       # ~30 s, rebuilds every method for labelled + held-out
python scripts/build_harmonised.py --materialise hybrid  # optional: write cache/harmonised/hybrid/half/*.npz (~280 MB)
python scripts/eval_harmonisation.py                     # ~8 min, regenerates outputs/harmonisation/
pytest -q tests/test_harmonise.py                        # synthetic round-trip tests + real-data checks (-m data)
```

The materialised `hybrid` arrays are committed (`cache/harmonised/hybrid/half/*.npz`, 276 MB, plus
`cache_heldout/harmonised/hybrid/half/`) for pipelines that read npz files directly, e.g. Modal GPU
runs; they are byte-identical to `load_site(..., normalise="none", harmonise="hybrid").image`. Other
methods are LUT-only (applied in < 1 ms at load time); `--materialise <method>` writes them if needed.

## 6. Limitations

* Anchors come from the BSE masks of the repo segmenter; a site with almost no Si or no pores
  would give a noisy anchor. All 34 sites have ≥ 0.6 % of each phase, and the `graphite` anchor is
  a median over > 50 % of the pixels, so this is not an issue here, but `anchors.csv` should be
  checked for new batches.
* The reference is the labelled-set median, so the absolute grey scale is a convention; only
  differences between sites are meaningful.
* Harmonisation does not touch curtaining, noise, focus or 0-clipping differences; those remain
  as documented in `docs/data-processing.md`. The Modelling session's artefact-injection ablation
  (`outputs/v2/injection_ablation/`, `outputs/v2/RESULTS.md` §9.5 on its branch) quantifies this:
  the +22 grey / 0.69× artefact is removed to the pixel and a `hybrid`-trained classifier is
  invariant to it, but injected σ = 6 grey noise or a 1 px blur pass straight through (a 1 px blur
  flips 4/6 fields to Batch 3, same as raw), Batch 3's lower BSE noise (−20 %) and sharpness
  (−25 %) remain a batch shortcut (two scalar statistics recognise 82–88 % of Batch 3 fields), and
  the BSE zero-clipped fraction (1.5–1.7 % in Batches 1/2 vs 0.4 % in Batch 3) alone gives 84 %
  Batch-3-vs-rest accuracy — no grey-level map can change a pixel that is already 0. For a route
  that equalises resolution and noise (downwards only) and flags clipping per pixel, use the
  physical pipeline `pmdb.clean` / `load_clean(..., kind="harm")` (`docs/clean.md`).
* `histmatch` (and therefore the Inlens channel of `hybrid`) changes gated KPIs by 5–12 % per
  field and alters the noise texture differently per batch (same ablation); `affine2` is the safer
  choice for texture-sensitive models, `hybrid` for models that need the Inlens grey scale aligned.
* `histmatch`/`hybrid` LUTs are fitted on the half-resolution histograms and `load_site` only
  accepts them with `resolution="half"`; a 2×2 mean changes the histogram, so the same LUT would
  not match the full-resolution Inlens distribution. The affine methods work at both resolutions.
* `load_site(..., harmonise=<m>)` defaults to `normalise="fixed"`; passing `normalise="percentile"`
  re-stretches each image and cancels the affine correction (a warning is raised).
