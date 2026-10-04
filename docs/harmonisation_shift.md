# Shift (non-intensity) harmonisation routes

The LUT route (`docs/harmonisation.md`), the physics route (`docs/clean.md`) and the imported
intensity pipelines (`docs/harmonisation_ext.md`) all act on grey-level *values*. What survives
them is the rest of the acquisition fingerprint: focus/blur, pixel noise, noise texture — the
*spectral* shift. Measured on the raw half-res data: BSE amplitude at Nyquist is ~25 % lower on the
four strong-session Batch-3 fields than on Batches 1–2, Gaussian noise σ is 0.24 vs 0.21, edge
width is per-site (78–100 nm) rather than group-structured. This document adds four published
methods that target that shift, from a wide search of the SEM, EM, CT, PET/MRI and domain-adaptation
literature (survey summary in §6). None of them is SEM-validated except CUT/CycleGAN-style
translation (one 2025 steel paper); all are labelled accordingly and kept as separate inputs for the
modelling fan-out.

| method | what it moves | published source | operates on | status |
|---|---|---|---|---|
| `spectrum` | radial amplitude spectrum (blur + noise texture) to the labelled median | Ohkubo et al., *Med. Phys.* 38 (2011) 3915 (MTF-ratio kernel conversion); Mackin et al., *Tomography* 5 (2019) 61 (NPS homogenisation filters) — CT | pixels, frequency domain, deterministic | image route, recommended for the shift ablation |
| `fda` | low-frequency amplitude window (β = 0.01) to the reference | Yang & Soatto, CVPR 2020, "Fourier Domain Adaptation" (github.com/YanchaoYang/FDA) | pixels, frequency domain | control (moves shading/black level, not texture) |
| `combat` | additive + multiplicative session effects in feature tables, batch protected | Johnson et al., *Biostatistics* 2007; Pomponio et al., *NeuroImage* 2020 — `neuroHarmonize` | KPI / fingerprint features | modelling-side route (no pixels) |
| `cut` | whole instrument "style" of the strong session | Park et al., ECCV 2020 (CUT/FastCUT); SEM precedent Tsutsui et al., *Mater. Trans.* 66 (2025) 941 (CycleGAN FE↔W SEM) | pixels, learned on Modal T4 | experimental, follow-up PR |

## 1. Inputs and policy

* Input to the image routes is the half-res raw grey (mask-aware 2×2 mean of the raw TIFF) plus the
  clean-pipeline exclusion masks, exactly as for the imported pipelines (`harmonise_ext.load_half_raw`).
  No in-house LUTs or dark/graphite normalisation are applied first.
* Excluded pixels (border/markers, bad bands, charging, clipping) are filled with a normalised-
  convolution local mean before any FFT and written back unchanged afterwards, so they never leak
  into a spectrum or a filter. Spectra are averaged over 512 px Hann-windowed interior tiles that
  are ≥ 90 % valid.
* Reference spectra are fitted on the labelled sites only; held-out sites are transform-only.
* Outputs: `cache/harmonised_shift/<method>/half/<batch>__<site>.npz` (`image` uint8 (H, W, 3)
  BSE/Inlens/SE_type, `mask` uint16), `model.json` (+ `model.npz` for `fda`), `params.csv`,
  repo-relative `manifest.csv`; held-out under `cache_heldout/harmonised_shift/`. Arrays are
  gitignored and rebuilt with one command (~4 min):

```
python scripts/build_harmonise_shift.py            # needs outputs/clean (scripts/build_clean.py)
python scripts/run_combat_features.py              # feature-level ComBat tables
python scripts/eval_harmonise_ext.py --methods none spectrum fda hybrid --out outputs/harmonisation_shift
```

```python
from pmdb.harmonise_shift import load_shift
img, mask = load_shift("Batch_3", "71vgq3fw", method="spectrum")   # (H, W, 3) uint8, (H, W, 3) uint16, 50 nm/px
valid = clean.valid_for_stats(mask[..., 0])
```

## 2. `spectrum` — kernel-conversion / NPS-homogenisation filter

For site *s* and detector *d* the radially averaged amplitude spectrum `A_s(f)` (64 bins to Nyquist)
is measured; the reference is the per-bin median over the 31 labelled sites (log domain). The site is
filtered in the Fourier domain with `H_s(f) = A_ref(f) / A_s(f)` (3-bin moving average, clamped to
[0.33, 3], `H(0) = 1`). Radial bins cover 0 – 0.5 c/px only; the FFT corners (0.5 < r ≤ 0.707 c/px,
not measured radially) are kept out of the fit and receive the Nyquist-bin gain on application. This
is the image-domain analogue of Ohkubo's MTF-ratio filter and of the
NPS-matching filters used to homogenise CT reconstruction kernels. Because DC is untouched, the mean
grey level, the black level and the per-phase medians are preserved to within the filter's ringing;
only the blur / noise texture is moved. The filter is *not* restricted to downward (blur) changes —
sites sharper or noisier than the median are smoothed, blurrier ones are sharpened by at most 3× in
amplitude, which is the published behaviour; the clamp bounds the noise amplification.

Fitted filters (`cache/harmonised_shift/spectrum/params.csv`): the four strong-session sites get
`H ≈ 1.25` at 0.04 c/px rising to `1.40–1.45` at Nyquist on BSE, the re-quantised mild sites
(`9luzk4jm`, `hzumfsms`, `ufdvpb81`) `1.1 → 1.37`, most Batch 1–2 sites `0.9–1.1` across the band.
The flat ~1.25 part is the strong group's 0.69× detector gain seen from the frequency side (DC is
excluded, so it shows up as a uniform amplitude deficit); the rising part is the real texture shift.
`spectrum` therefore also partially corrects the gain, but not the black level (DC) — for a
pure texture correction run it after `hybrid` (not built here).

## 2b. `hybrid_spectrum` — the recommended single modelling input

`spectrum` keeps the DC term, so on its own it only halves the Batch-3 black-level gap (§6); `hybrid`
(`pmdb/harmonise.py`) fixes black level and gain but leaves the texture shift. `hybrid_spectrum` stacks
the two: the per-site hybrid LUT is applied first, then the spectrum filter is **refit on the LUT-corrected
grey** (a filter fitted on raw grey would compensate the 0.69× gain a second time) and applied with the
same policy as §2 (labelled sites only, held-out transform-only, invalid pixels filled before the FFT and
restored afterwards). Models/params in `cache/harmonised_shift/hybrid_spectrum/`, same loader:

```python
from pmdb.harmonise_shift import load_shift
img, mask3 = load_shift("Batch_3", "71vgq3fw", method="hybrid_spectrum")   # (H, W, 3) uint8, (H, W, 3) uint16
```

Rebuild: `python scripts/build_harmonise_shift.py --methods hybrid_spectrum` (~5 min, needs `outputs/clean` and
`cache/harmonised/hybrid`).

## 3. `fda` — Fourier Domain Adaptation

Yang & Soatto's `FDA_source_to_target_np` as published: the centred low-frequency window of half-width
`floor(β · min(H, W))` px of the source amplitude spectrum is replaced by the reference amplitude
(mean fftshifted amplitude of the labelled sites, each divided by its pixel count so that DC equals
the mean grey and the model is independent of field size; remapped to the source grid by frequency and
rescaled to the source pixel count), phase kept,
β = 0.01 (the paper's default). With β = 0.01 the window is ~10 px, i.e. only shading and the black
level are swapped; the texture bins are untouched. It is kept as the published "style" baseline and
as a control for `spectrum`.

## 4. `combat` — feature-level ComBat

`scripts/run_combat_features.py` runs `neuroHarmonize.harmonizationLearn/Apply` on the fingerprint
features (`outputs/fingerprint/features.csv`) and the v1 KPI table (`outputs/kpis/site_kpis.csv`)
with `SITE` = acquisition session (`strong` = the four 1030-px fields, `mild` = the re-quantised /
offset fields, `clean` = all others) and the material batch as a protected one-hot covariate, so
ComBat regresses out the imaging session and keeps batch. Models are learnt on the labelled sites and
applied to the held-out tables (batch unknown → reference level). Caveat: `strong`/`mild` sessions
occur only in Batch 3, so session and batch are partially collinear; empirical-Bayes shrinkage keeps
the estimates finite but the Batch-3-only part of the session effect is identifiable only through the
protected covariate. Outputs: `outputs/harmonisation_shift/combat/*_combat*.csv`, `summary.json`.

`summary.json` reports three sets of leave-one-parent-out (LOPO, `outputs/parent_groups.csv`)
logistic-regression accuracies: `before` (raw features), `insample` (full-table ComBat — every row's
batch label entered the transform, so these are descriptive only) and `nested` (ComBat refitted
inside each fold on the training parents, held-out parent transformed with batch unknown, exactly as
a real held-out site is). The nested numbers are the honest ones:

| table | metric | before | ComBat in-sample | ComBat nested |
|---|---|---|---|---|
| fingerprint (16) | session shortcut | 0.52 | 0.45 | **0.52** |
| fingerprint (16) | strong-vs-rest | 0.81 | 0.77 | **0.81** |
| fingerprint (16) | batch | 0.52 | 0.61 | **0.52** |
| KPIs (42) | session shortcut | 0.52 | 0.39 | **0.42** |
| KPIs (42) | strong-vs-rest | 0.81 | 0.61 | **0.74** |
| KPIs (42) | batch | 0.39 | 0.65 | **0.39** |

Verdict: on the fingerprint table nested ComBat changes nothing (the session effect is estimated from
4 + 5 sites and shrunk to the prior); on the KPI table it removes part of the session signal
(0.52 → 0.42, strong-vs-rest 0.81 → 0.74) with batch accuracy unchanged (0.39 → 0.39). The
apparent "batch kept and sharpened" result of the in-sample run (0.61 / 0.65) was leakage of the
protected covariate and is not reproduced. ComBat is therefore a modest, feature-level complement
for KPI models, not a replacement for the image routes.

## 5. `cut` — unpaired translation (experimental)

Domain A = 624 BSE 256-px tiles of the four strong-session sites (hybrid-corrected, so intensity is
already fixed and the network only has texture to learn), domain B = 4 498 tiles of the other 27
labelled sites; FastCUT (`--CUT_mode FastCUT`, grey replicated to 3 channels because CUT's loader is RGB-only, output
averaged back to one channel; 256 crops, 60 + 20 epochs) on a Modal T4;
the trained generator is applied to the full strong-site BSE images in 512-px overlapping tiles. The
output is evaluated with the same material-preservation metrics as the other routes plus an explicit
structure check (pore/Si masks before vs after). Results land in a follow-up PR; the script is
`scripts/modal_cut.py`.

## 6. Results

`python scripts/eval_harmonise_ext.py --methods none spectrum fda hybrid --out outputs/harmonisation_shift`
(full table, per-site metrics and figures in `outputs/harmonisation_shift/`; `hybrid` = the in-house LUT
route for reference). Texture metrics are measured on the valid BSE pixels of every labelled site.

| metric (labelled, BSE unless stated) | none | spectrum | fda | hybrid |
|---|---|---|---|---|
| hf-ratio gap strong vs rest (0 = texture shift removed) | −0.064 | **−0.002** | −0.066 | −0.060 |
| hf-ratio gap strong vs rest, Inlens | −0.41 | **−0.02** | −0.42 | −0.16 |
| hf-ratio CV across sites | 0.111 | **0.007** | 0.111 | 0.110 |
| noise σ CV across sites | 0.153 | **0.030** | 0.153 | 0.125 |
| edge σ SD across sites (px) | 0.024 | **0.001** | 0.024 | 0.023 |
| texture-only shortcut: strong-vs-rest LOO acc (chance 0.87) | 1.00 | **0.84** | 1.00 | 0.97 |
| texture-only shortcut: batch LOO acc (chance 0.45) | 0.68 | 0.68 | 0.68 | 0.65 |
| grey shortcut: batch LOO acc | 0.61 | 0.65 | 0.68 | **0.55** |
| black-level gap strong (grey) | 22.0 | 11.6 | 11.7 | **1.2** |
| p90−p10 gap strong (gain) | −9.3 | **−2.1** | −6.6 | 3.9 |
| Si/graphite contrast change on clean sites | 0 | 1.6 % | 4.1 % | 0.3 % |
| fixed-threshold Si fraction vs segmenter, r | 0.94 | 0.90 | 0.92 | 0.91 |
| mean abs pixel change, clean / strong (grey) | 0 / 0 | 1.2 / 2.7 | 4.6 / 6.8 | 3.5 / 6.8 |

Held-out (`3e122cbj`, `fn0mhxef`, `xrv9xvzb`, models fitted on labelled sites only): `spectrum` moves
the BSE hf-ratio of the three fields to 0.19 (labelled median), graphite anchors unchanged (52–60 →
53–60), BSE p1 within ±4 grey.

Reading: `spectrum` does what it is for — the strong group's high-frequency deficit (−6 % BSE, −41 %
Inlens) and the across-site spread of blur / noise texture vanish (CV 0.11 → 0.007, noise CV 0.15 →
0.03), the texture-only strong-vs-rest classifier drops from 1.00 to 0.84 (chance 0.87), and it
changes clean sites by ~1 grey level with a 1.6 % contrast change. Because DC is kept it is *not* an
intensity fix: the black-level gap only halves (22 → 12, via the uniform amplitude part of the filter)
and the grey-statistics batch shortcut is unchanged, so for modelling it should be stacked on an
intensity route (`hybrid` or `clean`), which this build does not do yet. `fda` with β = 0.01 is a pure
low-frequency/shading swap: texture metrics are identical to `none`, black-level gap 22 → 12, but it
moves clean-site contrast by 4 % and pixels by 4.6 grey, and on Inlens it *creates* a strong-group
black-level gap (1 → 37 grey) because the swapped low-frequency window carries the reference's shading
into a detector whose strong sites are not affine — kept as the published control, not recommended. Texture does not identify the *batch* under any route (0.68 → 0.68, the three batches
differ in texture only through the strong session), which is consistent with the shift being an
acquisition-session effect. `combat` is feature-level only (see §4).

## 7. Survey — methods considered for the shift problem

| family | examples / code | why (not) used |
|---|---|---|
| Resolution matching by post-filter | PET/SPECT EARL harmonisation (Gaussian post-filters to common recovery curves); our `clean` route (ESF edge width, downward-only) | already covered by `clean`; `spectrum` generalises it to the full spectrum |
| Kernel / NPS conversion (CT) | Ohkubo 2011, Mackin 2019, Juntunen 2022, `hsu-lab/ctnorm` | **implemented as `spectrum`** |
| Fourier style methods | FDA (Yang & Soatto 2020), amplitude-mix augmentation (Xu et al. 2021), phase matching | **FDA implemented**; augmentation variants are modelling-side |
| Blind deconvolution / PSF estimation for SEM | APEX (Carasso), Bayesian SEM deconvolution (arXiv 1810.09739), `sdeconv` | needs a trusted PSF; sharpening risks hallucinated edges — not used |
| Self-supervised denoising | Noise2Void (`n2v` SEM example), `careamics` | denoise-then-renoise removes real fine texture; `clean` already matches noise upward — not used |
| Unpaired translation | CycleGAN (Tsutsui 2025 SEM; Janelia `transfer_em`; CT kernel CycleGANs), CUT/FastCUT | **FastCUT run as experimental `cut`** |
| Feature-level harmonisation | ComBat / `neuroHarmonize`, sphering, Harmony (Cell Painting benchmark, Arevalo 2024); PLOS One 2025 found ComBat > GAN for CT radiomics reproducibility | **ComBat implemented** |
| Model-side | AdaBN, BEN (batch-wise BN for microscopy), BigAug, AdverIN, CALAMITI, DANN | training recipes for the Modelling session, not data products |
