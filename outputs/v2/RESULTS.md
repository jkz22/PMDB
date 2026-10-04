# PMDB v2 — what the models learned, in plain language

This is the readable version. Every number comes from a file in `outputs/v2/`; the long auto-generated
tables are in `results_tables.md` and the earlier, exhaustive write-up is kept as `RESULTS_detailed.md`.

Four separate questions get mixed up easily, so they are kept apart throughout:

| Question | What answers it | Where |
|---|---|---|
| A. Do Kevin's KPIs measure the same thing on raw and on harmonised images? | recompute KPIs on each harmonised cache | §2 |
| B. Does a learned representation *contain* the KPI information? | ridge probe R² on held-out fields (not the model's own output) | §6 |
| C. Can an image model tell the three batches apart, and from what? | 3-class field accuracy, attribution, perturbation tests | §3–§4 |
| D. Is the NASA MicroNet model bad for this data, or badly used? | input-format and pooling ablations, fine-tuning | §5 |
| E. Does the physics-based clean route (PR #27) or the PR #16 LUT actually remove the microscope effects, and at what cost? | retrain on clean input, calibration, SAE, artefact-injection ablations | §9 |

**R² here always means**: embed every crop with the frozen model, fit a ridge regression from embedding to
z-scored KPI on training fields, score its predictions on crops from fields the model never saw.
R² = 1: the embedding fully determines the KPI; 0: no better than guessing the mean. It measures what the
representation *encodes*, not how good Kevin's KPI itself is.

---

## 1. The short answers

1. **The KPI targets are fine.** Kevin's segmenter sets its thresholds from each image's own percentiles,
   so an offset/gain change does not move them. Recomputed on `hybrid` or `affine2` images the five gated
   KPIs are identical to the raw-image values (rank correlation 1.000, batch means change < 0.2 %). Only the
   non-linear `histmatch` method moves them (≈ 5–12 % per field). The KPI-alignment leaderboard stands.
2. **Classification is poor because the batches are mostly *not* different in the images, and the part
   that is different is microscope texture.** The same fields are misfiled by every architecture and
   every harmonisation: 3 of 7 Batch_1 fields are confidently called Batch_2 and 3 of 7 Batch_2 fields
   are called Batch_1 or Batch_3. The two Batch_1 fields that *are* recognised have 2–3× the Si fraction of
   every other field, i.e. the classifier learned "lots of Si ⇒ Batch_1" from two outlier fields. Batch_3
   is recognised through lower BSE noise (−20 %) and lower sharpness (−25 %): those two numbers alone
   recognise 82–88 % of Batch_3 fields, confidence per field tracks them, adding noise of 6 grey levels at
   test time destroys the Batch_3 recognition, and no grey-level LUT changes them.
3. **Harmonisation is not bad, but it fixes the wrong thing for classification.** It removes the black-level
   and gain artefact correctly (Batch_3 BSE 1st percentile 8 → 2 grey levels, Inlens 17 → 9) and leaves the
   KPIs untouched (§2). It leaves a comb of unused grey levels on the four strongly corrected sites, but a
   direct test shows the classifiers do not use that comb. It cannot touch noise or focus, which is what
   actually separates Batch_3. One real cost: the KPI-*conditioned* VAE-B collapses on harmonised input
   (R² 0.49 → ≈ 0), because it had been reading the KPIs off grey level.
4. **Attribution (occlusion + Grad-CAM) says the classifiers look at the graphite matrix texture, not at
   Si particles or pores.** Over 3,994 attributed crops the evidence is spread across phases roughly in proportion to
   their area (graphite 0.94–1.03× its area share, pore 0.8–1.4×, Si 0.8–1.8× for confident predictions), and the attribution maps correlate
   only weakly (|r| < 0.2) with local brightness, edges or noise. Combined with the perturbation test, the
   consistent reading is: a diffuse texture/noise fingerprint, not microstructure.
5. **NASA MicroNet disagreed because of how we fed it.** MicroNet was trained on grey-scale micrographs
   replicated into three channels. Given our three *different* detectors as RGB its frozen embedding scores
   KPI R² 0.26. The same frozen weights on BSE alone score 0.51, and 0.66 with multi-stage mean+std pooling —
   level with fine-tuned DINOv2 and above frozen DINOv2-S and ViT-MAE-B. ImageNet normalisation and the
   choice of pooling matter little; the "RGB = three detectors" input format is what hurt it.

---

## 2. Are the KPI targets biased by the raw-image artefact? — No (`kpi_harm/crop_kpis_by_harm.csv`)

Kevin's BSE segmenter (`segment_bse`) smooths the image, then puts the pore threshold at a fixed fraction
between the 1st and 50th percentile and the Si threshold at a fixed fraction between the 99th and 50th. An
affine grey-level change (offset, gain) moves all those percentiles together, so the masks do not change.
We verified this by recomputing all gated KPIs on all 31 fields (6,084 crops) for each cache:

| Method | What it does | Change vs raw KPIs (per crop) |
|---|---|---|
| `hybrid` (PR #16 default) | affine LUT on BSE and SE, histogram match on Inlens | Spearman 1.000; mean abs. change ≤ 0.0004 in a fraction; batch means move ≤ 0.2 % |
| `affine2` | affine LUT on all channels | identical to `hybrid` on BSE (same LUT) |
| `histmatch` | non-linear histogram match | Spearman 0.95–0.99; field means move 5–12 % (pore up to 14 %) |

Only BSE enters the KPIs, and `hybrid` and `affine2` apply the same affine LUT to BSE, so for the KPI
targets "raw" and "harmonised" are the same labels. Representation R² against raw KPIs is therefore a
valid target for every model, including the harmonised-input ones.

Caveat that still stands: this shows the KPIs are *stable under* the imaging change, not that they are
*correct*. Their validity rests on the Phase 0 gates (phantom, scale, resolution) in `RESULTS_detailed.md` §3.

---

## 3. Why is batch classification so poor? (`cls_field_consistency.csv`, `cls_leaderboard.csv`)

Setup reminder: 3 classes, 31 fields (7/7/17), stratified grouped 5-fold by field so a field is never in
both train and test, 45 configurations (5 architectures × views × 4 harmonisations), 225 folds. Field
accuracy ranges 0.55–0.72; chance is 0.33 (or 0.55 if you always say Batch_3). Batch_3 recall 0.76–0.94,
Batch_1 0.29–0.57, Batch_2 0.14–0.43.

### 3.1 It is the same fields every time

Mean probability assigned to the *true* batch, averaged over all stack-view configurations:

| Field | raw | hybrid | affine2 | histmatch | note |
|---|---|---|---|---|---|
| Batch_1/4ih2ggld | 0.93 | 0.92 | 0.92 | 0.80 | Si fraction 0.139 (others ≈ 0.05–0.08) |
| Batch_1/5n1q8atc | 0.93 | 0.88 | 0.91 | 0.91 | Si fraction 0.164 |
| Batch_1/uhdslk0o | 0.45 | 0.37 | 0.48 | 0.43 | |
| Batch_1/ffwubibz | 0.38 | 0.58 | 0.43 | 0.63 | |
| Batch_1/iv6g2oq0 | 0.32 | 0.16 | 0.24 | 0.21 | called Batch_2 (0.59) |
| Batch_1/fzrt2k6r | 0.13 | 0.20 | 0.15 | 0.19 | called Batch_2 (0.83) |
| Batch_1/f1vzngrs | 0.11 | 0.14 | 0.12 | 0.21 | called Batch_2 (0.84) |
| Batch_2/3806gxp0 | 0.82 | 0.82 | 0.83 | 0.67 | |
| Batch_2/avn74qx1 | 0.75 | 0.79 | 0.76 | 0.70 | |
| Batch_2/i9jiqjwl | 0.29 | 0.14 | 0.24 | 0.15 | |
| Batch_2/epqdaau9 | 0.26 | 0.25 | 0.26 | 0.35 | called Batch_1 (0.56) |
| Batch_2/b3esycq1 | 0.14 | 0.15 | 0.17 | 0.16 | called Batch_1 (0.79) |
| Batch_2/r17byphk | 0.12 | 0.16 | 0.16 | 0.13 | split Batch_1/Batch_3 |
| Batch_2/rxax5ozo | 0.11 | 0.12 | 0.14 | 0.11 | called Batch_3 (0.75) |
| Batch_3 (17 fields) | 0.08–0.86 | 0.08–0.86 | | | see §3.2 |

The rankings barely move with harmonisation or architecture (full table in the CSV). A wrong answer that is
reproduced by a ResNet, an EfficientNet and a DINOv2 on four different input versions is not model noise:
these fields *look like* the other batch to any image model. For QC this means either (a) the batch labels
do not correspond to a visible microstructural difference at 12.8 µm scale, or (b) some fields are
genuinely out of character for their batch (you mentioned Batches 1 and 2 contain outliers — the two
high-Si Batch_1 fields are the clearest candidates).

### 3.2 What Batch_3 recognition is made of: noise and sharpness, not grey level

Per-field imaging statistics (BSE, `imaging_stats/per_image.csv`) against the classifiers' confidence:

| | Batch_1 | Batch_2 | Batch_3 |
|---|---|---|---|
| BSE noise σ (grey levels) | 10.5 ± 0.3 | 10.1 ± 0.4 | **8.2 ± 1.3** |
| BSE sharpness (Laplacian var / range²) | 0.150 ± 0.028 | 0.135 ± 0.010 | **0.109 ± 0.015** |
| BSE 1st percentile (black level) | 0 | 0 | 7.2 ± 10 (4 sites at 23–25) |

* A two-feature logistic regression on BSE noise + sharpness alone, leave-one-field-out, recognises
  **88 % of Batch_3 fields** (Batch_1/2 at 43 %, i.e. chance). Black level alone does the same (82 %), but
  black level is what harmonisation removes — and the classifiers did not lose accuracy when it was removed.
* Within Batch_3, the classifiers' confidence follows the imaging statistics: the four strong-offset,
  low-noise sites (`71vgq3fw`, `kbdh4tri`, `tuy3zymq`, `x7u69zsw`) and the other low-noise sites get
  P(Batch_3) 0.83–0.86; the two Batch_3 fields with Batch_1/2-like imaging (`pl8uabbv`: noise 10.2,
  sharpness 0.142; `cfe5vt7s`) get 0.09 and 0.23 — they are *misclassified as Batch_1/2*.
* **Test-time perturbation of the trained classifiers** (`cls_testtime_perturbation.csv`): a ±8 grey-level
  offset or a 0.85× gain changes nothing, and ±1.5 grey dequantisation noise changes nothing. Gaussian noise
  of σ = 6 grey levels (below the images' own σ ≈ 8–10) drops Batch_3 recall from 0.83 to 0.17 (ResNet-18)
  and 0.50 (EfficientNet-B4) for the raw-input models: added noise makes a Batch_3 field look like Batch_1/2.
  A 0.7 px blur does the opposite — every field is called Batch_3 (recall 1.0, Batch_1/2 → 0) because lower
  sharpness *is* the Batch_3 signature. The hybrid-trained ResNet is less noise-sensitive but equally
  blur-sensitive. A model keyed on microstructure would not care about either.

  | test-time perturbation | resnet18_imnet raw: field acc / B1 / B2 / B3 recall | resnet18_imnet hybrid: field acc / B1 / B2 / B3 recall | effb4_imnet raw: field acc / B1 / B2 / B3 recall |
  |---|---|---|---|
  | none | 0.62 / 0.50 / 0.25 / 0.83 | 0.54 / 0.50 / 0.25 / 0.67 | 0.62 / 0.50 / 0.25 / 0.83 |
  | dequant_1.5grey | 0.62 / 0.50 / 0.25 / 0.83 | 0.54 / 0.50 / 0.25 / 0.67 | 0.62 / 0.50 / 0.25 / 0.83 |
  | gauss_noise_3grey | 0.62 / 0.50 / 0.25 / 0.83 | 0.62 / 0.50 / 0.25 / 0.83 | 0.62 / 0.50 / 0.25 / 0.83 |
  | gauss_noise_6grey | 0.31 / 0.50 / 0.25 / 0.17 | 0.62 / 0.50 / 0.75 / 0.67 | 0.45 / 0.50 / 0.25 / 0.50 |
  | blur_0.7px | 0.46 / 0.00 / 0.00 / 1.00 | 0.46 / 0.00 / 0.00 / 1.00 | 0.46 / 0.00 / 0.00 / 1.00 |
  | blur_1.2px | 0.46 / 0.00 / 0.00 / 1.00 | 0.46 / 0.00 / 0.00 / 1.00 | 0.46 / 0.00 / 0.00 / 1.00 |
  | offset_+8grey | 0.62 / 0.50 / 0.25 / 0.83 | 0.54 / 0.50 / 0.25 / 0.67 | 0.62 / 0.50 / 0.25 / 0.83 |
  | gain_0.85 | 0.62 / 0.50 / 0.25 / 0.83 | 0.62 / 0.50 / 0.25 / 0.83 | 0.62 / 0.50 / 0.25 / 0.83 |
  (ResNet-18 folds 0–1 of the saved classifiers; chance field accuracy 0.33)
* After `hybrid` harmonisation the noise/sharpness statistics still classify Batch_3 at 71–88 %, and the
  Inlens histogram matching actually *increases* the batch difference in Inlens texture (Batch_3 Inlens
  Laplacian variance 1415 → 3061 vs 1470 → 2220 for Batch_2), because stretching a flat histogram amplifies
  noise.

Whether lower noise and lower sharpness are a *material* property (e.g. different polishing, charging) or an
*SEM* property (dwell time, averaging, focus) cannot be decided from the images; the session logs can.

### 3.3 What the classifiers look at: occlusion and Grad-CAM (`attribution_confident_summary.csv`, galleries in `cls_runs/<hash>/attrib_*.png`)

For the ResNet-18, EfficientNet-B4, DINOv2 and fine-tuned MicroNet classifiers (raw and hybrid input,
folds 0–1; MicroNet raw fold 0 only, a 100-crop subsample run on CPU after a hook fix), every test crop was attributed two ways: **occlusion** (replace each 32 px patch by the crop mean; how much does
P(predicted class) fall?) and **Grad-CAM** on the last convolutional map. Each map was then projected onto
Kevin's Si/graphite/pore masks for that crop and correlated with local brightness, contrast, edge
strength and noise. "SHAP-like" here means occlusion: for an image classifier, patch-removal is the
tractable version of a Shapley attribution; per-pixel SHAP on 3×256×256 inputs was not attempted.

Findings over 3,994 crops, 13 classifier runs (confident = P ≥ 0.8):

* Evidence is spread across the three phases **roughly in proportion to their area**: occlusion attribution
  per unit area is 0.94–1.03× for graphite in every group (graphite is 80–90 % of the image, so it carries
  most of the decision), 0.8–1.4× for pore and 0.8–1.8× for Si. Si is somewhat over-weighted in a few groups
  (DINOv2 confident-correct Batch_2 1.8×, ResNet-hybrid confident Batch_3 1.6×), consistent with the high-Si
  Batch_1 fields in §3.1, but there is no group where Si or pore dominates: the classifier is not looking
  "at the Si particles" or "at the pores".
* The maps correlate only weakly with local image statistics: occlusion |r| ≤ 0.19 with brightness, edges
  or noise in every group (mean −0.02). Grad-CAM is the one exception: for EfficientNet-B4 and DINOv2 the
  confident-correct Batch_1 maps *avoid* edges (r −0.4), i.e. they sit on smooth graphite interiors. There is
  no "bright patch ⇒ Batch_3" rule, consistent with harmonisation not hurting accuracy.
* Occlusion and Grad-CAM agree only moderately with each other (r 0.0–0.4), which is typical when the
  decision is diffuse rather than localised.
* In the galleries the hot spots sit on graphite flakes and along flake edges, with no consistent phase
  preference between confident-correct and confident-wrong crops.

Reading all of §3 together: the models predict batch from a **diffuse texture signature of the graphite
matrix** that is destroyed by small added noise and blur, left intact by every grey-level LUT, and
measured almost as well by two scalar noise/sharpness statistics. Batch_1 vs Batch_2 adds nothing beyond two
high-Si fields.

### 3.4 Is the comb left by the LUTs a shortcut? — No (`cls_dequant_comparison.csv`)

An affine LUT with gain > 1 leaves gaps in the 8-bit histogram: the four strongly corrected Batch_3 sites use
only 69–70 % of the grey levels in their range after `hybrid` (others 93–100 %), and the Inlens histogram
match leaves 71–79 % everywhere. A CNN could in principle read that. Test: ResNet-18 and DINOv2 retrained on
`hybrid` input with uniform ±1.5 grey-level noise added at train and test time (fills every gap; far below
the images' own noise). Result: field accuracy, per-batch recall and Batch_3 confidence unchanged (0.566/0.599
with and without; 94–95 % of crop predictions identical). The comb is not what they use.

---

## 4. Is the harmonisation bad?

What PR #16 `hybrid` does well:
* Removes the black-level lift and gain on the strong sites (BSE p1 24 → 0 on `71vgq3fw`), see
  `figs/harmonisation_before_after.png`. Phase contrast is preserved: KPIs unchanged (§2).
* Lets the KPI-alignment comparison be made on equal footing: VAE-C improves slightly (0.824 → 0.844),
  MAE-adapted/-scratch hold or improve, and their batch-kNN leakage drops.

What it does not do, or does badly:
* It cannot change noise or focus, which is where the batch signature lives (§3.2). Expecting classification
  accuracy to fall after harmonisation was the wrong expectation.
* Inlens histogram matching amplifies noise texture on flat-histogram sites and changes it differently per
  batch (§3.2). For texture-sensitive models `affine2` (affine on all channels) is the safer choice; for the
  KPIs it makes no difference.
* It leaves grey-level combs on strongly corrected sites. Harmless for the classifiers tested (§3.4), but
  cheap to remove by adding ±0.5 grey dequantisation noise in training — `classify.py --dequant` now exists.
* Any model that *conditions* on KPIs via grey level breaks: VAE-B/Inlens 0.49 → ≈ 0. That is a model
  failure the harmonisation exposed, not caused.

Verdict: harmonisation is correct for its stated purpose (grey levels) and neutral for the KPI targets; the
remaining Batch_3 separability is a noise/sharpness difference that is outside its scope.

---

## 5. NASA MicroNet (`micronet/probe_variants.csv`)

Checkpoint: `jstuckner/microscopy-efficientnet-b4-imagenet-micronet` (EfficientNet-B4, ImageNet →
MicroNet), loaded into the segmentation-models-pytorch encoder; frozen embedding = last-stage feature map,
global average pooled, 448-d; input scaled to [0,1] then ImageNet mean/std — the same path as the other
off-the-shelf models. Held-out fields, ridge probe:

| Input to MicroNet | Pooling | KPI R² (gated mean) | Si frac. | pore frac. | K04 agglom. | imaging-stat R² | batch kNN |
|---|---|---|---|---|---|---|---|
| 3 detectors as RGB (what we did) | last stage, mean | **0.26** | 0.30 | 0.34 | 0.08 | 0.04 | 0.40 |
| 3 detectors as RGB | last stage, GeM-3 | 0.29 | 0.32 | 0.48 | −0.01 | −0.01 | 0.37 |
| 3 detectors as RGB | last stage, max | 0.19 | 0.22 | 0.35 | −0.01 | −0.07 | 0.39 |
| 3 detectors as RGB | stages 3–5, mean | 0.37 | 0.47 | 0.36 | 0.12 | 0.09 | 0.39 |
| 3 detectors as RGB | stages 2–5, mean+std | 0.37 | 0.44 | 0.45 | 0.13 | 0.13 | 0.39 |
| 3 detectors as RGB, 2× upsampled | last stage, mean | 0.36 | 0.54 | 0.17 | 0.01 | 0.12 | 0.38 |
| 3 detectors as RGB, no ImageNet norm | last stage, mean | 0.25 | 0.37 | 0.16 | 0.01 | 0.04 | 0.45 |
| **BSE only, copied to 3 channels** | last stage, mean | **0.51** | 0.62 | 0.52 | 0.14 | 0.35 | 0.48 |
| **BSE only, copied to 3 channels** | stages 2–5, mean+std | **0.66** | 0.78 | 0.62 | 0.38 | 0.47 | 0.48 |
| BSE only, `hybrid` input | stages 2–5, mean+std | 0.67 | 0.80 | 0.60 | 0.36 | 0.45 | 0.46 |
| *for reference:* frozen DINOv2-S, stack | CLS+mean | 0.42 | 0.59 | 0.29 | 0.09 | 0.05 | 0.42 |
| *for reference:* frozen ViT-MAE-B, stack | mean | 0.46 | 0.62 | 0.44 | 0.10 | 0.31 | 0.39 |
| *for reference:* VAE-C (trained), stack | latent | 0.84 | 0.94 | 0.75 | 0.66 | −0.03 | 0.17 |

So:
* MicroNet was not "disagreeing with the KPIs"; it was given three different detectors in the RGB slots
  of a network trained on grey-scale micrographs. On BSE alone it is the best frozen encoder we have.
* The 0.26 → 0.66 gain also brings more imaging-stat leakage (0.04 → 0.47) and higher batch kNN, as expected
  for an encoder trained to be sensitive to micrograph texture; VAE-C remains far ahead on KPI alignment
  with no leakage.
* ImageNet normalisation (±0.01) and last-layer pooling choice (±0.03) are not the issue; multi-stage
  mean+std pooling helps (+0.1–0.15) because the KPIs are area fractions, which live at mid-level features.
* Fine-tuned MicroNet as a batch classifier (`effb4_micronet`, stack view): fine-tuned end to end
  for 1,500 steps like the other classifiers (`cls_micronet_ft.csv`, stratified grouped 5-fold): field accuracy
  **0.64 raw / 0.60 hybrid** (Batch_1 0.40, Batch_2 0.20, Batch_3 0.87 / 0.82), i.e. inside the 0.57–0.63 band
  of ResNet-18, EfficientNet-B4-ImageNet and DINOv2 and with the same per-field pattern. MicroNet weights are
  neither a handicap nor an advantage once the model is allowed to adapt; as a frozen encoder they only
  looked bad because of the RGB-as-detectors input.

Recommendation if MicroNet is to be used: BSE-only (or one detector per forward pass), stages 2–5 mean+std
pooling, and treat it like the other frozen encoders — a reasonable, imaging-sensitive baseline, not a
substitute for the KPI-aligned VAE-C.

---

## 6. Representation leaderboard, recap (unchanged; `leaderboard.csv`)

Gated-KPI R² on 5 held-out fields (221 crops), best config per family, raw / `hybrid` input:

| Family | raw | hybrid | imaging leakage (hybrid) | note |
|---|---|---|---|---|
| VAE-C (KPI head), stack, Batch_3 baseline | 0.824 | **0.844** | R² −0.03, batch kNN 0.17 | recommended; cross-fit 0.84–0.88, LOBO 0.76–0.83 |
| MAE-adapted (ViT-B) | 0.617 | 0.624 | high | |
| DINOv2-S fine-tuned | 0.520 | 0.473 | high | |
| MAE-scratch (ViT-S) | 0.321 | 0.361 | | |
| VAE-B (KPI-conditioned), Inlens | 0.490 | ≈ −0.04 | | collapses; drop |
| VAE-A (plain) | 0.070 | 0.171 | | |
| frozen ViT-MAE-B | 0.461 | 0.480 | | |
| frozen DINOv2-S | 0.424 | 0.342 | | |
| frozen MicroNet, stack (as run) | 0.260 | 0.301 | | see §5: 0.66 on BSE-only |

Why most of these are "poor": they are self-supervised models asked, after the fact, to linearly expose
five area-fraction KPIs. The one model that was told the KPIs exist (VAE-C's auxiliary head) exposes them;
the others encode what dominates pixel variance — texture, noise, session — which is exactly what §3 shows
the classifiers key on. The stack view beats single detectors in every family; BSE alone is second.

---

## 7. What is still open, and what would settle it

| Open question | Would be settled by |
|---|---|
| Is Batch_3's lower noise/sharpness an SEM setting or a sample property? | microscope session logs (dwell, averaging, focus, detector bias) for the Batch_3 session, or one Batch_3 sample re-imaged under Batch_1/2 settings |
| Are `f1vzngrs`, `fzrt2k6r`, `iv6g2oq0` (B1) and `b3esycq1`, `epqdaau9`, `rxax5ozo` (B2) mislabelled, outliers, or just ordinary? | sample provenance for those six fields; if they are the known outliers, drop them and re-run — the classification result would change qualitatively |
| Is the Batch_1 "high-Si" signature material? | the KPIs say Si fraction 0.14–0.16 on two fields vs 0.05–0.08 everywhere else; Kevin's measurement is unchanged by harmonisation, so this is real in the image, but whether it is a batch property or two odd samples needs more Batch_1 fields |
| Does any batch difference exist beyond texture and two fields? | with 7 fields per batch the 95 % CI on a per-batch recall is ± 35 points; more fields, not more models |

---

## 8. Files

* `kpi_harm/crop_kpis_by_harm.csv` — gated KPIs + segmenter thresholds per crop for none/hybrid/affine2/histmatch.
* `cls_field_consistency.csv` — per-field P(true batch) averaged over configurations, by harmonisation.
* `imaging_stats/per_image_by_harm.csv`, `imaging_stats/grey_levels_by_harm.csv` — noise/sharpness and
  grey-level occupancy before/after harmonisation.
* `cls_testtime_perturbation.csv` — saved classifiers under noise/blur/offset/gain at test time.
* `cls_dequant_comparison.csv` — comb-removal retraining.
* `cls_runs/<hash>/attribution.csv`, `attribution_summary.json`, `attrib_<batch>_{correct,wrong}.png` —
  per-crop occlusion/Grad-CAM metrics and galleries; `attribution_confident_summary.csv` aggregates them.
* `micronet/probe_variants.csv` — MicroNet input/pooling ablation; `cls_micronet_ft.csv` — fine-tuned MicroNet classifier.
* `src/v2/attribution.py`, `src/v2/testtime_perturb.py`, `src/v2/micronet_probe.py` — the code behind §3.3, §3.2 and §5.
* `figs/harmonisation_before_after.png` — raw vs hybrid on a Batch_1 site, a strong Batch_3 site and a clean Batch_3 site.
* `calibration/cls_calibration.csv` — ECE/NLL/temperature/prior-correction per classifier config (§9.2).
* `sae/sae_all_runs.csv`, `sae/<hash>/sae_features.csv`, `sae_gallery.png` — sparse-autoencoder feature audit (§9.4).
* `cls_probe_allfields.csv` — LOFO probes on the all-fields embeddings (§9.3); `attribution_by_route.csv` — occlusion/Grad-CAM summaries per input route.
* `injection_ablation/residual.csv`, `site_spread.csv`, `model_ablation.csv`, `zero_clipping.csv`, `residual.png` — artefact-injection ablations (§9.5); code in `src/v2/injection_ablation.py`, `src/v2/injection_model.py`.
* `leaderboard.csv`, `cls_leaderboard.csv`, `results_tables.md`, `RESULTS_detailed.md` — everything else.

Reproduce: `python -m src.v2.attribution <cls_run_hash>` (CPU ok), `python -m src.v2.classify --dequant 1.5 ...`,
KPI recomputation and MicroNet probes are the scripts recorded in the session (`kpi_harm/`, `micronet/`).

---

## 9. Round 4 — the physical "clean" route (PR #27/#28), calibration, SAE, and injection ablations

New in this round: every model family was retrained on the physics-based preprocessing
(`load_clean(..., kind="norm"|"harm")`, see `docs/clean.md`), alongside raw and PR #16 `hybrid`, with
representation models trained on **all 31 labelled fields** (no field held out — the embedding is then scored
by leave-one-field-out probes) and classifiers on stratified grouped 5-fold. Invalid pixels (border, collector,
scan bands, charging) are masked everywhere and crops with < 90 % valid pixels are rejected.
Files: `cls_leaderboard.csv`, `calibration/cls_calibration.csv`, `sae/sae_all_runs.csv`, `cls_probe_allfields.csv`,
`attribution_by_route.csv`, `injection_ablation/*.csv`.

### 9.1 Representations (KPI probe R², frozen embedding → ridge → gated KPIs)

| family | input | aug | KPI R² | imaging R² | batch kNN (chance 0.33) |
|---|---|---|---|---|---|
| VAE-C | raw | aug1 | **0.872** | −0.08 | 0.21 |
| VAE-C | hybrid | aug1 | 0.867 | −0.01 | 0.18 |
| VAE-C | clean harm | aug2 | 0.839 | 0.04 | 0.26 |
| VAE-C | clean norm | aug1 | 0.830 | −0.01 | 0.22 |
| VAE-C | clean harm | aug1 | 0.828 | 0.00 | 0.18 |
| VAE-C BSE only | clean harm | aug1 | 0.789 | 0.15 | 0.21 |
| MAE-adapted | clean harm / norm | aug1 | 0.67 / 0.63 | 0.07 / 0.06 | 0.41 |
| DINO-FT | clean norm / harm | aug1 | 0.55 / 0.53 | 0.04 / 0.01 | 0.36 / 0.35 |

Read with two caveats. (i) These all-fields runs score slightly higher than last round's held-out-trained
ones (0.87 vs 0.84 for the same VAE-C/hybrid config) partly because the encoder has reconstructed the
evaluation fields during training; the ranking, not the absolute gap, is the result. (ii) The KPI targets
are Kevin's raw-image KPIs for every route, so the clean routes are being asked to predict the same quantity.
With that, the clean route does **not** improve KPI alignment (0.83–0.84 vs 0.87); the physical units cost
~0.03–0.04 R² and the ranking of families is unchanged (VAE-C ≫ MAE > DINO).

### 9.2 Classification (stratified grouped 5-fold, 31 fields)

| arch | input | field acc | balanced acc | recall B1 / B2 / B3 | pred. share B3 (true 0.55) |
|---|---|---|---|---|---|
| ResNet-18 | raw | 0.61–0.71 | 0.49–0.66 | 0.29–0.71 / 0.29–0.43 / 0.82–0.88 | 0.55 |
| ResNet-18 | hybrid | 0.58 | 0.49 | 0.43 / 0.29 / 0.76 | 0.48 |
| ResNet-18 | clean norm | 0.68 | 0.58 | 0.57 / 0.29 / 0.88 | 0.58 |
| ResNet-18 | clean harm | **0.52** | 0.43 | 0.29 / 0.29 / 0.71 | 0.45 |
| ResNet-18 | clean harm, aug2 | 0.61 | 0.49 | 0.29 / 0.29 / 0.88 | 0.55 |
| DINOv2-FT | raw / hybrid / clean norm / clean harm | 0.65 / 0.61 / 0.65 / 0.65 | 0.53 / 0.51 / 0.53 / 0.53 | 0.43 / 0.29 / 0.82–0.88 (all) | 0.52–0.58 |
| MicroNet-FT | raw / hybrid / clean norm / clean harm | 0.65 / 0.61 / 0.65 / 0.61 | 0.53 / 0.51 / 0.53 / 0.49 | 0.29–0.43 / 0.29 / 0.82–0.88 | 0.55 |
| MicroNet-FT, BSE | clean norm / clean harm | 0.71 / 0.58 | 0.63 / 0.49 | 0.71 / 0.29 / 0.88 and 0.43 / 0.29 / 0.76 | 0.55 / 0.48 |

* **Nothing moves outside the noise.** Fold-to-fold SD of field accuracy is 0.10–0.17 and a 31-field
  accuracy has a 95 % CI of about ± 0.17, so none of the raw/hybrid/clean differences is significant. The
  one visible pattern — `clean harm` is the lowest for every architecture (0.52–0.61) — is consistent with
  the ablation in §9.5: the `harm` stage is the only route that removes the focus/noise shortcut, so the
  classifier loses its Batch_3 cue and has nothing else to use.
* **Batch_2 recall is 0.29 (2 of 7 fields) in every single configuration**, raw or harmonised, clean or
  not. Batch_1 recall varies 0.29–0.71 and is driven by the two high-Si fields (§3.1). Batch_3 recall is
  0.71–0.88. So the models are not "all Batch_3" (predicted Batch_3 share 0.45–0.58, true 0.55; a
  constant-Batch_3 classifier scores 0.55 accuracy / 0.33 balanced), but their above-chance balanced accuracy
  (0.43–0.66 vs 0.33) is almost entirely Batch_3 recognition plus the two Batch_1 outliers.
* **Calibration is poor everywhere.** Crop-level ECE 0.14–0.26, fitted temperatures 1.7–2.9 (i.e. logits
  2–3× too sharp), and the mean confidence on *wrong* fields (0.57–0.79) is nearly as high as on right
  fields (0.72–0.84): confidence does not flag the misfiled fields. Cross-fold temperature scaling brings
  field ECE to 0.04–0.18 but changes no decision; prior correction (dividing out the 7/7/17 class prior)
  changes balanced accuracy by ≤ 0.05 in either direction. Treat any P(batch) from these models as a ranking,
  not a probability.

### 9.3 LOFO probes on the all-fields embeddings (`cls_probe_allfields.csv`)

Frozen embedding → kNN/logistic probe, leave-one-field-out (31 folds). Best probe per embedding:

| embedding | field acc | B1 / B2 / B3 recall |
|---|---|---|
| VAE-C clean harm (aug1 / aug2) | 0.71 / 0.71 | 1.00 / **0.00** / 0.88, 0.86 / 0.00 / 0.94 |
| MAE-adapted clean norm | 0.71 | 0.71 / 0.29 / 0.88 |
| DINO-FT clean harm | 0.68 | 0.57 / 0.29 / 0.88 |
| VAE-C hybrid | 0.55 | 0.00 / 0.00 / 1.00 |
| VAE-C raw | 0.52 | 0.00 / 0.00 / 0.94 |

The KPI-aligned VAE-C embeddings on raw/hybrid input contain *no* usable Batch_1/2 information (both recalls
0, everything is called Batch_3). The clean-input embeddings recover Batch_1 (through the Si signal that the
fixed physical thresholds make explicit) but **no embedding or classifier anywhere in this project
recognises more than 2 of the 7 Batch_2 fields**.

### 9.4 Sparse autoencoder on the embeddings (`sae/sae_all_runs.csv`, galleries in `sae/<hash>/`)

TopK SAE (8× over-complete, k = 8, reconstruction R² 0.93–0.96) on each all-fields embedding; every feature is
labelled by whether its activation tracks the material KPIs, the imaging statistics (p1, noise, sharpness), both,
or neither (rare). Activation mass:

| embedding | material | imaging | mixed | rare | batch-selective features |
|---|---|---|---|---|---|
| VAE-C raw | 0.41 | 0.25 | 0.09 | 0.26 | 0 |
| VAE-C hybrid | 0.45 | 0.20 | 0.13 | 0.22 | 0 |
| VAE-C clean norm | 0.41 | 0.27 | 0.09 | 0.24 | 0 |
| VAE-C clean harm (aug1 / aug2) | 0.37 / 0.40 | 0.30 / 0.27 | 0.09 / 0.07 | 0.25 / 0.25 | 0 |
| MAE-adapted clean norm / harm | 0.15 / 0.16 | **0.61 / 0.60** | 0.03 | 0.20 / 0.21 | 0 |
| DINO-FT clean norm / harm | 0.24 / 0.24 | **0.63 / 0.63** | 0.04 | 0.09 | 0 |

* VAE-C is the only family whose embedding is majority material: 37–45 % of its activation mass is KPI
  features and 20–30 % imaging. The self-supervised MAE and DINO embeddings are ~60 % imaging features
  on *every* input route — which is why they classify batches better and predict KPIs worse.
* **The clean route does not reduce the imaging share.** For VAE-C it is 0.27–0.30 on clean input vs 0.20
  on hybrid and 0.25 on raw. Physical units remove the offset/gain but the noise and sharpness statistics
  are still encoded (they are real image properties, and `harm` only equalises them downwards to the Batch_3
  reference, §9.5).
* No feature in any model is batch-selective (fires on one batch only), so none of them has a "Batch_3
  detector" unit; the batch information is distributed over imaging features.

### 9.5 Injection ablations: does each route remove a known microscope effect? (`injection_ablation/`)

Method: take the raw half-resolution images of six sites (2 per batch, incl. the strong-offset Batch_3 site
`71vgq3fw` and the heavily 0-clipped Batch_1 site `4ih2ggld`), inject one artefact, run the result through each
route (raw, `affine2`, `hybrid`, `clean norm`, `clean harm`), and measure (a) how much of the artefact
survives in the image statistics, in units of the natural between-site SD of that statistic, and (b) what the
out-of-fold ResNet-18 classifier for that route does with it (`model_ablation.csv`, every site scored by the
fold model that never saw it). Injections: `session_strong` (0.69× gain, +22 grey — the Batch_3 artefact),
`offset6` (+6 grey), `noise6` (σ = 6 grey), `blur1.0` (1 px Gaussian), `requant3` (every 3rd grey level),
`all_strong` (all of them).

| effect | raw | affine2 / hybrid | clean norm | clean harm |
|---|---|---|---|---|
| black level + gain (p1, BSE) | passes through (3/6 fields flip to B3; P(B3) 0.05 → 0.6–0.85) | **removed** (0 flips, P(B3) unchanged) | removed in p1, but see offset row | removed in p1, but see offset row |
| +6 grey offset, side effects | pore fraction −0.5 to −0.9 SD (percentile segmenter) | none (exact) | pore fraction +0.1 to +0.45 SD, noise/sharpness +0.1 to +0.8 SD on the clipped sites; 0 flips | same as clean norm, slightly smaller; 0 flips |
| noise σ = 6 | passes (noise +1.0–1.5 SD); 1 flip | passes (identical to raw) | passes, looks *larger* in z units (+1.6–2.7 SD) | absorbed on 4/6 sites (−0.6 to +1.0 SD) but **+4 and +6 SD on `3806gxp0` and `71vgq3fw`**; 0 flips |
| blur 1 px | passes; **4/6 fields flip to Batch_3** (P(B3) 0.05 → 0.5–0.85) | passes; 4/6 flip | passes; 4/6 flip | sharpness residual −0.0 to −0.3 SD on 4 sites, −0.5 on the reference site; **2/6 flip** |
| requant (step 3) | nothing | nothing | 1 flip | nothing |
| all of the above | 4/6 flip | 0 flips (hybrid) | 1 flip | 2 flips |

What this says, route by route:

* **PR #16 LUTs (`affine2`, `hybrid`) do exactly what they claim and nothing more.** The black-level/gain
  artefact is removed to the pixel, the classifiers trained on them are invariant to it, and KPIs are
  untouched. They are, by construction, transparent to noise, blur and requantisation; the hybrid classifier
  flips 4 of 6 fields to Batch_3 under a 1 px blur. **Where it still fails: any focus or noise difference
  between sessions is read as "Batch_3".**
* **`clean norm`** removes offset/gain in the first percentile but is **not offset-invariant on 0-clipped
  sites**: on `4ih2ggld` (3.8 % of BSE pixels at exactly 0) a +6 grey offset moves its pore fraction by
  0.45 SD and its noise/sharpness statistics by 0.7–0.8 SD, while on the unclipped strong Batch_3 site the
  residual is exactly 0. The dark-level fit is a Tobit (censored) fit, so this is not a naive truncated-mode
  error; it is that a positive offset *un-clips* pixels that were at 0, changing the pore-mode shape and the
  noise texture in a way no later step can know about. The clipping itself is a batch fingerprint: Batch_1/2
  have 1.5–1.7 % BSE zeros, Batch_3 0.4 %, and a logistic model on the three detectors' zero fractions alone
  gets 84 % Batch_3-vs-rest field accuracy (`zero_clipping.csv`). No grey-level route can undo clipping —
  the information is gone. Noise and blur pass straight through `clean norm` (4/6 blur flips).
* **`clean harm` is the only route that attacks the focus/noise shortcut**, and it does so by degrading
  every site *down* to the four Batch_3 reference sites. It removes a 1 px blur on 4 of 6 sites and halves
  the classifier flips (2/6, and one of the two is a Batch_2 field that was already at P(B3) = 0.45 before
  injection). **Where it still fails:** (i) on the reference sites themselves there is no headroom, so an
  injected blur or noise passes through — and, because `harm` compresses the natural between-site spread,
  the same σ = 6 noise shows up as a +4 to +6 SD outlier on `3806gxp0`/`71vgq3fw` where the LUT routes show
  +1.5 SD; (ii) it inherits the 0-clipping sensitivity of `clean norm`; (iii) it adds noise to the clean
  Batch_1/2 images, so it is the lowest-accuracy input for every classifier (§9.2) and does not improve KPI
  alignment (§9.1) — the price of invariance.

### 9.6 Critical assessment — what still looks off

1. **Classifier differences between routes are within fold noise.** We should not claim that any preprocessing
   helps or hurts batch classification by more than ± 0.1 on 31 fields. What the round does establish, with the
   ablations rather than the accuracies, is *which* cue each route leaves available.
2. **Batch_2 is unrecognisable by every method** (2/7 fields, always the same two). Either Batch_2 is
   genuinely not distinct from Batch_1 in these micrographs or 5 of its 7 fields are atypical; only provenance
   can tell.
3. **Noise/sharpness are real image properties, so no "harmonisation" removes them without destroying
   information.** `clean harm` demonstrates the cost. If the goal is a classifier that cannot cheat on focus,
   the honest path is a *loss* on invariance (aug2-style blur/noise randomisation during training, which the
   ablation shows is still incomplete at aug2) or an imaging-stat-conditioned model, not a preprocessing step.
4. **0-clipping is unfixable downstream** and is a Batch_3 fingerprint on its own (84 % Batch_3-vs-rest from
   zero fractions). The only remedy is upstream: acquire with a non-zero black level so pores are not clipped.
5. **The all-fields representation training is transductive** for the KPI probe (eval fields were
   reconstructed in training). It is the right regime for "embed everything, then classify new fields", but
   the 0.87 KPI R² should be quoted next to the held-out 0.84, not instead of it.
6. **Calibration must be fixed before any P(batch) is shown to a user**: temperature 1.7–2.9 and wrong-field
   confidence ≈ right-field confidence. Temperature scaling is implemented (`calibration/`), but the deeper
   problem — confident errors on the same six fields — is a data problem.

Adjustments made this round in response: representation runs switched to all-fields training with LOFO
probing; `p99` removed from the SAE's imaging-confounder list (it tracks Si content); a noise-robust sharpness
statistic added so blur is not confounded with noise in the ablation; residuals expressed in between-site SD
units rather than relative to the raw effect; every ablation site scored with an out-of-fold model.
