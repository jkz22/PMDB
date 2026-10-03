# PMDB: the story so far, in one line of argument

*Written as the overnight synthesis of the repository and every parallel session (KPIs, GP maps,
harmonisation, physical clean, representation learning / classification, FEM review, functional
morphology). Every claim points at the file or document that holds the numbers.*

**The question.** Three batches of a graphite/Si anode were sectioned and imaged (31 labelled fields,
7 / 7 / 17; three held-back fields to assign). Batch 3 is the supplier baseline. Are the batches
different in a way that matters for the electrode, and can that difference be measured from SEM
cross-sections reproducibly enough to serve as QC?

## 1. First, make the images comparable (`docs/harmonisation.md`, `docs/clean.md`)
Before any material statement, the Batch 3 images had to be shown *not* to differ for trivial
reasons. They do differ trivially: four Batch 3 sites have a +19–23 grey-level black level and
0.69× gain on every detector, five more a milder offset, and three sites were re-quantised after
capture. Two routes remove this. The LUT route (`load_site(..., harmonise="hybrid")`) fixes the
affine artefact per site; the physics route (`load_clean`) measures dark level and graphite
flat-field, masks scan bands / charging / clipping, and harmonises resolution and noise downward
to the four unflagged Batch 3 references. After cleaning, an acquisition-only classifier falls
from 0.61 to chance (0.42); what remains separable is the material intensity distribution, which
is left alone on purpose (`outputs/clean/REPORT.md`). The segmenter is percentile-anchored, so
the KPIs are *identical* on raw and harmonised images (Spearman 1.000; `outputs/v2/RESULTS.md` §2).
Everything downstream therefore stands on grey-level-invariant masks.

## 2. Composition is the same (`docs/kpis/`, `outputs/kpis/site_kpis.csv`)
Kevin's spec-002 KPI pipeline (25 KPIs, 43 columns, 31/31 sites, 64 s) gives Si area fraction
K01 = 4.4–16.5 % with batch medians 6.5 / 5.9 / 5.9 % (Kruskal–Wallis p = 0.38). Porosity is
3–5 % everywhere. Two Batch 1 fields (`4ih2ggld`, `5n1q8atc`) carry 2–3× the Si of any other
field. Nothing in composition separates the batches.

## 3. Arrangement is not (`docs/fingerprint.md`)
Field-level geometry does: relative Si depth-band profile, directional pair-correlation excess
(x vs z) and the tile spread of Si–graphite contact (K15). A weight-free robust naive-Bayes
fingerprint on these 16 features reaches leave-one-site-out accuracy 0.677 against a 0.548
majority baseline, permutation p = 0.002 (10 000 permutations, `outputs/overnight/permutation/`),
with Batch 2 the hardest (recall 0.43). "Composition is identical, arrangement is not" is the
repository's central material claim, and the 31-site sample size is its central caveat.

## 4. Where in the field, and how sure (GP session)
The Gaussian-process track turns the same tile KPIs into spatial maps: one GP per KPI over
64-px tile positions (RBF + white noise), giving a smoothed KPI surface, a pointwise uncertainty,
and a fitted lengthscale that acts as a clump-size proxy. It is the tool for *within-field*
heterogeneity and for equivalence-style batch decisions with confidence intervals, not a
classifier.

## 5. What an image model learns, and what it should not (`outputs/v2/RESULTS.md`)
Representation learning asked whether a neural network *contains* the KPI information. The
KPI-headed VAE-C on the harmonised three-detector stack exposes the gated KPIs with ridge-probe
R² = 0.84 on held-out fields with no imaging-statistic leakage; self-supervised and off-the-shelf
encoders (MAE, DINOv2, NASA MicroNet) score 0.26–0.66 and encode texture first. Batch
classification tells the cautionary half of the story: field accuracy 0.55–0.72 across five
architectures, Batch 1 vs 2 at chance, and Batch 3 recognised through **lower BSE noise and
sharpness** — two scalar statistics alone recognise 88 % of Batch 3 fields, adding 6 grey levels
of noise destroys the recognition, and occlusion / Grad-CAM attributions sit diffusely on the
graphite matrix in proportion to its area. The visible "batch difference" an unconstrained model
finds is microscope texture, not microstructure, and six fields are misfiled by every model.

## 6. What the arrangement means for the electrode (`docs/functional.md`, this session)
The missing layer was function: given where the Si sits, what happens when it lithiates? The
pore phase does not percolate in any 2D section (0 / 31, 3–5 % porosity), so tortuosity is not
measurable from a slice and is recorded as such rather than reported. What is measurable is the
swelling budget: grown to its lithiated area, 77–88 % of Si growth lands on graphite, 5–6 % on
pore. The constrained share is the one functional quantity that differs by batch (B1 < B2 < B3,
p = 0.009) and it is the lithiation reading of the K15 contact fraction that the fingerprint
already uses; pore loss (13–25 %) is a function of Si fraction alone, so the high-Si Batch 1
fields are the pore-closure risk. These columns do not improve batch identification (LOSO 0.71
vs 0.68, same permutation p); they explain what the identified difference would do.

## 7. From geometry to mechanics (`docs/fem/literature-review.md`, `analytical_benchmarks/`)
The FEM review sets out the plane-strain finite-strain lithiation model (eigenstrain, neo-Hookean,
SOC-dependent properties) that would turn "Si pressing on graphite" into stress, and the
analytical benchmarks give order-of-magnitude swelling and cycling estimates. Neither is
calibrated to this material; §6 tells them which fields and which batch to simulate first.

## 8. What would settle it
The chain is consistent — artefacts removed, composition equal, arrangement different, the
difference material rather than textural, its functional consequence identified — but every link
runs on 7 fields per batch. A plug-in power analysis (`outputs/functional/power.csv`) says the
strongest KPI effects need ≈ 10–15 fields per batch for 80 % power and most need more than 50;
with the current sample the power to confirm even the largest effect is ≈ 0.6. The two things
that would move the project most are therefore not models: (i) more Batch 1 and Batch 2 fields,
and (ii) the Batch 3 microscope session logs (or one Batch 3 sample re-imaged under Batch 1/2
settings) to close the noise/sharpness question. On the held-back sites the fingerprint assigns
3e122cbj → Batch 1 (robust under jackknife), fn0mhxef → Batch 3 and xrv9xvzb → Batch 2
(`outputs/fingerprint/heldout_predictions.csv`, `outputs/overnight/stability/`); the functional
columns agree on the first and are too weak to vote on the other two.

### Where each piece lives
| Layer | Session / branch | Files |
|---|---|---|
| Loader, cache, harmonisation | main | `pmdb/io.py`, `pmdb/harmonise.py`, `docs/harmonisation.md` |
| Physical clean | main | `pmdb/clean.py`, `docs/clean.md`, `outputs/clean/` |
| KPIs (spec 002) | Kevin's KPI session, PRs #1/#4/#5 | `pmdb/kpis/`, `docs/kpis/`, `outputs/kpis/` |
| Fingerprint + overnight validation | main | `pmdb/fingerprint.py`, `docs/fingerprint.md`, `outputs/fingerprint/`, `outputs/overnight/` |
| GP tile maps, physics estimates | `analytical-benchmarks` | `analytical_benchmarks/` |
| Representation learning, classification, attribution | `devin/1791038010-v2-representation` | `outputs/v2/RESULTS.md`, `src/v2/` |
| FEM lithiation model review | `fem-lit-review`, `fem-sim` | `docs/fem/literature-review.md` |
| Functional morphology | this session | `pmdb/functional.py`, `docs/functional.md`, `outputs/functional/` |
