# What to do: the decision pipeline

*Supplementary session, 2026-10-04. Entry point for the deliverable; the reasoning behind each component is in
`docs/story.md`, `docs/reliability.md`, `docs/acceptance.md`, `docs/crosswalk.md`.*

## 0. The verdict in one paragraph

*Revised 2026-10-04 after the organiser released the held-out labels (§5).* Thirty-one labelled fields
support exactly one statistically significant **within-dataset** batch signal — the arrangement of Si through
the electrode thickness (the 16 fingerprint features, LOSO 0.68 vs 0.55 chance, p ≈ 0.005) — and no learner
gets more out of them (§1). **That signal did not transfer**: on the three released held-out labels the
fingerprint, XGBoost, both RF+FEM arms and the shipped consensus card score **0/3**, and 24 three-class
methods average 0.58 correct of 3 (chance 1.0). The two cues every model had learned — "very high Si ⇒
Batch 1" (from two fields) and "mid-depth Si dip ⇒ Batch 2" — were each contradicted by one held-out field.
What *did* hold: the off/baseline reading. Material deviation (KPIs + swelling) correctly puts 3e122cbj
outside and xrv9xvzb inside Batch 3; the one field it misses, fn0mhxef (Batch 1), is at the Batch 3 baseline
on every mask-based measure and is flagged only by the acquisition-sensitive readers (Modelling CNN
off-detector 3/3; grey-level nearest mean 2/3). So the project "works" as: an **off/baseline detector with
separate material and imaging columns**, a **per-field explanation that quotes only deviations larger than
the field's measurement noise**, and a **lithiation-consequence reading (pore loss) validated against the
FEM** — all three label-independent. Do not report a three-class batch accuracy from this dataset.

## 1. XGBoost on the reliable set (`scripts/run_xgb_reliable.py`, `outputs/xgb/`)

Asked directly: does a non-linear learner on *only the reliable inputs* beat the robust naive-Bayes
fingerprint? Reliable = v1 KPIs with 16-tile ICC ≥ 0.75 (K01, K02, K03 d50/d90/max, K05, K09, K14 p95, K15;
K07 / K04 agglomerate / K14 p50 / K16 dropped as noise), the 16 fingerprint features, SOC-1 pore loss and its
mid-depth contrast (segmenter-robust); constrained share excluded. Depth-2 trees, 150 rounds, leave-one-site-out,
200-permutation null on the whole LOSO pipeline.

| features | n | LOSO acc | perm p (null 95 %) | recall B1 / B2 / B3 | Batch 3-vs-rest AUC (perm p) |
|---|---|---|---|---|---|
| reliable KPIs | 10 | 0.45 | 0.41 (0.58) | 0.29 / 0.14 / 0.65 | 0.37 (0.68) |
| fingerprint 16 | 16 | **0.68** | **0.005** (0.58) | 0.71 / 0.43 / 0.76 | 0.57 (0.19) |
| reliable KPIs + functional | 12 | 0.52 | 0.20 (0.55) | 0.43 / 0.29 / 0.65 | 0.67 (0.08) |
| all reliable | 28 | 0.61 | 0.04 (0.58) | 0.57 / 0.43 / 0.71 | 0.55 (0.26) |
| tile-level XGB, 16 tiles × 31 sites → site posterior | 10 | 0.61 | 0.10 (0.61) | — | — |

(Exact recalls in `outputs/xgb/results.json`.) Readings:

- XGBoost on the 16 arrangement features reproduces the fingerprint's 0.677 to the field — same confusion
  structure (Batch 2 is the hard class). The model class is not the limit; the information is.
- Composition KPIs alone are chance under XGBoost (0.45) exactly as under the fingerprint model (0.26) and as
  the one-class acceptance test said (AUC 0.50). Adding them to the arrangement features dilutes (0.68 → 0.61),
  as adding them to the naive-Bayes model did (0.68 → 0.48).
- Tile-level training (496 rows) does not help: tiles within a field are near-copies for the KPIs that matter
  (K02 ICC 0.82), so the effective sample size stays 31.
- Gain shares (`outputs/xgb/importance.csv`, `figures/importance.png`): the trees split first on the Si depth
  profile (band 2, mid-dip), SOC-1 pore loss and its depth contrast, K15 contact, then the in-plane lateral
  curves — the same chain the story draws by hand.
- Batch 3-vs-rest: no family reaches significance; the functional one is again the closest (0.67, p 0.08),
  consistent with `docs/acceptance.md` §4.

## 1b. Tile voting with a confidence cutoff (`scripts/run_tile_vote.py`, `outputs/tilevote/`)

The natural next idea — run the KPIs on tiles, classify every tile, let the tiles vote and use the vote share
as the field's confidence, abstaining below a cutoff — was tested as stated. Tiles are 16 or 32 equal-width,
full-height strips (`pmdb.kpis.tile_slices`), so both the reliable tile KPIs and *arrangement* features (Si
fraction in 5 depth bands relative to the tile mean, slope, mid-dip, SOC-1 pore loss) exist per tile.
Logistic regression and XGBoost, leave-one-site-out (all tiles of the held-out field removed), plurality vote,
100-permutation null, accuracy-vs-coverage for cutoffs on the vote share.

| tiles | features | model | LOSO vote accuracy | perm p | vote share, correct calls | vote share, wrong calls |
|---|---|---|---|---|---|---|
| 16 | reliable KPIs | LR | 0.61 | 0.05 | 0.95 | 0.95 |
| 16 | arrangement | LR / XGB | 0.55 (= majority class) | 1.0 | 0.94 | 0.95 |
| 16 | all | LR / XGB | 0.61 | 0.06 | 0.87 | 0.92 |
| 32 | all | LR / XGB | 0.61 | — | 0.93 | 0.97 |

| cutoff on vote share (all, LR, 16) | coverage | accuracy among covered |
|---|---|---|
| 0.5 | 100 % | 0.61 |
| 0.8 | 90 % | 0.57 |
| 0.9 | 42 % | 0.46 |
| 0.94 | 23 % | 0.57 |

Readings (`outputs/tilevote/loso_votes.csv`, `coverage.csv`, `figures/coverage.png`):

- **The vote share is not a confidence.** Wrong calls are made with the same tile agreement as correct ones;
  14 fields have ≥ 15 of 16 tiles agreeing and 7 of them are wrong (every one called Batch 3). Raising the cutoff
  removes coverage without adding accuracy. Tiles of one field are near-copies of each other (K02 ICC 0.82), so
  they agree with *each other*, not with the truth: 16 votes carry about one vote of information.
- **Arrangement does not survive tiling.** Per-tile depth profiles classify at the majority-class rate: the
  Si depth profile is a whole-field quantity (it is averaged over the full width in the fingerprint for a
  reason); on a 3 µm strip it is sampling noise. Hence tile models can only use composition, which is exactly
  the information that carries no batch signal (§1).
- Held-out under the rule (all, LR, 16): 3e122cbj → Batch 1 (16/16 tiles), fn0mhxef → Batch 3 (16/16),
  xrv9xvzb → Batch 3 (14/16, mean posterior 0.57) — the tile view, being composition-only, sides with the
  scalar methods on xrv9xvzb and cannot see the depth arrangement the fingerprint calls Batch 2 on.
- Conclusion: the usable confidence remains the field-level conformal p / credibility of the fingerprint,
  which is a statement about *how unusual this field is for each batch*, not about how many sub-images agree.

## 1c. The simplest predictor: nearest batch mean (`scripts/run_mean_predictor.py`, `outputs/meanpred/`)

"Compare the mean of the new image to the mean of each batch." Each field is one small vector, the batch
means come from the training fields only, a field goes to the nearest mean (Euclidean after standardising on
the training fold). Leave-one-site-out, 500-permutation null. The flavours differ only in what "the mean of
the image" is taken over:

| "mean of the image" | p | LOSO acc | perm p | recall B1 / B2 / B3 |
|---|---|---|---|---|
| raw BSE grey: moments + 16-bin histogram | 20 | 0.61 | 0.008 | 0.57 / 0.43 / 0.71 |
| **harmonised** BSE grey, same | 20 | 0.68 | 0.004 | 0.71 / 0.29 / 0.82 |
| harmonised BSE histogram only | 16 | 0.68 | 0.004 | 0.71 / 0.29 / 0.82 |
| harmonised BSE grey **inside graphite pixels only** | 19 | **0.74** | 0.002 | 0.86 / 0.71 / 0.71 |
| harmonised BSE grey inside pore pixels only | 19 | 0.71 | 0.002 | 0.86 / 0.43 / 0.76 |
| raw BSE mean grey in 15 depth bands | 15 | 0.55 | 0.026 | 0.29 / 0.86 / 0.53 |
| harmonised, same | 15 | 0.42 | 0.31 | — |
| segmentation overlay means (Si / graphite / pore / binder fraction) | 4 | 0.48 | 0.12 | 0.29 / 0.43 / 0.59 |
| overlay Si + graphite depth profiles (15 bands, relative) | 30 | 0.48 | 0.14 | 0.57 / 0.43 / 0.47 |
| fingerprint 16 features, nearest centroid instead of robust NB | 16 | 0.65 | 0.014 | 0.71 / 0.57 / 0.65 |

Readings (`outputs/meanpred/results.json`, `loso.csv`, `figures/accuracy.png`):

- **The best "classifier" in the repository is the grey-level distribution inside a single phase.** Inside
  graphite pixels no composition and no arrangement can enter; the vector is noise width, residual gain and
  quantisation. It reaches 0.74 — above the fingerprint — and it does so *after* the hybrid harmonisation,
  which by design only matches black level and gain (graphite anchor SD 0.0 in `docs/harmonisation.md`)
  and leaves the noise and the within-phase distribution alone. What differs by batch: graphite IQR (Batch 2
  wider, 14 vs 12 DN, KW p 0.02), pore-pixel mean and SD (Batch 3 pores 7 DN vs 3.5–4.3, KW p 1e-4 — the
  non-affine remnant of the Batch 3 artefact: pixels at 0 cannot be mapped). This is the
  microscope-session fingerprint v2 found as noise/sharpness leakage, isolated to its purest form.
- **Composition means are chance** (0.48, p 0.12), as every other route found. The overlay depth profiles
  as a nearest-centroid are chance too; the fingerprint's 16 engineered features get 0.65 with the same
  nearest-centroid rule, so the fingerprint result is not model-specific either — it is the features.
- **Any image-based model with batch accuracy above ~0.68 must be suspected of reading the microscope**, not
  the electrode. The three held-back calls from the graphite-only vector are B1 / B2 / B3 — it agrees with the
  consensus on 3e122cbj (whose Si is visible in any statistic), and disagrees on both others (fn0mhxef → B2,
  xrv9xvzb → B3), which is how a session fingerprint rather than a material fingerprint would behave. Used
  the other way round, it is a cheap *QC* statistic: "was this image acquired like the Batch 3 images?"
- Mask-based features (fingerprint, KPIs, functional) are insulated from it only to the extent the segmenter
  is (Si IoU ≥ 0.992 raw vs harmonised, `docs/story.md` §6); the clean pipeline's noise harmonisation
  (`docs/clean.md`, arrays rebuilt locally) is the route to test whether the graphite-only signal can be
  removed at all — not done here.

## 2. The pipeline (`scripts/run_decision.py`, `outputs/decision/`)

One card per held-back field, every line pointing at a committed file:

| line | source | why this and not something else |
|---|---|---|
| **call** | fingerprint conformal NB on the 16 arrangement features (`outputs/fingerprint/heldout_predictions.csv`) | only model with a significant LOSO result; XGBoost on everything reliable agrees on all three |
| **confidence** | fingerprint credibility / confidence, XGBoost posterior (`outputs/xgb/heldout_predictions.csv`), consensus over the methods on main (`outputs/crosswalk/heldout_consensus.csv`) | three different notions of confidence shown side by side rather than one over-interpreted number |
| **inside Batch 3?** | conformal p(Batch 3) + family percentiles vs Batch 3 (`outputs/acceptance/heldout_joint.csv`) | the organiser's question; answered as membership, not as a rule |
| **how it differs** | KPIs with \|robust z\| > 2 vs Batch 3 **and** deviation > 2 × the field's sampling SD (`outputs/acceptance/deviation_long.csv`) | deviations inside the measurement noise are not evidence (`docs/reliability.md` §1) |
| **arrangement** | Si depth mid-dip, SOC-1 pore-loss depth profile (`outputs/functional/depth_swelling.csv`) | the signal the call is actually made from, in physical terms |
| **consequence** | SOC-1 geometric pore loss vs Batch 3 | validated against FEM pore closure (ρ 0.73, `docs/crosswalk.md` §2); segmenter-robust |

Result (`outputs/decision/heldout_cards.md`, figure `figures/decision.png`) — kept as written before the labels
were released, so the pipeline can be judged honestly; see §5 for the score:

- **3e122cbj → Batch 1** *(released label: **Batch 2** — wrong)*. 6/6 batch-calling methods agree; XGBoost P(B1) 0.63. *Outside* the Batch 3 baseline on
  composition and function (100th percentile) — Si fraction +10 robust SD (8× its sampling SD), number
  density +9, contact −5, spacing −5; SOC-1 pore loss 0.22, above every Batch 3 field. Arrangement alone cannot
  separate it from Batch 3 (conformal p 1.0): it is the functional twin of the two high-Si Batch 1 fields.
- **fn0mhxef → Batch 3** *(released label: **Batch 1** — wrong)*. 5/6 agree; XGBoost P(B3) 0.83. Inside the baseline on every family (24th–65th
  percentile); no reliable KPI is off; pore loss at the Batch 3 median. The one field where "Batch 3" is a
  comfortable call.
- **xrv9xvzb → Batch 2** *(released label: **Batch 3** — wrong; the scalar readers were right)*. 3/6 agree, split along the method type: everything that reads depth arrangement says
  Batch 2 (fingerprint credibility 1.0, confidence 0.5; XGBoost P(B2) 0.54), everything scalar says Batch 3.
  No reliable KPI is off and it is inside the Batch 3 cloud on every family, so if it *is* Batch 2 it is only
  by its mid-depth Si depletion (mid-dip −0.72 vs Batch 2 median −0.89, Batch 3 −0.10) and the matching
  mid-depth pore-loss minimum. Report as "Batch 2, arrangement-only evidence, inside Batch 3 on composition".

## 3. What not to do, and why

- **No further classifiers on these 31 fields.** Five model families (robust NB, two-stage RF, frozen
  DINOv2/MicroNet heads, XGBoost, tile voting) land between 0.55 and 0.68 on material features; the only
  thing that scores higher (0.74) is the grey-level distribution inside graphite, i.e. the microscope (§1c); the permutation null's 95th percentile is 0.58.
  Anything higher reported from here on is a search artefact unless pre-registered
  (`docs/eval-plan-oct4.md` showed how that goes).
- **No accept/reject rule for Batch 3.** Best-of-84 column AUC 0.74 = chance (p 0.50); pre-specified family
  scores ≤ 0.71 (p ≈ 0.08 corrected); XGBoost binary ≤ 0.67. Batch 3 is the wide batch.
- **No K07, K04 agglomerate, K14 p50, K08 in any explanation** (sampling noise / undefined MAD), and **no
  constrained-swelling share** (unvalidatable class, `docs/reliability.md` §3).
- **No fine-tuned image models**: v2 showed they learn BSE noise and sharpness first; the harmonisation makes
  the grey levels safe, not the noise spectrum.

## 4. What would raise the ceiling

In order of leverage (`docs/reliability.md` §4, `docs/functional.md` §3):

1. **More fields, not bigger fields**: sampling noise is 12–22 % of the field variance for the KPIs that
   matter, so K15 needs ≈ 50 fields per batch for 80 % power even at infinite area; Batch 2 (7 fields,
   recall 0.43) is the batch to add first.
2. **Batch 3 session logs or one Batch 3 sample re-imaged under Batch 1/2 settings** to close the
   noise/sharpness confound for any future image model.
3. **A measured identity for binder/carbon** (labelled patches or an EDS C/F map) — the only way to find out
   whether the constrained-swelling difference is real.
4. **FEM at matched SOC on the two high-Si Batch 1 fields and 3e122cbj** — the geometric pore-loss ranking says
   those are where mechanics would differ; the FEM is where that becomes a number.

## 5. Scored against the released labels (`scripts/run_heldout_score.py`, `outputs/decision/heldout_scored.md`)

On 2026-10-04 (11:04 UTC) the organiser gave the held-out labels in the Modelling-session chat:
**3e122cbj = Batch 2, fn0mhxef = Batch 1, xrv9xvzb = Batch 3.** Nothing was refitted; every call in this
branch and on main was scored as it stood.

| method (kind) | 3e122cbj | fn0mhxef | xrv9xvzb | correct | LOSO |
|---|---|---|---|---|---|
| fingerprint conformal NB (arrangement) | B1 | B3 | B2 | 0/3 | 0.68 |
| XGBoost all reliable (arrangement) | B1 | B3 | B2 | 0/3 | 0.61 |
| XGBoost reliable KPIs (scalar) | B1 | B3 | **B3** | 1/3 | 0.45 |
| two-stage RF, KPI arm (scalar, main) | B1 | B3 | **B3** | 1/3 | — |
| two-stage RF, FEM / KPI+FEM arms (arrangement, main) | B1 | B3 | B2 | 0/3 | — |
| tile vote 16 (scalar + per-tile depth) | B1 | B3 | **B3** | 1/3 | 0.61 |
| nearest mean, grey inside graphite (acquisition) | B1 | B2 | **B3** | 1/3 | 0.74 |
| nearest mean, grey inside pores (acquisition) | B1 | **B1** | **B3** | 2/3 | 0.71 |
| nearest mean, overlay fractions (scalar) | B1 | B2 | B2 | 0/3 | 0.48 |
| **decision card (consensus)** | B1 | B3 | B2 | **0/3** | — |
| one-class KPI / functional score ≥ 95th pct of B3 (off/baseline) | **off** | baseline | **baseline** | 2/3 | — |
| noise-aware KPI deviation, any KPI off (off/baseline) | **off** | baseline | **baseline** | 2/3 | — |
| Modelling session CNN off-detector, every route (off/baseline) | **off** | **off** | **baseline** | 3/3 | field acc 0.84–0.90 |

Full table (24 three-class methods, 5 binary) in `outputs/decision/heldout_scored.md`: 11 methods at 0/3,
12 at 1/3, one at 2/3, none at 3/3; mean 0.58 correct of 3 against 1.0 for uniform guessing.

**What three fields can and cannot say.** One method at 0/3 is unremarkable (probability 8/27 ≈ 0.30 under
chance), and three fields cannot validate or refute a 0.68 LOSO estimate on their own. What matters is the
*pattern*: every three-class method, arrangement-based or scalar, made the **same** two mistakes — 3e122cbj →
Batch 1 (its 13.7 % Si and swelling behaviour are the twin of the two high-Si Batch 1 fields; it is Batch 2)
and fn0mhxef → Batch 3 or Batch 2 (it sits inside Batch 3 on every reliable KPI, the fingerprint, the
functional score and the FEM features; it is Batch 1). The methods did not fail independently; they all
learned the same two cues from 7 + 7 non-baseline fields, and both cues were wrong about the next field.
Concretely:

1. **High Si fraction is not a Batch 1 property.** Two Batch 1 fields and now one Batch 2 field are
   high-Si (13.7–16.4 %); the rest of every batch is 5–8 %. Every "Batch 1" call in this project that
   rested on Si (consensus, XGBoost, RF, the material deviation table's *batch* reading) was reading an
   outlier axis that is orthogonal to the labels. The deviation itself is real and useful — it just says
   "off baseline", not "Batch 1".
2. **The mid-depth Si-depletion signature did not transfer.** xrv9xvzb has the Batch 2-like mid-dip
   (−0.72) and the matching mid-depth pore-loss minimum, and is Batch 3. The arrangement-vs-scalar
   disagreement recorded in `docs/crosswalk.md` §1 resolved in favour of the scalar readers.
3. **fn0mhxef is Batch 1 with no material signature.** No reliable KPI is off, pore loss is at the Batch 3
   median, the fingerprint conformal p(B3) is 0.44, FEM features are baseline; the Modelling session's
   imaging statistics put its sharpness at −4 SD and noise at −2 SD, and its CNN off-detector called it
   "off" with P 0.87–0.94 on every harmonisation route. Whatever makes it Batch 1 is visible to the
   acquisition-sensitive readers and invisible to every mask-based one.

**Combined with the organiser's other new information** (Modelling chat, 10:35 UTC: the batches are
artificial groupings of crops from larger original images, built from "some feature they've calculated or
observed"), the parsimonious reading is that the batch label tracks something that co-varies with the
acquisition session at least as much as with any microstructure KPI we compute — the Modelling session
reached the same conclusion from its side (its off-detector is reproduced by two noise/sharpness scalars,
field acc 0.90). We cannot tell from three fields whether that something is a material property that none
of our mask-derived KPIs captures, or an imaging property. Three fields also cannot establish the CNN's
3/3 as validation: for a binary with prior ≈ 0.45 the chance of 3/3 is ≈ 0.11, and the Modelling session
itself flags the fn0mhxef call as imaging-driven.

**What stands, what falls, what to report.**

- *Stands (label-independent):* the KPI reliability/ICC and fields-vs-area results; the noise-aware
  deviation table; the geometric swelling test and its FEM crosswalk (3e122cbj's pore-closure risk is real
  whichever batch it is); the microscope-session leakage control (§1c); the finding that composition
  overlays are at chance; the LOSO 0.68 *as a statement about these 31 fields* (permutation p ≈ 0.005).
- *Falls:* the fingerprint/XGBoost/consensus **as a batch predictor** for new fields (0/3, same errors as
  every other method); "arrangement is the batch signal" as an out-of-sample claim; every sentence in
  `docs/crosswalk.md`, `docs/story.md` and this document that read 3e122cbj as "Batch 1-like" or xrv9xvzb
  as "Batch 2 by arrangement" — they are kept as written and marked, not rewritten.
- *Report:* an **off/baseline card with two independent columns** — *material* (reliable-KPI deviation +
  functional score vs Batch 3: 3e122cbj off, fn0mhxef baseline, xrv9xvzb baseline) and *imaging*
  (noise/sharpness/grey-level session fingerprint: 3e122cbj off, fn0mhxef off, xrv9xvzb baseline) — with
  the explicit statement that the held-out labels are matched by the union, not by the material column
  alone. No three-class accuracy. The organiser's definition of the batches is the single piece of
  information that would turn this into a model: if it is a computed feature of the crops, regress our
  KPI and imaging statistics on it directly instead of classifying.
