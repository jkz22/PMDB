# What to do: the decision pipeline

*Supplementary session, 2026-10-04. Entry point for the deliverable; the reasoning behind each component is in
`docs/story.md`, `docs/reliability.md`, `docs/acceptance.md`, `docs/crosswalk.md`.*

## 0. The verdict in one paragraph

Thirty-one labelled fields support exactly one statistically significant batch signal — the **arrangement**
of Si through the electrode thickness and in the plane (the 16 fingerprint features) — and no learner, however
flexible, gets more out of them. Composition KPIs carry no batch information under any model; half of the
KPI catalogue is sampling noise at this field size; the one functional quantity that looked batch-specific
(constrained swelling) rests on an unvalidatable segmentation class. So the project "works" as what it is: a
**batch-labelling tool at ≈ 0.68 leave-one-site-out accuracy (chance 0.55, p ≈ 0.003–0.005)**, a
**per-field explanation that only quotes deviations larger than the field's own measurement noise**, and a
**lithiation-consequence reading (pore loss) validated against the FEM**. Do not promise a Batch 3
accept/reject rule; do not add models; add fields.

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

Result (`outputs/decision/heldout_cards.md`, figure `figures/decision.png`):

- **3e122cbj → Batch 1.** 6/6 batch-calling methods agree; XGBoost P(B1) 0.63. *Outside* the Batch 3 baseline on
  composition and function (100th percentile) — Si fraction +10 robust SD (8× its sampling SD), number
  density +9, contact −5, spacing −5; SOC-1 pore loss 0.22, above every Batch 3 field. Arrangement alone cannot
  separate it from Batch 3 (conformal p 1.0): it is the functional twin of the two high-Si Batch 1 fields.
- **fn0mhxef → Batch 3.** 5/6 agree; XGBoost P(B3) 0.83. Inside the baseline on every family (24th–65th
  percentile); no reliable KPI is off; pore loss at the Batch 3 median. The one field where "Batch 3" is a
  comfortable call.
- **xrv9xvzb → Batch 2.** 3/6 agree, split along the method type: everything that reads depth arrangement says
  Batch 2 (fingerprint credibility 1.0, confidence 0.5; XGBoost P(B2) 0.54), everything scalar says Batch 3.
  No reliable KPI is off and it is inside the Batch 3 cloud on every family, so if it *is* Batch 2 it is only
  by its mid-depth Si depletion (mid-dip −0.72 vs Batch 2 median −0.89, Batch 3 −0.10) and the matching
  mid-depth pore-loss minimum. Report as "Batch 2, arrangement-only evidence, inside Batch 3 on composition".

## 3. What not to do, and why

- **No further classifiers on these 31 fields.** Five model families (robust NB, two-stage RF, frozen
  DINOv2/MicroNet heads, XGBoost, tile voting) land between 0.55 and 0.68; the permutation null's 95th percentile is 0.58.
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
