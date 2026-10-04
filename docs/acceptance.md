# The acceptance question: is a new field inside the Batch 3 baseline?

The organiser framed the task as a manufacturer's decision: Batch 3 is what the supplier promised,
Batches 1 and 2 are the kinds of deviation to catch, and the held-out fields should be judged
"in or out of distribution" against that baseline (`data_heldout/README.md`). That is a one-class
test, not a three-way classification, and it is worth checking separately how well the evidence
collected so far answers *it*.

`python scripts/run_acceptance.py` → `outputs/acceptance/`.

## 1. One column at a time (`per_feature.csv`)

Every candidate column (34 v1 KPI / A columns, 16 fingerprint features, 35 functional F, 11 stretch
S; 84 remain after dropping near-constant or mostly-missing ones) is turned into an acceptance score: |robust z| of the
field against the Batch 3 fields, leave-one-out when the field is itself Batch 3. ROC AUC for
"Batch 1 or 2" vs "Batch 3":

| best columns | AUC |
|---|---|
| `S01_c2_anisotropy`, `S02_euler_merge_radius_um` | 0.74 |
| `F02_soc100_si_objects_ratio` (Si merging on lithiation) | 0.73 |
| `F01_si_pore_dist_p90_um`, `K05_local_af_cv` | 0.70–0.71 |
| median over all 84 columns | 0.55 |

**Permutation test on the maximum AUC over all 84 columns: p = 0.50.** Picking the best of 84
columns on 31 fields gives a best AUC of ≈ 0.73 by chance alone. No single KPI, fingerprint,
functional or stretch column rejects a non-baseline field better than a coin toss once the
search is accounted for.

## 2. The fingerprint classifier's own view (`fingerprint_ovr.csv`)

From its committed leave-one-site-out predictions (`outputs/fingerprint/loo_predictions.csv`),
one-vs-rest:

| batch | AUC (conformal p) | AUC (likelihood) | recall | precision |
|---|---|---|---|---|
| Batch 1 | 0.74 | 0.74 | 0.71 | 0.50 |
| Batch 2 | 0.38 | 0.30 | 0.43 | 0.60 |
| **Batch 3** | **0.56** | **0.54** | 0.77 | 0.81 |

The fingerprint's 0.68 three-way accuracy is earned by recognising Batch 1 (AUC 0.74) and by the
Batch 1 / Batch 2 mid-depth contrast inside the three-way vote (Batch 2 on its own is *not*
recognised one-vs-rest, AUC 0.30–0.38); its ability to say "this is *not* the baseline" is 0.54–0.56 AUC. The 0.77 recall / 0.81
precision on Batch 3 are what a 17-of-31 majority buys, not discrimination.

## 3. Why, and what it means

- Batch 3 is the *wide* batch: 17 fields, the largest spread on most columns (see the strip plots
  in `docs/figures/story.png`). Batches 1 and 2 are each 7 fields that sit inside that spread on
  composition and on almost every arrangement and functional column; what sets them apart is where
  they sit *relative to each other* (Batch 2 mid-depth depleted, Batch 1 not), which is a two-batch
  contrast, not a distance from the baseline.
- Consistent with the power analysis (`docs/functional.md` §2.7): at 7 / 7 / 17 fields even the
  largest effects are detected with power ≈ 0.3–0.6. An acceptance rule needs the baseline
  distribution pinned down — more Batch 3 fields, from more electrodes — before any
  threshold on these columns means anything.
- For the held-out sites this says: the batch *label* we attach (fingerprint, `docs/fingerprint.md`
  §4) is the defensible deliverable; an "accept / reject against Batch 3" verdict is not supported
  by the labelled data, except for `3e122cbj`, which sits outside *every* Batch 3 field on three
  functional columns at once (`docs/functional.md` §2.5) — a joint outlier that no single column's
  AUC captures.
- What would change this: a few more baseline fields (Batch 3 is the cheapest batch to extend),
  or a 3D/second-slice measurement of the one quantity that is physically tied to the batch
  difference (through-thickness Si placement) rather than more 2D descriptors.

## 4. Pre-specified multivariate score (no column search)

`scripts/run_acceptance_joint.py` (→ `outputs/acceptance/joint*.csv`, `joint_summary.json`). Three
families were fixed before looking: all v1 KPI/A columns (33), the 16 fingerprint features, and all
functional F columns (31). A field's score in a family is the RMS of its leave-one-out |robust z|
against Batch 3 over every column of the family, so no column is picked after the fact. Rejecting
Batch 1+2 from Batch 3 (1000 label permutations of the same statistic, Bonferroni × 3 families):

| family | columns | AUC | perm p | × 3 | median score B3 / B1+2 |
|---|---|---|---|---|---|
| v1 KPIs | 33 | 0.50 | 0.49 | 1.0 | 1.50 / 1.50 |
| fingerprint | 16 | 0.54 | 0.38 | 1.0 | 1.00 / 1.23 |
| functional | 31 | 0.71 | 0.027 | 0.08 | 1.20 / 1.86 |

Reading: the composition KPIs and the fingerprint give *no* joint one-class signal against Batch 3,
confirming §2–3 (the fingerprint's accuracy is a Batch 1 / Batch 2 contrast, not a Batch 3 boundary).
The functional family is the only one that moves, and it is borderline once the three families are
accounted for (p ≈ 0.08). With 14 non-baseline fields this is a hint that the lithiation-geometry
columns are the right place to look for an acceptance rule, not a rule. The mechanism is the one in
`docs/functional.md` §2.3/§2.6: Batch 1/2 fields sit at lower constrained share and higher pore loss,
driven by Si fraction and object size.

Held-out sites (score vs the Batch 3 reference statistics; percentile among the 17 LOO Batch 3 scores):

| site | v1 KPIs | fingerprint | functional |
|---|---|---|---|
| 3e122cbj | 4.16 (100th) | 0.57 (0th) | 5.34 (100th) |
| fn0mhxef | 1.19 (24th) | 1.18 (65th) | 1.02 (29th) |
| xrv9xvzb | 1.17 (24th) | 1.23 (65th) | 1.05 (35th) |

3e122cbj is outside every Batch 3 field on both the composition and functional families (consistent
with `docs/functional.md` §2.5: twice the Si, swelling into free space); fn0mhxef and xrv9xvzb are
inside the Batch 3 cloud on all three families, so the functional/joint evidence gives them no reason
to be rejected from the baseline — their batch labels rest on the fingerprint's Batch 1/2/3 classifier
alone and should carry its (moderate) confidence, not a one-class rejection.

Cross-check on 3e122cbj: its per-column fingerprint score is lower than any Batch 3 field (0.57), and
the fingerprint's own conformal output says the same thing — `outputs/fingerprint/heldout_predictions.csv`
gives p_Batch_3 = 1.0 with confidence 0.0 for a Batch 1 call of credibility 0.875. The fingerprint
therefore cannot separate 3e122cbj from the baseline; the evidence that it is *not* a Batch 3 field is
composition (twice the Si) and lithiation geometry, both at the 100th percentile of Batch 3. That is the
defensible held-out statement for this site: "Batch 1-like by composition and swelling behaviour,
arrangement indistinguishable from Batch 3".

> Caveat (`docs/reliability.md` §3): the functional family's AUC 0.71 rests mainly on the constrained share, whose batch difference does not reproduce under an independently written segmenter. Read the family result as segmentation-dependent.

## 5. Noise-aware "why is this field off?" table (`scripts/run_deviation_table.py`)

The organisers ask, per held-out field, *how it differs from Batch 3*. `outputs/acceptance/deviation_{long,z,flags}.csv`,
`deviation_why.md` and `figures/deviation_heatmap.png` answer that for all 34 fields and every v1 site KPI
(+ the two headline swelling columns): robust z vs Batch 3 (leave-one-out for Batch 3 fields) **and** the
deviation in units of the field's own sampling SD (SD of a 16-tile site mean, `outputs/reliability/icc.csv`).
A column is `off` only if |z| > 2 *and* the deviation exceeds 2 sampling SDs; `within_sampling_noise` if the
z is large but the field size cannot resolve it; `off_noise_unknown` where no tile-level noise estimate exists
(K05 z-scores, K06, K10–K13 — whole-field descriptors).

| | n `off` columns, median (max) |
|---|---|
| Batch 3 fields, leave-one-out | 1 (3) |
| Batch 1 fields | 1 (10) — the two high-Si fields 9–10, the other five 0–2 |
| Batch 2 fields | 1 (2) |
| held-out 3e122cbj / fn0mhxef / xrv9xvzb | 8 / 0 / 0 |

Reading:

- **Apart from the two high-Si Batch 1 fields, a Batch 1 or Batch 2 field deviates from the Batch 3 baseline
  on as many KPIs as a Batch 3 field does from the rest of Batch 3** (median 1 column, usually a size or
  density column at |z| 2–3). At KPI level there is no per-field "how it differs" for most off-baseline
  fields — which is the same statement as the AUC ≈ 0.5 of §4, made per field. Explanations for these fields
  have to come from the arrangement features (fingerprint) or from the classifier's attributions, and should
  be presented as such rather than as a KPI deviation.
- 3e122cbj is off on 8 columns (K01 +10, K02 +9, K03 max, K09, K15 −5; i.e. Si-rich, dense, closer-packed,
  less graphite contact) — the composition story of §2.5 in `docs/functional.md`, now with the noise floor.
  fn0mhxef and xrv9xvzb are inside Batch 3 on every resolvable column.
- Of the ten |z| > 2 readings on K07 (Clark–Evans), seven are within sampling noise — the ICC caution in
  `docs/reliability.md` applied: K07 should not appear in any "why" explanation.
- K08 (pair-correlation peak) and K14 p50 have MAD = 0 within Batch 3 (mostly zero / identical values) so no
  z is defined; they carry no per-field information at this field size.

