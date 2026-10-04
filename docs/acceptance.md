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

Caveat worth a follow-up: 3e122cbj's *fingerprint* joint score is lower than any Batch 3 field (0.57)
although the fingerprint classifier assigns it to Batch 1. A field can be central in the robust-z sense
on every depth/pair-correlation column individually while its *combination* is Batch 1-like; the RMS-z
score is per-column and does not see covariance. This is a limitation of the pre-specified score, not
evidence against the Batch 1 call, but it should be checked against the fingerprint's conformal p for
that site before anything is quoted.
