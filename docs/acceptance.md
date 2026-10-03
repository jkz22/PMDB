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
