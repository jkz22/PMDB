# Batch fingerprint: who made this electrode, and should you accept it?

`pmdb/fingerprint.py` assigns every unlabelled site to a batch with a
calibrated confidence, and flags sites that are like *no* known batch — the
accept/reject decision a manufacturer faces when a new shipment arrives.

Run everything with:

```bash
python scripts/run_fingerprint.py --heldout-dir outputs/heldout/kpis
python scripts/plot_fingerprint.py      # demo figures
python scripts/demo_reject.py           # live "reject a drifting batch" demo
```

Running `run_fingerprint.py` without `--heldout-dir` deletes any held-out outputs from an earlier run in the output directory.

## 1. The finding: composition is identical, arrangement is not

None of the 51 scalar site KPIs separates the batches: the best reaches
Kruskal–Wallis p = 0.065 (uncorrected, across 51 tests), and leave-one-site-out
classification on the screened KPI set stays **below** the 55% majority-class
baseline. By every per-site summary statistic — Si fraction, particle sizes,
densities, porosity — the three batches are the same material.

The differences live in *where* the Si sits, which only the full curves retain:

| Signal | Where | What it says |
|---|---|---|
| Si depth profile (`band_si_frac`) | mid-depth band, KW p ≈ 0.001 | **Batch 3 (baseline): uniform through the coating. Batch 2: top-heavy with a depleted mid-depth** (consistent with Si migration during drying). **Batch 1: bottom-heavy** (consistent with sedimentation) **and far more variable site-to-site.** |
| Pair correlation `g_obs_x/z` at 2–7 µm | medium-range order, KW p ≈ 0.01–0.04 | the lag where Si stops looking "clustered" differs by batch. |
| Tile spread of K15 (Si–graphite contact) | within-site heterogeneity, KW p ≈ 0.01 | batches differ in how *uniform* each electrode is, not just in its average. |

The fingerprint is therefore 16 curve-shape features per site: the five
relative depth-band fractions (profile shape, normalised by the site mean so
composition cancels), depth slope and mid-depth dip, the pair-correlation
level in four lag bins per axis, and the K15 tile spread.

A deliberately *excluded* signal: Batch 3's elevated BSE black level
(`outputs/raw_intensity_stats.csv`). It appears in only 10 of 17 Batch 3 sites
and is an acquisition setting, not a material property; a pixel-trained
classifier would happily learn it and be wrong about the material. The
fingerprint uses segmentation-derived geometry only.

## 2. The model: no fitted weights, calibrated confidence

31 labelled sites (7/7/17) forbid anything with real capacity, so:

- **Classifier:** robust per-batch naive Bayes. Each batch contributes a
  median and a robust scale per feature (shrunk 50% toward the pooled scale).
  Per-batch scales matter because the batches differ in *dispersion* as much
  as location — Batch 1 is the loose one; a nearest-centroid rule is blind to
  that and measurably worse.
- **Confidence:** class-conditional (Mondrian) *full* conformal p-values. For each batch the site is provisionally added to it, and every one of the n_b+1 points is scored leave-one-out against that batch refitted without it (the pooled standardization is re-estimated with the site included). The p-value is the site's rank among those scores. All points are scored by the same rule, so the p-value is finite-sample valid, quantised to multiples of 1/(n_b+1), which is coarse at n_b = 7, and honestly so. *Credibility* = p of the assigned batch; *confidence* = 1 − highest p among the other batches. That is the textbook 1 − second-highest p only when the assigned batch also has the highest p; otherwise it is lower, on purpose, because a rival that stays typical is not excluded.
- **Out-of-distribution:** a batch is rejected when its p-value is below
  α = 0.1 **or at its achievable floor** 1/(n_b+1), i.e. the site is more
  nonconforming than every labelled site of that batch. All batches rejected
  ⇒ the site is like no known batch ⇒ reject the shipment.

## 3. Validation (leave-one-site-out, n = 31)

Numbers from `outputs/fingerprint/evaluation.json` (seed 0, 500 label
permutations):

- **LOO accuracy 0.677** vs 0.548 majority-class baseline.
- **Label-permutation test: p = 0.002** (null mean 0.369) — the whole
  pipeline, not a lucky feature.
- Per-batch recall: Batch 1 5/7, Batch 2 3/7, Batch 3 13/17.

| true \ assigned | Batch 1 | Batch 2 | Batch 3 |
|---|---|---|---|
| Batch 1 | **5** | 0 | 2 |
| Batch 2 | 3 | **3** | 1 |
| Batch 3 | 2 | 2 | **13** |

Assignment is by best likelihood score, not by highest conformal p-value: the
p-values are per-batch *typicality* (each calibrated against that batch's own
spread) and deliberately do not compare across batches — a loose batch is
"typical" of almost anything, and assigning by max-p drops LOO accuracy to
0.355. The two numbers answer different questions and both appear on the
cards.

Honest caveats:

- Feature families were chosen by exploring the labelled data; the permutation
  test validates the final pipeline, but the exploration itself is a selection
  step a larger dataset would let us nest.
- With 7-site batches the conformal p-grid is coarse (steps of 1/8); the
  credibility numbers are honest but low-resolution.
- n = 31. The LOO accuracy has a wide confidence interval; the permutation
  test, not the point estimate, is the claim.

## 4. Held-out assignments

`outputs/fingerprint/heldout_predictions.csv`, with per-feature evidence in
`heldout_explain.csv` and one fingerprint card per site in
`outputs/fingerprint/figures/`.

| site | assigned | credibility | confidence | p(B1) / p(B2) / p(B3) | reading |
|---|---|---|---|---|---|
| `3e122cbj` | **Batch 1** | 0.88 | 0.00 | 0.88 / 0.88 / 1.00 | the honest hard case: typical of *every* batch (every p ≥ 0.88, Batch 3 highest); Batch 1 wins on likelihood only. Confidence 0 is the model saying "this one could be anyone" — not a bug, a correctly calibrated shrug. |
| `fn0mhxef` | **Batch 3** | 0.44 | 0.12 | 0.12 / 0.88 / 0.44 | mid-band Si at 1.18 — a region only Batch 3 occupies; every Batch 2 site is mid-depth depleted (< 0.7). Batch 1 effectively excluded (p = 0.12). Confidence is low because Batch 2 stays typical overall (p = 0.88) even though it loses on likelihood. |
| `xrv9xvzb` | **Batch 2** | 1.00 | 0.50 | 0.50 / 1.00 / 0.39 | mid-depth dip −0.72, inside the Batch 2 cluster and below every Batch 3 site; several pair-correlation bins agree. |

Side-channel worth stating out loud: `xrv9xvzb` is the one held-out site with
the elevated BSE black level (p1 = 6) that otherwise appears only in Batch 3
acquisitions. Its *material* fingerprint still says Batch 2. Either it is a
Batch 2 electrode imaged in a Batch-3-settings session, or the black level is
not the batch marker it looks like — exactly why the model refuses to use
intensity (see `scripts/demo_confound.py`).

## 5. Demo assets

- `outputs/fingerprint/figures/depth_profiles.png` — the money shot: per-batch
  Si depth profiles plus the held-out sites against the batch medians.
- `outputs/fingerprint/figures/card_<site>.png` — per held-out site: the most
  decisive features as strip plots, the held-out value as a black diamond, and
  the conformal verdict.
- `scripts/demo_reject.py` — a shipment drifts from the Batch 3 baseline
  toward heavy sedimentation; the model first re-assigns it to Batch 1 (the
  sedimentation-prone batch), then rejects it outright as out-of-distribution.
