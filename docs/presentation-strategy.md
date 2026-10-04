# Presentation strategy (2026-10-04)

How to present the fingerprint work against teams likely to show
higher-but-confounded accuracy numbers. Order matters: the confound reveal goes
**before** our accuracy slide, so every large number shown after us is already
suspect.

## Deck order

1. **The trap (confound reveal).** Live or pre-rendered `scripts/demo_confound.py`:
   an intensity-only model reaches 0.71 LOO — *better than ours* — then flips its
   assignments under a +7 grey-level offset, i.e. a microscope setting, not a
   material change. Message: on this dataset, the obvious high-capacity approach
   learns the acquisition session, not the electrode. (The elevated BSE black
   level appears in only 10/17 Batch 3 sites: an imaging setting correlated with
   batch, not a property of the material.)
2. **The finding.** No scalar KPI separates the batches (best of 51: KW p = 0.065;
   classifiers below the 55% majority baseline). Composition is identical — the
   *arrangement* is not. Money shot: `outputs/fingerprint/figures/depth_profiles.png`
   (Batch 3 flat, Batch 2 top-heavy with depleted mid-depth, Batch 1 bottom-heavy
   and variable).
3. **The model.** 16 curve-shape features, robust per-batch naive Bayes (no fitted
   weights — 31 sites forbid them), Mondrian conformal p-values, OOD = reject-all.
4. **The validation.** LOO 0.677 vs 0.548 baseline; **10,000-permutation p = 0.0025**
   (null mean 0.373; the null's 95th percentile is exactly the majority baseline).
   Stability: 1,496 jackknife refits and 45 hyperparameter configs barely move the
   held-out assignments. And the discipline exhibit: we pre-registered a single
   extension of the feature set (`docs/eval-plan-oct4.md`, committed before the
   run) — it failed its own rule (LOO 0.613, p 0.019) and was rejected, no second
   attempt. State the honest caveats out loud (feature families were explored on
   the labelled data; conformal grid is coarse at n_b = 7) — before anyone asks.
5. **The calls** (cards from `outputs/fingerprint/figures/`):
   - `fn0mhxef` → Batch 3 (mid-band Si in a region only Batch 3 occupies).
   - `xrv9xvzb` → Batch 2 (see the defense below).
   - `3e122cbj` → Batch 1: *stable decision, ambiguous typicality* — Batch 1 in
     87–94% of jackknife refits with Batch 3 the only alternative, yet conformal
     confidence 0.00 because the site is typical of every batch. That is the
     model refusing to manufacture certainty — the accept/reject behaviour a
     manufacturer actually needs.
6. **The reject demo.** `scripts/demo_reject.py`: a drifting shipment is first
   re-assigned to the sedimentation-prone batch, then rejected outright as
   out-of-distribution.

## The xrv9xvzb defense (one slide, hold until asked or present proactively)

`xrv9xvzb` is the one held-out site with the Batch-3-style elevated BSE black
level (p1 = 6). Its *material* fingerprint says Batch 2 (mid-depth dip −0.72,
inside the Batch 2 cluster, below every Batch 3 site; pair-correlation bins
agree; Batch 2 in 97.7–100% of jackknife refits and 45/45 configs).

Two hypotheses, stated honestly:

- **H1 (ours): a Batch 2 electrode imaged in a Batch-3-settings session.** The
  black level is an acquisition knob; it already fails to track batch inside the
  training set (present in only 10/17 Batch 3 sites). The task is to fingerprint
  the material, so we bet on the material.
- **H2: the black level is a genuine batch marker** (e.g. all Batch 3 shipments
  were imaged in one campaign). Then intensity-based models get this site right
  for a reason that will not survive the next microscope recalibration.

What would change our mind: acquisition metadata tying black level to supplier
rather than session, or a labelled Batch 2 site with p1 > 0. Neither exists in
the released data.

If we are wrong on this site, the error is the *right kind* of error: we chose
the material signal over a session artifact, documented the alternative before
scoring, and the same choice is what makes the model deployable. If we are
right, every pixel-trained model in the room got it wrong.

## Posture

- Never lead with 0.677; lead with *why the bigger numbers in the room are wrong*.
- Quote the permutation test, not the point estimate, as the claim.
- Volunteer our caveats before questions do; nothing a judge finds should be
  something we did not already say.
