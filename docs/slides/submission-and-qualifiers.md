# Submission (due 3 PM) + qualifiers run-of-show (3 min pitch / 2 min Q&A)

Format per organisers (2026-10-04): submission = repo link + short description +
≤2-min video (YouTube/Loom). Qualifiers = 3 min pitch + 2 min questions, live
demo strongly encouraged, do not just show slides.

## 1. Short description (paste into the Projects tab)

> **PMDB — batch fingerprinting that can't be fooled by the microscope.**
> Three electrode batches, identical by every composition KPI — we show the
> difference is *where* the silicon sits, not how much. An intensity model
> scores higher than ours in cross-validation, then flips its prediction when
> we turn a brightness knob: it learned the imaging session, not the material.
> Our model uses segmentation geometry only — 16 curve-shape features, robust
> per-batch statistics (31 sites forbid fitted weights), and conformal
> p-values that say "I don't know" when that is the true answer. Validated by
> a 10,000-label-permutation test (p = 0.0025), 1,496 jackknife refits, and a
> pre-registered feature evaluation that failed its own rule and was rejected
> — one evaluation, no second attempt. The same arithmetic assigns the three
> held-out sites (Batch 1 / Batch 2 / Batch 3) and rejects a drifting
> shipment outright as out-of-distribution: the accept/reject decision a
> battery manufacturer actually needs.

## 2. Video script (≤2 min; aim 1:50. Screen recording + voiceover)

Record: terminal with font cranked up + the two figures. No talking head needed.

- **0:00–0:20 — hook (terminal, `demo_confound.py` running).**
  "Here's a model that tells battery electrode batches apart with 71%
  accuracy. And here's the same model after we add 7 grey levels to the image
  — a microscope setting, the material untouched. Its prediction flips. It
  learned the camera, not the electrode. Most models trained on this dataset
  have this exact flaw. Ours doesn't — and that's the project."
- **0:20–0:45 — the finding (`depth_profiles.png`).**
  "Every composition statistic — silicon fraction, particle size, porosity —
  is identical across the three batches. What differs is arrangement: the
  baseline batch is uniform through the coating, Batch 2's silicon floated to
  the top during drying, Batch 1's sank. Real manufacturing physics, visible
  only in the full depth profiles."
- **0:45–1:15 — the model + the proof.**
  "With 31 labelled images you can't fit weights, so we don't: 16 curve-shape
  features, per-batch robust statistics, conformal confidence. Stress-tested
  instead of leaderboard-tuned: ten thousand label shuffles, p = 0.0025;
  1,500 jackknife refits barely move the answers. And when we pre-registered
  a richer feature set, it failed its own acceptance rule — so we kept the
  honest model."
- **1:15–1:40 — the calls (fingerprint cards).**
  "The three held-out sites: one to each batch. The spicy one is xrv9xvzb —
  it carries the baseline batch's camera signature, so every intensity model
  will call it Batch 3. Its material says Batch 2, in 98% of refits. We bet
  the material over the microscope, and documented the alternative before
  scoring."
- **1:40–1:55 — the reject demo (`demo_reject.py` output).**
  "And when a shipment drifts beyond every known batch, the same arithmetic
  rejects it outright — no retraining, no tuned thresholds. Identical
  composition, different arrangement, calibrated doubt. That's PMDB."

## 3. Qualifiers pitch (3:00, demo-led)

Pre-flight: run both demos once on the venue machine; keep screenshot
fallbacks open in a background tab. Figures ready: `depth_profiles.png`,
three `card_<site>.png`.

- **0:00–0:35 — The trap, live.** Run `demo_confound.py`.
  "Everyone's first model on this dataset is an intensity model. Watch: 71%
  accuracy, better than ours. Now the same real site, +7 grey levels — a
  microscope knob." *(point at flip)* "Batch 1 becomes Batch 3. Nothing about
  the electrode changed. It learned the imaging session. Keep that in mind
  when you see big accuracy numbers today — ours is geometry-only; this knob
  cannot move a single feature."
- **0:35–1:00 — The finding.** Show `depth_profiles.png`.
  "All 51 composition KPIs are batch-blind — same material. What differs is
  where the silicon sits: baseline uniform, Batch 2 top-heavy — drying
  migration — Batch 1 bottom-heavy — sedimentation. Physics a manufacturer
  can act on. And the dataset was engineered to contain trends: we believe
  this is the one that was planted."
- **1:00–1:30 — The proof.**
  "Thirty-one sites forbid fitted weights, so: 16 curve-shape features,
  per-batch robust statistics, conformal confidence. The claims are
  stress-tested, not tuned: permutation p = 0.0025 over ten thousand
  shuffles; the calls survive 1,496 jackknife refits and 45 hyperparameter
  settings. Yesterday we pre-registered a richer feature set — it failed its
  own rule, so we kept the honest model. One evaluation, no second attempt —
  it's all in the git history."
- **1:30–2:00 — The calls.** Show the three cards.
  "fn0mhxef: Batch 3, the baseline. xrv9xvzb: Batch 2 — it carries a
  Batch-3-style black level, so every intensity model in this room will call
  it Batch 3; we bet the material over the microscope, documented before
  scoring. 3e122cbj: Batch 1 — stable in 90% of refits, confidence zero,
  because it's typical of every batch. The model refuses to manufacture
  certainty. Which brings us to the thing a manufacturer actually needs —"
- **2:00–2:45 — The reject, live.** Run `demo_reject.py`.
  "A shipment drifts from the baseline toward heavy sedimentation. First it's
  accepted; then re-assigned to Batch 1 — the sedimentation-prone batch, the
  model telling you *what kind* of wrong it is — and past that, every batch
  rejects it: out of distribution, reject the shipment. Same arithmetic as
  the three calls. No retraining, no thresholds tuned."
- **2:45–3:00 — Close.**
  "Identical composition, different arrangement, calibrated doubt — and a
  model that can't be fooled by a brightness knob. Thank you."

## 4. Q&A pocket answers (2 min)

- **"Why only 0.677?"** The honest number beats the confounded one — slide 1
  showed what 0.71+ buys you. The claim is the permutation test (p = 0.0025),
  not the point estimate.
- **"What if xrv9xvzb is really Batch 3?"** Two hypotheses, documented before
  scoring (docs/presentation-strategy.md): ours is a Batch 2 electrode imaged
  with Batch-3 settings — the black level already fails to track batch inside
  the training set (10/17 sites). If we're wrong, it's the error a deployable
  QC model should make: material over session artefact.
- **"Several teams say Batch 3 for that site."** Models sharing the same
  confound make correlated errors — agreement among them is not evidence.
- **"Is it sensitive to noise?"** Per-site noise, blur, and grey quantisation
  were measured for the clean pipeline; no fingerprint feature correlates
  with any of them within batches (all |r| ≤ 0.33, n.s.). Brightness/gain is
  excluded by construction.
- **"Why is 3e122cbj confidence zero?"** Batch 1's trend is variability
  itself — a batch defined by inconsistency can't pin a single site by
  typicality. Confidence zero is the calibrated-correct output, not a failure.
- **"Why no deep model?"** n = 31. Nothing with capacity can be validated
  here, and the confound demo shows exactly what capacity buys on this data.

## 5. Before 3 PM checklist

1. Join https://iterate.inc/invite/3NZSl9CHz, join/create team, Projects tab.
2. Record the 2-min video (script above), upload to YouTube/Loom, test link
   in private window.
3. Submit: repo link + description (section 1) + video link.
4. Rehearse the 3-min pitch once against a stopwatch; capture demo-output
   screenshots as fallback.
