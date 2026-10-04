# Run of show — 90 s slides + 90 s live demo

Deck: open `docs/slides/pitch.html` in a browser, F11 for fullscreen.
Advance: `→` / click right half. Rehearse once against a stopwatch; the slide
timings below total 85 s to leave 5 s of slack.

## Slides (90 s)

**Slide 1 — The trap (25 s).**
"Everyone's first model on this dataset is a pixel or intensity model. It scores
0.71 — better than ours. Then you add 7 grey levels — a microscope knob, the
material untouched — and its prediction flips batches. It learned the imaging
session, not the electrode. Keep that in mind when you see big accuracy numbers
today. Ours is geometry-only: that offset cannot move a single feature."

**Slide 2 — The finding (20 s).**
"All 51 composition KPIs are batch-blind — these batches are the same material.
What differs is where the silicon sits: the baseline is uniform, Batch 2 is
top-heavy with a depleted mid-depth — drying migration — Batch 1 is bottom-heavy —
sedimentation. That's physics a manufacturer can act on."

**Slide 3 — The proof (25 s).**
"Thirty-one sites forbid fitted weights, so the model is 16 curve-shape features
and per-batch robust statistics, and the claims are stress-tested instead:
permutation p = 0.0025 over ten thousand shuffles; the calls survive 1,496
jackknife refits and 45 hyperparameter settings. And when we pre-registered a
richer feature set yesterday, it failed its own rule — so we kept the honest
model. One evaluation, no second attempt."

**Slide 4 — The calls (20 s).**
"fn0mhxef: Batch 3. xrv9xvzb: Batch 2 — it carries a Batch-3-style black level
and we bet the material over the microscope, documented before scoring.
3e122cbj: Batch 1 — a stable decision with confidence zero, because it's typical
of every batch; the model refuses to manufacture certainty. Which brings us to
the thing a manufacturer actually needs —"

## Live demo (90 s) — terminal, font cranked up

Pre-flight (before going on stage): run both commands once so imports are warm
and you know they work on the venue machine.

**Part 1 — the confound, live (±35 s):**

    .venv/Scripts/python scripts/demo_confound.py

Say while it runs (~3 s): "This is the intensity model from slide 1, live. LOO
0.71. Now the same real site with a +7 grey offset…" — point at the flip line:
"Batch 1 becomes Batch 3. Nothing about the electrode changed."

**Part 2 — reject the unknown shipment (±45 s):**

    .venv/Scripts/python scripts/demo_reject.py

Say: "Now a shipment drifts from the baseline toward heavy sedimentation. Watch
the conformal p-values: first it's accepted as the baseline, then it's
re-assigned to Batch 1 — the sedimentation-prone batch, the model telling you
*what kind* of wrong it is — and past that, every batch rejects it: out of
distribution, reject the shipment. Same arithmetic as the three calls you just
saw. No retraining, no thresholds tuned."

Close (±10 s): "Identical composition, different arrangement, calibrated
doubt — and a model that can't be fooled by a brightness knob. Thank you."

## Fallbacks

- Both demos run in ~3 s on the dev machine; if the venue machine is slow or
  python is broken, screenshots of both outputs are the fallback — capture them
  during rehearsal (`demo_confound.txt` / `demo_reject.txt` via `> file` work too)
  and keep them open in a background tab.
- If only one demo fits (time pressure), drop Part 1 — slide 1 already told the
  confound story; the reject demo is the one nobody else will have.
- Likely questions:
  - "Why only 0.677?" → slide 1; the honest number beats the confounded one.
    The permutation test (p = 0.0025), not the point estimate, is the claim.
  - "What if xrv9xvzb is really Batch 3?" → two-hypothesis slide in
    `docs/presentation-strategy.md`: we chose the material signal over a session
    artifact, stated falsifiers before scoring; if wrong, it's the error a
    deployable QC model should make.
  - "Why no deep model?" → n = 31; nothing with capacity can be validated here,
    and the confound demo shows what capacity buys you on this data.
