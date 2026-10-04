# Run of show — 90 s slides + 90 s live demo

Deck: open `docs/slides/pitch.html` in a browser, F11 for fullscreen.
Advance: `→` / click right half. Rehearse once against a stopwatch; the slide
timings below total 85 s to leave 5 s of slack.

Dashboard (live demo): `node demo/server.mjs` (Node ≥ 18, no `npm install`
needed) → http://localhost:8080. For another device, add `--host 0.0.0.0`
and browse to the host’s LAN address. Keys `1`–`6` switch views, `←`/`→` move
the slider on the current view. It re-reads `outputs/` every 2 s, so a fresh
pipeline run shows up without a reload. Backup video:
`demo/backup/pmdb-dashboard-demo.mp4` (regenerate: `python demo/record_backup.py`).

## Slides (90 s)

**Slide 1 — The trap (20 s).**
"Everyone's first model on this dataset is a pixel or intensity model. It scores
0.71 — better than ours. Then you add 7 grey levels — a microscope knob, the
material untouched — and its prediction flips batches. It learned the imaging
session, not the electrode. Keep that in mind when you see big accuracy numbers
today. Ours is geometry-only: that offset cannot move a single feature."

**Slide 2 — The finding (15 s).**
"All 51 composition KPIs are batch-blind — these batches are the same material.
What differs is where the silicon sits: the baseline is uniform, Batch 2 is
top-heavy with a depleted mid-depth — drying migration — Batch 1 is bottom-heavy —
sedimentation. That's physics a manufacturer can act on."

**Slide 3 — The physics (15 s).**
"We then simulated charging inside every real microstructure — 68 finite-element
runs on Modal for $2.61. Swelling tracks how much silicon there is, R² 0.97, and
the batches hold the same amount, p 0.38. So swelling can't separate batches —
adding FEM features failed our pre-registered rule. The signal is the arrangement."

**Slide 4 — The proof (20 s).**
"Thirty-one sites forbid fitted weights, so the model is 16 curve-shape features
and per-batch robust statistics, and the claims are stress-tested instead:
permutation p = 0.0025 over ten thousand shuffles; the calls survive 1,496
jackknife refits and 45 hyperparameter settings. And when we pre-registered a
richer feature set yesterday, it failed its own rule — so we kept the honest
model. One evaluation, no second attempt."

**Slide 5 — The calls (15 s).**
"fn0mhxef: Batch 3. xrv9xvzb: Batch 2 — it carries a Batch-3-style black level
and we bet the material over the microscope, documented before scoring.
3e122cbj: Batch 1 — a stable decision with confidence zero, because it's typical
of every batch; the model refuses to manufacture certainty. Which brings us to
the thing a manufacturer actually needs —"

## Live demo (90 s) — dashboard, browser fullscreen

Pre-flight: start the server, open http://localhost:8080, press `1`, check the
top-right dot is green ("live · results v1"). Keep the backup video open in a
second tab.

**Part 1 — the confound, live (±35 s), view `1`:**
Tap `→` once (+1), then on to +7. Say: "This is the intensity model from
slide 1, running live. Same real site, one grey level brighter — so no pixel
clips to black any more…" — point at the red card: "Batch 1 becomes Batch 3, and
so does every Batch 1 and 2 site, 14 of 14. By +7, roughly the Batch 3 average
black level, it's still Batch 3. Our model's call doesn't move; it has no
grey-level input."

**Part 2 — reject the unknown shipment (±45 s), view `5`:**
Drag the drift slider slowly. Say: "Now a shipment drifts from the baseline
toward heavy sedimentation. First it's accepted as the baseline, then it's
flagged as Batch 1 — the sedimentation-prone batch, the model telling you *what
kind* of wrong it is — and past t ≈ 1.15 every batch rejects it: out of
distribution, reject the shipment. This is the same model that made the three
calls, running in the browser, with parity tests against the Python code. No retraining, no thresholds tuned."

Close (±10 s): "Identical composition, different arrangement, calibrated
doubt — and a model that can't be fooled by a brightness knob. Thank you."

If asked: views `2` (amount vs arrangement), `3` (null distribution, pre-registered
tests), `4` (held-out calls, recomputed live) and `6` (any site's overlay + FEM GIF).

## Fallbacks

- Dashboard fails (no Node, port taken, browser issue) → play
  `demo/backup/pmdb-dashboard-demo.mp4` and narrate the same script over it.
- Second fallback: the terminal demos (`python scripts/demo_confound.py`,
  `python scripts/demo_reject.py`, ~3 s each) or the pitch clips in
  `outputs/clips/01_confound_flip.mp4` … `04_verdict_reject.mp4`.
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
