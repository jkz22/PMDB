# Run of show: 3 min qualifier (≈ 20 s framing + 2:40 live demo) + 2 min Q&A

The organisers want a live demo, not slides, and not the video. Open the dashboard
fullscreen and stay on it. The deck (`docs/slides/pitch.html`) is for Q&A or a
dashboard failure only.

Pre-flight: `node demo/server.mjs` (Node ≥ 18, no `npm install`) → http://localhost:8080,
F11, press `1`, check the top-right dot is green ("live · results v…"). Keys `1`–`6`
switch views and `←`/`→` move the slider on the current view. Keep
`demo/backup/pmdb-dashboard-demo.mp4` open in a second tab (to present from another device:
`node demo/server.mjs --host 0.0.0.0`).

## 0:00–0:20 Framing (dashboard on view 1, slider at 0)
"{{batch_count_word}} batches of a silicon/graphite anode that are chemically identical. A manufacturer
needs to know which batch a sample came from, and whether a new shipment is out of spec.
We built a model that answers from the electrode's structure, not from the microscope."

## 0:20–0:55 The trap (view `1`)
"Everyone's first model reads pixel intensity. It scores {{intensity_acc}}, which is better than ours."
Tap `→` once. "One grey level brighter, a microscope setting with the material untouched, and
it calls Batch 1 *Batch 3*. So does every Batch 1 and 2 site: {{flip}}." Go on to +7.
"It learned the imaging session. Ours is geometry-only, so this offset cannot move a single
feature."

## 0:55–1:25 Amount vs arrangement (view `2`)
"All 51 composition KPIs are batch-blind. We ran {{fem_runs}} finite-element charging simulations on
Modal for {{fem_cost}}. Swelling follows how much silicon there is, R² {{fem_r2}}, and the batches hold the
same amount. The difference is *where* it sits." Point at the curves: "the baseline is uniform,
Batch 2 is top-heavy (drying migration), Batch 1 is bottom-heavy (sedimentation)."

## 1:25–1:50 Proof (view `3`)
"{{fp_features}} curve-shape features, no fitted weights: {{fp_score_words}} leave-one-out, p = {{perm_p}} over {{perm_n_words}} label shuffles. Every challenger was pre-registered: FEM features, and XGBoost on
24 KPIs, which got {{xgb_kpi_only_words}}. All of them failed their own rule, so we kept the honest model."

## 1:50–2:15 Held-out calls (view `4`)
"These are recomputed live in the browser and are identical to the Python output. xrv9xvzb is
{{call_xrv9xvzb_batch}}, even though it carries a Batch-3-style black level: we bet on the material, not the
microscope. 3e122cbj is {{call_3e122cbj_batch}} with confidence zero, because it looks typical of every batch
and the model refuses to manufacture certainty."

## 2:15–2:50 Reject a shipment (view `5`)
Drag the slider slowly. "A shipment drifts toward sedimentation. First it's accepted, then it's
flagged as Batch 1-like, which tells you *what kind* of wrong it is, and past t ≈ 1.15 it is rejected
as out of distribution. Same model, no retraining, no tuned thresholds."

## 2:50–3:00 Close
"Identical composition, different arrangement, calibrated doubt, and a model a brightness knob
can't fool. Thank you." (View `6`, the site explorer, is there for questions.)

## Fallbacks
1. Dashboard fails: play `demo/backup/pmdb-dashboard-demo.mp4` (69 s, silent) and narrate this script over it.
2. Browser dead: `python scripts/demo_confound.py` then `python scripts/demo_reject.py` in a terminal.
3. Pre-rendered clips: `outputs/clips/01_confound_flip.mp4` … `04_verdict_reject.mp4`.

Q&A prep: `docs/slides/qa.md`.
