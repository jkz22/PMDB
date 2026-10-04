# pitch.writer report

## Draft
- `docs/pitch-brief.md` (new; no other file edited). Contents: a 3-line headline, 12 slides (problem, data, approach, physics, engineering, model story / D20, swelling validation, ablation, held-out predictions, limitations, next steps, close), and a numbers checklist (number, value, source path). Each slide has a title, a one-sentence key message, 2-4 bullets with numbers and source paths, a figure path and a speaker note.

## Rules applied
- No project CLAUDE.md exists. The repo's AGENTS.md has no prose rules. I used the rules from the dispatch prompt instead: plain, direct, technical English; short sentences; no hype; numbers copied exactly as the source shows them; source path next to each number. Normally a missing CLAUDE.md means I stop and escalate. I went ahead because the prompt supplied the rules.
- AGENTS.md task framing: every held-out site gets a batch, a confidence and an explanation against Batch_3. Slide 9 covers all three sites.

## Grounding
- Every number in the brief has a source path written next to it, and the numbers checklist repeats them. Sources: docs/fem/{literature-review,method,results}.md, docs/classifier/{method,results}.md, outputs/classifier/{metrics,heldout_predictions}.csv, outputs/fem/{run_log.json,validation.csv}, outputs/pooling_checks/check3_results.md, AGENTS.md, data_heldout/README.md, docs/kpis/README.md (the provider's "must not cluster and must not be localised" criterion), .claude/plans/fem-swelling.md.
- Derived by counting or comparing, not copied:
  - batch sizes 7/7/17 are row counts of validation.csv;
  - "lowest swelling of all 34 sites" for xrv9xvzb (0.117689) and "highest of the held-out sites" for 3e122cbj come from comparing validation.csv rows;
  - "28 of 31 end-to-end confidences in [0.5,0.7)" comes from the calibration n values;
  - "under $3" compares against the 2.608 ledger.
- Not from any file:
  - the chance level of 0.5 for two-class balanced accuracy. It is definitional and labelled that way in the checklist.
  - "3D / SliceGAN" as a next step. It comes only from the dispatch prompt and is marked "proposed; not yet in the repo docs".
- Could not source:
  - a total campaign wall-clock runtime. No file states one, so the deck uses per-case wall-time medians and maxima (results.md) instead.
  - an explanation of which FEM features drove the FEM arm's xrv9xvzb call. heldout_zscores.csv holds KPI-arm rows only.
  - whether 3e122cbj falls inside Batch_1's range. Only z-scores against Batch_3 exist.
- GIF choice: the highest swelling_sym per batch in validation.csv (B1 5n1q8atc, B2 b3esycq1, B3 x7u69zsw), plus held-out 3e122cbj. I did not view the GIFs. I viewed the three FEM PNGs. I did not open fig_confusion.png or fig_importance.png, so the brief describes their content only from results.md.
- No gap markers left.

## Contradictions between sources
1. docs/fem/method.md section 2 and its physics table still describe finite-strain kinematics (F = Fe·Fλ, neo-Hookean, Newton solve). Production is linear small-strain with a log eigenstrain (D20; run_log params.mechanics = "linear"; results.md Limitations). D20 says the switch should be documented in method.md limitations, but method.md section 9 does not mention it. The deck follows D20 and results.md.
2. docs/classifier/method.md section 2 says "10 µm margins (200 px at 0.1 µm)". 200 px at 0.1 µm is 20 µm, and run_log features.edge_um = 20.0 and D9 both say 20 µm. The deck does not quote a margin.
3. Rounding differences between the two SOC tables, method.md (generated) vs literature-review.md:
   - ε_c at s = 1: 0.094 vs 0.095;
   - J_Si at s = 0.7: 2.663 vs 2.662, and at s = 0.9: 3.048 vs 3.047;
   - ε_c at s = 0.7: 0.060 vs 0.061, and at s = 0.9: 0.083 vs 0.084.

   The deck uses method.md.
4. Precision differences, not real conflicts:
   - per-case cost: method.md median 0.02, max 0.02 vs results.md 0.019 / 0.023;
   - ledger: 2.61 (method.md) / 2.608 (results.md) / 2.6081 (run_log).

   The deck uses results.md and also shows the run_log figure.
5. Stale or superseded, noted only:
   - the plan's status line still reads "scoping";
   - the D15 budget estimate was ~$40-70, against $2.608 actually spent after the D17/D20 changes;
   - lit review verdict 12 says "finite strain REQUIRED", which D20 explicitly supersedes.

## Brief
- No brief file named by the project.

## Judgment calls
- I framed "FEM helped stage 1" as "scored highest on stage 1 (0.626 vs KPI 0.548), within the stated ~0.1 noise level". I did not present it as a gain.
- I described stage 2 (0.286-0.357) as below the 0.5 of a coin flip, rather than "at chance".
- On Slide 9, 3e122cbj carries the warning that z = +8 is far from Batch_3 and that the classifier has no "new supplier" output (classifier method section 7).
- I removed two sentences from my own draft because I could not ground them: a causal reading of the knee in swelling_vs_soc.png, and "if Si clusters, swelling concentrates".

## Notes
- The fix for contradictions 1 and 2 belongs in the generated docs (scripts/fem_docs.py for method.md; the classifier doc for the margin). I did not touch them.
