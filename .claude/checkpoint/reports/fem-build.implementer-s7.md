# fem-build Step 7 resume (implementer-s7)

## State at start (verified)
- e517e32 already contains the predictor fallback in solver.py (failed predictor attempt -> retry same step from u_n, `pred_ok`).
- stage3 (old solver): failed_at_s = 0.0265625 both orientations; substeps show reason -4 (DIVERGED_FNORM_NAN) with its=0 for every attempt after s=0.025: NaN residual at the initial guess = predictor overshoot (hypothesis confirmed by the record: its=0, reason -4).
- stage3_r1 (predictor fallback + rung 1 ds_min=1.95e-4): bottom failed_at_s=0.04316, top 0.02695. Failures are now real Newton failures: reason -5 (max_it 25) and -6 (line search) with its 18-25; no more -4. So the NaN-predictor problem is fixed but Newton itself does not converge at s~0.03-0.04.

## Debug log

### Diag 2 (bottom; default / pore_E 1e-3 / basic line search; ledger 3 diag rows later)
- Hypothesis: pore elements are genuinely crushed (J -> 0) within a few % SOC because Si volume growth (Jlam = 1.16 at s = 0.05, 3.24 at s = 1) exceeds what the soft pores can absorb; the barrier ln J then blocks Newton, independent of the predictor.
- Result (default, converged frames): min pore J = 0.59 (s=0.0125), 0.18 (s=0.025); 25 of 8388 pore cells have J < 0.5 at s = 0.025. With pore_E 1e-3: min pore J = 0.0017 at s = 0.05 (19 cells J < 0.1, 103 < 0.5), i.e. pore stiffness barely matters until J ~ 1e-3 (stress ~ lam*lnJ/J = 350 MPa only at J = 1e-3). Max step that converges from a converged state is tiny and shrinks: ds = 0.0125 from s=0 fails (25 its), 0.00625 converges in 4 its.
- Basic line search is worse: fails (reason -6) already at ds = 2e-4 near s = 0.0244.
- Conclusion so far: the problem is genuine pore closure (J_pore -> 0) in a soft-void model, not (only) the predictor. The predictor fallback fixed the NaN-initial-guess failures (reason -4 at its=0), but the remaining failures (-5/-6) come from crushed pores.

### Diag 2 (bottom; default / pore_E 1e-3 / basic line search)
- Hypothesis: pore elements are genuinely crushed (J -> 0) within a few % SOC because Si volume growth (Jlam = 1.16 at s = 0.05, 3.24 at s = 1) exceeds what the soft pores can absorb; the ln J barrier then blocks Newton, independent of the predictor.
- Result (converged frames): min pore J = 0.59 (s=0.0125), 0.18 (s=0.025); 25 of 8388 pore cells have J < 0.5 at s = 0.025. With pore_E 1e-3: min pore J = 0.0017 at s = 0.05 (19 cells J < 0.1, 103 < 0.5), so pore stiffness barely matters until J ~ 1e-3 (stress ~ lam*lnJ/J is only ~350 MPa at J = 1e-3). Largest step that converges shrinks: ds = 0.0125 from s=0 fails (25 its), 0.00625 converges in 4 its.
- Basic line search is worse: fails (reason -6) already at ds = 2e-4 near s = 0.0244.
- Conclusion so far: genuine pore closure (J_pore -> 0) in a soft-void model, not (only) the predictor. The predictor fallback fixed the NaN-initial-guess failures (reason -4 at its=0); the remaining failures (-5/-6) come from crushed pores.

### Diag 3 (bottom, full s 0..1, pore_E 1e-3 = rung 2 on top of rung 1) and Diag 4 (same + snes_max_it 100)
- Hypothesis: rung 2 (stiffer pores) and/or more Newton iterations get the window to s = 1.
- Result diag 3: failed_at_s = 0.0754 (rung 1 alone: 0.0432). Reasons -5/-6 at ds down to 2e-4; pore min J reaches 1.7e-3 already at s = 0.05.
- Result diag 4 (max_it 100): WORSE, failed_at_s = 0.0557 (steps took 40-100 its and still end -5/-6, fnorm 5-134). More iterations do not help; the crawl is not slow convergence but a non-convergent basin.
- Rung 3 evidence: basic line search failed at s = 0.0244 (diag 2), worse than bt.
- Conclusion: the ladder (rungs 1-3) cannot reach s = 1; rung 4 (accept NaN frames after s ~ 0.05-0.08) would leave 1-2 usable frames of 11 and the G4w swelling gate at s = 1 is undefined. Not climbed further: it would only produce a stage-4 failure the plan labels a bug.

## STOPPED: DEVIATION (design decision required) -- see final hand-back
Working tree: only modal_fem.py modified (diag mode: monitor option, pore J stats, timeout 10000). solver.py, fem.yaml, tests unchanged; no unit rerun needed. Nothing committed. Step 7 stage 3 / stage 4 / VT numbers / GIF frames NOT produced (no s=1 solution exists).
Ledger total $0.63 (21 rows; this session's diag rows ~ $0.37).

## Round 4 / P30 implementation (approved)
- solver.py: `compaction_energy(J, Jc, kappa)`, `simulate(..., compaction=(Jc, kappa))`, DG0 pore indicator weights the barrier (J = total det F2). Stress post-processing (sxx/szz/vm) EXCLUDES the barrier term (pore stress is not a feature input; features use J over pore).
- run.py and diag pass the config values; fem.yaml pore keys + remediation entry; test_p30_compaction_barrier + T6 passes compaction.
- Round-4 results: stage3_p30 (Jc 0.3, kappa 500) FAILED earlier (bottom 0.0229, top 0.0199): substeps end reason -5 with small fnorm after 25 its even at tiny ds; hypothesis: barrier Hessian jumps 0 -> kappa/Jc^2 at J=Jc (only C1) so Newton cycles at the kink. Cost 0.298, ledger 0.945. Next: approved fallback (Jc 0.5, kappa 5000) as bottom diag.
- Unit after P30: 47 passed (ledger 20261003T233223Z).
- Fallback (Jc 0.5, kappa 5000) bottom diag: failed_at_s 0.0164 (WORSE; -5 with fnorm 0.003-30 after 25 its at ds down to 2e-4). Consistent with the kink hypothesis (stiffer jump -> harder). Ladder exhausted; STOPPED before rung 4/Stage 4 since rung 4 leaves ~0 usable frames. Working tree: solver.py/run.py/fem.yaml/tests/modal_fem.py modified (P30 as approved, Jc 0.3 kappa 500 in fem.yaml), unit green.

## Round 5 / P31 (linear mechanics)
- Implemented _simulate_linear in solver.py (simulate(..., mechanics='linear')), config `mechanics: linear`, run.py/diag pass it. P30 code kept (finite path only, no interference); finite tests unchanged. Tests: linear T1a, clamped-column closed form; key-set test updated.
- Unit (linear + finite): 49 passed (ledger ~20261004). Fix notes: T1a linear needs ly=1 (plane strain makes ly!=1 stressed); float32 fields so rtol 1e-6.

## Stage 3 / 4 with linear mechanics (tag stage3_lin, Batch_3/vc2whyaq, 40 um, 100 nm, 206800 cells)
- bottom wall 27 s, top 39 s, 11 solves each (2.5-3.5 s per solve), peak RSS 1.83 GB, cost ~$0.003 each; failed_at_s NaN both (no failure).
- G4w: swelling_sym(s=1) = 0.1326 (bottom 0.1338, top 0.1314): gate_ok True, lit_band_ok True.
- VT2 sxx_mean(s=1) = -2888 / -2871 MPa (bottom/top); ratio sxx/(-10) = 288.8 / 287.1 (MPa-scale, reported only).
- VT4 porosity(s=1) 0.0337 / 0.0310; porosity_change -0.0069 / -0.0096; porosity_rel_change -0.170 / -0.236.
- VT5 J_si_mean(s=1) = 1.693 / 1.712 (free Si Jlam = 3.24; log small strain J = 1+exx+ezz is a linear measure, so not directly comparable).
- si_yield_frac = 1.0, vm_si p50 = 18.3-19.6 GPa, p95 = 26.2-26.5 GPa (unphysical stress in Si: small-strain linear with 3.24x volume eigenstrain; a known consequence of D20 -> stress features are saturated/over-large, flag), pore_closed_frac 0.092 / 0.107, first_pore_closure_s = 0.1 (both).
- GIF (frames 0, 5, 10 viewed): frame 0 greyscale BSE, flat top at ~52 um; frame 5 (SOC 50%, swelling 9.3%) and 10 (SOC 100%, 13.4%) show the top surface rising to ~57 and ~59 um, large bright Si grains (stress-saturated, pale yellow at the 1e4 MPa top of the colour range) expanding, black pore regions shrinking/closing; von Mises overlay smooth in graphite (brown, 1-100 MPa); no checkerboarding visible; some fine hatched texture at Si-graphite interfaces (hatching = elevated vm, not inverted cells; J of linear path can be <= 0 in principle but J only reported). 
- Ledger total after this: see ledger.csv (about $1.14).
- Step 8 not started. Nothing committed.
