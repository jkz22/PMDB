## Plan ready (revision round 1)
- **Plan file**: `/Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/plans/fem-build.md`
- **Steps**: 10 steps, est. complexity HIGH
- **Key decisions**:
  - The swelling stop window is [0.03, 0.39]. It applies to the windows, the benchmark, the full si run (G4) and the full siox run (new gate G4x, checked before the sweeps). It does not apply to sweep variants. The literature band [0.09, 0.39] is only reported, as `lit_band_ok`. T1a is named as the user's free-expansion test.
  - Modal-only (D16). The image is `ghcr.io/fenics/dolfinx/dolfinx:v0.10.0`, built with `pip install uv` and `uv pip install --system` with pinned deps: numpy==2.2.6 (the image's own), scipy 1.14.1 and scikit-image 0.25.2 (the same versions as the KPI pipeline). This tag and the dolfinx 0.10 API (`NonlinearProblem(F,u,bcs=,petsc_options_prefix=,petsc_options=)`, `problem.solver.getIterationNumber()`) were checked by the spike. Everything runs as `modal run modal_fem.py --mode probe|unit|window|bench|full|sweeps`. Pure-numpy modules are tested locally and also inside the image.
  - Segmentation keeps the un-harmonised `segment(load_site(..., normalise="none"))` path (P24). On 4 sites (2 strong, 1 mild, 1 clean), hybrid and none segmentation differ on 0.035-0.14% of pixels, and phase fractions agree within 0.001. The masks are therefore identical to the ones behind the KPI features in the D10 joint model. A data test and the probe's image-vs-local label check both guard this.
- **Open questions for dispatcher**: none

### Review comments 1-7
1. **Gate amended.** P25 replaces old P23. G4w, G4, the benchmark and ES7 are restated. G4x is new for siox. G5 stays si-only. Gate numbers are also kept in `fem.yaml`.
2. **Branch.** The plan uses `fem-sim` throughout. "Never merge any PR" replaces the #18 line. Step 10 pushes and runs `gh pr create --base main --head fem-sim` and never merges. It does not open a second PR if one is already open.
3. **Environment.** Superseded by D16. There is no conda env or `environment-fem.yml`, and the image is pinned. The real/double PETSc question is settled by `PETSC_ARCH=linux-gnu-real64-32`.
4. **Cost.** The exponent is 1.5 everywhere. Projections use a 1.25 safety factor plus a $5 dev allowance. Every Modal call, including test iterations, goes into one ledger, and calls that return nothing are charged at their timeout. Retries are 0. Full and sweep runs are split into chunks, with the cap re-checked before each chunk. STOP B and C report the expected and the worst-case cost. A $15 guard stops test iterations from running away before the benchmark.
5. **Coarsening test.** The asserts are hard and a failure means stop-and-report. Measured margins: Si +1.9-2.4%, graphite -0.2-0.3%.
6. **Test count.** There are now 9 named test functions (T2 is split in two).
7. **T6.** It now runs to s=1. It asserts at frame 1 and reports the rest through a `T6_REPORT` line (ES14).

### Dispatcher changes and coordinator facts
- **A (Modal-only).** Applied as above.
- **B (harmonisation).** Decided as P24.
- **C (branch and PR).** Applied as in comment 2.
- **Volume layout.** `/data/half` and `/data/heldout/half` are already uploaded, so Step 4 now checks them with `modal volume ls` instead of uploading.
- **Spike.** The tag, install recipe and API are pinned in P20 and P12. The `bt` line search is kept, with a pre-authorised fallback to the spike's `basic`.

### Notes
- The planning-time measurements used a throwaway `/tmp` script that only read `cache/`.
- `scripts/modal_fem_spike.py` (from the spike) and `scripts/pooling_compare.py` are untracked in the worktree. The plan does not reference them as deliverables. The dispatcher should decide whether to commit or delete them.
- Revision log round 1 is appended to the plan with a response to every point.
