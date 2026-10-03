# Plan review: fem-build.md (round 1)

**VERDICT: REVISE**

**Assessment**: The physics and numerics hold up. I re-derived the energy, the eigenstretch laws, the SOC mapping and every analytic test's closed form, and all of them are correct and falsifiable. Two binding-context contradictions block execution. The swelling gate was amended by the user after the plan was written (commit 00422db), and the plan commits to the wrong branch. A third problem blocks Step 1: it assumes a `pmdb-fem` conda env that does not exist. Everything else is minor.

## Comments

1. [CRITICAL] The swelling gate contradicts the binding file as amended at HEAD (commit 00422db, "amend swelling stop gate to 3-39%").
   - `.claude/plans/fem-swelling.md` now says: "the literature band [9%, 39%] is REPORTED as a comparison only. The STOP condition is swelling < 3% or > 39% … The analytic free-expansion unit test (J = Jλ, zero stress) must still pass exactly."
   - The plan still gates on the old band in P23 (`swelling … in [0.09, 0.39]`), Stage 4 G4w (`Pass if it is in [0.09, 0.39]`), Step 9 G4 and Expected surprise 4.
   - Why it blocks: the plan's own preamble says "Where this plan and those files disagree, those files win: stop and report". The implementer would stop at Step 7 on the contradiction, or would stop wrongly at about 7-9% swelling.
   - Suggested direction:
     - Restate P23, G4w, G4 and ES4 with the stop window [0.03, 0.39].
     - Keep [0.09, 0.39] as a reported column, for example `lit_band_ok` in `validation.csv`.
     - Decide explicitly whether the 3-39% stop window also applies to siox. The amendment does not limit it to si, and the plan currently makes siox "informational only" with no stop point after the siox run (Step 10).
     - T1a already satisfies "J = Jλ, zero stress" because λy = 1. Saying so in G1 would close the loop.

2. [CRITICAL] Wrong branch, and stale PR context.
   - Step 10 says `Commit to branch fem-lit-review and push`, and the Done criteria say `Committed on fem-lit-review`.
   - The binding pre-approval says "committing and pushing to branch fem-sim (PR #18 merged 2026-10-03; new PR from fem-sim)". The current branch is `fem-sim`.
   - §3 says "No merging of PR #18", but #18 is already merged.
   - Why it blocks: the implementer would push work to a merged branch, or would stop on the contradiction.
   - Suggested direction:
     - Use `fem-sim` throughout.
     - State whether Step 10 opens the new PR from `fem-sim`. The pre-approval and the user's memory ("work lands via PR") suggest yes, never merged.

3. [MAJOR] Step 1 and Expected surprise 1 assume an existing `pmdb-fem` env, but there is none (`conda env list` shows no pmdb-fem). The planner's report says the same.
   - Step 1 says "`<V>` is the exact version reported by `conda list -n pmdb-fem fenics-dolfinx`" and "Install any of these packages missing from the existing `pmdb-fem` env".
   - Why it blocks: `<V>` is undefined, and the implementer must decide how to create the env and which dolfinx to pick. conda-forge has osx-64 builds of both 0.10.0 and 0.11.0 (verified via the anaconda.org API), and their APIs differ in the places ES1 lists. The host is x86_64 with no Docker, so there is no local container fallback.
   - Suggested direction:
     - Say to create the env from `environment-fem.yml`.
     - Fix the version rule, for example "the newest fenics-dolfinx ≥ 0.10 that resolves on osx-64, with the same version on linux-64 for Modal".
     - Consider pinning the PETSc real/double variant so conda cannot pick a complex build.

4. [MINOR] Cost projection is inconsistent and lacks a worst-case bound.
   - Stage 3 extrapolates with `(n_cells_site/n_cells_window)^1.5`, but STOP B scales "per site by n_cells/n_cells_bench", which is linear.
   - Site heights range from 806 to 1158 half-res rows (`cache/half/manifest.csv`), so the exponent matters by about 1.2-1.7×.
   - P20 also sets `retries=1`, and the timeout can be 3× the benchmark wall. A projection from the benchmark alone does not bound spend if cases hang or retry.
   - Suggested direction: pick one exponent. Also report a worst-case figure (n_cases × timeout × rate) beside the expected figure at STOP B/C, so the dispatcher can judge the $180 hard cap.

5. [MINOR] Step 3's data test (`coarse Si area fraction is within ±20% relative`) has no Expected-surprise entry for failure.
   - Majority-vote coarsening of 1-px Si features can plausibly miss the ±20% band.
   - Suggested direction: state "stop and report" (consistent with G2), or say whether the tolerance is advisory.

6. [MINOR] Step 4 Verification says `all 7 tests must pass`, but the plan defines 8 (T1a, T1b, T1c, T2, T3, T4, T5, T6). This is mechanical but worth fixing.

7. [MINOR] T6 only goes to s = 0.25. The fragile point for this model is pore-element inversion (Q1 quadrature-point J ≤ 0, giving NaN in `ln Je`) next to strongly swelling Si at s ≥ 0.5.
   - That is what would drive G5 (> 3 of 34 failures). Today it would first surface in the Stage 3/4 window, which ES3 handles by stopping at A.
   - Suggested direction (optional): extend T6 to s = 1 with a report-only assertion on `failed_at_s`, so the risk shows up in Stage 1 rather than in the window.

## Physics and numerics check (no findings)

- **P6 energy.** `Fe = F₃Fλ⁻¹`, `Ce₃₃ = 1/λy²`, `Je = detF₂/(λxλzλy)` and the Jλ weighting match R5:E13 and the lit review's "For the planner". The P13 Cauchy stress `(1/Je)[μ(Be−I)+λ ln Je I]` follows correctly from that energy.
- **Eigenstretch laws (P7).** Si `(1+2.8u)^(1/3)`, SiOx β = 1.6 and graphite `(1+εa, 1+εc, 1+εa)` with c ∥ z and εa = 0.01εc/0.103 all match the lit review.
- **SOC mapping (P9).** The general-s* normalisers reproduce 0.675/0.325, and 0.7168 at s* = 0.36. I recomputed all 11 table rows. The largest deviation is ε_zz at s = 1 (0.0944 vs the tabulated 0.095), inside the 1.5e-3 tolerance.
- **Shah mapping.** `C/Cmax = βu/3` and σY(u = 1) = 513.16 MPa are correct.
- **T1b/T1c closed forms.** Correct: the in-plane elastic stretch is equal in x and z by the σxx = σzz = 0 symmetry, and brentq has a sign change on (0.5, 2).
- **T2.** The ε* = 1e-2 finite-strain deviation is about 0.24%, well inside 3e-2. Q1 represents the piecewise-linear exact solution exactly.
- **T3.** The sign is correct.
- **T4.** The off-centre pixel makes it falsifiable against a row or column flip.
- **T5.** Max_it = 1 cannot converge a nonlinear increment, so `failed_at_s` ≤ 0.1 either way.
- **BCs.** Free-edge pressure residual sign `+p_eff·v·n` is correct.
- **Coarsening tie-break.** `counts*10 + bonus` gives all 7 stated block results.
- **Tiles.** `fem_tile_slices(1747, 0.1)` gives 200…1547 with 22.45 µm tiles, matching D9.
- **Orientation sym.** The sym rule is invariant to a vertical flip, including the band metrics (maxdev and |slope| are both flip-invariant).
- **Shared helpers.** `band_profile` and `band_summary` (pmdb/kpis/fields.py:71, 79) exist with the assumed semantics (NaN when the mean ≤ 0, so frame 0 is NaN; the sweeps use frames 1-10).
- **Load path.** `load_site(..., cache_root=…)` builds `<root>/half/<batch>__<site>.npz` and reads `<root>/half/manifest.csv` (pmdb/io.py:394-414), so the Modal `/data` and `/data/heldout` layout works. Volume name `pmdb-data` matches `modal_app.py:10`.
- **D10.** 6 fixed tiles × 11 frames × 81 metrics per (config, orientation, sym) in `tile_curves.csv` is exactly what feature-level [mean, max, spread] pooling needs.

**Swelling-gate assessment (requested; not a change request).** The planner's ~7% is a lower bound.
- With lateral u_x = 0 and plane strain, graphite's through-thickness strain picks up a Poisson term, about εc + 2ν/(1−ν)·εa ≈ 0.094 + 0.006 ≈ 0.10. That gives about 7.6% at 76% graphite.
- Plane-strain Si has an area ratio of about 2.66 (I solved this with the T1b closed form), so 6% Si can add up to several percent before pores absorb it.
- I expect a median of roughly 8-14%. Under the old [9, 39] band a low-side trip was plausible. Under the amended [3, 39] stop window it is very unlikely unless something is broken, which is the amendment's intent.

## Assumption ratings (FRAGILE)

- **Pores stay uninverted to s = 1 on most sites** (ersatz E = 0.05 MPa next to Si with J = 3.24). Verify with the Stage 3 window run to s = 1, or with an extended T6.
- **Benchmark-based projection bounds spend.** This fails if a few sites need many halvings or hit the timeout and retry. See comment 4.
- **numpy<2 pin (ES2) resolves with dolfinx 0.10/0.11.** It probably will not. ES2's "stop and escalate" fallback is acceptable, but check first whether pmdb simply runs under numpy 2.
- **GIF vm range [1, 1e4] MPa.** The plane-strain out-of-plane constraint on Si gives σyy of order 10-20 GPa, so ES8 will likely trigger. That is already handled.

## What's missing

- The amended gate (comment 1), the correct branch and PR step (comment 2), and env creation (comment 3).
- Pooling checks 3-4 are in the user's pre-approved scope but are excluded here. That is acceptable; one line saying they follow in a separate plan would avoid confusion.
