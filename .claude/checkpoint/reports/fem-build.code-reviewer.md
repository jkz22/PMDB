## Code review — commit 0b28ff7 (fem-build Steps 1-6b)

**VERDICT: APPROVE** (one MEDIUM and three LOW findings for the dispatcher to triage; none blocks Step 7)

**Plan compliance**: complete for Steps 1-6b.
- Step 1: `configs/fem/fem.yaml` matches the plan text exactly. `config.py` matches.
- Step 2 `materials.py`: P7/P8/P9 formulas are correct. The SOC-table test checks all 11 rows.
- Step 3 `geometry.py`: label order, the 2x2 majority vote (count*10 + rank bonus, bonus < 10, so the tie-break never overrides a count), trailing-row drop and `central_cols` all match. The data tests are present (both sites' checks are in one function; this is reported deviation 2).
- Step 4: `result.py`, `solver.py`, `modal_fem.py` (probe/unit only) and the 9 solver tests are present. Two deviations, both reported and both mechanical: ES5 `--python /dolfinx-env/bin/python`, and probe labels returned as bytes.
- Steps 5, 6 and 6b match. I checked `method.md`: outside the AUTO blocks the hand-written text contains only identifiers, no numbers (P28).
- Out of scope was respected. Nothing outside the FEM files was touched in this commit.

**Verification**: I re-ran `pytest tests/test_fem_unit.py tests/test_fem_docs.py` on a clean `git archive 0b28ff7` export with the default python: 40 passed. I did not re-run the 9 dolfinx tests (Modal only). The implementer's pytest.txt path and the T6_REPORT are recorded in the report.

**What I checked against the code and found correct.** These are the targeted silent-corruption risks the G1 tests would not catch.
- Energy (P6). `Fe2 = F2·diag(1/λx, 1/λz)` and `Fe33 = 1/λy` (F33 = 1, plane strain). `Je = det Fe2 / λy`. `tr Ce = tr(Fe2ᵀFe2) + 1/λy²`. ψ is weighted by `Jλ = λx λz λy` (solver.py:78-85).
- Cauchy stress: `(1/Je)[μ(Be−I) + λ ln Je I]` with `Be33 = 1/λy²` (solver.py:121-130). This equals `P Fᵀ / J` for the multiplicative split. `J = det F2` is the total J.
- Graphite stretch tuple (x, z, y) = (1+εa, 1+εc, 1+εa), with c ∥ mesh axis 1 (z) (materials.py:75-76, solver.py:79).
- Cell→pixel map. `row = H−1−floor(z/h)`. DG0 assignment uses `label_by_dof[cell_dofs] = labels.ravel()[pix]`, and readout uses `a[pix] = x[cell_dofs]`; the two are inverses (solver.py:52-64, 147). The node map `row = H − rint(z/h)` matches P13 (solver.py:135-142).
- The property table is indexed by label code, so no phase can be swapped (solver.py:68-75).
- BCs. `bottom` puts uz = 0 at z = 0 (image last row). `top` puts it at z = H. Rollers fix ux on both lateral edges. The `isclose` tolerances are far below h (solver.py:102-108).
- Stepping (P11). The predictor, halve-on-fail with u restored, the ds_min boundary, and snapping s_n to the target all work. Every step is a dyadic fraction of 0.05, so a truncated remainder never drops below ds_min. Frames at and after `failed_at_s` stay NaN (solver.py:209-245).
- Features. Free-edge row is 0 for bottom and H for top, with `n_free` +1 and −1. Edge nodes are inclusive. The porosity, `porosity_rel_change` and empty-phase NaN rules hold. `first_pore_closure_s` is computed over the whole domain. `p = −(sxx+szz+syy)/3`. Tier-4 masking is correct. In `symmetrise`, metrics are averaged with NaN propagating, `converged` is an AND, and `failed_at_s` / `first_pore_closure_s` use `fmin`, the correct "min, NaN = never" semantics (features.py:51-167).
- GIF. Corner z0 is `(H−ri)·h`, consistent with "uz positive toward row 0". The block means for `reduceat` and the C-array shape (n−1, m−1) are correct. The LogNorm is fixed. After a failure, the last converged frame is repeated (gif.py:43-101).
- Units are µm and MPa throughout. Tiles use `tile_x0_um = start·h`.

**Findings** (numbered, severity-rated; these go verbatim to an implementer):

1. [MEDIUM] `pmdb/fem/solver.py:112-118, 175` — The code accepts any SNES reason > 0, and PETSc's default `snes_stol` is 1e-8. `SNESConvergedDefault` therefore reports `SNES_CONVERGED_SNORM_RELATIVE` (reason 4) whenever ‖λ·δu‖ < 1e-8‖u‖, and that can happen when the `bt` line search has cut λ to a tiny value while the residual is still far from converged. That is exactly the stiff pore-collapse regime that drives `pore_closed_frac`, `first_pore_closure_s` and the `q*_J_pore` features. Such a step is silently accepted as a converged frame. No test covers it: the analytic tests run with rtol/atol 1e-12 on problems that converge quadratically.
   - Fix: add `"snes_stol": 0.0` to the `petsc_options` dict at solver.py:112-118. With it, convergence can only come from the fnorm tests (`snes_rtol` / `snes_atol`), so `reason > 0` really means the residual converged. Also add `"fnorm": float(problem.solver.getFunctionNorm())` to the substep record `rec` (solver.py:176), wrapped in the same try/except as `getConvergedReason`, so the run log shows the final residual. Dispatcher note: this adds one PETSc option beyond the P12 list. It is physics-neutral; it only tightens what "converged" means.

2. [LOW] `pmdb/fem/solver.py:167-174` — `except Exception` turns every PETSc/MUMPS error into reason −99. That includes a MUMPS workspace or memory error (INFOG(1) = −9/−8) or a JIT failure. The stepper then halves Δs down to ds_min and records `failed_at_s`, so an infrastructure fault looks like non-convergence, and only the bare −99 code is left behind to tell them apart.
   - Fix: change to `except Exception as exc:` and set `rec["error"] = repr(exc)[:500]` in the substep record when an exception occurred (add `"error": None` to the record otherwise). Keep the stepping behaviour unchanged.

3. [LOW] `tests/test_fem_solver.py:113-127` (test value) — No test detects a cell map that is consistently flipped vertically in both directions (assignment and readout) relative to the BCs. T4 (argmax at (1,6)) would still pass, because a flip in assignment is undone by the same flip in readout. The T2/T3 free-edge displacement does not depend on which layer sits next to the collector. The code is correct today (solver.py:53, 64, 147), but a regression in the midpoint→row formula would go unnoticed.
   - Fix: in `_check_bilayer`, also assert the interface node row 5. For `bottom` (A unstrained next to the fixed edge): `abs(r.u_nodes[1, 5, :, 1].mean()) < 1e-3 * uz_free`. For `top` (B next to the collector at row 0): `r.u_nodes[1, 5, :, 1].mean() == pytest.approx(-uz_free, rel=rel)`. A flipped map gives `uz_free` and 0 there instead.

4. [LOW] `tests/test_fem_solver.py:149-150` — The plan's T4 says "a 7×9 label map with a single SI pixel at (row 1, col 6)". The test builds a 9×7 map (9 rows, 7 cols), so col 6 is the last column, next to the x = W roller. The test still catches row and column flips. However, the probe pixel now sits on a BC rather than in the interior, which is weaker than the plan's case.
   - Fix: change `np.full((9, 7), ...)` to `np.full((7, 9), ...)`. Keep `lab[1, 6] = SI` and the assertions unchanged.

**Notes to the dispatcher (not code findings)**
- `gif.py:74-75`: if frame 0 itself failed (`failed_at_s == 0`), `last_ok = 0` points at NaN `u_nodes` and `pcolormesh` will likely raise. Step 7's `run_case` must catch this as `gif_error` (P19). This is relevant for Step 7, not this commit.
- `symmetrise` gives NaN `sym` rows when `top` rows are absent. The §2.1 "drop_top" deviation (sym = copy of bottom) must be handled in `fem_collect` (Step 8), not here.
- Implementer note 5 (the run_log key names `cases`, `ledger_total_usd`) must be honoured by Step 8, or `fem_docs._cost_block` stays pending.

**Positive observations**: The finite-strain kinematics, the plane-strain Fe33 = 1/λy treatment, det-Fλ weighting and the DG0 index mapping are all correct. The code is compact and close to the plan, which makes these easy to check by reading.
