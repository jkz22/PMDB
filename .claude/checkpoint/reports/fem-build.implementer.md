# fem-build implementer report: Steps 1 to 6b (stopped before Step 7)

## Changes made (all new, nothing committed or pushed)
- `configs/fem/fem.yaml` - Step 1 (exact plan content).
- `pmdb/fem/__init__.py`, `config.py` (Step 1), `materials.py` (Step 2), `geometry.py` (Step 3), `result.py` and `solver.py` (Step 4), `features.py` (Step 5), `gif.py` (Step 6).
- `modal_fem.py` (repo root) - Step 4: image per P20, volumes, ledger and cost helpers, `budget_check`, `probe` and `run_unit_tests` functions, entrypoint modes `probe` and `unit` only.
- `tests/test_fem_unit.py` (config, materials, geometry, features, gif; 37 tests), `tests/test_fem_solver.py` (9 solver tests T1a, T1b, T1c, T2 small, T2 finite, T3, T4, T5, T6), `tests/test_fem_data.py` (2 data tests), `tests/test_fem_docs.py` (3 tests).
- `scripts/fem_docs.py` (`--method` works; `--results` prints a message and exits 2 until Step 10), `docs/fem/method.md` (skeleton plus the 5 generated blocks filled).
- Nothing else in the tree was touched (the other modified/untracked files in `git status` belong to the classifier lane).

## Verification
- Step 1/2/3 local: `pytest -q tests/test_fem_unit.py` pass. `pytest -q -m data tests/test_fem_data.py` -> 2 passed (coarse/fine ratio-1: Si +0.019, graphite -0.0025, pore +0.072, binder -0.220; harmonisation invariance max differing fraction 0.00137 on 9luzk4jm, max area-fraction diff 0.00091).
- `pytest -q tests/test_fem_solver.py` in default python: skipped (importorskip).
- Step 4 `modal run modal_fem.py --mode probe` -> exit 0. python 3.12.3, dolfinx 0.10.0, petsc4py 3.24.0, numpy 2.2.6, scipy 1.14.1, scikit-image 0.25.2, mumps True. Label disagreement image vs local: 0.000e+00 on all 5 probe sites (vc2whyaq, 71vgq3fw, kbdh4tri, tuy3zymq, x7u69zsw). `NonlinearProblem.__init__(self, F, u, *, petsc_options_prefix, bcs=None, J=None, P=None, kind=None, petsc_options=None, form_compiler_options=None, jit_options=None, entity_maps=None)`; the `solve()` docstring says the caller must check `getConvergedReason()`, which solver.py does. Probe JSON: `outputs/modal/fem/probe/20261003T213523Z.json`.
- `modal run modal_fem.py --mode unit` (final run, after Steps 5-6) -> exit 0, 46 passed (37 unit + 9 solver). Gate G1 passed (the first unit run, with only the solver and Steps 1-3 tests, gave 34 passed). pytest.txt: `outputs/modal/fem/tests/20261003T214241Z/pytest.txt`.
- T6_REPORT: `{"failed_at_s": null, "min_pore_J": [1.0, 0.4370802640914917, 0.30594298243522644, 0.15417274832725525, 0.02613675408065319], "n_substeps": 34}` (no failure; min pore J reaches 0.026 at s=1, below the 0.1 closure threshold, so pores do close in this toy case).
- Step 5: `pytest -q tests/test_fem_unit.py -k features` -> 11 passed. Step 6: `-k gif` -> 1 passed (synthetic 580x1748, 11 frames, <= 2 MB, width in {900,720,600}), also passes in the image.
- Step 6b: `pytest -q tests/test_fem_docs.py` -> 3 passed. `python scripts/fem_docs.py --method` exit 0; `grep -c "AUTO:" docs/fem/method.md` = 10; `grep -c "_pending:"` = 1 (only the cost block).
- Full local `pytest -q -m "not data"` -> 149 passed, 1 skipped (solver module), 16 deselected.

## Modal spend so far (ledger `outputs/modal/fem/ledger.csv`, gitignored)
Ledger total $0.0161 (probe ok $0.0029, probe failed attempt $0.0029 added by hand, unit $0.0048, unit $0.0055). Image builds and container start outside function wall time are not in the ledger (covered by DEV_ALLOWANCE_USD = 5.0). Cap $180, no budget refusal. Pricing page was not re-checked (constants CPU_RATE 0.0472, MEM_RATE 0.0080 unchanged).

## Notes (deviations, all mechanical)
1. ES5 applied (pre-authorized): the first probe failed with `ModuleNotFoundError: skimage`, i.e. `uv pip install --system` did not install into the interpreter Modal runs. Replaced `--system` with `--python /dolfinx-env/bin/python` in `modal_fem.py`. The failed first probe call (about 10 s) was ledgered by hand as status `error`, attempt 0.
2. `tests/test_fem_data.py`: the harmonisation test loops over the 5 sites inside one test function so the file has the 2 tests the plan expects.
3. `budget_check` implements P22 rules (a) and (d) only (probe, unit, bench, window); modes `full`/`full_rerun` (rules b, c) raise NotImplementedError and belong to Step 8.
4. `probe` returns each label map as `{shape, bytes}` rather than a numpy array, to avoid numpy 2.2 to 1.26 pickle problems on the local client.
5. The `cost` block of `method_blocks` reads these run_log keys defensively: `cost_model`, `production`, `cases` (list of dicts with `wall_s`, `cost_usd`), `ledger_total_usd` (falls back to `ledger_total`). The plan does not name the per-case key or the ledger-total key, so Step 8's `run_log.json` should use these names (or the block should be adapted).
6. `scripts/fem_docs.py` params block renders values as `dotted.key = value` for every row; solver `ds_growth` logic: step doubles on success with <= grow_if_its_le iterations, as in P11.
7. The analytic tests set `snes_rtol=1e-12, snes_atol=1e-12` (test-level option) so the 1e-8 / 1e-6 relative tolerances in T1a/T1b/T1c are met.
8. The GIF was only tested on synthetic data, not eyeballed (the Stage-3 visual check is Step 7).
9. Not in scope but noticed: nothing outside the FEM files was modified.
