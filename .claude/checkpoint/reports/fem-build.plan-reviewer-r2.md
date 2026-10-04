# Plan review: fem-build.md (round 2)

**VERDICT: REVISE**

**Assessment**: All seven round-1 comments are resolved in the plan text itself, not just in the Revision log. The Modal-only design is executable: the image, the recipe, the ledger, the cap check and the per-chunk stop points are concrete enough for an implementer to follow. The physics is unchanged from the version approved in round 1. The harmonisation decision (P24) is justified and checked against the code. The plan is blocked by one thing only: D17, the user's scope cut recorded in `fem-swelling.md` during this review. The plan still schedules the SiOx full run and the D14 sweeps. Its own preamble says "Where this plan and those files disagree, those files win: stop and report", so the implementer would stop at once. Everything else is MINOR.

## Prior comments (round 1): verification

| # | Status | Evidence in the plan text |
|---|---|---|
| 1 swelling gate | Resolved | The window is [0.03, 0.39] in P25, the `gates:` block of `fem.yaml`, Step 5 (g), Stage 4 G4w, Step 9 G4 and ES7. `lit_band_ok` is reported only. T1a is named as the free-expansion test. |
| 2 branch | Resolved | Line 5, §3, Step 10.5 (`gh pr create --base main --head fem-sim`, never merge, no duplicate PR) and Done criteria. |
| 3 env | Superseded by D16, resolved | P20 pins the image tag the spike used (`scripts/modal_fem_spike.py:4`). Conda references are gone. |
| 4 cost | Resolved | One exponent (1.5) is used in P22 and Stage 3. Lost calls are charged at the timeout. `retries=0`. Runs are chunked with a re-check before each chunk. STOP B/C report the worst case. |
| 5 coarsening | Resolved | The asserts are hard, and a failure means stop and report (Step 3). |
| 6 test count | Resolved | 9 named functions; every verification says "9 passed". |
| 7 T6 to s = 1 | Resolved | T6 runs frames [0, 0.25, 0.5, 0.75, 1.0], adds the `T6_REPORT` line, and ES14 covers it. |

## Comments

1. [CRITICAL] **D17 (scope cut) contradicts the plan.** `.claude/plans/fem-swelling.md:70-72` says "D13 SiOx full run and D14 robustness sweeps are REMOVED … no SiOx gate (G4x), no sweep config/mode, no Spearman robustness filter."
   - Why it blocks: the plan's preamble tells the implementer to stop on any contradiction with the binding files. Steps 1, 7, 8, 9 and 10 all schedule SiOx or sweep work.
   - Suggested direction: apply the cut list below. Re-derive every row count and cost projection for si only. State explicitly whether STOP POINT C is still a mandatory stop. Once SiOx and the sweeps are gone, the only work left after C is collection, docs and the PR, all of which the user pre-approved.

2. [MINOR] **The probe cannot detect whether uv moved numpy.** P20 pins `numpy==2.2.6` because it is "the image's own" numpy, so that uv "cannot move it under the compiled dolfinx/petsc4py".
   - The probe's pass criterion "prints … numpy 2.2.6" (Step 4) will always hold, because numpy is pinned. If the image actually ships a different 2.x, uv silently replaces it, and the probe still prints 2.2.6.
   - The spike report gives no numpy version. I could not query the registry config from this sandbox.
   - Why it matters: if numpy was moved, the failure shows up later as an ABI or import error inside dolfinx or petsc4py, and no Expected surprise maps that symptom to its cause.
   - Suggested direction: assert the image's numpy version before the uv layer runs (one build-time check), or have uv refuse to change any package that is already installed. If the check fails, it is a stop under ES2.

3. [MINOR] **The FEM/KPI mask-identity guard runs on one site only.** P24 rationale (3) relies on "identical masks" between the FEM and KPI pipelines, but those are different environments:
   - the image runs numpy 2.2.6 on Python 3.12;
   - the KPI pipeline ran numpy 1.26.4 on Python 3.10 (`modal_app.py:14-17`).

   The probe compares only `Batch_3/vc2whyaq`. This does not block, because ES3 sets a clear threshold. If the identity claim is meant to hold for all 34 sites, adding the 4 strong Batch-3 sites to the probe would cost cents. Planner's call.

## Cut list for D17 (sections to delete or simplify)

- **Context (§1):** drop "Si and SiOx on all 34 sites" and "D14 sweeps on 6 sites".
- **P7, P8, P9:** drop the SiOx eigenstretch and moduli, the Shah E variant, the isotropic-graphite variant, the proportional split and the "SiOx uses the same σY" sentence. The general-s* normaliser can stay, but nothing exercises s* ≠ 0.25 any more.
- **P10 and the Step 4 BCSpec / solver BC bullet:** the `confined` and `pressure` free-edge modes, and the solid-only pressure facet tagging, were used only by the sweeps. Removing them is a real simplification of `solver.py`. Keep `free` and the test-only `lateral="left"`.
- **P18:** delete the whole decision (sweeps, Spearman, `robust`).
- **P20:** the `outputs/kpis/site_kpis.csv` `add_local_file` was needed only for the sweep-site test, so it can go.
- **P22:**
  - drop `f_cfg` and the siox branch;
  - chunks become si bottom then si top only, and the "sweeps: variants 1-6, then 7-12" chunking goes;
  - remove "and sweeps" from the projected-cost formula.
- **P25:** drop the siox window clause, G4x, the siox failure-count reporting and the sweep-variant exemption.
- **P26:** the modes list loses `sweeps`.
- **Step 1:**
  - in `fem.yaml`, remove the si/siox fields named in the P7/P8/P9 bullet above, `graphite.strain_mode`, `configs.siox`, the whole `sweeps:` block, and `bc.pressure_MPa` / `bc.top` if only `free` remains;
  - in `config.py`, remove `list_sweep_configs`, `sweep_sites`, `select_sweep_sites` and the sweep branch of `load_params`. `list_full_configs` becomes trivial or goes;
  - remove the tests for `siox`, `si__umax_0.60`, `len(list_sweep_configs()) == 12` and `select_sweep_sites`.
- **Step 2:** remove tests (b) s* = 0.36 and (c) proportional if those modes are removed, (e)'s isotropic-graphite half, and (f) Shah.
- **Step 7:**
  - Stage 3: drop "then the same with `--config siox`" and the siox/si wall ratio;
  - Stage 4: drop G4w for siox;
  - STOP A: drop the `f_cfg` report line.
- **Step 8:**
  - delete mode `sweeps`;
  - in `fem_collect.py`, drop `sweep_site_curves.csv`, `sweep_tile_curves.csv`, `robustness.csv` and `robust_summary.csv`;
  - `validation.csv` becomes "per site" (si only);
  - the gate summary drops siox;
  - STOP B projections use 68 si cases instead of 208 (68 + 68 + 72), and the largest chunk becomes 34.
- **Step 9:** STOP C's "projection for siox + sweeps" goes, and the decision on whether STOP C remains a stop is needed (comment 1).
- **Step 10:**
  - delete sub-steps 1 (SiOx / G4x) and 2 (sweeps);
  - in the docs list, drop the "robustness flag (P18)";
  - drop "robust-feature counts" from the PR body and "G4x" from the gate list;
  - in Verification, `site_curves.csv` becomes 34×3×11 = 1122 rows and `tile_curves.csv` 34×3×6×11 = 6732; delete the sweep and robustness row counts and the "Report G4x … robust pairs" line.
- **Expected surprises:** delete ES13 (sweep variants). In ES7, drop the siox mention, if there is one.
- **Done criteria:** drop G4x and the sweep and robustness CSVs; the row counts follow Step 10.
- **§3 Out of scope:** add SiOx and sweeps (D17). The "never appear in sweeps" held-out clause, and the "absent from the sweep tables" check in Done, become moot.

## Checks with no finding

- **Harmonisation (P24): justified.**
  - `load_site` accepts `normalise="none"` with `harmonise="hybrid"`; only `percentile` is rejected (`pmdb/io.py:382`).
  - The hybrid LUTs exist (`cache/harmonised/hybrid/luts.npz`).
  - `docs/harmonisation.md:140` says the segmenter "is unaffected by any method".
  - The KPI pipeline segments `normalise="none"` input (`scripts/run_kpis.py:218-219`).
  - FEM features never read grey levels, and the GIF's per-site p1-p99 stretch is affine-invariant.
  - The data-test guard is falsifiable: a ≤ 0.005 bound against a measured value of ≤ 0.0014.
- **Image and pins.**
  - scipy 1.14.1 and scikit-image 0.25.2 match `modal_app.py:16`.
  - scipy 1.14.1, scikit-image 0.25.2 and pandas 2.2.3 all have cp312 / numpy-2 wheels.
  - The `pmdb` import chain (`pmdb/__init__` → io → harmonise → segment; `pmdb.kpis` → scipy/skimage) needs only packages in the P20 list.
  - The KPI catalogue CSV is read only inside a function (`pmdb/kpis/__init__.py:62`), so leaving it out of the image is safe.
  - `pytest.ini` has `pythonpath = .`, so the subprocess pytest with `cwd=/root` finds `pmdb`.
  - `outputs/modal/` is gitignored (`.gitignore:12`).
  - `cache_heldout/half` holds the 3 npz files plus the manifest.
- **Cost constants.** 0.0472 and 0.0080 are $0.0000131 and $0.00000222 converted to per-hour rates.
  - The ledger and cap check give an upper bound per chunk.
  - The $15 dev guard and the exit codes are unambiguous.
  - After D17 the expected spend roughly halves, so the $180 cap has ample headroom.
- **Physics regressions.** P6-P13 and the T1a-T6 closed forms are unchanged from round 1. I re-checked the T1b in-plane condition μ(f²−1)+λ_L ln(f²/λ) = 0 and the T2 εzz factor (1+ν)/(1−ν).

## Assumption ratings (FRAGILE)

- **The image's numpy is 2.2.6.** Nothing in the run itself can detect a mismatch (comment 2). Verify with a pre-uv build-time assertion.
- **`uv pip install --system` packages are importable.** The spike installed pandas but never imported it, so this is unverified. ES5 pre-authorises the `--python /dolfinx-env/bin/python` fallback, and `--mode unit` exercises every import. Covered.
- **Pores stay uninverted to s = 1** (carried over from round 1). T6_REPORT and the Stage-3 window now surface this early. Covered.

## What's missing

- Applying D17 (comment 1, cut list above).
- Nothing else significant.
