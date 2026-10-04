# FEM build: 2D plane-strain finite-strain lithiation swelling, features, GIFs, Modal runner

Binding inputs: `.claude/plans/fem-swelling.md` (D1-D19, budget, D19 pitch-deck documentation with script-generated numbers (this plan delivers its FEM part: `docs/fem/method.md`, `docs/fem/results.md`), D18 classifier architecture (separate plan, consumes §2.1), user pre-approval with the $180 HARD CAP, the AMENDED swelling gate, D10 FINAL = class-weighted mean of tile probabilities in a separate classifier plan, D16 Modal-only environment, D17 scope cut: only the default pure-Si configuration, no SiOx, no sweeps) and `docs/fem/literature-review.md` (parameters, SOC mapping, validation targets VT1-VT6, "For the planner"). The pre-approval section's **FULL AUTONOMY** policy governs execution: nothing waits for the user, the only hard stop is the $180 Modal cap, every gate is fix-and-continue (P25), and test or swelling-gate failures are bugs escalated to the dispatcher for a fix round. Where this plan and those files disagree, those files win: escalate to the dispatcher.

Branch: `fem-sim` (PR #18 from `fem-lit-review` is already merged). All work is committed and pushed to `fem-sim`, and a new PR `fem-sim → main` is opened at the end. Never merge it.

## 1. Context

The project wants batch-discriminating features from a mechanics simulation of each SEM cross-section. Every site's segmentation (`pmdb.segment.segment`) becomes a pixel-based finite-element mesh. The mesh is lithiated from 0 to 100% SOC under uniform-per-phase eigenstretches. Si swells isotropically, graphite anisotropically (c ∥ z), and binder and pores do not swell. The solve is plane-strain, finite-strain, compressible neo-Hookean, with FEniCSx/dolfinx. Fields are snapshotted at 11 SOC frames and reduced to tile and site curve features (D8 tiers 1-4) on the D9 tile grid, plus one GIF per site (D6, D15). All dolfinx code runs on Modal in the pinned official dolfinx Docker image (D16): analytic tests, the resolution check, the smoke window, the benchmark and production. Production is the single default pure-Si configuration on all 34 sites in both collector orientations, 68 cases (D17). Pure-numpy modules are tested locally as well. The classifier and site pooling (final D10) are a separate plan that consumes `tile_curves.csv` / `site_curves.csv` exactly as specified in §2.1.

## 2. Design decisions

Conventions used throughout:
- Mesh axis 0 = **x** (coating width = image columns).
- Mesh axis 1 = **z** (thickness). z points toward image row 0: `z = (H - row - 0.5)·h` at a cell centre.
- **y** = out-of-plane.
- Lengths in µm, stresses and moduli in MPa (so E_Si = 96 000 MPa), forces per unit depth in MPa·µm.
- "Default python" = the repo's usual interpreter (`python`, currently 3.10.3 with numpy 1.26.4, pandas 2.1.4, scipy 1.14.1, scikit-image 0.25.2, pyyaml, matplotlib, Pillow, modal 1.6.0 all installed). "Image" = the Modal image of P20 (Python 3.12, numpy 2.2.6). Pure-numpy modules must run in both.
- There is exactly one parameter set (`configs/fem/fem.yaml`, D17). No config names, no `config` argument, no `config` column anywhere.

| # | Decision | Rationale |
|---|---|---|
| P1 | Module layout. `pmdb/fem/{__init__,config,materials,geometry,result,solver,features,gif,run}.py`; `configs/fem/fem.yaml`; `scripts/fem_collect.py`; `modal_fem.py` (new, at repo root). No `environment-fem.yml`, no local dolfinx, no local runner script (D16). Only `solver.py` imports dolfinx/ufl/petsc4py/mpi4py, and `run.py` imports `solver` lazily inside the function that needs it. Every `pmdb/fem` module starts with `from __future__ import annotations` and uses no numpy-2-only API. | Everything except the solve is unit-testable in the default python and in the image. This matches how `pmdb.kpis` stays independent of the segmenter. |
| P2 | Mesh resolution 100 nm/px (coarsen the 50 nm label map 2×2). Native 50 nm is used only for the Stage-2 resolution check. | A full site is about 580 × 1750 coarse px, so about 1.0 M quads and 2.0 M DOFs. At 50 nm it would be 4× the cells and about 8× the direct-solve cost. The Stage-2 gate tests the convergence of this choice. |
| P3 | Label map `uint8`: BINDER=0 (unassigned solid), SI=1, GRAPHITE=2, PORE=3, ARTEFACT=4. Coarsening is a 2×2 majority vote with ties broken by priority SI > PORE > GRAPHITE > BINDER > ARTEFACT. An odd trailing row or column is dropped. | Si is the rarest phase and the main driver, so ties keep Si. Thin pores carry most of the accommodation (lit review C16), so pore beats the solids. Artefact is ersatz-like anyway, so it yields. Measured at planning time on vc2whyaq / fzrt2k6r / 3806gxp0: coarse/fine area ratio − 1 = Si +0.019/+0.024/+0.020, graphite −0.002/−0.003/−0.003, pore +0.072/+0.071/+0.075, binder −0.22/−0.21/−0.24. |
| P4 | Mesh is a structured quadrilateral `create_rectangle` over `[0, W·h] × [0, H·h]` with one cell per coarse pixel. The cell→pixel map comes from cell midpoints. Materials are DG0 functions indexed through that map. | The mesh matches the pixel grid exactly, so there is no meshing step and phase tags are exact. DG0 makes per-phase updates a plain array assignment. |
| P5 | Displacement space is vector Q1 (`("Lagrange", 1, (2,))` on quads). Quadrature degree is 2. | Q1 nodes sit at pixel corners, so nodal displacement maps exactly to the corner grid. ν ≤ 0.34 everywhere means no volumetric locking. Degree 2 (2×2 Gauss) is exact enough and stops UFL from estimating a huge degree for the log terms. |
| P6 | Energy per reference volume ψ₀ = J_λ·[μ/2 (tr Ce − 3) − μ ln Je + λ/2 (ln Je)²]. Here Fe = F₃·Fλ⁻¹, F₃ = [[F₂, 0], [0, 1]] (total out-of-plane stretch 1 = plane strain), Fλ = diag(λx, λz, λy), Je = det F₂ /(λx λz λy), Ce = Feᵀ Fe (3×3), so Ce₃₃ = 1/λy². Residual = derivative of ∫ψ₀ dx; Jacobian = its derivative. | Exactly the lit-review energy (R5:E13), its plane-strain I1 and its det-Fλ weighting. |
| P7 | Eigenstretches (λx, λz, λy). **Si**: J_λ^(1/3)·(1,1,1) with J_λ = 1 + β·u, β = 2.8. **Graphite (anisotropic, the only mode)**: (1+ε_a, 1+ε_c, 1+ε_a), with ε_c from np.interp over the breakpoints and ε_a = 0.01·ε_c/0.103. **Binder, pore and artefact**: (1,1,1). | Lit review "For the planner". The out-of-plane graphite stretch is the a-axis value (c ∥ z). SiOx and isotropic-graphite variants are removed (D17). |
| P8 | Moduli. **Si (Qi, the only law)**: E = 96 000 − 55 000·u MPa, ν = 0.29 − 0.04·u. **Graphite**: E = 32 000 + 77 000·y MPa, ν = 0.32 − 0.08·y. **Binder**: 500 MPa, ν 0.34. **Pore and artefact**: E = 1e-4·E_binder, ν 0.3. Lamé: μ = E/(2(1+ν)), λ = Eν/((1+ν)(1−2ν)). | Lit review table and D11. Shah and SiOx moduli are removed (D17). |
| P9 | SOC mapping (Yao split, general form, recovers the lit-review table at s* = 0.25): f_Si(s) = [0.96·min(s,s*) + 0.58·max(s−s*,0)] / [0.96 s* + 0.58(1−s*)] and f_Gr(s) = [0.04·min(s,s*) + 0.42·max(s−s*,0)] / [0.04 s* + 0.42(1−s*)]. Then u = u_max·f_Si (0.80) and y = y_max·f_Gr (0.91). Si yield flag: σY = 3000 − 3150·x/(1+x) MPa with x = 3.75·u. | Lit review "Overall SOC". The general normaliser is kept because it is one formula with s* read from the config; only s* = 0.25 is run. The proportional split is removed (D17). |
| P10 | BCs. Lateral edges x = 0 and x = W: u_x = 0. Orientation `bottom`: collector at z = 0 (image last row) with u_z = 0; free edge at z = H. Orientation `top`: collector at z = H (image row 0) with u_z = 0; free edge at z = 0. The free edge is always traction-free (no term). A test-only lateral mode `left` (u_x = 0 on x = 0 only) gives roller-only support for free-expansion tests. | Lit review BCs and D12 default. The confined and stack-pressure modes existed only for the D14 sweeps and are removed (D17). |
| P11 | Load stepping. Mandatory targets are the 10 frame values plus s*. Δs starts at 0.05, Δs_max = 0.05, Δs_min = `solver.ds_min` (default 0.1/64; remediation rung 1 lowers it, P25). On failure (SNES reason ≤ 0, NaN, or a PETSc exception), restore the last converged u and halve Δs. Below Δs_min the run has failed: `failed_at_s` = the attempted s, later frames are NaN, and the solve stops. On success with ≤ 5 Newton iterations, double Δs (capped). Predictor: u ← u_n + (Δs_try/Δs_last)(u_n − u_{n−1}) once two converged states exist, otherwise u_n. | D11 failure semantics. Linear extrapolation cuts Newton iterations, which dominate cost. |
| P12 | Nonlinear solver: dolfinx **0.10.0** (pinned by P20), API as verified by the Modal spike (`.claude/reports/fem-spike.implementer.md`, `scripts/modal_fem_spike.py`): `NonlinearProblem(F_res, u, bcs=bcs, petsc_options_prefix="fem_", petsc_options={...})` from `dolfinx.fem.petsc` with no `J` argument (the Jacobian is UFL's automatic derivative of `F_res`); `problem.solve()`; convergence from `problem.solver.getConvergedReason()` (> 0 converged) and `problem.solver.getIterationNumber()`. Component Dirichlet BCs follow the spike: `Vs, _ = V.sub(i).collapse()`, `locate_dofs_topological((V.sub(i), Vs), fdim, facets)`, `dirichletbc(zero_fn_on_Vs, dofs, V.sub(i))`. DG0 post-processing uses `fem.Expression(expr, V0.element.interpolation_points)` (a property in 0.10). Eigenstretch and modulus coefficients are DG0 Functions updated by assigning `x.array` before each `solve()` (the spike updated a `fem.Constant` the same way). Options: `snes_type newtonls`, `snes_linesearch_type` = `solver.linesearch` (default `bt`; the spike verified `basic`; `bt` is chosen for the large Si eigenstrains; ES1 and remediation rung 3 switch it), `snes_rtol 1e-8`, `snes_atol 1e-10`, `snes_max_it 25`, `ksp_type preonly`, `pc_type lu`, `pc_factor_mat_solver_type mumps`, `mat_mumps_icntl_14 100`. The run is serial: assert `MPI.COMM_WORLD.size == 1`. `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are set to the container CPU count before dolfinx/petsc4py are imported (the image defaults OPENBLAS_NUM_THREADS=1). | D5 (SNES + MUMPS). The spike proved this exact API on Modal (neo-Hookean with det-Fλ weighting, 5 steps, 4 Newton its/step, mean J exact, max|P| 1.7e-14). Serial keeps the pixel mapping trivial. Parallelism comes from running many cases at once on Modal. |
| P13 | Post-processing at each frame. Interpolate UFL expressions into DG0 at cell midpoints: total J = det F₂ and the Cauchy stress σ = (1/Je)[μ(Be − I) + λ ln Je I], Be = Fe Feᵀ (3×3). Components are sxx, szz, sxz and syy (out-of-plane). von Mises vm = sqrt(½[(sxx−szz)² + (szz−syy)² + (syy−sxx)²] + 3 sxz²). Pressure p = −(sxx+szz+syy)/3 is derived, not stored. Nodal u is mapped to corners: col = rint(x/h), row = H − rint(z/h). | Fields are on the reference (coarse) pixel grid, as D15 requires. Cell-centre values are exact for DG0 and need no averaging. |
| P14 | Storage. One `fields.npz` (np.savez_compressed) per (run_tag, orientation, site), on Modal volume `pmdb-fem-out` under `/fields/{tag}/{orientation}/{batch}__{site}.npz`. Keys: `labels` uint8 (H,W); `s` (11,); `converged` bool (11,); `u_nodes` float32 (11,H+1,W+1,2) as (ux, uz) in µm; `J, sxx, szz, sxz, syy, vm` float32 (11,H,W), in MPa except J. A small `result.json` (meta + feature rows) goes under `/results/{tag}/{orientation}/` and GIFs under `/gifs/{tag}/`. | Fields are kept so the tiers can be re-extracted (D8). Large fields are kept apart from the small results so `modal volume get` of the results stays small. |
| P15 | Feature regions (D9). 6 tiles per site with fixed edges `linspace(edge_px, W − edge_px, 7).round()`, edge_px = round(20 µm / h). Full height, no overlap. The site-level region is the union of the 6 tiles (central region), so site and tile metrics share one definition and exclude the lateral-BC bands. Cropped windows have no tiles, and their site region is all columns. | D9 tiling with a fixed tile count (D10 risk note). |
| P16 | Feature set per (region, frame). Tier 1 (17 named metrics, §Step 5). Tier 3: quantiles q ∈ {5,25,50,75,95,99} of {vm, p, J} over each of {si, gr, binder}, plus J over pore (60 metrics). Tier 4: depth-band (5 bands, reusing `pmdb.kpis.fields.band_profile` and `band_summary`) maxdev and absslope of vm and J over solid cells (4 metrics). Tier 5 (per-particle) is not built. Metric code is identical for tiles and site. | D8 tiers 1-4. Tier 4 is magnitude-only like K12 because the depth sign is unknown. Fields are stored, so tier 5 can be added later without new runs. Final D10 classifies each tile on its full feature vector and averages tile probabilities per site (separate plan); this needs exactly the per-tile 81-metric × 11-frame rows of §2.1. |
| P17 | Orientation combination. Both orientations are simulated. CSVs carry `orientation ∈ {bottom, top, sym}`. `sym` = the arithmetic mean of bottom and top per metric per (batch, site, [tile,] frame). `sym` is NaN if either is NaN, `converged` = both, `first_pore_closure_s` = min, `failed_at_s` = min; `s`, `heldout`, `tile_x0_um`, `tile_x1_um` are copied (identical in both). The primary downstream feature set is `sym`. | No foil is visible (docs/data-processing.md l.55, spec 002 "Directions"), so the collector side is unknown per site. The mean is invariant to a vertical flip of the image, so features cannot encode an arbitrary orientation guess. |
| P18 | Removed by D17 (sweeps and the Spearman robustness flag dropped). Number kept so later references stay stable. | D17. |
| P19 | GIF (D15). One per site, rendered in-container for orientation `bottom` only (labelled "collector assumed at bottom"). Single panel. The BSE (coarse block-mean of the un-harmonised uint8 BSE, grey, per-site p1-p99 stretch) is warped by displacement via `pcolormesh` on deformed corner coordinates. von Mises is overlaid on solid cells only (alpha 0.55, `magma`, LogNorm fixed at [1, 1e4] MPa, the same for every site and frame). Title: site, scenario, SOC %, swelling %. Axes fixed per site: x [−1, W+1] µm, z [−0.05H, 1.45H]. 11 frames, 500 ms, loop. Frames after a failure repeat the last converged frame with the banner "solver failed at SOC x%". Size ladder (900 px,128 colours) → (900,64) → (720,64) → (600,32); the first ≤ 2 000 000 bytes wins, otherwise `gif_error` is recorded. | D6/D15. The per-site p1-p99 stretch is affine-invariant, so the GIF background looks the same with or without harmonisation (P24); it is visual only and feeds no feature. |
| P20 | **Modal image (D16).** `modal.Image.from_registry("ghcr.io/fenics/dolfinx/dolfinx:v0.10.0")`, the tag verified by the Modal spike (Modal accepts it without `add_python`; python 3.12.3, dolfinx 0.10.0, petsc4py 3.24.0). Its registry config (checked at planning time) has a Python 3.12 venv at `/dolfinx-env` first on PATH, dolfinx under `/usr/local/dolfinx-real/lib/python3.12/dist-packages` via PYTHONPATH, PETSc 3.24.0 `PETSC_ARCH=linux-gnu-real64-32` built with `--download-mumps`, mpich, numpy 2.2.6 and g++ for the FFCx JIT. Then `run_commands`, in this order: (0) **numpy build-time assertion, before any install**: `python -c "import numpy, sys; v = numpy.__version__; sys.exit(0 if v == '2.2.6' else 'image numpy ' + v + ' != 2.2.6')"` (a non-zero exit fails the image build); (a) `pip install uv`; (b) `uv pip install --system numpy==2.2.6 scipy==1.14.1 scikit-image==0.25.2 pandas==2.2.3 matplotlib==3.10.3 pillow==11.3.0 pyyaml==6.0.2 tifffile==2025.5.10 imagecodecs==2025.3.30 pytest==8.3.5`. Then `.env({"PMDB_CACHE": "/data", "PYTHONPATH": "/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib", "PETSC_ARCH": "linux-gnu-real64-32"})`, then the local files: `pytest.ini`, `configs/fem/fem.yaml`, `tests/test_fem_unit.py`, `tests/test_fem_solver.py` (each `add_local_file` to the same relative path under `/root`), and `add_local_python_source("pmdb")`. Volumes: `pmdb-data` read-only at `/data` (`/data/half` labelled, `/data/heldout/half` held-out, already uploaded); `pmdb-fem-out` (`create_if_missing=True`) at `/out`. App name `pmdb-fem`. `retries=0` on every function. | D16 requires the official image with a pinned tag; this exact tag and install recipe (`pip install uv`, `uv pip install --system`) passed the spike. Step (0) proves numpy 2.2.6 is the image's own version, so the `numpy==2.2.6` pin in (b) is a no-op and uv cannot move numpy under the compiled dolfinx/petsc4py (review r2 comment 2: the probe alone could not detect this). `--system` resolves to the first python on PATH, `/dolfinx-env/bin/python`, the interpreter Modal runs; `--mode unit` (which imports every dep) confirms it, and ES5 covers a switch to `--python /dolfinx-env/bin/python`. 0.11.0 exists but 0.10.x was requested and P12's API is the 0.10 one. scipy/scikit-image match the versions that produced `outputs/kpis/*` (modal_app.py, default python), so segmentation masks match the KPI pipeline (P24). pandas 2.2.3 is the oldest line with numpy-2 wheels. `retries=0` because the P29 orchestrator performs the single retry itself, with 1.5× timeout and memory. |
| P21 | **Cost accounting.** `cost_usd = (wall_s + 120)/3600 × (cpu × 0.0472 + mem_GiB × 0.0080)` (Modal's published $0.0000131 per core-second and $0.00000222 per GiB-second; the +120 s covers container start, image load and JIT outside the measured wall; check the rates on modal.com/pricing in Step 4 and update the constants if they differ). `wall_s` is measured inside the remote function from entry to return. Local ledger `outputs/modal/fem/ledger.csv` (gitignored), columns `utc, mode, tag, batch, site, orientation, attempt, cpu, memory_mb, timeout_s, wall_s, cost_usd, status` (`status` ∈ ok, error, lost). `probe` and `unit` rows are appended by the entrypoint directly. Case rows (every attempt) and one `orchestrator` row per `run_cases` call are appended by `sync(tag)` from the volume status files (P29), deduplicated on (tag, status file, batch, site, orientation, attempt), so rerunning sync never double-counts. A lost attempt (timeout, OOM kill, exception) is charged at `wall_s = timeout_s`. | One ledger covers test iterations and production, as the cap requires. Building case rows from volume status files keeps the ledger correct when a detached run outlives the local client. Charging lost calls at the timeout keeps it an upper bound. |
| P22 | **Budget** (HARD CAP $180, the only hard stop under FULL AUTONOMY). `DEV_ALLOWANCE_USD = 5.0` reserves image builds, orchestrator containers and anything billed outside function wall time. `worst(calls)` = Σ (timeout_s + 120)/3600 × rate. `expected(cases)` = Σ 1.25 × c_ref × (n_cells_site/n_cells_ref)^1.5, where c_ref/n_cells_ref = the mean actual per-case cost and cells of `full` cases already finished if any exist, else of the successful benchmark cases at the launch cpu; `n_cells_site = (height//2)·(width//2)` from the manifests. R (remaining planned spend) = expected(production chunks not yet launched, including the one about to launch) + worst(the 2 Stage-2 windows at cpu 4, 16 GiB, timeout 7200) unless Stage 2 is done or dropped. Rules, evaluated by the local entrypoint before every launch: (a) probe, unit, bench, and window tags not starting `stage2`: exit 3 if ledger_total + worst(calls) + DEV_ALLOWANCE > 180. (b) `full` with tag `full`: if ledger_total + R + DEV_ALLOWANCE > 180, set `drop_top` (the top chunk leaves R; launching top while `drop_top` exits 0 with `top dropped by budget`); recompute; if still > 180 set `drop_stage2`; recompute; if still > 180 exit 3. (c) tag `full_rerun`: if ledger_total + expected(rerun cases) + R + DEV_ALLOWANCE > 180, exit 0 with `reruns skipped by budget` (rung 4 applies). (d) window tags starting `stage2`: exit 0 with `stage2 dropped by budget` if `drop_stage2`, else rule (a). Decisions persist in `outputs/modal/fem/budget.json` `{"drop_top": bool, "drop_stage2": bool, "log": [str]}` and are never reverted. One exponent (1.5) everywhere, including the Stage-3 window → full-site extrapolation. | FULL AUTONOMY cost rule: re-project before each chunk; drop `top`, then the 50 nm check, and stop only if even that exceeds $180. The 1.5 exponent is the 2D nested-dissection LU flop scaling; the 1.25 factor and per-chunk re-projection bound the overshoot to one chunk. |
| P23 | Provenance follows the `scripts/run_kpis.py` pattern: git commit, command, package versions (dolfinx, petsc4py, numpy, scipy, scikit-image), image tag, params_hash and the full params dict in each `result.json` and in `outputs/fem/run_log.json`, which also records the ledger total, per-mode subtotals, `cost_model` (`cpu_rate`, `mem_rate`, `overhead_s`, `cap_usd`, `exponent` = 1.5) and `production` (`cpu`, `memory_mb`, `timeout_s` of the `full` cases). | Same reproducibility pattern as the KPI tables. |
| P24 | **Segmentation input: un-harmonised, identical to the KPI pipeline.** `run_case` loads `load_site(batch, site, resolution="half", normalise="none", cache_root=…)` with the default `harmonise="none"` and calls `segment(site)` unchanged. It does NOT use `harmonise="hybrid"`. | (1) D4 binds the input to `segment(site)` on `cache/half`, and spec 002 D-011 / `pmdb/segment.py` define it on un-normalised BSE with affine-invariant percentile-anchored thresholds; `docs/harmonisation.md` §4.3 states the segmenter "is unaffected by any method". (2) Measured at planning time, `segment()` on `harmonise="hybrid"` vs `"none"` differs on 0.035% / 0.050% / 0.137% / 0.038% of pixels for 71vgq3fw (strong) / kbdh4tri (strong) / 9luzk4jm (mild) / fzrt2k6r (B1), with every phase fraction within 0.001: uint8 rounding noise, not a material change. (3) The KPI features that join the FEM features downstream (`outputs/kpis/site_kpis.csv`, `tile_kpis.csv`) were segmented exactly this way (`scripts/run_kpis.py:218`, `modal_app.py`), so the FEM mesh and the KPIs share identical masks; switching would introduce a gratuitous mask mismatch for no gain. (4) FEM features depend only on the label map and geometry, never on grey levels, so the Batch-3 grey-level artefact cannot leak into them; the only grey-level use is the GIF background (P19), which is visual. Guards: the Step-3 data test asserts the harmonisation invariance on the four strong sites plus one mild site, and the Step-4 probe asserts that in-image segmentation equals local segmentation on `Batch_3/vc2whyaq` plus the four strong Batch-3 sites (the image runs numpy 2.2.6 / Python 3.12, the KPI pipeline ran numpy 1.26.4 / Python 3.10). |
| P25 | **Validation gates and failure policy** (FULL AUTONOMY; numbers fixed here). Gates: **G1** = the 9 solver tests (T1a is the user's free-expansion test, J = Jλ, zero stress, exact) plus all unit/data tests; **G2** = the 50 nm resolution check (Step 9); **swelling stop window** [0.03, 0.39] on swelling_sym(site region, s = 1) for the Stage-4 window (G4w), the benchmark, and the production median (G4); **G5** = sites with `failed_at_s` < 1 in either orientation ≤ 3. The **literature band** [0.09, 0.39] (VT1) is reported only, as `lit_band_ok`; VT2/VT4/VT5 are reported only. Policy on failure, nothing waits for the user: (1) G1 or any test failure is a bug: escalate to the dispatcher with the failing values and the pytest.txt path for a fix round, then re-run. (2) Swelling window failure at the Stage-4 window or the benchmark is a bug: escalate to the dispatcher with diagnostics (per-phase J means at s = 1, `phase_properties(1.0, p)` stretches, mean free-edge uz sign per orientation, result JSON paths); if the dispatcher returns "no bug found" after one investigation round, continue. G4 failure in production does not stop: continue to Step 10, the results.md `flags` block records it, and the final hand-back lists it as a suspected bug. (3) G2 failure: continue at 100 nm; flagged. (4) Non-convergence, **remediation ladder**, applied globally and logged in `fem.yaml` `remediation`: rung 1 `solver.ds_min` 0.1/64 → 0.1/512; rung 2 `pore.E_rel_binder` 1e-4 → 1e-3; rung 3 switch `solver.linesearch` (bt ↔ basic); rung 4 accept NaN frames after the failure SOC. Rungs are climbed only on the Stage-3 window, before production, because rung 2 changes the physics and must be identical for all sites. In production, each case with finite `failed_at_s` < 1 gets one physics-neutral rerun (tag `full_rerun`, ds_min ÷ 8 and the other line search); if it still fails, rung 4. G5 > 3 is reported and flagged, never a stop. (5) Modal infra errors and timeouts: P29 retry, then the case is recorded as missing (§2.1 deviations) and the run continues. (6) Cost: P22. | FULL AUTONOMY section of `fem-swelling.md`, plus the amended swelling gate. The ladder is applied pre-production so features stay comparable across sites; production reruns only use rungs that do not change the converged solution. |
| P26 | **Modal-only workflow (D16), detached.** All dolfinx code is exercised only through `modal run modal_fem.py --mode <m>` from the repo root with the default python's `modal` client. Modes: `probe`, `unit` (attached, short), `window`, `bench`, `full` (case modes, always launched as `modal run --detach ...`), and `sync` (local only: pulls a tag's results, status files and GIFs from the volume, writes ledger rows, prints the summary). Locally written (gitignored) under `outputs/modal/fem/`: `tests/<UTC>/pytest.txt`, `probe/<UTC>.json`, `results/{tag}/{orientation}/{batch}__{site}.json`, `status/{tag}/*.json`, `gifs/{tag}/`, `ledger.csv`, `budget.json`. A case-mode entrypoint blocks on the orchestrator, then runs `sync(tag)` itself and prints `SYNC DONE <tag>`; if the local process ends before that line, poll the volume and run `--mode sync --tag <tag>` (never relaunch). Exit codes: 1 test failure or case still failing after retry, 3 cap refusal, 4 swelling gate failure (window/bench only). Pure-numpy modules are tested locally with `pytest` in the default python and again inside the image by `--mode unit`. | Test env = production env (D16). `--detach` keeps runs alive if the laptop disconnects (FULL AUTONOMY); sync makes local state reproducible from the volume. |
| P27 | **Output contract.** `outputs/fem/site_curves.csv` and `outputs/fem/tile_curves.csv` follow §2.1 exactly. Nothing else in this plan may change their columns, keys, value sets or row counts. | A parallel classifier plan (final D10) builds against this schema. |
| P28 | **Pitch-deck docs (D19).** Two docs, `docs/fem/method.md` and `docs/fem/results.md`, each a hand-written skeleton (headings and number-free prose) with generated blocks delimited by `<!-- AUTO:name -->` … `<!-- /AUTO:name -->`. One script, `scripts/fem_docs.py`, fills the blocks: `--method` fills method.md, `--results` fills results.md and writes the PNG figures to `outputs/fem/figures/`. **Rule**: in these two docs, every parameter value, result, cost, count, percentage or threshold sits inside an AUTO block; hand text outside the blocks contains no numbers other than identifiers (D17, P25, G4, VT2, section numbers). A block whose inputs do not exist yet is filled with the single line `_pending: generated after the full run_`. The parameter table is rendered from `load_params()` and its Source / Evidence-strength cells are copied at runtime from the `## 1. Final parameter table` in `docs/fem/literature-review.md` by exact match on the Parameter cell, with a link to `literature-review.md#1-final-parameter-table`. `docs/fem/README.md` stays operational (commands, volume layout, ledger mechanics, links) and is not subject to the rule. | D19 requires every number to be script-generated and the docs to one-shot a deck; markers let prose be hand-written while numbers cannot drift. Copying citations from the lit review at runtime keeps one source of truth. Lean: one script, two modes, three figures. |
| P29 | **Remote orchestrator and retry.** `run_cases` (cpu 0.25, memory 1024, timeout 86400, volume `/out`) runs on Modal: (1) `run_case_remote.with_options(cpu, memory, timeout).map(cases, return_exceptions=True)`; (2) every case that raised or returned `meta.error` is retried once with timeout = min(86400, ceil(1.5·timeout)) and memory = ceil(1.5·memory/1024)·1024; (3) a case still failing gets a stub `/out/results/{tag}/{orientation}/{batch}__{site}.json` = `{"meta": {batch, site, orientation, heldout, "error": msg, "status": "error" or "lost"}, "site_rows": [], "tile_rows": []}`; (4) it writes `/out/status/{tag}/{UTC}.json` (one record per attempt: batch, site, orientation, attempt, cpu, memory_mb, timeout_s, wall_s, status; plus the orchestrator's own wall_s) after the first pass and again after the retry, commits the volume and returns the status. `run_case_remote` writes result.json, fields and GIF only on success. Every expected case therefore ends with exactly one result.json, so a poll can count them: `until [ "$(modal volume ls pmdb-fem-out /results/<tag>/<orientation> 2>/dev/null \| grep -c '\.json')" -ge <N> ]; do sleep 300; done; echo POLL DONE` (run it in the background). Production timeout is capped at 28800 s so first attempt + retry (2.5×) fits the orchestrator's 86400 s. | FULL AUTONOMY: "retry once per case with 1.5× timeout; then record as failed and continue"; long runs detached with progress polled from the volume. Server-side orchestration survives a lost local client; the stub result makes the poll terminate. |

### 2.1 Output contract: `site_curves.csv` and `tile_curves.csv` (P27)

Both files: one header row, comma-separated, floats written with `float_format="%.6g"`, booleans as `True`/`False`, NaN as an empty field (pandas default). They contain the 31 labelled sites and the 3 held-out sites (34 sites), production tag `full` only. Rows are sorted by `batch`, `site`, `orientation` in the order (bottom, top, sym), then `tile` (tile file only), then `frame`. Consumers must select metric columns by name, not by position.

Key and metadata columns, in this order:

| column | type | values / meaning |
|---|---|---|
| `batch` | str | `Batch_1`, `Batch_2`, `Batch_3`, `Batch_heldout` |
| `site` | str | site id, e.g. `vc2whyaq` |
| `heldout` | bool | `True` iff `batch == "Batch_heldout"` |
| `orientation` | str | `bottom` (collector at image last row), `top` (collector at image row 0), `sym` (P17 mean of the two; primary feature set) |
| `tile` | int | **tile_curves only**: 0..5, left to right in image columns |
| `tile_x0_um`, `tile_x1_um` | float | **tile_curves only**: tile column bounds in µm from image column 0 on the 100 nm grid (P15) |
| `frame` | int | 0..10 |
| `s` | float | SOC = frame/10 (0.0, 0.1, …, 1.0) |
| `converged` | bool | frame solved (sym: both orientations solved) |
| `failed_at_s` | float | SOC at which the solve failed, NaN if it reached s = 1 (sym: min); same on every row of a (site, orientation) |
| `first_pore_closure_s` | float | first frame s with any PORE cell J < 0.1 over the whole domain, NaN if never (sym: min); same on every row of a (site, orientation) |

Then the 81 metric columns (§Step 5 definitions), in this order:
1. Tier 1 (17): `swelling, surface_rough, sxx_mean_MPa, syy_mean_MPa, porosity, porosity_change, porosity_rel_change, J_si_mean, J_gr_mean, J_binder_mean, vm_si_p50_MPa, vm_si_p95_MPa, vm_gr_p95_MPa, vm_binder_p95_MPa, p_si_mean_MPa, si_yield_frac, pore_closed_frac`.
2. Tier 3 (60): for field in (`vm`, `p`, `J`), for phase in (`si`, `gr`, `binder`), for q in (5, 25, 50, 75, 95, 99): `q{q}_{field}_{phase}` (54 columns, e.g. `q5_vm_si`), then `q5_J_pore, q25_J_pore, q50_J_pore, q75_J_pore, q95_J_pore, q99_J_pore`. `vm` and `p` in MPa; `J` dimensionless.
3. Tier 4 (4): `band_vm_maxdev, band_vm_absslope, band_J_maxdev, band_J_absslope`.

Metric NaN rules: unconverged frame → all 81 NaN; empty phase in the region → that phase's metrics NaN; `porosity_rel_change` NaN if porosity at frame 0 is 0; `sym` NaN if either orientation is NaN.

Unique keys and row counts:
- `site_curves.csv`: key (`batch`, `site`, `orientation`, `frame`); region = union of the 6 tiles (P15); 34 × 3 × 11 = **1122 rows**; 9 key/meta columns (`batch, site, heldout, orientation, frame, s, converged, failed_at_s, first_pore_closure_s`; no `tile`, `tile_x0_um`, `tile_x1_um`) + 81 metrics = **90 columns**.
- `tile_curves.csv`: key (`batch`, `site`, `orientation`, `tile`, `frame`); 34 × 3 × 6 × 11 = **6732 rows**; 12 key/meta columns + 81 metrics = **93 columns**.

No `config`, `region` or `tag` column exists in either file.

**Allowed deviations under FULL AUTONOMY** (recorded in `run_log.json`, flagged in `docs/fem/results.md`):
- A case with no successful result after its retry and rerun still has all its rows: `converged = False`, `failed_at_s = 0.0`, `first_pore_closure_s` NaN, all 81 metrics NaN, tile bounds from the manifest width. Row counts are unchanged; the case is listed in `run_log.json["missing_cases"]`.
- Frames after a solver failure are NaN (remediation rung 4); the classifier imputes in-fold.
- If the budget drops orientation `top` (P22): no `top` rows, `sym` rows are copies of the `bottom` rows, and row counts become site_curves 34 × 2 × 11 = **748** and tile_curves 34 × 2 × 6 × 11 = **4488**; `run_log.json["orientations_run"] == ["bottom"]` (otherwise `["bottom", "top"]`).

## 3. Out of scope (do NOT touch)

- `data/`, `data_heldout/`: never write. Held-out sites are simulated for features only and are flagged `heldout=True` in every CSV.
- `pmdb/io.py`, `pmdb/segment.py`, `pmdb/harmonise.py`, `pmdb/kpis/**`, `modal_app.py`, `requirements.txt`, `cache*/` (including `cache/harmonised/`), and the existing tests: read or import only, never modify.
- No local FEniCSx/dolfinx/conda environment and no `environment-fem.yml` (D16). No local Docker.
- No use of `harmonise=` other than `"none"` in the FEM pipeline (P24); the Step-3 data test is the only place that loads `harmonise="hybrid"`, read-only.
- D17 removals: no SiOx configuration or run, no D14 robustness sweeps, no Spearman robustness flag, no Shah E law, no isotropic-graphite strain, no proportional SOC split, no confined or stack-pressure boundary condition, no named configs or `config` column.
- D19 parts owned elsewhere: `docs/classifier/*` and `docs/pitch-brief.md` (classifier plan and writer lane). This plan writes only `docs/fem/{README,method,results}.md`, `scripts/fem_docs.py` and `outputs/fem/figures/*.png`.
- The classifier and site pooling (final D10, D18; separate plan), any LOSO, recomputing KPIs on the new tile grid, tier-5 per-particle features, viscoplasticity, electrochemistry, generalised plane strain, periodic BCs.
- Never merge any PR. The only PR action is opening `fem-sim → main` in Step 10.
- No Modal launch other than the modes and run lists in Steps 4 and 7-10.

## 4. Steps

### Step 1 — Config file and config loader (local)

Targets: `configs/fem/fem.yaml`, `pmdb/fem/__init__.py` (docstring only), `pmdb/fem/config.py`, `tests/test_fem_unit.py` (new).

- `configs/fem/fem.yaml`, with exactly this content (comments optional):

```yaml
version: 1
mesh: {res_nm: 100.0}
soc: {frames: 11, s_star: 0.25, si_ratio: [0.96, 0.58], gr_ratio: [0.04, 0.42], u_max: 0.80, y_max: 0.91}
si: {beta: 2.8, E_MPa: [96000.0, -55000.0], nu: [0.29, -0.04], yield_MPa: [3000.0, 3150.0], x_per_u: 3.75}
graphite: {eps_c_points: [[0.0, 0.0], [0.25, 0.055], [0.5, 0.055], [1.0, 0.103]],
           eps_a_over_c: [0.01, 0.103], E_MPa: [32000.0, 77000.0], nu: [0.32, -0.08]}
binder: {E_MPa: 500.0, nu: 0.34}
pore: {E_rel_binder: 1.0e-4, nu: 0.3, closure_J: 0.1}
solver: {ds_max: 0.05, ds_min: 0.0015625, grow_if_its_le: 5, snes_rtol: 1.0e-8, snes_atol: 1.0e-10,
         snes_max_it: 25, quadrature_degree: 2, linesearch: bt}
remediation: []   # P25 ladder: one string appended per applied rung
features: {n_tiles: 6, edge_um: 20.0, quantiles: [5, 25, 50, 75, 95, 99], n_depth_bands: 5}
gif: {width_px: 900, ladder: [[900, 128], [900, 64], [720, 64], [600, 32]], vm_range_MPa: [1.0, 10000.0],
      frame_ms: 500, max_bytes: 2000000, alpha: 0.55}
gates: {swelling_stop: [0.03, 0.39], swelling_lit_band: [0.09, 0.39], max_failed_sites: 3}
```

- `pmdb/fem/config.py`:

```python
FEM_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "fem" / "fem.yaml"   # same pattern as CATALOGUE_PATH
def load_params(path: Path = FEM_CONFIG_PATH) -> dict          # yaml.safe_load of the file; no merging, no named configs (D17)
def params_hash(p: dict) -> str                                # sha1(json.dumps(p, sort_keys=True)).hexdigest()[:12]
```

Verification: `pytest -q tests/test_fem_unit.py -k config` in the default python. These tests must pass:
- `load_params()["gates"]["swelling_stop"] == [0.03, 0.39]`;
- `set(load_params()) == {"version", "mesh", "soc", "si", "graphite", "binder", "pore", "solver", "remediation", "features", "gif", "gates"}`;
- `params_hash(load_params())` is 12 hex characters and equal on two calls.

### Step 2 — Materials and SOC driver (pure numpy, local)

Target: `pmdb/fem/materials.py`; tests in `tests/test_fem_unit.py`.

```python
BINDER, SI, GRAPHITE, PORE, ARTEFACT = 0, 1, 2, 3, 4
PHASE_NAMES = {BINDER: "binder", SI: "si", GRAPHITE: "gr", PORE: "pore", ARTEFACT: "artefact"}
SOLID = (SI, GRAPHITE, BINDER)
@dataclass(frozen=True)
class PhaseProps:
    E: float            # MPa
    nu: float
    stretch: tuple[float, float, float]   # (lambda_x, lambda_z, lambda_y)
def soc_fractions(s: float, p: dict) -> tuple[float, float]          # (f_si, f_gr) per P9 (Yao split only)
def lithiation_state(s: float, p: dict) -> tuple[float, float]      # (u, y) = (u_max f_si, y_max f_gr)
def graphite_strains(y: float, p: dict) -> tuple[float, float]       # (eps_a, eps_c)
def phase_properties(s: float, p: dict) -> dict[int, PhaseProps]     # all 5 labels, per P7/P8
def si_yield_MPa(u: float, p: dict) -> float                         # P9 formula
def lame(E: float, nu: float) -> tuple[float, float]                 # (mu, lam)
```

Verification: `pytest -q tests/test_fem_unit.py -k materials` in the default python. Tests that must pass:
- (a) Every row of the lit-review table at s = 0, 0.1, …, 1.0 for f_Si, f_Gr, u, y, J_Si and ε_c matches the table to abs 1.5e-3. Hard-code the 11 rows from `docs/fem/literature-review.md` §"Overall SOC"; the table is rounded to 3 decimals.
- (b) `si_yield_MPa(0.0, p) == 3000` and `si_yield_MPa(1.0, p)` = 3000 − 3150·3.75/4.75 ≈ 513.16 (abs 0.01).
- (c) Si stretch at s = 1 cubed = 3.24 (abs 2e-3).
- (d) Pore E = 0.05 MPa under defaults.

### Step 3 — Geometry: labels, coarsening, crop; segmentation-input guard (pure numpy, local)

Target: `pmdb/fem/geometry.py`; tests in `tests/test_fem_unit.py` and a new `tests/test_fem_data.py` (module-level `pytestmark = pytest.mark.data`).

```python
def labels_from_masks(m: Masks) -> np.ndarray                 # uint8; BINDER default, then SI, GRAPHITE, PORE, ARTEFACT
def coarsen_labels(labels: np.ndarray, factor: int = 2,
                   priority: tuple[int, ...] = (SI, PORE, GRAPHITE, BINDER, ARTEFACT)) -> np.ndarray
    # drop trailing rows/cols so shape divisible; counts per label per block; argmax(counts*10 + bonus), bonus = len-1-rank
def coarsen_image(img: np.ndarray, factor: int) -> np.ndarray  # float32 block mean, same trailing drop
def central_cols(width_px: int, crop_um: float, nm_per_px: float) -> slice  # even width n = 2*floor(crop_um*1000/nm/2), even start c0 = 2*((width_px-n)//4)
```

factor = round(res_nm / site.nm_per_px). factor 1 returns the input unchanged.

`tests/test_fem_data.py` contains two tests:
- `test_coarsening_fractions`: on `Batch_3/vc2whyaq` (half, `normalise="none"`), coarse/fine area ratio − 1 must be within ±0.20 for Si and ±0.05 for graphite (measured at planning: +0.019 and −0.002). Print the pore and binder ratios (reported, not asserted; measured +0.072 and −0.22).
- `test_segmentation_harmonisation_invariant` (P24 guard): for each of `Batch_3/71vgq3fw, Batch_3/kbdh4tri, Batch_3/tuy3zymq, Batch_3/x7u69zsw, Batch_3/9luzk4jm`, compute `labels_from_masks(segment(load_site(b, s, resolution="half", normalise="none", harmonise=h)))` for h ∈ {"none", "hybrid"}. Assert the fraction of differing pixels ≤ 0.005 and every per-label area fraction differs by ≤ 0.002 (abs). Print the per-site numbers (measured at planning: ≤ 0.0014 and ≤ 0.001).

Verification:
- `pytest -q tests/test_fem_unit.py -k geometry` (default python). The block cases must pass: [SI,SI,GR,GR]→SI, [SI,GR,GR,B]→GR, [SI,P,GR,B]→SI, [P,P,GR,GR]→P, [GR,GR,B,B]→GR, [A,A,B,B]→B, and a 5×7 input gives a 2×3 output.
- `pytest -q -m data tests/test_fem_data.py` (default python): 2 passed. A failure of either test is a bug (it would invalidate P2 or P24): escalate to the dispatcher for a fix round (P25).

### Step 4 — Solver, Modal image, probe, analytic tests on Modal (Stage 1, gate G1)

Targets: `pmdb/fem/result.py` (pure numpy), `pmdb/fem/solver.py` (dolfinx), `tests/test_fem_solver.py` (new; module-level `pytest.importorskip("dolfinx")`), `modal_fem.py` (new: image, volumes, ledger, cap check, `probe` and `unit` modes; the case modes are added in Steps 7-8).

`result.py`:

```python
FIELD_KEYS = ("J", "sxx", "szz", "sxz", "syy", "vm")
@dataclass
class SimResult:
    labels: np.ndarray; px_um: float; s: np.ndarray; converged: np.ndarray
    u_nodes: np.ndarray            # (n, H+1, W+1, 2) float32, (ux, uz), uz positive toward image row 0
    fields: dict[str, np.ndarray]  # FIELD_KEYS -> (n, H, W) float32, NaN for unconverged frames
    failed_at_s: float             # nan if none
    substeps: list[dict]           # {s, ds, its, reason, ok, wall_s}
    wall_s: float
def save_npz(r: SimResult, path: Path) -> None ; def load_npz(path: Path) -> SimResult   # P14 keys; substeps/wall_s go to result.json, not npz
```

`solver.py`:

```python
@dataclass(frozen=True)
class BCSpec:
    orientation: Literal["bottom", "top"]
    lateral: Literal["both", "left"] = "both"      # "left" = test-only roller support
def simulate(labels: np.ndarray, px_um: float, props_fn: Callable[[float], dict[int, PhaseProps]],
             bc: BCSpec, solver_opts: dict, frames: np.ndarray, extra_targets: Sequence[float] = (),
             log: Callable[[dict], None] | None = None) -> SimResult
```

Solver implementation requirements. The algorithms are fixed by the design decisions; mechanics are up to the implementer.
- Mesh and cell→pixel map: P4. DG0 dof for cell c is `V0.dofmap.list[c, 0]`. Assert the pixel map is a permutation of range(H·W).
- DG0 coefficients: `mu, lam, lx, lz, ly`. Before every solve, set them from `props_fn(s_try)` by phase via precomputed DG0-dof index arrays per label.
- Energy, residual and Jacobian: P6, with `dx` metadata `{"quadrature_degree": solver_opts["quadrature_degree"]}`.
- BCs: P10. The free edge has no term (traction-free). The solve at s = 0 is always performed; it converges in 0 iterations.
- Stepping and failure: P11, with substep targets = sorted(set(frames) ∪ extra_targets). `run.py` passes `extra_targets=(s_star,)`.
- Solver options: P12. Read rtol/atol/max_it from `solver_opts`. Detect NaN residuals as failure (SNES reason `DIVERGED_FNORM_NAN` or a non-finite `u.x.array`).
- Frame outputs: P13 into float32 arrays. Unconverged frames are all NaN, including u_nodes.
- `log(dict)` is called once per substep attempt with the substep record.

Analytic tests in `tests/test_fem_solver.py`, exactly these 9 test functions. Small meshes; the module should run in under 3 min on 2 CPUs.
- `test_t1a_free_eigenstretch` (**the user's free-expansion test, J = Jλ, zero stress**): 6×4 px all one label, custom props with stretch (1.2, 1.1, 1.0), E = 1000, ν = 0.3, `lateral="left"`, orientation bottom, frames [0, 1]. At frame 1: J = 1.32 (rel 1e-8) in every cell, and max |σ| < 1e-6 MPa. (λy = 1, so plane strain imposes no out-of-plane constraint and this is exact free expansion.)
- `test_t1b_si_plane_strain`: same mesh. `props_fn(s)` gives every label Si properties at u = 0.8·s, so frame 1 has u = 0.8 (λ = 3.24^(1/3), E = 52 000, ν = 0.258). Closed form: in-plane σ = 0 ⇒ μ(f²−1) + λ_L ln(f²/λ_y) = 0 for the in-plane elastic stretch f. Solve with `scipy.optimize.brentq` on f ∈ (0.5, 2). Expected: J = λ²·f², sxx = szz = 0 (abs 1e-6·E), syy = (μ(1/λ²−1) + λ_L ln Je)/Je with Je = f²/λ (rel 1e-6).
- `test_t1c_graphite_plane_strain`: same as T1b with graphite at y = 0.91. Here fe = in-plane elastic stretch with μ(fe²−1) + λ_L ln(fe²/λy) = 0, J = λx λz fe² (rel 1e-6), and sxx = szz = 0.
- `test_t2_bilayer_small`: 8 px wide × 10 px tall. Image rows 0-4 (top, layer B) carry isotropic eigenstretch 1+ε* with ε* = 1e-4. Rows 5-9 (layer A) have stretch 1. Both layers E = 1000, ν = 0.3, `lateral="both"`, orientation bottom, free top. Expected in B: sxx = syy = −E ε*/(1−ν) (rel 1e-3), szz ≈ 0 (abs 1e-3·|sxx|). In A: all |σ| < 1e-6·|sxx_B|. Top-edge mean uz = 5·h·ε*(1+ν)/(1−ν) (rel 1e-3).
- `test_t2_bilayer_finite`: the same with ε* = 1e-2; same signs, magnitudes within rel 3e-2.
- `test_t3_orientation_top`: `test_t2_bilayer_small` with orientation top. B is now adjacent to the collector; the free edge is the bottom node row. Expect u_nodes[1, H, :, 1] mean = −5·h·ε*(1+ν)/(1−ν) (rel 1e-3), the same stress in B, and zero stress in A.
- `test_t4_pixel_mapping`: a 7×9 label map with a single SI pixel at (row 1, col 6), the rest BINDER. `props_fn = lambda s: phase_properties(s, load_params())`, frames [0, 0.2], lateral both, orientation bottom. At frame 1, the pixel with the largest vm is (1, 6), and `result.labels` equals the input.
- `test_t5_failure_path`: T1b geometry with solver_opts snes_max_it = 1, ds_min = 0.05, frames linspace(0,1,11). Expect `failed_at_s` finite and ≤ 0.1, `converged[0]` True, every later frame False with NaN fields.
- `test_t6_soft_pores`: 12×12 px. A 4×4 SI block in the centre, a 1-px PORE ring around it, BINDER outside. Default props (`phase_properties(s, load_params())`), lateral both, orientation bottom, free top, frames [0, 0.25, 0.5, 0.75, 1.0]. Asserted: `converged[1]` True and min J over PORE cells at frame 1 < 0.9. Report-only (no assertion): print one line `T6_REPORT ` + `json.dumps({"failed_at_s": ..., "min_pore_J": [per frame], "n_substeps": ...})`. The closure flag itself (J < 0.1) is computed only in `features` (Step 5).

`modal_fem.py` (this step):
- Image and volumes exactly per P20, including the step-(0) numpy assertion before `pip install uv`. Constants `DOLFINX_IMAGE = "ghcr.io/fenics/dolfinx/dolfinx:v0.10.0"`, `PY = "python"` (the image's PATH python, `/dolfinx-env/bin/python`), `CPU_RATE = 0.0472`, `MEM_RATE = 0.0080`, `OVERHEAD_S = 120`, `DEV_ALLOWANCE_USD = 5.0`, `CAP_USD = 180.0`, `LEDGER = outputs/modal/fem/ledger.csv` (repo-relative), `PROBE_SITES = [("Batch_3", "vc2whyaq"), ("Batch_3", "71vgq3fw"), ("Batch_3", "kbdh4tri"), ("Batch_3", "tuy3zymq"), ("Batch_3", "x7u69zsw")]`.
- Helpers (local side): `cost_usd(wall_s, cpu, memory_mb) -> float` (P21), `ledger_total() -> float`, `append_ledger(rows: list[dict])`, `budget_check(mode: str, tag: str, items: list, cpu: float, memory_mb: int, timeout_s: int) -> str` (P22; returns `"go"` or `"skip"` (the caller prints the reason and exits 0), or raises `SystemExit(3)`; reads and updates `budget.json`).
- `@app.function(cpu=1.0, memory=4096, timeout=1800, volumes={"/data": pmdb-data read-only}) def probe() -> dict`: returns python, dolfinx, petsc4py, numpy, scipy, scikit-image versions; `PETSc.Sys.hasExternalPackage("mumps")`; `str(inspect.signature(dolfinx.fem.petsc.NonlinearProblem.__init__))`; `dolfinx.fem.petsc.NonlinearProblem.solve.__doc__`; `labels`: a dict `"{batch}/{site}" -> labels_from_masks(segment(load_site(batch, site, resolution="half", normalise="none", cache_root="/data")))` for every entry of `PROBE_SITES`; wall_s.
- `@app.function(cpu=2.0, memory=4096, timeout=1800) def run_unit_tests(k: str = "") -> dict`: sets `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS="2"` in the subprocess env and runs `[PY, "-m", "pytest", "-q", "-rP", "tests/test_fem_unit.py", "tests/test_fem_solver.py"] + (["-k", k] if k else [])` with `cwd="/root"`; returns `{"returncode", "stdout", "stderr", "wall_s"}` (stdout/stderr truncated to their last 200 000 characters).
- `@app.local_entrypoint() def main(mode: str, k: str = "", ...)` (later steps add the case parameters). `probe`: budget check, call, write `outputs/modal/fem/probe/<UTC>.json` (labels excluded, per-site differing fractions included), compute the same labels locally with the default python for every `PROBE_SITES` entry and print the fraction of differing pixels per site; exit 1 if dolfinx ≠ 0.10.0 or mumps is False; differing fractions > 1e-4 are printed as `WARN` and do not change the exit code (ES3). `unit`: budget check, call, write stdout+stderr to `outputs/modal/fem/tests/<UTC>/pytest.txt`, print the last 40 lines and every `T6_REPORT` line, exit with the pytest return code. Both append a ledger row.

Before the first launch (one-off, read-only on local data):
- `modal volume ls pmdb-data /half` must list 31 npz + `manifest.csv` (at planning time `/half` exists). If `/half` is missing, run `modal volume put pmdb-data cache/half /half`.
- `modal volume ls pmdb-data /heldout/half` must list `Batch_heldout__{3e122cbj,fn0mhxef,xrv9xvzb}.npz` + `manifest.csv` (uploaded by the coordinator on 2026-10-03). Only if missing: `modal volume put pmdb-data cache_heldout/half /heldout/half` (reads only `cache_heldout/`, never `data_heldout/`).
- Check the P21 rates on modal.com/pricing and update the constants if they differ.

Verification (Stage 1):
- `pytest -q tests/test_fem_solver.py` in the default python: the module is skipped, not errored.
- `modal run modal_fem.py --mode probe`: the image builds (the numpy assertion passes) and the run exits 0; prints dolfinx 0.10.0, mumps True, numpy 2.2.6, and the label disagreement for each of the 5 probe sites (expected 0; > 1e-4 is logged, ES3). Adapt `solver.py` to the printed 0.10.0 signatures before the next command.
- `modal run modal_fem.py --mode unit`: exits 0; `tests/test_fem_unit.py` all pass in the image (numpy 2.2.6) and `tests/test_fem_solver.py` reports 9 passed. Record the `T6_REPORT` line for report point A.

**Gate G1**: any failure of the 9 solver tests or the unit tests is a bug: escalate to the dispatcher with the failing assertion values and the pytest.txt path for a fix round (P25; not a user stop). A `T6_REPORT` with `failed_at_s` < 1 is not a G1 failure; it is reported at report point A as an early warning (Expected surprise 14).

### Step 5 — Features (pure numpy/pandas, local)

Target: `pmdb/fem/features.py`; tests in `tests/test_fem_unit.py`.

```python
def fem_tile_slices(width_px: int, px_um: float, n_tiles: int = 6, edge_um: float = 20.0) -> list[slice]   # P15
def region_metrics(r: SimResult, frame: int, cols: slice, orientation: str, p: dict) -> dict[str, float]
def run_curves(r: SimResult, orientation: str, p: dict, window: bool) -> tuple[list[dict], list[dict]]
    # (site_rows, tile_rows). site_rows: one per frame with keys frame, s, converged, failed_at_s,
    # first_pore_closure_s + the 81 metrics. tile_rows: one per (tile, frame) with the same keys plus
    # tile (0..5), tile_x0_um, tile_x1_um. window=True -> site region = all cols, tile_rows empty.
def symmetrise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame   # P17; returns df + orientation="sym" rows
def swelling_gate(swelling_sym: float, p: dict) -> dict              # {"swelling_sym", "gate_ok", "lit_band_ok"} per P25, NaN -> both False
```

The Tier-1 metric names are exact, and the definitions are per region R (cells in `cols`, all rows). H_um = H·h. n_free is the free-edge outward normal sign: +1 for bottom, −1 for top. The free-edge node row is 0 for bottom and H for top, with node cols `cols.start .. cols.stop` inclusive.
1. `swelling` = mean(n_free·uz over free-edge nodes)/H_um
2. `surface_rough` = std(n_free·uz over the same nodes)/H_um
3. `sxx_mean_MPa` = mean sxx over all R cells
4. `syy_mean_MPa` = mean syy over all R cells
5. `porosity` = Σ_{PORE∩R} J / Σ_R J
6. `porosity_change` = porosity(frame) − porosity(0)
7. `porosity_rel_change` = porosity(frame)/porosity(0) − 1 (NaN if porosity(0) = 0)
8. `J_si_mean`, 9. `J_gr_mean`, 10. `J_binder_mean`
11. `vm_si_p50_MPa`, 12. `vm_si_p95_MPa`, 13. `vm_gr_p95_MPa`, 14. `vm_binder_p95_MPa`
15. `p_si_mean_MPa` (p = −(sxx+szz+syy)/3)
16. `si_yield_frac` = fraction of SI cells with vm > `si_yield_MPa(u(s))`
17. `pore_closed_frac` = fraction of PORE cells with J < `pore.closure_J`

Tier 3 metrics are named `q{q}_{vm|p|J}_{si|gr|binder}` and `q{q}_J_pore`. Tier 4 metrics are `band_vm_maxdev`, `band_vm_absslope`, `band_J_maxdev`, `band_J_absslope`: `band_profile(field*solid, solid, n_bands)` over R, then `band_summary`, keeping (maxdev, absslope). Metric dict order is the §2.1 column order. Any metric over an empty phase set is NaN. Unconverged frames give NaN metrics with `converged=False`. `first_pore_closure_s` is the first frame s where `pore_closed_frac > 0` over the whole domain, NaN if never; it is the same value on every row of the run.

Verification: `pytest -q tests/test_fem_unit.py -k features` (default python). Tests must pass on synthetic SimResults with no dolfinx:
- (a) `fem_tile_slices(1747, 0.1)` gives 6 slices with first start 200, last stop 1547, and contiguous edges.
- (b) An all-GRAPHITE 20×600 grid with uz = 0.1·z (bottom), J = 1.1 and sxx = −5 gives swelling 0.1 (rel 1e-6), surface_rough 0, sxx_mean −5, J_gr_mean 1.1, J_si_mean NaN.
- (c) The same with orientation top and uz = −0.1·(H_um − z) gives swelling 0.1.
- (d) A PORE pixel with J = 0.05 at frame 3 gives pore_closed_frac > 0 and first_pore_closure_s = s[3].
- (e) `symmetrise` on two rows with swelling 0.1 and 0.2 gives sym 0.15, and NaN propagates.
- (f) The metric keys equal the §2.1 list of 81 names, in order (17 + 60 + 4).
- (g) `swelling_gate`: 0.05 → gate_ok True, lit_band_ok False; 0.02 → both False; 0.20 → both True; 0.40 → both False.

### Step 6 — GIF rendering (local)

Target: `pmdb/fem/gif.py`; tests in `tests/test_fem_unit.py`.

```python
def render_site_gif(bse_coarse: np.ndarray, r: SimResult, title_prefix: str, swelling: Sequence[float],
                    cfg: dict) -> tuple[bytes, dict]   # (gif bytes, {"width_px", "colors", "bytes"}); raises GifTooLarge after ladder
```

Implementation per P19. matplotlib Agg; each frame drawn to an RGB array. Each frame is quantised with Pillow `quantize(colors=c, method=Image.Quantize.MEDIANCUT)`. Saved with `save_all=True, append_images=..., duration=frame_ms, loop=0, optimize=True`. Downsample for drawing with stride f = ceil(W/width_px): corners `u_nodes[:, ::f, ::f]` with the last row/col included, cell values block-mean over f×f.

Verification: `pytest -q tests/test_fem_unit.py -k gif` (default python). On a synthetic 580×1748 case (Gaussian-smoothed σ = 1 noise BSE, linear uz, log-uniform vm in [1, 1e4], last 2 frames unconverged), the output has `PIL.Image.open(...).n_frames == 11`, ≤ 2 000 000 bytes, and width ∈ {900, 720, 600}. Then `modal run modal_fem.py --mode unit` exits 0 (Steps 5-6 also pass in the image).

### Step 6b — `docs/fem/method.md` and the docs generator (local, D19)

Targets: `scripts/fem_docs.py` (new), `docs/fem/method.md` (new), `tests/test_fem_docs.py` (new; local only, NOT added to the Modal image).

`scripts/fem_docs.py` (default python; argparse flags `--method`, `--results`, both allowed together):

```python
def fill_blocks(path: Path, blocks: dict[str, str]) -> None
    # Replace the text between "<!-- AUTO:{name} -->" and "<!-- /AUTO:{name} -->" (markers kept) with
    # "\n" + blocks[name] + "\n" for every name. Raise ValueError if a marker pair is missing or a
    # marker in the file has no entry in `blocks`. Writes the file only if all blocks resolve.
def lit_param_rows(lit_md: Path = Path("docs/fem/literature-review.md")) -> dict[str, dict[str, str]]
    # Parse the markdown table under "## 1. Final parameter table": Parameter cell -> {"default", "source", "evidence"}.
def method_blocks(p: dict, run_log: dict | None) -> dict[str, str]   # keys: params, physics, soc, gates, cost
```

method.md blocks:
- `params`: table `Parameter | Value used | Source | Evidence strength` preceded by the line `Sources: [literature review, parameter table](literature-review.md#1-final-parameter-table)` (relative link; both files are in `docs/fem/`). One row per entry of this fixed map (lit-review Parameter cell → `fem.yaml` keys rendered as `key = value`): "Si volumetric eigenstretch" → `si.beta`; "Si utilisation at 100% overall SOC, u_max" → `soc.u_max`; "Graphite lithiation at 100% overall SOC, y_max" → `soc.y_max`; "Overall SOC → per-phase normalised fraction" → `soc.s_star, soc.si_ratio, soc.gr_ratio`; "Si E(u)" → `si.E_MPa`; "Si ν(u)" → `si.nu`; "Si yield (flag only, v1 is elastic)" → `si.yield_MPa, si.x_per_u`; "Graphite c-axis strain ε_zz(y)" → `graphite.eps_c_points`; "Graphite a-axis strain ε_xx(y)" → `graphite.eps_a_over_c`; "Graphite E, ν (isotropic)" → `graphite.E_MPa, graphite.nu`; "Binder + CBD (unassigned solid)" → `binder.E_MPa, binder.nu`; "Pore / artefact" → `pore.E_rel_binder, pore.nu, pore.closure_J`; "Resolution" → `mesh.res_nm`. Source and Evidence cells come from `lit_param_rows`; a missing name raises KeyError.
- `physics`: table `Assumption | Model choice | Source | Evidence strength` copying the Default, Source and Evidence cells verbatim from `lit_param_rows` for "Kinematics", "Elastic energy (all phases)", "Graphite orientation", "Out-of-plane", "Current-collector edge", "Lateral edges", "Separator-side edge".
- `soc`: the 11-row table `s | f_Si | f_Gr | u | y | J_Si | ε_c` computed with `pmdb.fem.materials` (`soc_fractions`, `lithiation_state`, `graphite_strains`, J_Si = 1 + β·u), 3 decimals.
- `gates`: the `gates` block of `fem.yaml` rendered as a list (stop window, literature band, max failed sites), the solver settings (`solver.*`), and the `remediation` list (or `none applied`).
- `cost`: from `outputs/fem/run_log.json` if it exists: `cost_model` constants, `production` cpu/memory/timeout, median and max per-case `wall_s` and `cost_usd` of tag `full`, ledger total vs `cap_usd`. Otherwise the pending line (P28).

`docs/fem/method.md` skeleton, sections in this order, number-free prose (P28 rule) around the blocks:
1. **Pipeline**: a mermaid `flowchart LR` with nodes SEM site (BSE/Inlens/SE) → `segment()` label map → coarsened label map → quad mesh (one cell per pixel) → lithiation over SOC frames → Newton solve on Modal → fields (J, stresses, von Mises, displacement) → tile and site features (`tile_curves.csv`, `site_curves.csv`) → classifier (separate plan); a second edge from fields → GIF. Below it, one short paragraph per arrow naming the module that does it (`pmdb.segment`, `pmdb.fem.geometry`, `pmdb.fem.solver`, `pmdb.fem.features`, `pmdb.fem.gif`, `modal_fem.py`).
2. **Physics assumptions**: prose for plane strain, finite-strain neo-Hookean with eigenstretch, per-phase uniform lithiation, elastic only (yield is a flag), pores as an ersatz soft material; then `<!-- AUTO:physics -->`.
3. **Parameters**: `<!-- AUTO:params -->`.
4. **Boundary conditions**: prose for collector edge, lateral rollers, traction-free separator side, both orientations and the `sym` mean (P10, P17).
5. **SOC mapping**: prose (Yao two-region split, graphite staging) + `<!-- AUTO:soc -->`.
6. **Features and output tables**: prose for tiers 1/3/4 and tiles (P15, P16), and a link to §2.1's content reproduced as the column list (column names only, no counts).
7. **Validation gates**: prose (stop window vs literature band, G1/G2/G4/G5 meanings, the fix-and-continue policy and remediation ladder of P25) + `<!-- AUTO:gates -->`.
8. **Runtime and cost**: prose (Modal, detached runs, per-chunk re-projection and the drop-`top` / drop-resolution-check budget ladder) + `<!-- AUTO:cost -->`.
9. **Limitations**: number-free bullets: 2D plane strain; no electrochemistry or rate effects; elastic only; uniform SOC per phase; no particle rearrangement, binder creep or SEI growth (so electrode swelling is expected to sit below measured values); collector side unknown (hence `sym`); pure-Si assumption, SiOx not run (D17); parameter robustness not measured (sweeps dropped, D17); pore stiffness is a numerical choice without a source.

`tests/test_fem_docs.py` (imports `fill_blocks`, `lit_param_rows`, `method_blocks` via `importlib` from `scripts/fem_docs.py`):
- `fill_blocks` on a temp file with two marker pairs replaces both and keeps the markers; a missing pair raises ValueError and leaves the file unchanged;
- `lit_param_rows()` contains every name of the `params` and `physics` maps;
- `method_blocks(load_params(), None)["cost"]` is the pending line, and `["params"]` contains `96000`.

Verification:
- `pytest -q tests/test_fem_docs.py` (default python): 3 passed.
- `python scripts/fem_docs.py --method` exits 0; `grep -c "AUTO:" docs/fem/method.md` = 10 (5 opening + 5 closing markers); only the `cost` block is pending.

### Step 7 — Case runner, orchestrator, window and sync modes, Stages 3-4 → report point A

Targets: `pmdb/fem/run.py`, `modal_fem.py` (add `run_case_remote`, `run_cases`, `sync`, modes `window` and `sync`), `scripts/fem_collect.py` (new; this step adds only `--g2`, which Step 9 runs).

```python
def run_case(batch: str, site: str, orientation: Literal["bottom", "top"], *,
             cache_root: str | Path | None = None, res_nm: float | None = None, crop_um: float | None = None,
             fields_path: Path | None = None, render_gif: bool = False,
             solver_overrides: dict | None = None, log=print) -> dict
```

Pipeline:
1. `p = load_params()`; if `solver_overrides`, `p["solver"].update(solver_overrides)` (so `params` and `params_hash` in meta record it).
2. `site = load_site(batch, site, resolution="half", normalise="none", cache_root=cache_root)` — default `harmonise="none"` (P24).
3. `masks = segment(site)`.
4. labels = `labels_from_masks(masks)`, cropped to `central_cols` if `crop_um`, then coarsened with factor = round((res_nm or p["mesh"]["res_nm"]) / site.nm_per_px). The BSE (`site.image[..., 0]`) is cropped and coarsened the same way.
5. `simulate(..., props_fn=lambda s: phase_properties(s, p), bc=BCSpec(orientation), solver_opts=p["solver"], frames=linspace(0,1,11), extra_targets=(p["soc"]["s_star"],))`.
6. `save_npz` if `fields_path`.
7. `run_curves(window=crop_um is not None)`.
8. Render the GIF if `render_gif`, catching `GifTooLarge` into `gif_error`.

Returns `{"meta": {...P23 fields, batch, site, heldout: batch == "Batch_heldout", orientation, H, W, px_um, n_cells, crop_um, res_nm, failed_at_s, first_pore_closure_s, n_substeps, newton_its_total, substeps, wall_s, peak_rss_mb (resource.getrusage ru_maxrss), versions, params, params_hash, gif_error}, "site_rows": [...], "tile_rows": [...], "gif": bytes | None}`. Every row gets batch, site, heldout and orientation.

`modal_fem.py` additions:
- `@app.function(cpu=4.0, memory=16384, timeout=21600, retries=0, volumes={"/data": pmdb-data read-only, "/out": pmdb-fem-out}) def run_case_remote(case: dict, tag: str, threads: int, crop_um: float = 0.0, res_nm: float = 0.0, solver_overrides: dict | None = None) -> dict` with `case = {"batch", "site", "orientation"}`:
  - set `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` = str(threads) before importing `pmdb.fem.run`;
  - cache_root = `/data/heldout` if batch == "Batch_heldout" else `/data`; `crop_um`/`res_nm` of 0 mean None; `render_gif = orientation == "bottom"`;
  - call `run_case(..., fields_path=Path(f"/out/fields/{tag}/{orientation}/{batch}__{site}.npz"), render_gif=render_gif, solver_overrides=solver_overrides)`;
  - on success only: write `result.json` (gif bytes removed; cpu, memory_mb, timeout_s, wall_s added to meta) to `/out/results/{tag}/{orientation}/{batch}__{site}.json` and the GIF to `/out/gifs/{tag}/{batch}__{site}.gif`; `volume.commit()`;
  - return the result dict (gif included); any exception → return `{"meta": {..., "error": "Type: msg"}, "site_rows": [], "tile_rows": [], "gif": None}` without writing (modal_app.py D7 pattern; P29 writes the stub after the retry).
- `@app.function(cpu=0.25, memory=1024, timeout=86400, retries=0, volumes={"/out": pmdb-fem-out}) def run_cases(cases: list[dict], tag: str, cpu: float, memory: int, timeout: int, threads: int, crop_um: float = 0.0, res_nm: float = 0.0, solver_overrides: dict | None = None) -> dict`: exactly P29.
- Local `sync(tag: str) -> list[dict]`: via `modal.Volume.from_name("pmdb-fem-out")`, copy `/results/{tag}/**`, `/status/{tag}/*.json` and `/gifs/{tag}/*.gif` to the P26 local paths (overwrite), append ledger rows from the status files (P21 dedupe; cost computed locally, also stored as `meta.cost_usd` in the local JSON of the successful attempt), return the local result metas.
- Local `summarize(tag)`: prints per case wall_s, n_cells, n_substeps, newton_its_total, s/Newton-iteration, peak_rss_mb, cost, error/status, and Tier-1 site metrics at s = 0.5 and 1.0; for each site with both orientations, swelling_sym(s = 1) with `swelling_gate`'s gate_ok and lit_band_ok; then n ok / error / lost, n failed_at_s < 1, total and mean wall, tag cost, ledger total.
- `main` gains `orientation: str = "both", sites: str = "", crop_um: float = 0.0, res_nm: float = 0.0, tag: str = "", cpu: float = 4.0, memory: int = 16384, timeout: int = 7200, solver_overrides: str = ""` (a JSON object string, parsed with `json.loads`). Mode `window`: cases = `sites` (comma-separated Batch/id) × orientations (`both` = bottom, top); `budget_check`; `run_cases.remote(...)` with threads = int(cpu); `sync(tag)`; `summarize(tag)`; print `SYNC DONE <tag>`. Exit 4 if any printed gate_ok is False, else 1 if any case has an error/lost stub, else 0. Mode `sync`: `sync(tag)`, `summarize(tag)`, `SYNC DONE <tag>`, exit 0; no remote call.
- `scripts/fem_collect.py --g2 DIR100 DIR50`: reads the window result JSON found recursively under each directory (site region) and writes `outputs/modal/fem/g2.csv` with, at s = 0.5 and 1.0, both values and the relative difference for `swelling`, `sxx_mean_MPa`, `porosity_change`, (J_si_mean − 1) (limit 0.10) and `vm_si_p50_MPa`, `vm_si_p95_MPa` (limit 0.20), plus a `pass` column; relative difference = |a−b|/max(|b|, 1e-12) with b = the 50 nm value. Prints the table; exit 1 if any limit is exceeded.

Stages on site `Batch_3/vc2whyaq` (a Batch 3 site with typical K01), via Modal. (Stage 2, the resolution check, moved to Step 9 so the budget ladder can drop it, P22.)
- **Stage 3, end-to-end window with timings**: `modal run --detach modal_fem.py --mode window --sites Batch_3/vc2whyaq --orientation both --crop-um 40 --res-nm 100 --tag stage3`. Extrapolate to a full site per case as measured seconds per Newton iteration × (n_cells_site/n_cells_window)^1.5 × the measured iteration count (P22 exponent). Open `outputs/modal/fem/gifs/stage3/Batch_3__vc2whyaq.gif` and check frames 0, 5 and 10 visually: the warp goes upward and the colour bar is fixed.
- **Remediation ladder (P25)**: if either Stage-3 case has a finite `failed_at_s` < 1, apply the next rung in `configs/fem/fem.yaml` (rung 1 `solver.ds_min: 0.0001953125`; rung 2 `pore.E_rel_binder: 1.0e-3`; rung 3 `solver.linesearch` switched to the other of bt/basic), append `"rung<k>: <change> (stage3 failed_at_s bottom=<b>, top=<t>)"` to `remediation`, commit, re-run `modal run modal_fem.py --mode unit` (a failure is a bug, P25) and re-run Stage 3 with tag `stage3_r<k>`. Rungs are cumulative; stop climbing when both orientations reach s = 1. After rung 3, accept (rung 4: append `"rung4: accept NaN frames"`). The settings in `fem.yaml` after this point are global for the benchmark and production.
- **Stage 4, validation on the window (gate G4w)**: from the last Stage-3 run, swelling_sym(s = 1) must have gate_ok ([0.03, 0.39], P25). A failure is a bug: escalate to the dispatcher with the P25 diagnostics; continue only after the dispatcher's fix round or its "no bug found" reply. Report lit_band_ok and, without gating: sxx_mean_MPa(s = 1)/(−10) (VT2), porosity_change and porosity_rel_change at s = 1 (VT4), J_si_mean(s = 1) against the plane-strain expectation (VT5), si_yield_frac, pore_closed_frac, first_pore_closure_s and failed_at_s.

Verification:
- `pytest -q tests/test_fem_unit.py` and `pytest -q -m data tests/test_fem_data.py` pass in the default python; `modal run modal_fem.py --mode unit` exits 0.
- `outputs/modal/fem/results/stage3/{bottom,top}/Batch_3__vc2whyaq.json` exist; `outputs/modal/fem/status/stage3/` has a status file; `ledger.csv` has a row for every attempt so far plus one `orchestrator` row; running `modal run modal_fem.py --mode sync --tag stage3` a second time adds no ledger rows.

**Report point A (no waiting).** Append to the implementer report file (the path the dispatcher gave) and continue to Step 8:
- the G1 test summary and the `T6_REPORT` line;
- the probe versions and the per-site label disagreement for the 5 probe sites;
- the Stage-3 timings and full-site extrapolation, and the remediation rungs applied (or none);
- the G4w numbers (gate_ok, lit_band_ok) and the VT2/VT4/VT5 report;
- the ledger total;
- any Expected-surprise responses used.

### Step 8 — Benchmark, full mode, collector (Stage 5) → report point B

Targets: `modal_fem.py` (modes `bench`, `full`), `scripts/fem_collect.py` (full tables).

`modal_fem.py`:
- **bench**: on `Batch_3/vc2whyaq`, full site, memory 16384, timeout 21600: `run_cases.spawn([bottom, top], tag="bench", cpu=4.0, threads=4)` and `run_cases.spawn([bottom], tag="bench_cpu8", cpu=8.0, threads=8)`, then `.get()` both, `sync` + `summarize` both tags, `SYNC DONE bench`. `budget_check` rule (a) over the 3 calls. Exit 4 if the cpu-4 pair fails the swelling gate.
- **full**: `--orientation` must be `bottom` or `top` (one chunk per invocation). Cases = all 31 labelled sites (from `cache/half/manifest.csv`) + 3 held-out (from `cache_heldout/half/manifest.csv`), restricted by `--sites` if given. `--tag` defaults to `full`; tag `full_rerun` is used with `--sites` and `--solver-overrides` for convergence reruns. `budget_check` per P22 rules (b)/(c). Then `run_cases.remote(...)`, `sync`, `summarize`, `SYNC DONE <tag>`. Never exits 4 (G4/G5 are evaluated by the collector).

`scripts/fem_collect.py --results outputs/modal/fem/results`:
- Expected cases = 34 sites × `orientations_run` (`["bottom"]` if `outputs/modal/fem/budget.json` has `drop_top`, else both).
- Per expected (batch, site, orientation), choose among the local result JSONs of tags `full` and `full_rerun` the one without `error` with the largest `failed_at_s` (NaN counts as 1.0, i.e. best); ties go to `full`. If none is error-free, the case is missing (§2.1 deviation rows; tile bounds from `fem_tile_slices((width//2), 0.1)` with the manifest width).
- Writes to `outputs/fem/`:
  - `site_curves.csv` and `tile_curves.csv` exactly per §2.1, including its deviation rules (sym via `symmetrise` with keys `[batch, site, frame]` and `[batch, site, tile, frame]`; if `top` was dropped, `sym` rows are copies of `bottom`);
  - `validation.csv` (one row per site, 34 rows: batch, site, heldout, swelling_bottom, swelling_top (NaN if dropped), swelling_sym at s = 1, `gate_ok`, `lit_band_ok`, sxx_ratio_vt2, porosity_change, porosity_rel_change, J_si_mean, si_yield_frac, failed_at_s, first_pore_closure_s);
  - `run_log.json` (P23 provenance + per-case meta of the chosen results without the substeps list + `orientations_run`, `missing_cases`, `rerun_cases` (chosen from `full_rerun`), `budget` (budget.json contents), `remediation` (from params), `probe` (newest probe JSON), ledger total and per-mode subtotals, `cost_model`, `production`); substep logs go to the gitignored `outputs/modal/fem/substeps/`;
  - copies the bottom GIF of each chosen result (`outputs/modal/fem/gifs/{tag}/{batch}__{site}.gif`) to `outputs/fem/gifs/`, and `outputs/modal/fem/g2.csv` to `outputs/fem/g2.csv` if it exists.
- Prints the gate summary (P25): median swelling_sym with gate_ok and lit_band_ok, sites outside the stop window, G5 count, missing cases, ledger total. Floats with `float_format="%.6g"`. It always writes full-size tables (missing cases become deviation rows), never `_partial` files; exit 1 only if an input file is unreadable.

Run `modal run --detach modal_fem.py --mode bench`. If `SYNC DONE bench` is not printed, poll (P29) `/results/bench/bottom` for 1 and `/results/bench/top` for 1 and `/results/bench_cpu8/bottom` for 1, then `--mode sync --tag bench` and `--mode sync --tag bench_cpu8`.

Verification: 3 results with no `error`. A `fields.npz` from the volume (`modal volume get pmdb-fem-out /fields/bench/bottom/Batch_3__vc2whyaq.npz /tmp/`) loads with `load_npz` and the shapes are (11, H/2, W/2) and (11, H/2+1, W/2+1, 2). The benchmark GIF is ≤ 2 MB. A bench swelling-gate failure is a bug (P25 (2)): escalate to the dispatcher. If neither cpu-4 case succeeds after the P29 retry, ES11.

**Report point B (no waiting).** Append to the implementer report file and continue to Step 9:
- per case: wall_s, n_substeps, newton_its_total, peak_rss_mb, cost_usd, failed_at_s, swelling_sym(s = 1) with gate_ok and lit_band_ok, and VT2/VT4/VT5;
- the chosen production settings, by these fixed rules: cpu = the lower cost_usd of cpu 4 vs cpu 8 (within 10% → the lower wall time; if the cpu-8 case failed, cpu 4); memory = max(4096, ceil(1.5·peak_rss_mb/1024)·1024) MB; timeout = min(28800, max(7200, 3·max bench wall_s at the chosen cpu, rounded up to the hour));
- the **expected** projection: P22 `expected` over the 68 production cases + the Stage-2 worst case + ledger total + DEV_ALLOWANCE, compared with $180, and the budget decision this implies (none / drop top / drop stage2);
- the **worst-case** figure: Σ over the 68 cases of (2.5·timeout_s + 240)/3600 × rate (first attempt + 1.5× retry), + ledger total, and the worst case of one 34-case chunk — reported, not gated;
- whether the vm range [1, 1e4] MPa covered the benchmark (Expected surprise 9).

### Step 9 — Full run, convergence reruns, resolution check → report point C

1. **Bottom chunk**: launch `modal run --detach modal_fem.py --mode full --orientation bottom --cpu <C> --memory <M> --timeout <T>` in the background. Wait with the P29 poll (`/results/full/bottom`, N = 34). If the launch log lacks `SYNC DONE full`, run `modal run modal_fem.py --mode sync --tag full`.
2. **Top chunk**: the same with `--orientation top` (poll `/results/full/top`, N = 34). If `budget.json` has `drop_top`, the entrypoint exits 0 with `top dropped by budget`; skip to item 3.
3. **Convergence reruns** (P25 (4)): for each orientation run, list the cases in `outputs/modal/fem/results/full/<o>/` without `error` and with finite `failed_at_s` < 1. If any, launch once per orientation `modal run --detach modal_fem.py --mode full --orientation <o> --tag full_rerun --sites <comma list> --solver-overrides '{"ds_min": <current solver.ds_min / 8>, "linesearch": "<the other of bt/basic>"}' --cpu <C> --memory <M> --timeout <T>`, poll `/results/full_rerun/<o>` for the list length, sync. `reruns skipped by budget` (exit 0) means rung 4 applies. No second rerun.
4. **Stage 2, resolution check (G2)**: unless `budget.json` has `drop_stage2`: `modal run --detach modal_fem.py --mode window --sites Batch_3/vc2whyaq --orientation bottom --crop-um 20 --res-nm 100 --tag stage2_100`, the same with `--res-nm 50 --tag stage2_50`, then `python scripts/fem_collect.py --g2 outputs/modal/fem/results/stage2_100 outputs/modal/fem/results/stage2_50`. Exit 1 = G2 failed: continue at 100 nm (P25 (3)); the results.md `flags` block records it.
5. `python scripts/fem_collect.py --results outputs/modal/fem/results`.

Verification: `fem_collect.py --results` exits 0; `outputs/fem/site_curves.csv` and `tile_curves.csv` have the §2.1 row counts (deviations included); `validation.csv` has 34 rows; `run_log.json` has `orientations_run`, `missing_cases`, `rerun_cases`, `budget`, `remediation`. Gates computed (P25): G4 = median swelling_sym(s = 1) over 34 sites in [0.03, 0.39]; G5 = sites with failed_at_s < 1 in either orientation ≤ 3.

**Report point C (no waiting).** Append to the implementer report file and continue to Step 10:
- G4/G5 values and verdicts, the median's lit_band_ok, and sites individually outside [0.03, 0.39];
- the swelling distribution by batch;
- VT2/VT4 medians;
- failed sites with failed_at_s and first_pore_closure_s, the reruns and their outcome, missing cases;
- budget decisions, the G2 result (or "dropped");
- the ledger total.

A G4 or G5 failure does not stop the run (P25): continue, and list it in the final hand-back as a suspected bug.

### Step 10 — Results doc, figures, README, commit, push, PR

1. Extend `scripts/fem_docs.py` with `--results` (P28). Inputs: `outputs/fem/{site_curves,validation,g2}.csv` (`g2.csv` absent if Stage 2 was dropped), `outputs/fem/run_log.json` (includes `budget`, `remediation`, `missing_cases`, `rerun_cases`, `orientations_run`), `outputs/fem/gifs/`, and the `9 passed` summary line of the newest `outputs/modal/fem/tests/*/pytest.txt` (copied verbatim). Figures, written with matplotlib Agg at 150 dpi to `outputs/fem/figures/`:
   - `swelling_vs_soc.png`: site-region `swelling` vs `s`, orientation `sym`, one line per labelled site coloured by batch (held-out sites dashed grey), stop window and literature band at s = 1 drawn as shaded spans from `gates`;
   - `swelling_by_batch.png`: strip plot of `swelling` at s = 1 (`sym`) per batch including `Batch_heldout`, with the literature band shaded;
   - `example_frames.png`: 3 rows (per labelled batch, the site whose `swelling` at s = 1 (`sym`) is the batch median, lower-median on ties) × 3 columns (GIF frames 0, 5, 10 read with Pillow from `outputs/fem/gifs/{batch}__{site}.gif`), titled with batch/site and frame SOC.
   results.md blocks, from `def results_blocks(out_dir: Path = Path("outputs/fem"), pytest_txt: Path | None = None, figures: bool = True) -> dict[str, str]` (`figures=False` skips PNG writing and GIF reading; the `figures` block then lists paths only):
   - `flags`: one bullet per condition that holds, else `none`: G2 failed or not run (dropped by budget); G4 median outside the stop window; G5 count > `max_failed_sites`; `top` dropped by budget; each missing case; each production rerun and its outcome; each applied remediation rung.
   - `validation`: G1 line; G2 table from `g2.csv`; median `swelling_sym` at s = 1 over all 34 and per batch with the stop window and literature band and the G4 verdict and `lit_band_ok`; per-batch medians of `sxx_ratio_vt2`, `porosity_change`, `porosity_rel_change`, `J_si_mean` from `validation.csv`; sites outside the stop window.
   - `convergence`: per orientation from `run_log.json` (tag full): cases, cases with `failed_at_s` < 1, median and max `n_substeps`, `newton_its_total`, `wall_s`; G5 count and verdict; the failed sites with `failed_at_s` and `first_pore_closure_s`.
   - `cost`: ledger total vs cap, per-mode subtotals, median and max production `cost_usd` per case.
   - `figures`: the three PNGs embedded by relative path (`../../outputs/fem/figures/<name>.png`) and a list of all 34 GIF relative paths (`../../outputs/fem/gifs/{batch}__{site}.gif`).
   - `batch_diff`: over the 81 metrics at s = 1, orientation `sym`, site region, labelled sites only: Kruskal-Wallis H across Batch_1/2/3 (`scipy.stats.kruskal`, NaNs dropped; skip a metric with < 3 values in any batch); the top 8 metrics by H as a table `metric | median B1 | median B2 | median B3 | H | p`, labelled "descriptive, unadjusted, 31 sites" in the generated text.
   `docs/fem/results.md` skeleton, number-free prose around the blocks, sections in order: Flags (`flags`), Validation (`validation`), Convergence (`convergence`), Cost (`cost`), Figures and GIFs (`figures`), Batch differences (`batch_diff`), and a one-line pointer to `method.md`. Add one test to `tests/test_fem_docs.py`: `results_blocks` on a synthetic 2-site-per-batch `site_curves`/`validation`/`run_log` fixture returns all 6 keys with no pending line (fixture built in the test with pandas; no GIFs needed when `figures=False` is passed).
2. Run `python scripts/fem_docs.py --method --results` (fills method.md's `cost` block too).
3. Write `docs/fem/README.md` (operational, P28): the Modal-only workflow and commands (probe, unit, window, bench, full, collect, `fem_docs.py`) with the pinned image tag; the volume layout (P14); the ledger and cap mechanics (P21-P22); the segmentation-input decision (P24, citing `test_segmentation_harmonisation_invariant` and the probe); links to `method.md` and `results.md`.
4. Add a "FEM simulation" pointer line to the top-level `README.md` under the Modal section, linking `docs/fem/method.md` and `docs/fem/results.md`.
5. Commit on `fem-sim` and `git push origin fem-sim`. Then open the PR: `gh pr create --base main --head fem-sim --title "FEM lithiation-swelling simulation: solver, Modal runner, features, GIFs"` with a body summarising the gates (G1, G2, G4w, G4, G5 values), the `flags` block of results.md, the report-point-C summary, the final ledger total and pointers to `docs/fem/method.md` and `docs/fem/results.md`. Never merge it. If an open PR from `fem-sim` already exists, the push updates it; do not open a second one.

Verification:
- `outputs/fem/site_curves.csv` has 34×3×11 = 1122 rows and 90 columns; `tile_curves.csv` has 34×3×6×11 = 6732 rows and 93 columns (748 / 4488 rows if `budget.json` has `drop_top`, §2.1); both headers match §2.1 in order.
- `outputs/fem/gifs/` holds 34 files, each ≤ 2 000 000 bytes.
- `outputs/fem/figures/` holds `swelling_vs_soc.png`, `swelling_by_batch.png`, `example_frames.png`; `pytest -q tests/test_fem_docs.py` gives 4 passed; `grep -c "_pending:" docs/fem/method.md docs/fem/results.md` gives 0 for both.
- `pytest -q -m "not data"` and `pytest -q -m data` pass in the default python (FEM solver tests skipped there).
- `modal run modal_fem.py --mode unit` exits 0 on the final commit.
- `gh pr view fem-sim --json state,baseRefName` shows `OPEN`, base `main`.
- Report the final ledger total.

## 5. Expected surprises

Under FULL AUTONOMY "escalate" below always means: hand back to the dispatcher for a fix round. It never means waiting for the user. The only hard stop is a budget exit 3.

1. **dolfinx 0.10.0 API detail differs from P12** beyond what the spike verified (e.g. `dofmap.list` shape). Adapt mechanically to the signatures printed by the probe. If `snes_linesearch_type bt` is rejected or makes T1-T4 fail where `basic` passes, pre-authorized: set `solver.linesearch: basic` in `fem.yaml` and record it (rung 3 then means switching to `bt`). If MUMPS is unavailable, escalate (environment bug).
2. **The image build fails the P20 step-(0) numpy assertion (the image's numpy is not 2.2.6), or pmdb fails to import or run under numpy 2.2.6 in the image** (probe or unit mode). Do not edit `pmdb/io.py`, `segment.py` or `kpis/**`, and do not change numpy in the image (dolfinx/petsc4py are compiled against it). Escalate, quoting the version the assertion printed.
3. **Probe label disagreement between image and local segmentation > 1e-4 on any of the 5 probe sites.** Continue: FEM features stay internally consistent; only P24's FEM/KPI mask-identity claim weakens. Log the per-site numbers at report point A; they also reach `run_log.json["probe"]`.
4. **A pinned pip version does not install on Python 3.12 / numpy 2.2.6.** Pre-authorized: use the newest patch release of the same major.minor that installs, never changing numpy or the scipy/scikit-image minor versions; record it in run_log. If scipy 1.14.x or scikit-image 0.25.x cannot install, escalate.
5. **Packages installed with `uv pip install --system` are not importable at runtime, or PYTHONPATH is lost.** Pre-authorized: replace `--system` with `--python /dolfinx-env/bin/python`, and keep the P20 `.env` PYTHONPATH (already set explicitly). If ghcr.io is unavailable, the identical Docker Hub tag `dolfinx/dolfinx:v0.10.0` may be used. Any other image change: escalate (no micromamba/conda fallback, D16).
6. **Newton stagnates.** If the logs show quadratic convergence that stalls between 1e-8 and 1e-6 relative residual, causing max_it failures, set `snes_rtol: 1.0e-6` in `fem.yaml`, append it to `remediation`, and rerun `--mode unit` and Stage 3. Any other recurring Newton failure is handled by the P25 ladder.
7. **Window or benchmark swelling below 9%.** Likely (reviewer estimate 8-14%, planner lower bound ~7%). Not a failure: record lit_band_ok = False and report the per-phase J means. Only < 3% or > 39% is treated as a bug (P25).
8. **Stage 2 resolution gate fails.** Continue at 100 nm (P25 (3)); running at 50 nm would cost about 8×. Flagged in results.md.
9. **The benchmark vm range is not covered.** If the bottom benchmark has p99.9 of solid vm at s = 1 > 1e4 MPa, or p1 of solid vm at s = 0.1 < 1 MPa, set `gif.vm_range_MPa` to [10^floor(log10 p1), 10^ceil(log10 p99.9)] from the benchmark before the full run. Report it at report point B. (Likely: plane-strain syy on Si is of order 1e4 MPa.)
10. **GIF exceeds 2 MB after the ladder for some site.** Record `gif_error`, continue, and list the sites at the end. Do not change the ladder.
11. **Benchmark OOM or no cpu-4 bench case succeeds after the P29 retry (which already uses 1.5× memory).** Rerun `--mode bench` once with `--memory 32768` (add the `memory` parameter to bench if needed). If still no success, escalate: production cannot be costed. A budget exit 3 at any point is the one hard stop: report to the dispatcher with the three printed numbers.
12. **Pre-existing test failures in the default python** that also fail on `main`. Report them; do not fix them.
13. (Removed by D17: sweep variants no longer exist. Number kept so references stay stable.)
14. **`T6_REPORT` shows `failed_at_s` < 1 (pore inversion next to Si).** Not a G1 failure. Report it at report point A as an early warning; the Stage-3 window and its remediation ladder are the deciding evidence.
15. (Removed by FULL AUTONOMY: the $15 dev guard is dropped; only the $180 cap stops. Number kept.)
16. **`modal volume ls pmdb-data /half` lists other than 31 npz + manifest.** Re-upload `cache/half` with `modal volume put --force pmdb-data cache/half /half` and re-check; the local cache is the source of truth.
17. **The local client disconnects during a detached run.** Do not relaunch, because that would double-spend. Poll (P29), then `--mode sync --tag <tag>`. If the orchestrator itself is lost (its 86400 s have passed and fewer result.json files exist than expected), add ledger rows by hand for the missing cases at `wall_s = timeout_s`, status `lost`, then relaunch only those cases with `--sites` under the same tag.

## 6. Done criteria

- G1 (analytic tests, including T1a = free expansion) and all unit/data tests pass. A G4w or bench swelling failure went through one dispatcher round (fixed, or "no bug found"). G2, G4 and G5 are computed; each either passed or is listed in the results.md `flags` block.
- Committed on `fem-sim` and pushed:
  - code (`pmdb/fem/*`, `configs/fem/fem.yaml` including its `remediation` list, `scripts/fem_collect.py`, `scripts/fem_docs.py`, `modal_fem.py`);
  - tests (`tests/test_fem_unit.py`, `tests/test_fem_solver.py`, `tests/test_fem_data.py`, `tests/test_fem_docs.py`);
  - `outputs/fem/{site_curves,tile_curves,validation}.csv`, `outputs/fem/g2.csv` (unless Stage 2 was dropped), `outputs/fem/run_log.json`, 34 GIFs ≤ 2 MB in `outputs/fem/gifs/`, `outputs/fem/figures/*.png`;
  - `docs/fem/{README,method,results}.md` (no pending blocks; the P28 number rule holds) and the top-level README pointer.
- `site_curves.csv` and `tile_curves.csv` match §2.1 exactly, including its deviation rules (1122/6732 rows; 748/4488 if `top` was dropped).
- A PR `fem-sim → main` is open and unmerged.
- Ledger total in `run_log.json` (all modes, including probe/unit/window and orchestrators) ≤ $180.
- Both pytest invocations in the default python (`-m "not data"` and `-m data`) pass; `modal run modal_fem.py --mode unit` passes.
- No file under `data/` or `data_heldout/` changed (`git status --porcelain data data_heldout` is empty). Held-out rows are all `heldout=True`.
- Nothing waited for the user; every former stop point has its report-point section in the implementer report.

## 7. Revision log

### Round 1 (2026-10-03) — trigger: plan review `.claude/reports/fem-build.plan-reviewer.md` (VERDICT REVISE, comments 1-7) plus dispatcher changes A (D16 Modal-only), B (harmonisation decision), C (branch fem-sim / new PR)

Review comments:
1. [CRITICAL, swelling gate] **accepted.** New P25 replaces old P23: stop window [0.03, 0.39]; literature band [0.09, 0.39] reported only as `lit_band_ok` (validation.csv, window/bench printouts, every stop report). G4w, benchmark, G4 (Step 9) and Expected surprise 7 restated. Decided: the stop window also applies to siox (new gate G4x, checked before sweeps launch in Step 10), because the amendment is not restricted to si and its bug-level rationale is configuration-independent; it does not apply to sweep variants (top_confined is 0 by construction). G5 stays si-only per the pre-approval ("default run"); siox failures are reported. T1a is now named as the user's free-expansion test (λy = 1) in Step 4 and P25. Gate numbers are also in `fem.yaml` (`gates:`) so code and plan share one source. *(Siox/G4x/sweep parts superseded by Round 2, D17.)*
2. [CRITICAL, branch] **accepted.** `fem-sim` throughout; §3 now says "never merge any PR" (the PR #18 line is removed, as #18 is merged); Step 10 pushes to `fem-sim` and opens `fem-sim → main` with `gh pr create`, never merged, without opening a duplicate (also change C).
3. [MAJOR, no pmdb-fem env] **accepted, superseded by D16 (change A).** No conda env and no `environment-fem.yml`. The image is pinned to the official `ghcr.io/fenics/dolfinx/dolfinx:v0.10.0`, proven on Modal by the coordinator's spike (`.claude/reports/fem-spike.implementer.md`); its registry config was also checked at planning time (Python 3.12 venv, PETSc 3.24.0 real64 with MUMPS, numpy 2.2.6, g++). The real/double PETSc concern is settled by the image's `PETSC_ARCH=linux-gnu-real64-32`, re-asserted in `.env`. A probe (Step 4) verifies versions, MUMPS, the `NonlinearProblem` signature and in-image vs local segmentation before the solver is written. Old ES1/ES2/ES6 (conda version, numpy<2 pin, micromamba fallback) are replaced by ES1-ES5.
4. [MINOR, cost projection] **accepted.** One exponent, 1.5, everywhere (P22; Stage 3 and all projections). Projections carry a 1.25 safety factor and a $5 dev allowance; full and sweep runs launch in chunks with a cap re-check between chunks using actual chunk costs. `retries=0` (P20) removes the retry doubling. STOP B and C now report the worst case (Σ (timeout+120 s) × rate over all remaining cases, plus the largest-chunk worst case) beside the expected projection. Lost calls are charged at the timeout in the ledger (P21).
5. [MINOR, coarsening tolerance] **accepted.** The ±20% Si / ±5% graphite asserts are hard; failure is stop-and-report (it would invalidate P2). Planning-time measurement on 3 sites: Si +1.9-2.4%, graphite −0.2-0.3%, so the tolerances have ample margin; numbers recorded in P3 and Step 3.
6. [MINOR, test count] **accepted.** Step 4 now lists 9 named test functions (T2 split into small and finite) and every verification says "9 passed".
7. [MINOR, T6 to s = 1] **accepted.** `test_t6_soft_pores` now runs frames [0, 0.25, 0.5, 0.75, 1.0]; the frame-1 assertions stay; `failed_at_s`, min pore J per frame and substep count are printed as a `T6_REPORT` line (report-only), surfaced by `--mode unit` (`-rP`) and reported at STOP A (ES14).

Reviewer "What's missing": pooling checks 3-4 are now named in Context and §3 as following in a separate plan. Reviewer assumption "numpy<2 pin": moot under D16; the image's numpy 2.2.6 is kept and pmdb's numpy-2 compatibility is tested by the probe and by running `tests/test_fem_unit.py` in the image (ES2 stops if it fails).

Dispatcher changes:
- A. **accepted.** P1, P12, P20, P21, P22, P26 rewritten; Step 4 creates `modal_fem.py` with `probe` and `unit` modes; Stages 2-4 run through `--mode window` (Step 7); `scripts/fem_run_local.py` and all `conda run` commands removed. Exact commands: `modal run modal_fem.py --mode probe|unit|window|bench|full|sweeps ...`; payloads, pytest logs and the ledger return to `outputs/modal/fem/`. Pure-numpy modules keep local pytest in the default python and are also run in the image. Test-iteration spend is in the same ledger as production and counts against the $180 cap, with a $15 dev guard before the benchmark.
- B. **decided: keep the existing `segment()` path on un-harmonised `normalise="none"` input** (P24). Justification: D4 binding; D-011 affine-invariant rule; measured pixel disagreement hybrid vs none ≤ 0.14% with phase fractions within 0.001 on strong, mild and clean sites; identical masks to the KPI tables used jointly in D10; FEM features never read grey levels. Guarded by `test_segmentation_harmonisation_invariant` (Step 3) and the probe's image-vs-local label check (Step 4). Since segmentation does not change, no KPI recomputation is needed for consistency.
- C. **accepted** (see comment 2).

Coordinator facts received during the round (folded in, no reviewer item):
- Volume layout `pmdb-data`: `/half/*.npz` + `/half/manifest.csv` (31 labelled) and `/heldout/half/Batch_heldout__{3e122cbj,fn0mhxef,xrv9xvzb}.npz` + `/heldout/half/manifest.csv` (already uploaded). P20 and Step 4 now verify with `modal volume ls` instead of uploading; held-out cases use `cache_root="/data/heldout"`.
- Spike result (dolfinx 0.10.0 on Modal, passed): P20 pins the spike's tag `ghcr.io/fenics/dolfinx/dolfinx:v0.10.0` (replacing the Docker Hub `v0.10.0-r1` chosen earlier in this round) and its `pip install uv` + `uv pip install --system` recipe; P12 pins the spike-verified API (`NonlinearProblem(F, u, bcs=, petsc_options_prefix=, petsc_options=)`, automatic Jacobian, `problem.solve()`, `problem.solver.getIterationNumber()/getConvergedReason()`, component BCs via `V.sub(i).collapse()`, `interpolation_points` property). Line search stays `bt` with a pre-authorized fallback to the spike's `basic` (ES1). The probe is kept: it still checks MUMPS, dependency imports under numpy 2.2.6 and image-vs-local segmentation identity, which the spike did not cover.

### Round 2 (2026-10-03) — trigger: plan review `.claude/reports/fem-build.plan-reviewer-r2.md` (VERDICT REVISE, comments 1-3) plus dispatcher items 1-3 (D17 scope cut, STOP C, minors) and the final D10 in `fem-swelling.md`

Review comments:
1. [CRITICAL, D17 scope cut] **accepted, full cut list applied.** Context: SiOx and sweeps removed; production = 68 pure-Si cases. P7/P8/P9: SiOx eigenstretch and moduli, Shah E, isotropic graphite, proportional split and the SiOx σY sentence removed; the general s* normaliser kept (one formula, s* from config). P10 and Step 4 `BCSpec`: `top`/`pressure_MPa`, the confined and pressure modes and the solid-only pressure facet tagging removed; `free` and the test-only `lateral="left"` kept. P18 deleted (number kept as a stub). P20: the `outputs/kpis/site_kpis.csv` `add_local_file` removed. P22: `f_cfg`, the siox branch, the sweep chunking and "and sweeps" removed; two chunks of 34. P25: siox window, G4x, siox failure count and sweep exemption removed. P26: mode `sweeps` removed. Step 1: `fem.yaml` loses `soc.split`, `si.model/E_law/shah_*`, `siox`, `graphite.strain_mode`, `bc`, `configs`, `sweeps` (`si.qi_E_MPa` renamed `si.E_MPa` since it is the only law); `config.py` reduced to `load_params(path)` and `params_hash`; tests for siox, `si__umax_0.60`, sweep count, `select_sweep_sites` and the KeyError name lookup removed and replaced by a key-set test and a hash test. Step 2: tests (b) s* = 0.36, (c) proportional, the isotropic half of (e) and (f) Shah removed; remaining tests relettered (a)-(d). Step 7: Stage 3 siox run, siox/si wall ratio, siox G4w and the STOP-A `f_cfg` line removed. Step 8: mode `sweeps`, the sweep/robustness CSVs removed; `validation.csv` is per site (34 rows); gate summary si only; STOP B projections over 68 cases, chunk worst case over 34. Step 9: the siox + sweeps projection removed (see dispatcher item 2 for the STOP C decision). Step 10: old sub-steps 1-2 deleted; robustness flag, robust-feature counts and G4x removed from docs/PR; row counts 1122 and 6732; sweep/robustness counts and the "Report G4x … robust pairs" line removed. ES13 replaced by a removal stub (numbering kept for references); ES1 loses the pressure-BC facet-tag example; ES6 loses "(all configs)"; ES7 had no siox mention. Done criteria: G4x and the sweep/robustness CSVs removed; row counts follow Step 10. §3: D17 removals listed explicitly; the "never appear in sweeps" and "absent from the sweep tables" clauses removed.
2. [MINOR, numpy undetectable by the probe] **accepted.** P20 now runs a build-time step (0) before `pip install uv` that exits non-zero unless the image's own `numpy.__version__ == "2.2.6"`, which fails the image build. ES2 now covers that failure (stop and escalate with the printed version). The probe's "numpy 2.2.6" print stays as a cross-check.
3. [MINOR, mask identity on one site] **accepted.** `PROBE_SITES` = `Batch_3/vc2whyaq` plus the four strongly affected Batch-3 sites (`71vgq3fw`, `kbdh4tri`, `tuy3zymq`, `x7u69zsw`); the probe returns one label map per site, the local entrypoint compares each and exits 1 if any differs by > 1e-4. Probe memory raised to 4096 MB and timeout to 1800 s for the 5 segmentations (cents). ES3 and STOP A updated to per-site numbers.

Dispatcher items:
- 1. **Config column decided: removed.** With one parameter set, keeping `config` would carry a name argument, a merge function, a path level and a constant column through `config.py`, `run_case`, `run_case_remote`, `main`, the volume layout, the ledger and both CSVs; that is not free, so it is deleted everywhere (P14, P17, P21, P26, Steps 1, 4, 7, 8). The output schema is otherwise unchanged except that `run_curves` now emits `tile` (int) on tile rows instead of a mixed `region` column, and site rows carry no region/tile columns; the full contract is stated once in new §2.1 (P27), including column order, key columns, orientation values (bottom/top/sym), frames 0-10, NaN rules and row/column counts (site_curves 1122 × 90, tile_curves 6732 × 93). Step 5 test (f) and Step 10 verification check the header against §2.1.
- 2. **STOP C is no longer a user stop.** Step 9 ends at "checkpoint C": the report content is recorded for the final hand-back and execution continues to Step 10. The per-chunk cap check (P22) and the rerun's cap check remain; it still stops on G4 failure, G5 failure, a cap refusal, or a case that errors/is lost after its single rerun. STOP A and STOP B are unchanged (not in the dispatcher's instruction).
- 3. **Minors 2 and 3 applied** (see review comments 2 and 3).
- Final D10 (`fem-swelling.md`: class-weighted mean of tile probabilities, classifier in a separate plan): preamble, Context, P16 rationale and §3 updated; the earlier "feature-level [mean, max, spread] pooling" wording is removed. Pooling checks are no longer mentioned as future work in this plan (check 3 is done, check 4 dropped under D17).
- D19 (coordinator message during this round; `fem-swelling.md` D19, pitch-deck docs with script-generated numbers): **accepted, lean.** New P28 (AUTO-block docs, one generator script, three figures, number rule, citations copied at runtime from `literature-review.md` §1). New Step 6b writes `scripts/fem_docs.py --method`, `docs/fem/method.md` (mermaid pipeline, physics assumptions, parameter table with lit-review Source/Evidence cells and link, BCs, SOC table, gates, runtime/cost, limitations) and `tests/test_fem_docs.py` (local only). Step 10 adds `--results` (validation vs stop window and literature band, G1/G2, convergence, actual cost, figures + GIF paths, Kruskal-Wallis per-batch differences) writing `docs/fem/results.md` and `outputs/fem/figures/{swelling_vs_soc,swelling_by_batch,example_frames}.png`; `docs/fem/README.md` becomes operational only. P23 `run_log.json` gains `cost_model` and `production` so the cost block is generated. Step 8's collector also commits `outputs/fem/g2.csv`. `docs/classifier/*` and `docs/pitch-brief.md` are out of scope (§3). D18 is noted in the preamble as the consumer of §2.1; it changes nothing here.
- Also fixed: the §2.1 site_curves column-count sentence (9 key/meta + 81 metrics = 90 columns).
- FULL AUTONOMY (coordinator message during this round; `fem-swelling.md` pre-approval section): **accepted.** This supersedes the "checkpoint C" decision above and the STOP A/B text from earlier in this round. Changes:
  - Preamble: names the policy.
  - STOP A/B/C become report points A/B/C. The report-point content is appended to the implementer report and execution continues.
  - P25 rewritten as the single gate and failure-policy table:
    - test failures, and swelling-window failures at the window or benchmark, are bugs escalated to the dispatcher (one investigation round);
    - a production G4 failure, a G2 failure and G5 > 3 are flagged, not stops;
    - the remediation ladder is climbed on the Stage-3 window only, because rung 2 changes physics and must be global; production gets one physics-neutral rerun per failed case (tag `full_rerun`), then rung 4.
  - New P29: a remote orchestrator `run_cases` with one retry at 1.5× timeout and 1.5× memory, stub result.json for cases that still fail, status files on the volume, and a poll one-liner that ends when all result.json files exist. All case modes launch with `modal run --detach`; a new local `sync` mode rebuilds local results and the ledger from the volume (P21, P26).
  - P22 rewritten as the budget ladder (drop `top`, then drop Stage 2, then exit 3, the only hard stop), persisted in `budget.json`.
  - Stage 2 moved from Step 7 to Step 9 so the ladder can drop it.
  - The $15 dev guard is removed (ES15 stub).
  - §2.1 gains "Allowed deviations": missing-case NaN rows with `failed_at_s = 0.0`; the drop-top layout with `sym` = `bottom` and 748/4488 rows.
  - `fem.yaml` gains `solver.linesearch` and `remediation: []`, and the Step-1 key-set test is updated to match.
  - The probe's label disagreement no longer fails the probe (ES3: log and continue).
  - Production timeout is capped at 28800 s so attempt plus retry fits the orchestrator.
  - results.md gains a generated `flags` block; the `results_blocks` test expects 6 keys.
  - ES list and Done criteria rewritten for these changes; ES17 (client disconnect) added.

### Round 3 (2026-10-03, dispatcher patch) — trigger: code review of 0b28ff7 (`fem-build.code-reviewer.md`, APPROVE)
- P12 amended (approved, physics-neutral): `petsc_options` gains `"snes_stol": 0.0`; each substep record stores `fnorm` (SNES function norm) and `error` (repr of any caught exception, else None). Applied during Step 7.
- Tests: bilayer interface-row assertions added; T4 map shape fixed to (7, 9).
- Step 7: `run_case` catches GIF render failure on NaN frames as `gif_error`.

### Round 4 (2026-10-04, dispatcher decision under delegated physics judgment) — trigger: Step 7 deviation report (`fem-build.implementer-s7.md`)
Finding: Stage-3 non-convergence is physical, not numerical: the soft-void pore (E = 1e-4·E_binder, neo-Hookean ln J) has no stiffness until J ≈ 0, Si (Jλ up to 3.24) crushes adjacent pores to J → 0 and Newton cannot cross it. Ladder rungs 1-3 reach only s ≈ 0.024-0.075; rung 4 would leave 1-2 frames. The predictor NaN (reason -4) is already fixed by the fallback in e517e32.
Decision P30 — **pore compaction barrier** (PORE phase only): ψ_pore = ψ_NH(E_pore = 1e-4·E_binder, ν 0.3) + ψ_c, with
  ψ_c = conditional(J < J_c, (κ/2)·(ln(J/J_c))², 0),  J = det F (total), J_c = 0.3, κ = E_binder = 500 MPa.
  C1 at J = J_c (value and slope 0); barrier → ∞ as J → 0, so closed pores cannot invert. Physical reading: a pore squeezed
  below 30% of its area behaves like compacted binder (walls in contact). Weighted the same way as ψ_NH (det Fλ = 1 for pores).
  Pore-closure flag threshold (closure_J = 0.1) unchanged. Rung 2 (pore E 1e-3) NOT adopted; rung 1 (ds_min/8) and the
  predictor fallback stay. Config: `pore: {..., compaction_Jc: 0.3, compaction_kappa_MPa: 500.0}`; remediation entry added.
Fallback if the window still fails before s = 1: J_c 0.5 and κ = 5000 MPa (one run), then rung 4 (accept NaN frames).
Unit test: 1-cell pore under prescribed compression — energy and stress identical to plain NH for J ≥ J_c, finite and
  increasing for J < J_c; and test_t6_soft_pores must still pass (its frame-1 assertions at J ≥ 0.43 are unaffected).

### Round 5 (2026-10-04, dispatcher decision D20) — trigger: P30 made convergence worse (fem-build.implementer-s7.md)
P31 — production mechanics = small-strain linear elasticity, plane strain, logarithmic eigenstrain:
  ε_e(3D) = [sym∇u − diag(ln λx, ln λz)] in-plane, ε_e,yy = −ln λy; σ = λ_L tr(ε_e) I + 2μ ε_e (3D, report sxx, szz, sxz, syy);
  μ, λ_L per phase from E(s), ν(s) at the frame's s (DG0, same props_fn and label map). One dolfinx LinearProblem (MUMPS)
  per frame, frames = 11 targets (no substepping; s* extra target not needed). Same BCs/orientations, same SimResult schema:
  u_nodes, J := 1 + ε_xx + ε_zz (total small strain), vm, p from σ, converged = True unless the solve raises (then NaN frame,
  error recorded). No compaction barrier. fem.yaml: `mechanics: linear` (finite path selectable for tests).
Tests: free-expansion T1a for linear = total strain equals ln λ, stress < 1e-8·E; bilayer small-strain closed forms
  unchanged; finite-strain tests keep running against the finite solver. Pore-closure flag: J < closure_J (may be ≤ 0).
