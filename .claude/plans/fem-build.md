# FEM build: 2D plane-strain finite-strain lithiation swelling, features, GIFs, Modal runner

Binding inputs: `.claude/plans/fem-swelling.md` (D1-D16, budget, user pre-approval with the $180 HARD CAP, the AMENDED swelling gate, D10 feature-level pooling, D16 Modal-only environment) and `docs/fem/literature-review.md` (parameters, SOC mapping, validation targets VT1-VT6, "For the planner"). Where this plan and those files disagree, those files win: stop and report.

Branch: `fem-sim` (PR #18 from `fem-lit-review` is already merged). All work is committed and pushed to `fem-sim`, and a new PR `fem-sim → main` is opened at the end. Never merge it.

## 1. Context

The project wants batch-discriminating features from a mechanics simulation of each SEM cross-section. Every site's segmentation (`pmdb.segment.segment`) becomes a pixel-based finite-element mesh. The mesh is lithiated from 0 to 100% SOC under uniform-per-phase eigenstretches. Si swells isotropically, graphite anisotropically (c ∥ z), and binder and pores do not swell. The solve is plane-strain, finite-strain, compressible neo-Hookean, with FEniCSx/dolfinx. Fields are snapshotted at 11 SOC frames and reduced to tile and site curve features (D8 tiers 1-4) on the D9 tile grid, plus one GIF per site (D6, D15). All dolfinx code runs on Modal in the pinned official dolfinx Docker image (D16): analytic tests, the resolution check, the smoke windows, the benchmark and production (Si and SiOx on all 34 sites in both collector orientations, D14 sweeps on 6 sites). Pure-numpy modules are tested locally as well. Classification and pooling (D10, including pooling checks 3-4) are not part of this plan; they follow in a separate plan.

## 2. Design decisions

Conventions used throughout:
- Mesh axis 0 = **x** (coating width = image columns).
- Mesh axis 1 = **z** (thickness). z points toward image row 0: `z = (H - row - 0.5)·h` at a cell centre.
- **y** = out-of-plane.
- Lengths in µm, stresses and moduli in MPa (so E_Si = 96 000 MPa), forces per unit depth in MPa·µm.
- "Default python" = the repo's usual interpreter (`python`, currently 3.10.3 with numpy 1.26.4, pandas 2.1.4, scipy 1.14.1, scikit-image 0.25.2, pyyaml, matplotlib, Pillow, modal 1.6.0 all installed). "Image" = the Modal image of P20 (Python 3.12, numpy 2.2.6). Pure-numpy modules must run in both.

| # | Decision | Rationale |
|---|---|---|
| P1 | Module layout. `pmdb/fem/{__init__,config,materials,geometry,result,solver,features,gif,run}.py`; `configs/fem/fem.yaml`; `scripts/fem_collect.py`; `modal_fem.py` (new, at repo root). No `environment-fem.yml`, no local dolfinx, no local runner script (D16). Only `solver.py` imports dolfinx/ufl/petsc4py/mpi4py, and `run.py` imports `solver` lazily inside the function that needs it. Every `pmdb/fem` module starts with `from __future__ import annotations` and uses no numpy-2-only API. | Everything except the solve is unit-testable in the default python and in the image. This matches how `pmdb.kpis` stays independent of the segmenter. |
| P2 | Mesh resolution 100 nm/px (coarsen the 50 nm label map 2×2). Native 50 nm is used only for the Stage-2 resolution check. | A full site is about 580 × 1750 coarse px, so about 1.0 M quads and 2.0 M DOFs. At 50 nm it would be 4× the cells and about 8× the direct-solve cost. The Stage-2 gate tests the convergence of this choice. |
| P3 | Label map `uint8`: BINDER=0 (unassigned solid), SI=1, GRAPHITE=2, PORE=3, ARTEFACT=4. Coarsening is a 2×2 majority vote with ties broken by priority SI > PORE > GRAPHITE > BINDER > ARTEFACT. An odd trailing row or column is dropped. | Si is the rarest phase and the main driver, so ties keep Si. Thin pores carry most of the accommodation (lit review C16), so pore beats the solids. Artefact is ersatz-like anyway, so it yields. Measured at planning time on vc2whyaq / fzrt2k6r / 3806gxp0: coarse/fine area ratio − 1 = Si +0.019/+0.024/+0.020, graphite −0.002/−0.003/−0.003, pore +0.072/+0.071/+0.075, binder −0.22/−0.21/−0.24. |
| P4 | Mesh is a structured quadrilateral `create_rectangle` over `[0, W·h] × [0, H·h]` with one cell per coarse pixel. The cell→pixel map comes from cell midpoints. Materials are DG0 functions indexed through that map. | The mesh matches the pixel grid exactly, so there is no meshing step and phase tags are exact. DG0 makes per-phase updates a plain array assignment. |
| P5 | Displacement space is vector Q1 (`("Lagrange", 1, (2,))` on quads). Quadrature degree is 2. | Q1 nodes sit at pixel corners, so nodal displacement maps exactly to the corner grid. ν ≤ 0.34 everywhere means no volumetric locking. Degree 2 (2×2 Gauss) is exact enough and stops UFL from estimating a huge degree for the log terms. |
| P6 | Energy per reference volume ψ₀ = J_λ·[μ/2 (tr Ce − 3) − μ ln Je + λ/2 (ln Je)²]. Here Fe = F₃·Fλ⁻¹, F₃ = [[F₂, 0], [0, 1]] (total out-of-plane stretch 1 = plane strain), Fλ = diag(λx, λz, λy), Je = det F₂ /(λx λz λy), Ce = Feᵀ Fe (3×3), so Ce₃₃ = 1/λy². Residual = derivative of ∫ψ₀ dx; Jacobian = its derivative. | Exactly the lit-review energy (R5:E13), its plane-strain I1 and its det-Fλ weighting. |
| P7 | Eigenstretches (λx, λz, λy). **Si**: J_λ^(1/3)·(1,1,1) with J_λ = 1 + β·u, β = 2.8. **SiOx**: the same form with β = 1.6. **Graphite (anisotropic)**: (1+ε_a, 1+ε_c, 1+ε_a), with ε_c from np.interp over the breakpoints and ε_a = 0.01·ε_c/0.103. **Graphite (isotropic variant)**: (1+ε_iso)·(1,1,1), with (1+ε_iso)³ = (1+ε_a)²(1+ε_c). **Binder, pore and artefact**: (1,1,1). | Lit review "For the planner". The out-of-plane graphite stretch is the a-axis value (c ∥ z), and the isotropic variant preserves the lattice volume. |
| P8 | Moduli. **Si (Qi, default)**: E = 96 000 − 55 000·u MPa, ν = 0.29 − 0.04·u. **Si (Shah variant)**: E = 120 000 − 80 000·(β·u/3.0) MPa with the same ν. C/Cmax = β·u/3.0 because Shah's ΩCmax = 3.00 is volumetric. **SiOx**: 34 000 MPa, ν 0.17. **Graphite**: E = 32 000 + 77 000·y MPa, ν = 0.32 − 0.08·y. **Binder**: 500 MPa, ν 0.34. **Pore and artefact**: E = 1e-4·E_binder, ν 0.3. Lamé: μ = E/(2(1+ν)), λ = Eν/((1+ν)(1−2ν)). | Lit review table and D11/D13. u is not C/Cmax (Contradictions), so equal volumetric strain is the only defensible mapping. |
| P9 | SOC mapping (general form, recovers the lit-review table at s* = 0.25): f_Si(s) = [0.96·min(s,s*) + 0.58·max(s−s*,0)] / [0.96 s* + 0.58(1−s*)] and f_Gr(s) = [0.04·min(s,s*) + 0.42·max(s−s*,0)] / [0.04 s* + 0.42(1−s*)]. The proportional variant sets f_Si = f_Gr = s. Then u = u_max·f_Si (0.80) and y = y_max·f_Gr (0.91). Si yield flag: σY = 3000 − 3150·x/(1+x) MPa with x = 3.75·u. The SiOx run uses the same σY formula (borrowed, documented). | Lit review "Overall SOC". The s* = 0.36 sweep needs the normalisers recomputed so that f(1) = 1. |
| P10 | BCs. Lateral edges x = 0 and x = W: u_x = 0. Orientation `bottom`: collector at z = 0 (image last row) with u_z = 0; free edge at z = H. Orientation `top`: collector at z = H (image row 0) with u_z = 0; free edge at z = 0. Free-edge modes: `free` (traction-free, default), `confined` (u_z = 0), `pressure` (dead-load normal traction −p·n). The pressure is applied only on free-edge facets of solid cells (SI, GRAPHITE, BINDER) and scaled by W/W_solid so the total force equals p·W. It is applied at s = 0 as a preload, so frame 0 holds the preload state. A test-only lateral mode `left` (u_x = 0 on x = 0 only) gives roller-only support for free-expansion tests. | Lit review BCs and D12. Solid-only pressure stops the 1 MPa load from crushing 0.05 MPa ersatz pores that open onto the surface (the separator touches solids). |
| P11 | Load stepping. Mandatory targets are the 10 frame values plus s*. Δs starts at 0.05, Δs_max = 0.05, Δs_min = 0.1/64. On failure (SNES reason ≤ 0, NaN, or a PETSc exception), restore the last converged u and halve Δs. Below Δs_min the run has failed: `failed_at_s` = the attempted s, later frames are NaN, and the solve stops. On success with ≤ 5 Newton iterations, double Δs (capped). Predictor: u ← u_n + (Δs_try/Δs_last)(u_n − u_{n−1}) once two converged states exist, otherwise u_n. | D11 failure semantics. Linear extrapolation cuts Newton iterations, which dominate cost. |
| P12 | Nonlinear solver: dolfinx **0.10.0** (pinned by P20), API as verified by the Modal spike (`.claude/reports/fem-spike.implementer.md`, `scripts/modal_fem_spike.py`): `NonlinearProblem(F_res, u, bcs=bcs, petsc_options_prefix="fem_", petsc_options={...})` from `dolfinx.fem.petsc` with no `J` argument (the Jacobian is UFL's automatic derivative of `F_res`); `problem.solve()`; convergence from `problem.solver.getConvergedReason()` (> 0 converged) and `problem.solver.getIterationNumber()`. Component Dirichlet BCs follow the spike: `Vs, _ = V.sub(i).collapse()`, `locate_dofs_topological((V.sub(i), Vs), fdim, facets)`, `dirichletbc(zero_fn_on_Vs, dofs, V.sub(i))`. DG0 post-processing uses `fem.Expression(expr, V0.element.interpolation_points)` (a property in 0.10). Eigenstretch and modulus coefficients are DG0 Functions updated by assigning `x.array` before each `solve()` (the spike updated a `fem.Constant` the same way). Options: `snes_type newtonls`, `snes_linesearch_type bt` (the spike verified `basic`; `bt` is chosen for the large Si eigenstrains, ES1 covers a switch), `snes_rtol 1e-8`, `snes_atol 1e-10`, `snes_max_it 25`, `ksp_type preonly`, `pc_type lu`, `pc_factor_mat_solver_type mumps`, `mat_mumps_icntl_14 100`. The run is serial: assert `MPI.COMM_WORLD.size == 1`. `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are set to the container CPU count before dolfinx/petsc4py are imported (the image defaults OPENBLAS_NUM_THREADS=1). | D5 (SNES + MUMPS). The spike proved this exact API on Modal (neo-Hookean with det-Fλ weighting, 5 steps, 4 Newton its/step, mean J exact, max|P| 1.7e-14). Serial keeps the pixel mapping trivial. Parallelism comes from running many cases at once on Modal. |
| P13 | Post-processing at each frame. Interpolate UFL expressions into DG0 at cell midpoints: total J = det F₂ and the Cauchy stress σ = (1/Je)[μ(Be − I) + λ ln Je I], Be = Fe Feᵀ (3×3). Components are sxx, szz, sxz and syy (out-of-plane). von Mises vm = sqrt(½[(sxx−szz)² + (szz−syy)² + (syy−sxx)²] + 3 sxz²). Pressure p = −(sxx+szz+syy)/3 is derived, not stored. Nodal u is mapped to corners: col = rint(x/h), row = H − rint(z/h). | Fields are on the reference (coarse) pixel grid, as D15 requires. Cell-centre values are exact for DG0 and need no averaging. |
| P14 | Storage. One `fields.npz` (np.savez_compressed) per (run_tag, config, orientation, site), on Modal volume `pmdb-fem-out` under `/fields/...`. Keys: `labels` uint8 (H,W); `s` (11,); `converged` bool (11,); `u_nodes` float32 (11,H+1,W+1,2) as (ux, uz) in µm; `J, sxx, szz, sxz, syy, vm` float32 (11,H,W), in MPa except J. A small `result.json` (meta + feature rows) goes under `/results/...` and GIFs under `/gifs/...`. | Fields are kept so the tiers can be re-extracted (D8). Large fields are kept apart from the small results so `modal volume get` of the results stays small. |
| P15 | Feature regions (D9). 6 tiles per site with fixed edges `linspace(edge_px, W − edge_px, 7).round()`, edge_px = round(20 µm / h). Full height, no overlap. The site-level region is the union of the 6 tiles (central region), so site and tile metrics share one definition and exclude the lateral-BC bands. Cropped windows have no tiles, and their site region is all columns. | D9 tiling with a fixed tile count (D10 risk note). |
| P16 | Feature set per (region, frame). Tier 1 (17 named metrics, §Step 5). Tier 3: quantiles q ∈ {5,25,50,75,95,99} of {vm, p, J} over each of {si, gr, binder}, plus J over pore (60 metrics). Tier 4: depth-band (5 bands, reusing `pmdb.kpis.fields.band_profile` and `band_summary`) maxdev and absslope of vm and J over solid cells (4 metrics). Tier 5 (per-particle) is not built. Metric code is identical for tiles and site. | D8 tiers 1-4. Tier 4 is magnitude-only like K12 because the depth sign is unknown. Fields are stored, so tier 5 can be added later without new runs. D10 (updated) pools features as [mean, max, spread] over tiles downstream; this needs exactly the per-tile 81-metric × 11-frame rows produced here. |
| P17 | Orientation combination. Both orientations are simulated. CSVs carry `orientation ∈ {bottom, top, sym}`. `sym` = the arithmetic mean of bottom and top per metric per (site, config, region, frame). `sym` is NaN if either is NaN, `converged` = both, `first_pore_closure_s` = min, `failed_at_s` = min. The primary downstream feature set is `sym`. | No foil is visible (docs/data-processing.md l.55, spec 002 "Directions"), so the collector side is unknown per site. The mean is invariant to a vertical flip of the image, so features cannot encode an arbitrary orientation guess. |
| P18 | Sweeps (D14). Base config `si`, orientation `bottom` only, 12 variants, on the 6 sites `Batch_1/fzrt2k6r, Batch_1/uhdslk0o, Batch_2/3806gxp0, Batch_2/rxax5ozo, Batch_3/vc2whyaq, Batch_3/utfgcjfa` (2 per batch nearest the batch-median `K01_si_frac_adm` in `outputs/kpis/site_kpis.csv`, ties by site id). The comparator is the default `si`/`bottom` rows of the same 6 sites from the full run. Robustness is the Spearman ρ across the 6 site-level values between default and variant, per (metric, frame) for frames 1-10. `robust` = ρ ≥ 0.8 for all 12 variants. NaN ρ (constant or NaN input) counts as not robust. Features are flagged, not dropped. | D14 exactly, with one orientation, which halves sweep cost. Dropping features is a classification-plan decision. |
| P19 | GIF (D15). Rendered in-container only for config `si`, orientation `bottom` (labelled "collector assumed at bottom"). Single panel. The BSE (coarse block-mean of the un-harmonised uint8 BSE, grey, per-site p1-p99 stretch) is warped by displacement via `pcolormesh` on deformed corner coordinates. von Mises is overlaid on solid cells only (alpha 0.55, `magma`, LogNorm fixed at [1, 1e4] MPa, the same for every site and frame). Title: site, scenario, SOC %, swelling %. Axes fixed per site: x [−1, W+1] µm, z [−0.05H, 1.45H]. 11 frames, 500 ms, loop. Frames after a failure repeat the last converged frame with the banner "solver failed at SOC x%". Size ladder (900 px,128 colours) → (900,64) → (720,64) → (600,32); the first ≤ 2 000 000 bytes wins, otherwise `gif_error` is recorded. | D6/D15. The per-site p1-p99 stretch is affine-invariant, so the GIF background looks the same with or without harmonisation (P24); it is visual only and feeds no feature. |
| P20 | **Modal image (D16).** `modal.Image.from_registry("ghcr.io/fenics/dolfinx/dolfinx:v0.10.0")`, the tag verified by the Modal spike (Modal accepts it without `add_python`; python 3.12.3, dolfinx 0.10.0, petsc4py 3.24.0). Its registry config (checked at planning time) has a Python 3.12 venv at `/dolfinx-env` first on PATH, dolfinx under `/usr/local/dolfinx-real/lib/python3.12/dist-packages` via PYTHONPATH, PETSc 3.24.0 `PETSC_ARCH=linux-gnu-real64-32` built with `--download-mumps`, mpich, numpy 2.2.6 and g++ for the FFCx JIT. Then `run_commands` (a) `pip install uv`, (b) `uv pip install --system numpy==2.2.6 scipy==1.14.1 scikit-image==0.25.2 pandas==2.2.3 matplotlib==3.10.3 pillow==11.3.0 pyyaml==6.0.2 tifffile==2025.5.10 imagecodecs==2025.3.30 pytest==8.3.5`. Then `.env({"PMDB_CACHE": "/data", "PYTHONPATH": "/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib", "PETSC_ARCH": "linux-gnu-real64-32"})`, then the local files: `pytest.ini`, `configs/fem/fem.yaml`, `outputs/kpis/site_kpis.csv`, `tests/test_fem_unit.py`, `tests/test_fem_solver.py` (each `add_local_file` to the same relative path under `/root`), and `add_local_python_source("pmdb")`. Volumes: `pmdb-data` read-only at `/data` (`/data/half` labelled, `/data/heldout/half` held-out, already uploaded); `pmdb-fem-out` (`create_if_missing=True`) at `/out`. App name `pmdb-fem`. `retries=0` on every function. | D16 requires the official image with a pinned tag; this exact tag and install recipe (`pip install uv`, `uv pip install --system`) passed the spike. `--system` resolves to the first python on PATH, `/dolfinx-env/bin/python`, the interpreter Modal runs; `--mode unit` (which imports every dep) confirms it, and ES5 covers a switch to `--python /dolfinx-env/bin/python`. 0.11.0 exists but 0.10.x was requested and P12's API is the 0.10 one. numpy is pinned to the image's own 2.2.6 so uv cannot move it under the compiled dolfinx/petsc4py. scipy/scikit-image match the versions that produced `outputs/kpis/*` (modal_app.py, default python), so segmentation masks match the KPI pipeline (P24). pandas 2.2.3 is the oldest line with numpy-2 wheels. `retries=0` because errored cases are rerun once by hand and a hang that hit the timeout would hang again (comment 4). |
| P21 | **Cost accounting.** `cost_usd = (wall_s + 120)/3600 × (cpu × 0.0472 + mem_GiB × 0.0080)` (Modal's published $0.0000131 per core-second and $0.00000222 per GiB-second; the +120 s covers container start, image load and JIT outside the measured wall; check the rates on modal.com/pricing in Step 4 and update the constants if they differ). `wall_s` is measured inside the remote function from entry to return. Every remote call of every mode (probe, unit, window, bench, full, sweeps) appends one row to the local ledger `outputs/modal/fem/ledger.csv` (gitignored) with columns `utc, mode, tag, batch, site, config, orientation, cpu, memory_mb, timeout_s, wall_s, cost_usd, status` (`status` ∈ ok, error, lost). A call that returns no result (timeout, OOM kill, exception from `.map(return_exceptions=True)`) is `lost` and charged at `wall_s = timeout_s`. | One ledger covers the test iterations as well as production, as the user's cap requires. Charging lost calls at the timeout keeps the ledger an upper bound. |
| P22 | **Cap check** (HARD CAP $180). `DEV_ALLOWANCE_USD = 5.0` reserves image builds and anything Modal bills outside function wall time. Before every launch the local entrypoint computes `ledger_total + projected + DEV_ALLOWANCE_USD` and exits with code 3 (printing all three numbers) if it exceeds 180. `projected` for probe/unit/window/bench = Σ over the calls about to launch of `(timeout_s + 120)/3600 × rate` (worst case; these are cents to a few dollars). `projected` for full/sweeps = Σ over the cases about to launch of `1.25 × c_bench × (n_cells_site/n_cells_bench)^1.5 × f_cfg`, where `c_bench` = mean `cost_usd` of the benchmark cases whose `cpu` equals the launch cpu, `n_cells_site = (height//2)·(width//2)` from the manifests, `n_cells_bench` from the benchmark meta, `f_cfg = 1` for si and sweeps and `f_cfg = max(1, Σwall_siox/Σwall_si)` over the Stage-3 window results for siox. Full and sweep runs launch in chunks (full: all 34 sites `bottom`, then all 34 `top`; sweeps: variants 1-6, then 7-12; 34 or 36 cases per chunk) and the check is repeated before each chunk with `c_bench` replaced by the mean actual per-case cost of the chunks already finished in this run, scaled the same way. **Dev guard**: if `ledger_total` reaches $15 before the benchmark is launched, stop and report (test-iteration runaway). One exponent (1.5) is used everywhere, including the Stage-3 window → full-site extrapolation. | Pre-approval: "project cost from the measured benchmark; stop and ask if cumulative actual + projected would exceed $180". The 1.5 exponent is the 2D nested-dissection LU flop scaling, conservative for the 4-9× window→site upscaling; for site vs benchmark (cell ratio 0.78-1.12) it changes the projection by < 20%. The 1.25 factor and per-chunk re-check bound the overshoot to at most one chunk's worst case, which is reported (STOP B/C). |
| P23 | Provenance follows the `scripts/run_kpis.py` pattern: git commit, command, package versions (dolfinx, petsc4py, numpy, scipy, scikit-image), image tag, params_hash and the full params dict in each `result.json` and in `outputs/fem/run_log.json`, which also records the ledger total and per-mode subtotals. | Same reproducibility pattern as the KPI tables. |
| P24 | **Segmentation input: un-harmonised, identical to the KPI pipeline.** `run_case` loads `load_site(batch, site, resolution="half", normalise="none", cache_root=…)` with the default `harmonise="none"` and calls `segment(site)` unchanged. It does NOT use `harmonise="hybrid"`. | (1) D4 binds the input to `segment(site)` on `cache/half`, and spec 002 D-011 / `pmdb/segment.py` define it on un-normalised BSE with affine-invariant percentile-anchored thresholds; `docs/harmonisation.md` §4.3 states the segmenter "is unaffected by any method". (2) Measured at planning time, `segment()` on `harmonise="hybrid"` vs `"none"` differs on 0.035% / 0.050% / 0.137% / 0.038% of pixels for 71vgq3fw (strong) / kbdh4tri (strong) / 9luzk4jm (mild) / fzrt2k6r (B1), with every phase fraction within 0.001: uint8 rounding noise, not a material change. (3) The KPI features that join the FEM features in the D10 model (`outputs/kpis/site_kpis.csv`, `tile_kpis.csv`) were segmented exactly this way (`scripts/run_kpis.py:218`, `modal_app.py`), so the FEM mesh and the KPIs share identical masks; switching would introduce a gratuitous mask mismatch for no gain. (4) FEM features depend only on the label map and geometry, never on grey levels, so the Batch-3 grey-level artefact cannot leak into them; the only grey-level use is the GIF background (P19), which is visual. Guard: the Step-3 data test asserts this invariance on the four strong sites plus one mild site, and the Step-4 probe asserts that in-image segmentation equals local segmentation. |
| P25 | **Validation gates** (user-set; numbers fixed here). **Swelling stop window** (amended gate): swelling_sym(site region, s = 1) must lie in **[0.03, 0.39]**. It applies to: the Stage-4 windows (si and siox, single value), the benchmark (si, single value, sym of the bottom/top bench cases), the full si run (median over 34 sites, gate G4) and the full siox run (median over 34, gate G4x, checked before sweeps launch). It does not apply to sweep variants (`top_confined` has swelling 0 by construction). Per-site values outside the window are listed at the stop points but do not stop the run on their own. **Literature band** [0.09, 0.39] (VT1, Michael graphite-only 9% to Prado 15 wt% Si at 10 mV 39%) is REPORTED only, as `lit_band_ok` in `validation.csv` and in every stop-point report. **Failure gate G5**: a site counts as failed if either orientation has `failed_at_s` < 1.0; the gate fails at > 3 of 34 in the default `si` run. The siox failure count is reported, not gated. **G1**: T1a is the user's "analytic free-expansion unit test (J = Jλ, zero stress)" (λy = 1, so plane strain is free expansion) and must pass at its stated tolerances; all other Stage-1 tests are also gating. VT2 (sxx_mean / −10 MPa), VT4 (porosity change) and VT5 (J_si_mean) are reported, never gated. | `fem-swelling.md` amended swelling gate: stop only if < 3% or > 39% (bug-level: eigenstrain missing, wrong sign, runaway); lit band reported. The amendment is not restricted to si and its bug-level rationale is configuration-independent, so siox is gated too (comment 1). |
| P26 | **Modal-only test workflow (D16).** All dolfinx code is exercised only through `modal run modal_fem.py --mode <m>` from the repo root with the default python's `modal` client. Modes: `probe`, `unit`, `window`, `bench`, `full`, `sweeps`. Every mode writes the returned payloads locally under `outputs/modal/fem/` (gitignored): `tests/<UTC>/pytest.txt` for unit, `probe/<UTC>.json` for probe, `results/{tag}/{config}/{orientation}/{batch}__{site}.json` and `gifs/{tag}/` for case modes, plus the ledger row(s). The process exits non-zero on test failure (1), cap refusal (3) or gate failure (4). Pure-numpy modules (config, materials, geometry, result, features, gif) are tested locally with `pytest` in the default python and again inside the image by `--mode unit`. | Test env = production env, as D16 requires. Writing every payload locally makes stop-point reports reproducible without `modal volume get`. |

## 3. Out of scope (do NOT touch)

- `data/`, `data_heldout/`: never write. Held-out sites are simulated for features only, are flagged `heldout=True` in every CSV, and never appear in sweeps.
- `pmdb/io.py`, `pmdb/segment.py`, `pmdb/harmonise.py`, `pmdb/kpis/**`, `modal_app.py`, `requirements.txt`, `cache*/` (including `cache/harmonised/`), and the existing tests: read or import only, never modify.
- No local FEniCSx/dolfinx/conda environment and no `environment-fem.yml` (D16). No local Docker.
- No use of `harmonise=` other than `"none"` in the FEM pipeline (P24); the Step-3 data test is the only place that loads `harmonise="hybrid"`, read-only.
- D10 pooling, pooling checks 3-4 (follow in a separate plan), any classifier, LOSO, recomputing KPIs on the new tile grid, tier-5 per-particle features, viscoplasticity, electrochemistry, generalised plane strain, periodic BCs.
- Never merge any PR. The only PR action is opening `fem-sim → main` in Step 10.
- No Modal launch other than the modes and run lists in Steps 4 and 7-10.

## 4. Steps

### Step 1 — Config file and config loader (local)

Targets: `configs/fem/fem.yaml`, `pmdb/fem/__init__.py` (docstring only), `pmdb/fem/config.py`, `tests/test_fem_unit.py` (new).

- `configs/fem/fem.yaml`, with exactly this content (comments optional):

```yaml
version: 1
mesh: {res_nm: 100.0}
soc: {frames: 11, split: yao, s_star: 0.25, si_ratio: [0.96, 0.58], gr_ratio: [0.04, 0.42], u_max: 0.80, y_max: 0.91}
si: {model: si, beta: 2.8, E_law: qi, qi_E_MPa: [96000.0, -55000.0], nu: [0.29, -0.04],
     shah_E_MPa: [120000.0, -80000.0], shah_omega_cmax: 3.0, yield_MPa: [3000.0, 3150.0], x_per_u: 3.75}
siox: {beta: 1.6, E_MPa: 34000.0, nu: 0.17}
graphite: {strain_mode: anisotropic, eps_c_points: [[0.0, 0.0], [0.25, 0.055], [0.5, 0.055], [1.0, 0.103]],
           eps_a_over_c: [0.01, 0.103], E_MPa: [32000.0, 77000.0], nu: [0.32, -0.08]}
binder: {E_MPa: 500.0, nu: 0.34}
pore: {E_rel_binder: 1.0e-4, nu: 0.3, closure_J: 0.1}
bc: {top: free, pressure_MPa: 0.0}
solver: {ds_max: 0.05, ds_min: 0.0015625, grow_if_its_le: 5, snes_rtol: 1.0e-8, snes_atol: 1.0e-10,
         snes_max_it: 25, quadrature_degree: 2}
features: {n_tiles: 6, edge_um: 20.0, quantiles: [5, 25, 50, 75, 95, 99], n_depth_bands: 5}
gif: {width_px: 900, ladder: [[900, 128], [900, 64], [720, 64], [600, 32]], vm_range_MPa: [1.0, 10000.0],
      frame_ms: 500, max_bytes: 2000000, alpha: 0.55}
gates: {swelling_stop: [0.03, 0.39], swelling_lit_band: [0.09, 0.39], max_failed_sites: 3}
configs:
  si: {}
  siox: {si: {model: siox}}
sweeps:
  base: si
  orientation: bottom
  sites: [Batch_1/fzrt2k6r, Batch_1/uhdslk0o, Batch_2/3806gxp0, Batch_2/rxax5ozo, Batch_3/vc2whyaq, Batch_3/utfgcjfa]
  variants:
    umax_0.60: {soc: {u_max: 0.60}}
    umax_0.95: {soc: {u_max: 0.95}}
    binderE_0.05: {binder: {E_MPa: 50.0}}
    binderE_2.0: {binder: {E_MPa: 2000.0}}
    poreE_1e-6: {pore: {E_rel_binder: 1.0e-6}}
    poreE_1e-2: {pore: {E_rel_binder: 1.0e-2}}
    gr_isotropic: {graphite: {strain_mode: isotropic}}
    sstar_0.36: {soc: {s_star: 0.36}}
    split_proportional: {soc: {split: proportional}}
    top_confined: {bc: {top: confined}}
    top_stack_1MPa: {bc: {top: pressure, pressure_MPa: 1.0}}
    si_E_shah: {si: {E_law: shah}}
```

- `pmdb/fem/config.py`:

```python
FEM_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "fem" / "fem.yaml"   # same pattern as CATALOGUE_PATH
def load_params(name: str = "si", path: Path = FEM_CONFIG_PATH) -> dict
    # base = YAML minus keys {configs, sweeps}. name in configs -> deep_merge(base, configs[name]).
    # name == f"{sweeps.base}__{v}" with v in sweeps.variants -> deep_merge(base, configs[sweeps.base], variants[v]).
    # Anything else -> KeyError. Result also gets p["name"] = name. deep_merge = recursive dict merge, override wins, inputs not mutated.
def list_full_configs(path=FEM_CONFIG_PATH) -> list[str]      # ["si", "siox"] in YAML order
def list_sweep_configs(path=FEM_CONFIG_PATH) -> list[str]     # ["si__umax_0.60", ...] 12, YAML order
def sweep_sites(path=FEM_CONFIG_PATH) -> list[tuple[str, str]]
def params_hash(p: dict) -> str                               # sha1(json.dumps(p, sort_keys=True)).hexdigest()[:12]
def select_sweep_sites(site_kpis_csv: Path, per_batch: int = 2) -> list[tuple[str, str]]
    # per batch in sorted batch order: sort by (|K01_si_frac_adm - batch median|, site), take per_batch
```

Verification: `pytest -q tests/test_fem_unit.py -k config` in the default python. These tests must pass:
- `load_params("siox")["si"]["model"] == "siox"`;
- `load_params("si__umax_0.60")["soc"]["u_max"] == 0.6`, with `["si"]["model"] == "si"`;
- `len(list_sweep_configs()) == 12`;
- `select_sweep_sites(Path("outputs/kpis/site_kpis.csv")) == sweep_sites()`;
- `load_params("si")["gates"]["swelling_stop"] == [0.03, 0.39]`;
- `load_params("nope")` raises KeyError.

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
def soc_fractions(s: float, p: dict) -> tuple[float, float]          # (f_si, f_gr) per P9; split 'yao'|'proportional'
def lithiation_state(s: float, p: dict) -> tuple[float, float]      # (u, y) = (u_max f_si, y_max f_gr)
def graphite_strains(y: float, p: dict) -> tuple[float, float]       # (eps_a, eps_c)
def phase_properties(s: float, p: dict) -> dict[int, PhaseProps]     # all 5 labels, per P7/P8 (si.model si|siox, si.E_law qi|shah, graphite.strain_mode)
def si_yield_MPa(u: float, p: dict) -> float                         # P9 formula; same formula for siox
def lame(E: float, nu: float) -> tuple[float, float]                 # (mu, lam)
```

Verification: `pytest -q tests/test_fem_unit.py -k materials` in the default python. Tests that must pass:
- (a) Every row of the lit-review table at s = 0, 0.1, …, 1.0 for f_Si, f_Gr, u, y, J_Si and ε_c matches the table to abs 1.5e-3. Hard-code the 11 rows from `docs/fem/literature-review.md` §"Overall SOC"; the table is rounded to 3 decimals.
- (b) With s* = 0.36: f_Si(1) = f_Gr(1) = 1 and f_Si(0.36) = 0.96·0.36/0.7168 (rel 1e-12).
- (c) proportional: f = (s, s).
- (d) `si_yield_MPa(0.0, p) == 3000` and `si_yield_MPa(1.0, p)` = 3000 − 3150·3.75/4.75 ≈ 513.16 (abs 0.01).
- (e) Si stretch at s = 1 cubed = 3.24 (abs 2e-3). The graphite isotropic stretch cubed equals (1+ε_a)²(1+ε_c) (rel 1e-12).
- (f) Shah E at u = 0.8 = 120000 − 80000·2.24/3 MPa.
- (g) Pore E = 0.05 MPa under defaults.

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
- `pytest -q -m data tests/test_fem_data.py` (default python): 2 passed. A failure of either test is a stop-and-report (it would invalidate P2 or P24 respectively).

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
    top: Literal["free", "confined", "pressure"] = "free"
    pressure_MPa: float = 0.0
    lateral: Literal["both", "left"] = "both"      # "left" = test-only roller support
def simulate(labels: np.ndarray, px_um: float, props_fn: Callable[[float], dict[int, PhaseProps]],
             bc: BCSpec, solver_opts: dict, frames: np.ndarray, extra_targets: Sequence[float] = (),
             log: Callable[[dict], None] | None = None) -> SimResult
```

Solver implementation requirements. The algorithms are fixed by the design decisions; mechanics are up to the implementer.
- Mesh and cell→pixel map: P4. DG0 dof for cell c is `V0.dofmap.list[c, 0]`. Assert the pixel map is a permutation of range(H·W).
- DG0 coefficients: `mu, lam, lx, lz, ly`. Before every solve, set them from `props_fn(s_try)` by phase via precomputed DG0-dof index arrays per label.
- Energy, residual and Jacobian: P6, with `dx` metadata `{"quadrature_degree": solver_opts["quadrature_degree"]}`.
- BCs: P10. In `free` mode the free edge has no term. For `pressure`, mark free-edge facets whose single adjacent cell has a SOLID label as tag 1, and add `+ p_eff·dot(v, n)·ds(1)` to the residual, with p_eff = p·W/W_solid (n = outward reference normal; the sign gives compression). The solve at s = 0 is always performed: for free/confined it converges in 0 iterations; for pressure it is the preload.
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
- `test_t4_pixel_mapping`: a 7×9 label map with a single SI pixel at (row 1, col 6), the rest BINDER. `props_fn = lambda s: phase_properties(s, load_params("si"))`, frames [0, 0.2], lateral both, orientation bottom. At frame 1, the pixel with the largest vm is (1, 6), and `result.labels` equals the input.
- `test_t5_failure_path`: T1b geometry with solver_opts snes_max_it = 1, ds_min = 0.05, frames linspace(0,1,11). Expect `failed_at_s` finite and ≤ 0.1, `converged[0]` True, every later frame False with NaN fields.
- `test_t6_soft_pores`: 12×12 px. A 4×4 SI block in the centre, a 1-px PORE ring around it, BINDER outside. Default props (`phase_properties(s, load_params("si"))`), lateral both, orientation bottom, free top, frames [0, 0.25, 0.5, 0.75, 1.0]. Asserted: `converged[1]` True and min J over PORE cells at frame 1 < 0.9. Report-only (no assertion, comment 7): print one line `T6_REPORT ` + `json.dumps({"failed_at_s": ..., "min_pore_J": [per frame], "n_substeps": ...})`. The closure flag itself (J < 0.1) is computed only in `features` (Step 5).

`modal_fem.py` (this step):
- Image and volumes exactly per P20. Constants `DOLFINX_IMAGE = "ghcr.io/fenics/dolfinx/dolfinx:v0.10.0"`, `PY = "python"` (the image's PATH python, `/dolfinx-env/bin/python`), `CPU_RATE = 0.0472`, `MEM_RATE = 0.0080`, `OVERHEAD_S = 120`, `DEV_ALLOWANCE_USD = 5.0`, `CAP_USD = 180.0`, `DEV_GUARD_USD = 15.0`, `LEDGER = outputs/modal/fem/ledger.csv` (repo-relative).
- Helpers (local side): `cost_usd(wall_s, cpu, memory_mb) -> float` (P21), `ledger_total() -> float`, `append_ledger(rows: list[dict])`, `cap_check(projected_usd: float) -> None` (P22; `SystemExit(3)`).
- `@app.function(cpu=1.0, memory=2048, timeout=900, volumes={"/data": pmdb-data read-only}) def probe() -> dict`: returns python, dolfinx, petsc4py, numpy, scipy, scikit-image versions; `PETSc.Sys.hasExternalPackage("mumps")`; `str(inspect.signature(dolfinx.fem.petsc.NonlinearProblem.__init__))`; `dolfinx.fem.petsc.NonlinearProblem.solve.__doc__`; the label array `labels_from_masks(segment(load_site("Batch_3", "vc2whyaq", resolution="half", normalise="none", cache_root="/data")))`; wall_s.
- `@app.function(cpu=2.0, memory=4096, timeout=1800) def run_unit_tests(k: str = "") -> dict`: sets `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS="2"` in the subprocess env and runs `[PY, "-m", "pytest", "-q", "-rP", "tests/test_fem_unit.py", "tests/test_fem_solver.py"] + (["-k", k] if k else [])` with `cwd="/root"`; returns `{"returncode", "stdout", "stderr", "wall_s"}` (stdout/stderr truncated to their last 200 000 characters).
- `@app.local_entrypoint() def main(mode: str, k: str = "", ...)` (later steps add the case parameters). `probe`: cap check, call, write `outputs/modal/fem/probe/<UTC>.json` (labels excluded), compute the same labels locally with the default python and print the fraction of differing pixels; exit 1 if dolfinx ≠ 0.10.0, mumps is False, or the differing fraction > 1e-4. `unit`: cap check, call, write stdout+stderr to `outputs/modal/fem/tests/<UTC>/pytest.txt`, print the last 40 lines and every `T6_REPORT` line, exit with the pytest return code. Both append a ledger row.

Before the first launch (one-off, read-only on local data):
- `modal volume ls pmdb-data /half` must list 31 npz + `manifest.csv` (at planning time `/half` exists). If `/half` is missing, run `modal volume put pmdb-data cache/half /half`.
- `modal volume ls pmdb-data /heldout/half` must list `Batch_heldout__{3e122cbj,fn0mhxef,xrv9xvzb}.npz` + `manifest.csv` (uploaded by the coordinator on 2026-10-03). Only if missing: `modal volume put pmdb-data cache_heldout/half /heldout/half` (reads only `cache_heldout/`, never `data_heldout/`).
- Check the P21 rates on modal.com/pricing and update the constants if they differ.

Verification (Stage 1):
- `pytest -q tests/test_fem_solver.py` in the default python: the module is skipped, not errored.
- `modal run modal_fem.py --mode probe`: exits 0; prints dolfinx 0.10.0, mumps True, numpy 2.2.6, label disagreement 0 (≤ 1e-4 accepted). Adapt `solver.py` to the printed 0.10.0 signatures before the next command.
- `modal run modal_fem.py --mode unit`: exits 0; `tests/test_fem_unit.py` all pass in the image (numpy 2.2.6) and `tests/test_fem_solver.py` reports 9 passed. Record the `T6_REPORT` line for STOP A.

**Gate G1**: any failure of the 9 solver tests is a user gate: stop and report the failing assertion values (with the pytest.txt path). A `T6_REPORT` with `failed_at_s` < 1 is not a G1 failure; it is reported at STOP A as an early warning (Expected surprise 14).

### Step 5 — Features (pure numpy/pandas, local)

Target: `pmdb/fem/features.py`; tests in `tests/test_fem_unit.py`.

```python
def fem_tile_slices(width_px: int, px_um: float, n_tiles: int = 6, edge_um: float = 20.0) -> list[slice]   # P15
def region_metrics(r: SimResult, frame: int, cols: slice, orientation: str, p: dict) -> dict[str, float]
def run_curves(r: SimResult, orientation: str, p: dict, window: bool) -> tuple[list[dict], list[dict]]
    # (site_rows, tile_rows); one row per (region, frame); keys: region ("site" or tile index 0..5), frame, s, converged,
    # tile_x0_um, tile_x1_um, first_pore_closure_s, failed_at_s, + all metrics. window=True -> site region = all cols, no tiles.
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

Tier 3 metrics are named `q{q}_{vm|p|J}_{si|gr|binder}` and `q{q}_J_pore`. Tier 4 metrics are `band_vm_maxdev`, `band_vm_absslope`, `band_J_maxdev`, `band_J_absslope`: `band_profile(field*solid, solid, n_bands)` over R, then `band_summary`, keeping (maxdev, absslope). Any metric over an empty phase set is NaN. Unconverged frames give NaN metrics with `converged=False`. `first_pore_closure_s` is the first frame s where `pore_closed_frac > 0` over the whole domain, NaN if never; it is the same value on every row of the run.

Verification: `pytest -q tests/test_fem_unit.py -k features` (default python). Tests must pass on synthetic SimResults with no dolfinx:
- (a) `fem_tile_slices(1747, 0.1)` gives 6 slices with first start 200, last stop 1547, and contiguous edges.
- (b) An all-GRAPHITE 20×600 grid with uz = 0.1·z (bottom), J = 1.1 and sxx = −5 gives swelling 0.1 (rel 1e-6), surface_rough 0, sxx_mean −5, J_gr_mean 1.1, J_si_mean NaN.
- (c) The same with orientation top and uz = −0.1·(H_um − z) gives swelling 0.1.
- (d) A PORE pixel with J = 0.05 at frame 3 gives pore_closed_frac > 0 and first_pore_closure_s = s[3].
- (e) `symmetrise` on two rows with swelling 0.1 and 0.2 gives sym 0.15, and NaN propagates.
- (f) The number of metric keys = 17 + 60 + 4 = 81.
- (g) `swelling_gate`: 0.05 → gate_ok True, lit_band_ok False; 0.02 → both False; 0.20 → both True; 0.40 → both False.

### Step 6 — GIF rendering (local)

Target: `pmdb/fem/gif.py`; tests in `tests/test_fem_unit.py`.

```python
def render_site_gif(bse_coarse: np.ndarray, r: SimResult, title_prefix: str, swelling: Sequence[float],
                    cfg: dict) -> tuple[bytes, dict]   # (gif bytes, {"width_px", "colors", "bytes"}); raises GifTooLarge after ladder
```

Implementation per P19. matplotlib Agg; each frame drawn to an RGB array. Each frame is quantised with Pillow `quantize(colors=c, method=Image.Quantize.MEDIANCUT)`. Saved with `save_all=True, append_images=..., duration=frame_ms, loop=0, optimize=True`. Downsample for drawing with stride f = ceil(W/width_px): corners `u_nodes[:, ::f, ::f]` with the last row/col included, cell values block-mean over f×f.

Verification: `pytest -q tests/test_fem_unit.py -k gif` (default python). On a synthetic 580×1748 case (Gaussian-smoothed σ = 1 noise BSE, linear uz, log-uniform vm in [1, 1e4], last 2 frames unconverged), the output has `PIL.Image.open(...).n_frames == 11`, ≤ 2 000 000 bytes, and width ∈ {900, 720, 600}. Then `modal run modal_fem.py --mode unit` exits 0 (Steps 5-6 also pass in the image).

### Step 7 — Case runner, window mode, Stages 2-4 on Modal → STOP POINT A

Targets: `pmdb/fem/run.py`, `modal_fem.py` (add `run_case_remote` and mode `window`), `scripts/fem_collect.py` (new; this step adds only `--g2`).

```python
def run_case(batch: str, site: str, config: str, orientation: Literal["bottom", "top"], *,
             cache_root: str | Path | None = None, res_nm: float | None = None, crop_um: float | None = None,
             fields_path: Path | None = None, render_gif: bool = False, log=print) -> dict
```

Pipeline:
1. `p = load_params(config)`.
2. `site = load_site(batch, site, resolution="half", normalise="none", cache_root=cache_root)` — default `harmonise="none"` (P24).
3. `masks = segment(site)`.
4. labels = `labels_from_masks(masks)`, cropped to `central_cols` if `crop_um`, then coarsened with factor = round((res_nm or p["mesh"]["res_nm"]) / site.nm_per_px). The BSE (`site.image[..., 0]`) is cropped and coarsened the same way.
5. `simulate(..., props_fn=lambda s: phase_properties(s, p), bc=BCSpec(orientation, p["bc"]["top"], p["bc"]["pressure_MPa"]), frames=linspace(0,1,11), extra_targets=(p["soc"]["s_star"],))`.
6. `save_npz` if `fields_path`.
7. `run_curves(window=crop_um is not None)`.
8. Render the GIF if `render_gif`, catching `GifTooLarge` into `gif_error`.

Returns `{"meta": {...P23 fields, batch, site, heldout: batch == "Batch_heldout", config, orientation, H, W, px_um, n_cells, crop_um, res_nm, failed_at_s, first_pore_closure_s, n_substeps, newton_its_total, substeps, wall_s, peak_rss_mb (resource.getrusage ru_maxrss), versions, params, params_hash, gif_error}, "site_rows": [...], "tile_rows": [...], "gif": bytes | None}`. Every row gets batch, site, heldout, config and orientation.

`modal_fem.py` additions:
- `@app.function(cpu=4.0, memory=16384, timeout=21600, retries=0, volumes={"/data": pmdb-data read-only, "/out": pmdb-fem-out}) def run_case_remote(batch, site, config, orientation, tag, threads: int, render_gif: bool, crop_um: float = 0.0, res_nm: float = 0.0) -> dict`:
  - set `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` = str(threads) before importing `pmdb.fem.run`;
  - cache_root = `/data/heldout` if batch == "Batch_heldout" else `/data`; `crop_um`/`res_nm` of 0 mean None;
  - call `run_case(..., fields_path=Path(f"/out/fields/{tag}/{config}/{orientation}/{batch}__{site}.npz"), render_gif=render_gif)`;
  - write `result.json` (gif bytes removed) to `/out/results/{tag}/{config}/{orientation}/{batch}__{site}.json` and the GIF to `/out/gifs/{tag}/{batch}__{site}.gif`; `volume.commit()`;
  - add cpu, memory_mb, timeout_s, wall_s to meta (cost is computed locally into the ledger, P21, and also stored as `meta.cost_usd` in the locally written JSON);
  - return the result dict (gif included);
  - any exception → return `{"meta": {..., "error": "Type: msg"}, "site_rows": [], "tile_rows": [], "gif": None}` (modal_app.py D7 pattern).
- `main` gains parameters `config: str = "si", orientation: str = "both", sites: str = "", crop_um: float = 0.0, res_nm: float = 0.0, tag: str = "", cpu: float = 4.0, memory: int = 16384, timeout: int = 7200`. Mode `window`: cases = `sites` (comma-separated Batch/id) × orientations (`both` = bottom, top); all cases launched in one `.with_options(cpu=cpu, memory=memory, timeout=timeout).map(..., return_exceptions=True)` after the P22 cap check; GIF on for config si / orientation bottom; write each result locally per P26 and append ledger rows; print per case wall_s, n_cells, n_substeps, newton_its_total, s/Newton-iteration, peak_rss_mb, cost, and Tier-1 site metrics at s = 0.5 and 1.0; if both orientations ran, print swelling_sym(s = 1) with `swelling_gate`'s gate_ok and lit_band_ok. Exit 4 if any printed `gate_ok` is False, 1 if any case errored.
- `scripts/fem_collect.py --g2 DIR100 DIR50`: reads the two window result JSONs (site region) and writes `outputs/modal/fem/g2.csv` with, at s = 0.5 and 1.0, both values and the relative difference for `swelling`, `sxx_mean_MPa`, `porosity_change`, (J_si_mean − 1) (limit 0.10) and `vm_si_p50_MPa`, `vm_si_p95_MPa` (limit 0.20); relative difference = |a−b|/max(|b|, 1e-12) with b = the 50 nm value. Prints the table; exit 1 if any limit is exceeded.

Stages, all on site `Batch_3/vc2whyaq` (the Batch 3 sweep site, typical K01), all via Modal:
- **Stage 2, resolution check (gate G2)**:
  `modal run modal_fem.py --mode window --sites Batch_3/vc2whyaq --config si --orientation bottom --crop-um 20 --res-nm 100 --tag stage2_100`,
  the same with `--res-nm 50 --tag stage2_50`, then
  `python scripts/fem_collect.py --g2 outputs/modal/fem/results/stage2_100 outputs/modal/fem/results/stage2_50` — exit 0 = pass.
- **Stage 3, end-to-end window with timings**:
  `modal run modal_fem.py --mode window --sites Batch_3/vc2whyaq --config si --orientation both --crop-um 40 --res-nm 100 --tag stage3`, then the same with `--config siox`. Extrapolate to a full site per case as measured seconds per Newton iteration × (n_cells_site/n_cells_window)^1.5 × the measured iteration count (P22 exponent). Open `outputs/modal/fem/gifs/stage3/Batch_3__vc2whyaq.gif` and check frames 0, 5 and 10 visually: the warp goes upward and the colour bar is fixed.
- **Stage 4, validation on the window (gate G4w)**: from the Stage-3 output, swelling_sym(s = 1) for si and for siox. Each must have gate_ok (window [0.03, 0.39], P25). Report lit_band_ok and, without gating: sxx_mean_MPa(s = 1)/(−10) (VT2), porosity_change and porosity_rel_change at s = 1 (VT4), J_si_mean(s = 1) against the plane-strain expectation (VT5), si_yield_frac, pore_closed_frac, first_pore_closure_s and failed_at_s.

Verification:
- `pytest -q tests/test_fem_unit.py` and `pytest -q -m data tests/test_fem_data.py` pass in the default python; `modal run modal_fem.py --mode unit` exits 0.
- Result JSONs exist under `outputs/modal/fem/results/{stage2_100,stage2_50,stage3}/`; `ledger.csv` has a row for every call so far.

**STOP POINT A (mandatory; no benchmark spend yet).** Report back to the dispatcher with:
- the G1 test summary and the `T6_REPORT` line;
- the probe versions;
- the G2 table (both resolutions, both frames, relative differences, pass/fail);
- the Stage-3 timings and full-site extrapolation, plus the siox/si wall ratio (f_cfg);
- the G4w numbers (gate_ok, lit_band_ok) and the VT2/VT4/VT5 report;
- the ledger total;
- any Expected-surprise responses used.

Do not continue until the dispatcher replies. If G1, G2 or G4w fails, say so first.

### Step 8 — Benchmark, full/sweep modes, collector (Stage 5) → STOP POINT B

Targets: `modal_fem.py` (modes `bench`, `full`, `sweeps`), `scripts/fem_collect.py` (full tables).

`modal_fem.py`:
- **bench**: three cases on `Batch_3/vc2whyaq`, full site, config si: (bottom, cpu 4, tag `bench`), (top, cpu 4, tag `bench`), (bottom, cpu 8 via `run_case_remote.with_options(cpu=8.0, memory=16384, timeout=21600)`, tag `bench_cpu8`). GIF on for the first. Cap check with the worst-case projection (P22). Print the swelling_sym gate for the two cpu-4 cases.
- **full**: all 31 labelled sites (from `cache/half/manifest.csv`) + 3 held-out (from `cache_heldout/half/manifest.csv`) for `--config`, in two chunks (all 34 `bottom`, then all 34 `top`), each via `.with_options(cpu=cpu, memory=memory, timeout=timeout).map(..., return_exceptions=True)`, tag `full`. P22 cap check before each chunk (exit 3 on refusal). GIF only for si/bottom. After both chunks, print the swelling gate for the median swelling_sym and the G5 count; exit 4 if the median fails the window (for si: also if G5 fails).
- **sweeps**: `sweep_sites()` × `list_sweep_configs()` × {bottom}, tag `sweeps`, in two chunks (variants 1-6, then 7-12, YAML order), P22 check before each chunk. No swelling gate.
- All case modes: each returned result is written locally per P26; lost/errored calls get ledger rows per P21; print a summary: n ok, n error, n lost, n failed_at_s < 1, total and mean wall, chunk cost, ledger total.
- `--sites` (comma-separated Batch/id) restricts the case list; it is used for reruns.

`scripts/fem_collect.py --results outputs/modal/fem/results`: reads all JSONs and writes the committed tables to `outputs/fem/`:
- `site_curves.csv` and `tile_curves.csv` (tag full; configs si and siox; orientations bottom, top and sym via `symmetrise`);
- `sweep_site_curves.csv` and `sweep_tile_curves.csv` (tag sweeps, plus the si/bottom default rows of the 6 sites relabelled config `si`);
- `robustness.csv` (P18: columns metric, frame, s, variant, rho; plus `robust_summary.csv` with metric, frame, min_rho, robust);
- `validation.csv` (per site × config: swelling bottom/top/sym at s = 1, `gate_ok`, `lit_band_ok`, sxx_ratio_vt2, porosity_change, porosity_rel_change, J_si_mean, si_yield_frac, failed_at_s, first_pore_closure_s);
- `run_log.json` (P23 provenance + per-case meta without the substeps list + ledger total and per-mode subtotals; substep logs go to the gitignored `outputs/modal/fem/substeps/`);
- copies `gifs/full/*.gif` to `outputs/fem/gifs/`.

It prints the gate summary (P25): median swelling_sym for si and siox with gate_ok and lit_band_ok, sites outside the stop window, the count of failed sites for si and siox, and the ledger total. Floats are written with `float_format="%.6g"`. It never overwrites an existing table with fewer sites than it already has: it writes `<name>_partial.csv` instead and exits 1, the same rule as modal_app.py.

Run `modal run modal_fem.py --mode bench`.

Verification: 3 results with no `error`. A `fields.npz` from the volume (`modal volume get pmdb-fem-out /fields/bench/si/bottom/Batch_3__vc2whyaq.npz /tmp/`) loads with `load_npz` and the shapes are (11, H/2, W/2) and (11, H/2+1, W/2+1, 2). The benchmark GIF is ≤ 2 MB.

**STOP POINT B (mandatory).** Report:
- per case: wall_s, n_substeps, newton_its_total, peak_rss_mb, cost_usd, failed_at_s, swelling_sym(s = 1) with gate_ok and lit_band_ok, and VT2/VT4/VT5;
- the cpu 4 vs cpu 8 cost per case and the chosen cpu (the lower cost_usd; if the two are within 10%, choose the lower wall time);
- memory for production = max(4096, ceil(1.5·peak_rss_mb/1024)·1024) MB;
- timeout = min(86400, max(7200, 3·max bench wall_s at the chosen cpu, rounded up to the hour));
- the **expected** projection: P22 formula summed over 68 si + 68 siox + 72 sweep cases, + ledger total + DEV_ALLOWANCE, compared with $180;
- the **worst-case** figure: Σ over the same 208 cases of (timeout_s + 120)/3600 × rate at the chosen cpu/memory, + ledger total, and the worst case of the largest single chunk (36 cases) — reported, not gated (it bounds the overshoot between per-chunk checks);
- whether the vm range [1, 1e4] MPa covered the benchmark (Expected surprise 9).

Do not launch `full` until the dispatcher replies.

### Step 9 — Full Si run → STOP POINT C

Run `modal run modal_fem.py --mode full --config si --cpu <chosen> --memory <chosen> --timeout <chosen>`. Rerun errored or lost cases once with `--sites`. Then run `python scripts/fem_collect.py --results outputs/modal/fem/results`.

Verification: 68 si result JSONs with no `error`, and `outputs/fem/validation.csv` has 34 si rows. Gate check (P25):
- G4 = median swelling_sym(s = 1) over 34 sites in [0.03, 0.39];
- G5 = sites with failed_at_s < 1 in either orientation ≤ 3.

**STOP POINT C (mandatory).** Report:
- G4/G5 values and verdicts, the median's lit_band_ok, and sites individually outside [0.03, 0.39];
- the swelling distribution by batch;
- VT2/VT4 medians;
- the list of failed sites with failed_at_s and first_pore_closure_s;
- the ledger total;
- the expected and worst-case projection for siox + sweeps against the $180 cap.

Do not launch siox or sweeps until the dispatcher replies.

### Step 10 — SiOx full run (gate G4x), sweeps, collection, docs, commit, push, PR

1. `modal run modal_fem.py --mode full --config siox --cpu <chosen> --memory <chosen> --timeout <chosen>`; rerun errored/lost cases once; `python scripts/fem_collect.py --results outputs/modal/fem/results`. **Gate G4x**: median siox swelling_sym(s = 1) in [0.03, 0.39]. If it fails, stop and report before launching sweeps.
2. `modal run modal_fem.py --mode sweeps --cpu <chosen> --memory <chosen> --timeout <chosen>`; rerun errored/lost cases once; run `fem_collect.py` again.
3. Write `docs/fem/README.md`, which describes: the Modal-only workflow and commands (probe, unit, window, bench, full, sweeps, collect) with the pinned image tag; the segmentation-input decision (P24, with the measured invariance numbers); the output tables and their columns (metric definitions copied from Step 5); the orientation `sym` rule (P17); the volume layout (P14); the ledger and cap accounting (P21-P22); the robustness flag (P18); the gate results with numbers (stop window and literature band shown separately).
4. Add a "FEM simulation" pointer line to the top-level `README.md` under the Modal section.
5. Commit on `fem-sim` and `git push origin fem-sim`. Then open the PR: `gh pr create --base main --head fem-sim --title "FEM lithiation-swelling simulation: solver, Modal runner, features, GIFs"` with a body summarising the gates (G1, G2, G4w, G4, G4x, G5 values), the robust-feature counts, the final ledger total and a pointer to `docs/fem/README.md`. Never merge it. If an open PR from `fem-sim` already exists, the push updates it; do not open a second one.

Verification:
- `outputs/fem/site_curves.csv` has 34×2×3×11 = 2244 rows and `tile_curves.csv` has 13 464 rows.
- `sweep_site_curves.csv` has (12+1)×6×11 = 858 rows.
- `robustness.csv` has 81 metrics × 10 frames × 12 variants = 9720 rows.
- `outputs/fem/gifs/` holds 34 files, each ≤ 2 000 000 bytes.
- `pytest -q -m "not data"` and `pytest -q -m data` pass in the default python (FEM solver tests skipped there).
- `modal run modal_fem.py --mode unit` exits 0 on the final commit.
- `gh pr view fem-sim --json state,baseRefName` shows `OPEN`, base `main`.
- Report G4x and the siox failure count, the number of robust (metric, frame) pairs per tier, and the final ledger total.

## 5. Expected surprises

1. **dolfinx 0.10.0 API detail differs from P12** beyond what the spike verified (e.g. `dofmap.list` shape, facet-tag helpers for the pressure BC). Adapt mechanically to the signatures printed by the probe. If `snes_linesearch_type bt` is rejected or makes T1-T4 fail where `basic` passes, pre-authorized: switch to `basic` (the spike-verified value) in `solver.py` and record it. If MUMPS is unavailable, stop and escalate.
2. **pmdb fails to import or run under numpy 2.2.6 in the image** (probe or unit mode). Do not edit `pmdb/io.py`, `segment.py` or `kpis/**`, and do not change numpy in the image (dolfinx/petsc4py are compiled against it). Stop and escalate.
3. **Probe label disagreement between image and local segmentation > 1e-4.** Stop and escalate: it would break the FEM/KPI mask identity that P24 relies on. ≤ 1e-4 (including exactly 0) is accepted and logged.
4. **A pinned pip version does not install on Python 3.12 / numpy 2.2.6.** Pre-authorized: use the newest patch release of the same major.minor that installs, never changing numpy or the scipy/scikit-image minor versions; record it in run_log. If scipy 1.14.x or scikit-image 0.25.x cannot install, stop and escalate.
5. **Packages installed with `uv pip install --system` are not importable at runtime, or PYTHONPATH is lost.** Pre-authorized: replace `--system` with `--python /dolfinx-env/bin/python`, and keep the P20 `.env` PYTHONPATH (already set explicitly). If ghcr.io is unavailable, the identical Docker Hub tag `dolfinx/dolfinx:v0.10.0` may be used. Any other image change: stop and escalate (no micromamba/conda fallback, D16).
6. **Newton stagnates.** If the logs show quadratic convergence that stalls between 1e-8 and 1e-6 relative residual, causing max_it failures, set `snes_rtol: 1.0e-6` in `fem.yaml` (all configs), record it in the hand-back, and rerun Stage 1-3. Any other recurring Newton failure before s = 1 on the window: stop at STOP POINT A and report the substep log.
7. **Window or benchmark swelling below 9%.** Likely (reviewer estimate 8-14%, planner lower bound ~7%). Not a stop: record lit_band_ok = False and report the per-phase J means. Only < 3% or > 39% stops (P25).
8. **Stage 2 resolution gate fails.** Not pre-authorized, because running at 50 nm changes cost by about 8×. Stop and report.
9. **The benchmark vm range is not covered.** If the si/bottom benchmark has p99.9 of solid vm at s = 1 > 1e4 MPa, or p1 of solid vm at s = 0.1 < 1 MPa, set `gif.vm_range_MPa` to [10^floor(log10 p1), 10^ceil(log10 p99.9)] from the benchmark before the full run. Report it at STOP POINT B. (Likely: plane-strain syy on Si is of order 1e4 MPa.)
10. **GIF exceeds 2 MB after the ladder for some site.** Record `gif_error`, continue, and list the sites at the end. Do not change the ladder.
11. **Benchmark OOM at 16 GiB, or a cap check refuses (exit 3).** Stop and report.
12. **Pre-existing test failures in the default python** that also fail on `main`. Report them; do not fix them.
13. **The 1 MPa stack-pressure or confined variant fails early in the sweep, or `top_confined` swelling is identically 0.** Expected. It is recorded as `failed_at_s` / constant values and makes the affected (metric, frame) ρ NaN, so those features become not robust. No rerun.
14. **`T6_REPORT` shows `failed_at_s` < 1 (pore inversion next to Si).** Not a G1 failure and no parameter change is pre-authorized. Report it at STOP A as an early warning for G5; the Stage-3 window to s = 1 is the deciding evidence.
15. **Ledger reaches $15 before the benchmark** (dev guard, P22). Stop and report the per-mode subtotals.
16. **`modal volume ls pmdb-data /half` lists other than 31 npz + manifest.** Re-upload `cache/half` with `modal volume put --force pmdb-data cache/half /half` and re-check; the local cache is the source of truth.

## 6. Done criteria

- G1 (analytic tests, including T1a = free expansion), G2 (resolution), G4w, G4, G4x and G5 all pass, or each failure was reported at its stop point and the dispatcher instructed continuation.
- Committed on `fem-sim` and pushed:
  - code (`pmdb/fem/*`, `configs/fem/fem.yaml`, `scripts/fem_collect.py`, `modal_fem.py`);
  - tests (`tests/test_fem_unit.py`, `tests/test_fem_solver.py`, `tests/test_fem_data.py`);
  - `outputs/fem/{site_curves,tile_curves,sweep_site_curves,sweep_tile_curves,robustness,robust_summary,validation}.csv`, `outputs/fem/run_log.json`, 34 GIFs ≤ 2 MB;
  - `docs/fem/README.md` and the top-level README pointer.
- A PR `fem-sim → main` is open and unmerged.
- Row counts as in Step 10.
- Ledger total in `run_log.json` (all modes, including probe/unit/window) ≤ $180.
- Both pytest invocations in the default python (`-m "not data"` and `-m data`) pass; `modal run modal_fem.py --mode unit` passes.
- No file under `data/` or `data_heldout/` changed (`git status --porcelain data data_heldout` is empty). Held-out rows are all `heldout=True` and absent from the sweep tables.

## 7. Revision log

### Round 1 (2026-10-03) — trigger: plan review `.claude/reports/fem-build.plan-reviewer.md` (VERDICT REVISE, comments 1-7) plus dispatcher changes A (D16 Modal-only), B (harmonisation decision), C (branch fem-sim / new PR)

Review comments:
1. [CRITICAL, swelling gate] **accepted.** New P25 replaces old P23: stop window [0.03, 0.39]; literature band [0.09, 0.39] reported only as `lit_band_ok` (validation.csv, window/bench printouts, every stop report). G4w, benchmark, G4 (Step 9) and Expected surprise 7 restated. Decided: the stop window also applies to siox (new gate G4x, checked before sweeps launch in Step 10), because the amendment is not restricted to si and its bug-level rationale is configuration-independent; it does not apply to sweep variants (top_confined is 0 by construction). G5 stays si-only per the pre-approval ("default run"); siox failures are reported. T1a is now named as the user's free-expansion test (λy = 1) in Step 4 and P25. Gate numbers are also in `fem.yaml` (`gates:`) so code and plan share one source.
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
