# FEM build: 2D plane-strain finite-strain lithiation swelling, features, GIFs, Modal runner

Binding inputs: `.claude/plans/fem-swelling.md` (D1-D15, budget, user pre-approval and HARD CAP), `docs/fem/literature-review.md` (parameters, SOC mapping, validation targets VT1-VT6, "For the planner"). Where this plan and those files disagree, those files win: stop and report.

## 1. Context

The project wants batch-discriminating features from a mechanics simulation of each SEM cross-section. Every site's segmentation (`pmdb.segment.segment`) becomes a pixel-based finite-element mesh. The mesh is lithiated from 0 to 100% SOC under uniform-per-phase eigenstretches. Si swells isotropically, graphite anisotropically (c ∥ z), and binder and pores do not swell. The solve is plane-strain, finite-strain, compressible neo-Hookean, with FEniCSx/dolfinx. Fields are snapshotted at 11 SOC frames and reduced to tile and site curve features (D8 tiers 1-4) on the D9 tile grid, plus one GIF per site (D6, D15). Production runs on Modal: Si and SiOx on all 34 sites in both collector orientations, and D14 sweeps on 6 sites. Classification and pooling (D10) are not part of this plan.

## 2. Design decisions

Conventions used throughout:
- Mesh axis 0 = **x** (coating width = image columns).
- Mesh axis 1 = **z** (thickness). z points toward image row 0: `z = (H - row - 0.5)·h` at a cell centre.
- **y** = out-of-plane.
- Lengths in µm, stresses and moduli in MPa (so E_Si = 96 000 MPa), forces per unit depth in MPa·µm.

| # | Decision | Rationale |
|---|---|---|
| P1 | Module layout. `pmdb/fem/{__init__,config,materials,geometry,result,solver,features,gif,run}.py`; `configs/fem/fem.yaml`; `scripts/fem_run_local.py`, `scripts/fem_collect.py`; `modal_fem.py` (new, at repo root); `environment-fem.yml`. Only `solver.py` imports dolfinx/ufl/petsc4py/mpi4py, and `run.py` imports `solver` lazily inside the function that needs it. | Everything except the solve is unit-testable in the existing torch/numpy<2 env. This matches how `pmdb.kpis` stays independent of the segmenter. |
| P2 | Mesh resolution 100 nm/px (coarsen the 50 nm label map 2×2). Native 50 nm is used only for the Stage-2 resolution check. | A full site is about 580 × 1750 coarse px, so about 1.0 M quads and 2.0 M DOFs. At 50 nm it would be 4× the cells and about 8× the direct-solve cost. The Stage-2 gate tests the convergence of this choice. |
| P3 | Label map `uint8`: BINDER=0 (unassigned solid), SI=1, GRAPHITE=2, PORE=3, ARTEFACT=4. Coarsening is a 2×2 majority vote with ties broken by priority SI > PORE > GRAPHITE > BINDER > ARTEFACT. An odd trailing row or column is dropped. | Si is the rarest phase and the main driver, so ties keep Si. Thin pores carry most of the accommodation (lit review C16), so pore beats the solids. Artefact is ersatz-like anyway, so it yields. |
| P4 | Mesh is a structured quadrilateral `create_rectangle` over `[0, W·h] × [0, H·h]` with one cell per coarse pixel. The cell→pixel map comes from cell midpoints. Materials are DG0 functions indexed through that map. | The mesh matches the pixel grid exactly, so there is no meshing step and phase tags are exact. DG0 makes per-phase updates a plain array assignment. |
| P5 | Displacement space is vector Q1 (`("Lagrange", 1, (2,))` on quads). Quadrature degree is 2. | Q1 nodes sit at pixel corners, so nodal displacement maps exactly to the corner grid. ν ≤ 0.34 everywhere means no volumetric locking. Degree 2 (2×2 Gauss) is exact enough and stops UFL from estimating a huge degree for the log terms. |
| P6 | Energy per reference volume ψ₀ = J_λ·[μ/2 (tr Ce − 3) − μ ln Je + λ/2 (ln Je)²]. Here Fe = F₃·Fλ⁻¹, F₃ = [[F₂, 0], [0, 1]] (total out-of-plane stretch 1 = plane strain), Fλ = diag(λx, λz, λy), Je = det F₂ /(λx λz λy), Ce = Feᵀ Fe (3×3), so Ce₃₃ = 1/λy². Residual = derivative of ∫ψ₀ dx; Jacobian = its derivative. | Exactly the lit-review energy (R5:E13), its plane-strain I1 and its det-Fλ weighting. |
| P7 | Eigenstretches (λx, λz, λy). **Si**: J_λ^(1/3)·(1,1,1) with J_λ = 1 + β·u, β = 2.8. **SiOx**: the same form with β = 1.6. **Graphite (anisotropic)**: (1+ε_a, 1+ε_c, 1+ε_a), with ε_c from np.interp over the breakpoints and ε_a = 0.01·ε_c/0.103. **Graphite (isotropic variant)**: (1+ε_iso)·(1,1,1), with (1+ε_iso)³ = (1+ε_a)²(1+ε_c). **Binder, pore and artefact**: (1,1,1). | Lit review "For the planner". The out-of-plane graphite stretch is the a-axis value (c ∥ z), and the isotropic variant preserves the lattice volume. |
| P8 | Moduli. **Si (Qi, default)**: E = 96 000 − 55 000·u MPa, ν = 0.29 − 0.04·u. **Si (Shah variant)**: E = 120 000 − 80 000·(β·u/3.0) MPa with the same ν. C/Cmax = β·u/3.0 because Shah's ΩCmax = 3.00 is volumetric. **SiOx**: 34 000 MPa, ν 0.17. **Graphite**: E = 32 000 + 77 000·y MPa, ν = 0.32 − 0.08·y. **Binder**: 500 MPa, ν 0.34. **Pore and artefact**: E = 1e-4·E_binder, ν 0.3. Lamé: μ = E/(2(1+ν)), λ = Eν/((1+ν)(1−2ν)). | Lit review table and D11/D13. u is not C/Cmax (Contradictions), so equal volumetric strain is the only defensible mapping. |
| P9 | SOC mapping (general form, recovers the lit-review table at s* = 0.25): f_Si(s) = [0.96·min(s,s*) + 0.58·max(s−s*,0)] / [0.96 s* + 0.58(1−s*)] and f_Gr(s) = [0.04·min(s,s*) + 0.42·max(s−s*,0)] / [0.04 s* + 0.42(1−s*)]. The proportional variant sets f_Si = f_Gr = s. Then u = u_max·f_Si (0.80) and y = y_max·f_Gr (0.91). Si yield flag: σY = 3000 − 3150·x/(1+x) MPa with x = 3.75·u. The SiOx run uses the same σY formula (borrowed, documented). | Lit review "Overall SOC". The s* = 0.36 sweep needs the normalisers recomputed so that f(1) = 1. |
| P10 | BCs. Lateral edges x = 0 and x = W: u_x = 0. Orientation `bottom`: collector at z = 0 (image last row) with u_z = 0; free edge at z = H. Orientation `top`: collector at z = H (image row 0) with u_z = 0; free edge at z = 0. Free-edge modes: `free` (traction-free, default), `confined` (u_z = 0), `pressure` (dead-load normal traction −p·n). The pressure is applied only on free-edge facets of solid cells (SI, GRAPHITE, BINDER) and scaled by W/W_solid so the total force equals p·W. It is applied at s = 0 as a preload, so frame 0 holds the preload state. A test-only lateral mode `left` (u_x = 0 on x = 0 only) gives roller-only support for free-expansion tests. | Lit review BCs and D12. Solid-only pressure stops the 1 MPa load from crushing 0.05 MPa ersatz pores that open onto the surface (the separator touches solids). |
| P11 | Load stepping. Mandatory targets are the 10 frame values plus s*. Δs starts at 0.05, Δs_max = 0.05, Δs_min = 0.1/64. On failure (SNES reason ≤ 0, NaN, or a PETSc exception), restore the last converged u and halve Δs. Below Δs_min the run has failed: `failed_at_s` = the attempted s, later frames are NaN, and the solve stops. On success with ≤ 5 Newton iterations, double Δs (capped). Predictor: u ← u_n + (Δs_try/Δs_last)(u_n − u_{n−1}) once two converged states exist, otherwise u_n. | D11 failure semantics. Linear extrapolation cuts Newton iterations, which dominate cost. |
| P12 | Nonlinear solver: dolfinx ≥ 0.10 `dolfinx.fem.petsc.NonlinearProblem` (SNES). Options: `snes_type newtonls`, `snes_linesearch_type bt`, `snes_rtol 1e-8`, `snes_atol 1e-10`, `snes_max_it 25`, `ksp_type preonly`, `pc_type lu`, `pc_factor_mat_solver_type mumps`, `mat_mumps_icntl_14 100`. The run is serial: assert `MPI.COMM_WORLD.size == 1`. Threads (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) equal the container CPU count. | D5 (SNES + MUMPS). Serial keeps the pixel mapping and gathering trivial. 2D nested-dissection LU at 2 M DOFs is seconds to tens of seconds with threaded BLAS. Parallelism comes from running many cases at once on Modal. |
| P13 | Post-processing at each frame. Interpolate UFL expressions into DG0 at cell midpoints: total J = det F₂ and the Cauchy stress σ = (1/Je)[μ(Be − I) + λ ln Je I], Be = Fe Feᵀ (3×3). Components are sxx, szz, sxz and syy (out-of-plane). von Mises vm = sqrt(½[(sxx−szz)² + (szz−syy)² + (syy−sxx)²] + 3 sxz²). Pressure p = −(sxx+szz+syy)/3 is derived, not stored. Nodal u is mapped to corners: col = rint(x/h), row = H − rint(z/h). | Fields are on the reference (coarse) pixel grid, as D15 and the proposed decisions require. Cell-centre values are exact for DG0 and need no averaging. |
| P14 | Storage. One `fields.npz` (np.savez_compressed) per (run_tag, config, orientation, site), on Modal volume `pmdb-fem-out` under `/fields/...`. Keys: `labels` uint8 (H,W); `s` (11,); `converged` bool (11,); `u_nodes` float32 (11,H+1,W+1,2) as (ux, uz) in µm; `J, sxx, szz, sxz, syy, vm` float32 (11,H,W), in MPa except J. A small `result.json` (meta + feature rows) goes under `/results/...` and GIFs under `/gifs/...`. | Fields are kept so the tiers can be re-extracted (D8). Large fields are kept apart from the small results so `modal volume get` of the results stays small. |
| P15 | Feature regions (D9). 6 tiles per site with fixed edges `linspace(edge_px, W − edge_px, 7).round()`, edge_px = round(20 µm / h). Full height, no overlap. The site-level region is the union of the 6 tiles (central region), so site and tile metrics share one definition and exclude the lateral-BC bands. Cropped windows (local stages) have no tiles, and their site region is all columns. | D9 tiling with a fixed tile count (D10 risk note). |
| P16 | Feature set per (region, frame). Tier 1 (17 named metrics, §Step 5). Tier 3: quantiles q ∈ {5,25,50,75,95,99} of {vm, p, J} over each of {si, gr, binder}, plus J over pore (60 metrics). Tier 4: depth-band (5 bands, reusing `pmdb.kpis.fields.band_profile` and `band_summary`) maxdev and absslope of vm and J over solid cells (4 metrics). Tier 5 (per-particle) is not built. Metric code is identical for tiles and site. | D8 tiers 1-4. Tier 4 is magnitude-only like K12 because the depth sign is unknown. Fields are stored, so tier 5 can be added later without new runs. |
| P17 | Orientation combination. Both orientations are simulated. CSVs carry `orientation ∈ {bottom, top, sym}`. `sym` = the arithmetic mean of bottom and top per metric per (site, config, region, frame). `sym` is NaN if either is NaN, `converged` = both, `first_pore_closure_s` = min, `failed_at_s` = min. The primary downstream feature set is `sym`. | No foil is visible (docs/data-processing.md l.55, spec 002 "Directions"), so the collector side is unknown per site. The mean is invariant to a vertical flip of the image, so features cannot encode an arbitrary orientation guess. Per-orientation rows are kept for the later classification plan. |
| P18 | Sweeps (D14). Base config `si`, orientation `bottom` only, 12 variants, on the 6 sites `Batch_1/fzrt2k6r, Batch_1/uhdslk0o, Batch_2/3806gxp0, Batch_2/rxax5ozo, Batch_3/vc2whyaq, Batch_3/utfgcjfa`. These are the 2 per batch nearest the batch-median `K01_si_frac_adm` in `outputs/kpis/site_kpis.csv`, ties by site id, and were computed at planning time. The comparator is the default `si`/`bottom` rows of the same 6 sites from the full run. Robustness is the Spearman ρ across the 6 site-level values between default and variant, per (metric, frame) for frames 1-10. `robust` = ρ ≥ 0.8 for all 12 variants. NaN ρ (constant or NaN input) counts as not robust. Features are flagged, not dropped. | D14 exactly, with one orientation, which halves sweep cost. Robustness to a parameter is a property of the parameter, not of the BC orientation. Dropping features is a classification-plan decision. |
| P19 | GIF (D15). Rendered in-container only for config `si`, orientation `bottom` (labelled "collector assumed at bottom"). It is a single panel. The BSE (coarse block-mean, grey, per-site p1-p99 stretch) is warped by displacement via `pcolormesh` on deformed corner coordinates. von Mises is overlaid on solid cells only (alpha 0.55, `magma`, LogNorm fixed at [1, 1e4] MPa, the same for every site and frame). Title: site, scenario, SOC %, swelling %. Axes are fixed per site: x [−1, W+1] µm, z [−0.05H, 1.45H]. 11 frames, 500 ms, loop. Frames after a failure repeat the last converged frame with the banner "solver failed at SOC x%". The size ladder is (900 px,128 colours) → (900,64) → (720,64) → (600,32); the first ≤ 2 000 000 bytes wins, otherwise `gif_error` is recorded. | D6/D15: a fixed colour scale across sites, ≤ 2 MB and about 900 px wide. In-container rendering avoids downloading fields. Collector at the bottom is the conventional picture. |
| P20 | Modal. New `modal_fem.py`, app `pmdb-fem`. Image: `modal.Image.micromamba(python_version="3.12").micromamba_install(<conda-forge pkgs>, channels=["conda-forge"])` with the same dolfinx version as local `pmdb-fem`, plus `c-compiler` (FFCx JIT needs a C compiler), `add_local_python_source("pmdb")` and `add_local_file("configs/fem/fem.yaml", "/root/configs/fem/fem.yaml")`. The input volume `pmdb-data` (read-only) is mounted at `/data`, holding `/data/half` (labelled) and `/data/heldout/half` (held-out, uploaded in Step 8). The output volume `pmdb-fem-out` (`create_if_missing=True`) is mounted at `/out`. One function `run_case` handles one (site, config, orientation). Defaults are cpu=4, memory=16384, timeout=21600, retries=1, and are overridden with `.with_options` after the benchmark. | This follows `modal_app.py` (read-only volume, `add_local_python_source`, record-and-continue errors). The same conda-forge build runs locally and remotely. One case per container gives the cheapest parallelism for a serial solver. |
| P21 | Cost accounting. `cost_usd = wall_h × (cpu × 0.0472 + mem_GiB × 0.0080)`. These are Modal's published $0.0000131 per core-second and $0.00000222 per GiB-second; check them against modal.com/pricing in Step 8 and update the constants if they differ. Every result records cpu, memory_mb, wall_s, peak_rss_mb and cost_usd. `fem_collect.py` prints the cumulative cost. The cap check before each launch: cumulative actual + projected ≤ $180 (user HARD CAP). | User pre-approval terms. |
| P22 | Provenance follows the `scripts/run_kpis.py` pattern: git commit, command, package versions (dolfinx, petsc4py, numpy), params_hash and the full params dict in each `result.json` and in `outputs/fem/run_log.json`. | Same reproducibility pattern as the KPI tables. |
| P23 | Validation gates (user-set; the numbers are fixed here). **VT1 band**: swelling (sym, site region, s = 1) in [0.09, 0.39], from graphite-only 9% (Michael) to 15 wt% Si at 10 mV 39% (Prado). For windows (Stage 4) and the benchmark site it applies to the single value. For full runs it applies to the median over the 34 sites. **Failure gate**: a site counts as failed if either orientation has `failed_at_s` < 1.0, and the gate fails at > 3 of 34 (default `si` run). VT2 (sxx_mean / −10 MPa), VT4 (porosity change) and VT5 (J_si_mean) are reported, never gated. | User pre-approval: "simulated electrode swelling falls outside the literature bands" and "> 3 of 34". The 9-39% band spans the lit-review VT1 values relevant to Si/graphite blends at our Si content (K01 about 0.06). |

## 3. Out of scope (do NOT touch)

- `data/`, `data_heldout/`: never write. Held-out sites are simulated for features only, are flagged `heldout=True` in every CSV, and never appear in sweeps.
- `pmdb/io.py`, `pmdb/segment.py`, `pmdb/kpis/**`, `modal_app.py`, `requirements.txt`, the existing tests: read or import only, never modify.
- D10 pooling, any classifier, LOSO, recomputing KPIs on the new tile grid, tier-5 per-particle features, viscoplasticity, electrochemistry, generalised plane strain, periodic BCs.
- No merging of PR #18 or of anything else to main.
- No Modal launch outside Steps 8-10, and none beyond the run lists given there.

## 4. Steps

### Step 1 — Environment, config file, config loader

Targets: `environment-fem.yml`, `configs/fem/fem.yaml`, `pmdb/fem/__init__.py` (docstring only), `pmdb/fem/config.py`, `tests/test_fem_unit.py` (new).

- `environment-fem.yml`: name `pmdb-fem`, channel conda-forge. Packages: python=3.12, `fenics-dolfinx=<V>`, mpich, c-compiler, numpy, scipy, scikit-image, pandas, matplotlib, pillow, tifffile, imagecodecs, pyyaml, pytest, with pip: modal. `<V>` is the exact version reported by `conda list -n pmdb-fem fenics-dolfinx`; it must be ≥ 0.10 (see Expected surprises). Install any of these packages missing from the existing `pmdb-fem` env into it.
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

Verification: in the default env (`python` = the repo's usual interpreter), run `pytest -q tests/test_fem_unit.py -k config`. These tests must pass:
- `load_params("siox")["si"]["model"] == "siox"`;
- `load_params("si__umax_0.60")["soc"]["u_max"] == 0.6`, with `["si"]["model"] == "si"`;
- `len(list_sweep_configs()) == 12`;
- `select_sweep_sites(Path("outputs/kpis/site_kpis.csv")) == sweep_sites()`;
- `load_params("nope")` raises KeyError.

### Step 2 — Materials and SOC driver (pure numpy)

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

Verification: `pytest -q tests/test_fem_unit.py -k materials`. Tests that must pass:
- (a) Every row of the lit-review table at s = 0, 0.1, …, 1.0 for f_Si, f_Gr, u, y, J_Si and ε_c matches the table to abs 1.5e-3. Hard-code the 11 rows from `docs/fem/literature-review.md` §"Overall SOC"; the table is rounded to 3 decimals.
- (b) With s* = 0.36: f_Si(1) = f_Gr(1) = 1 and f_Si(0.36) = 0.96·0.36/0.7168 (rel 1e-12).
- (c) proportional: f = (s, s).
- (d) `si_yield_MPa(0.0, p) == 3000` and `si_yield_MPa(1.0, p)` = 3000 − 3150·3.75/4.75 ≈ 513.16 (abs 0.01).
- (e) Si stretch at s = 1 cubed = 3.24 (abs 2e-3). The graphite isotropic stretch cubed equals (1+ε_a)²(1+ε_c) (rel 1e-12).
- (f) Shah E at u = 0.8 = 120000 − 80000·2.24/3 MPa.
- (g) Pore E = 0.05 MPa under defaults.

### Step 3 — Geometry: labels, coarsening, crop (pure numpy)

Target: `pmdb/fem/geometry.py`; tests in `tests/test_fem_unit.py` and a new `tests/test_fem_data.py` (marked `data`).

```python
def labels_from_masks(m: Masks) -> np.ndarray                 # uint8; BINDER default, then SI, GRAPHITE, PORE, ARTEFACT
def coarsen_labels(labels: np.ndarray, factor: int = 2,
                   priority: tuple[int, ...] = (SI, PORE, GRAPHITE, BINDER, ARTEFACT)) -> np.ndarray
    # drop trailing rows/cols so shape divisible; counts per label per block; argmax(counts*10 + bonus), bonus = len-1-rank
def coarsen_image(img: np.ndarray, factor: int) -> np.ndarray  # float32 block mean, same trailing drop
def central_cols(width_px: int, crop_um: float, nm_per_px: float) -> slice  # even width n = 2*floor(crop_um*1000/nm/2), even start c0 = 2*((width_px-n)//4)
```

factor = round(res_nm / site.nm_per_px). factor 1 returns the input unchanged.

Verification:
- `pytest -q tests/test_fem_unit.py -k geometry`. The block cases must pass: [SI,SI,GR,GR]→SI, [SI,GR,GR,B]→GR, [SI,P,GR,B]→SI, [P,P,GR,GR]→P, [GR,GR,B,B]→GR, [A,A,B,B]→B, and a 5×7 input gives a 2×3 output.
- `pytest -q -m data tests/test_fem_data.py`. On `Batch_3/vc2whyaq` (half, normalise="none"): the coarse Si area fraction is within ±20% relative of the 50 nm fraction, and graphite within ±5% relative. Print the pore fraction both ways (reported, not asserted).

### Step 4 — Solver and analytic tests (Stage 1, gate G1)

Targets: `pmdb/fem/result.py` (pure numpy), `pmdb/fem/solver.py` (dolfinx), `tests/test_fem_solver.py` (new; module-level `pytest.importorskip("dolfinx")`).

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

Implementation requirements. The algorithms are fixed by the design decisions; mechanics are up to the implementer.
- Mesh and cell→pixel map: P4. DG0 dof for cell c is `V0.dofmap.list[c, 0]`. Assert the pixel map is a permutation of range(H·W).
- DG0 coefficients: `mu, lam, lx, lz, ly`. Before every solve, set them from `props_fn(s_try)` by phase via precomputed DG0-dof index arrays per label.
- Energy, residual and Jacobian: P6, with `dx` metadata `{"quadrature_degree": solver_opts["quadrature_degree"]}`.
- BCs: P10. In `free` mode the free edge has no term. For `pressure`, mark free-edge facets whose single adjacent cell has a SOLID label as tag 1, and add `+ p_eff·dot(v, n)·ds(1)` to the residual, with p_eff = p·W/W_solid (n = outward reference normal; the sign gives compression). The solve at s = 0 is always performed: for free/confined it converges in 0 iterations; for pressure it is the preload.
- Stepping and failure: P11, with substep targets = sorted(set(frames) ∪ extra_targets). `run.py` passes `extra_targets=(s_star,)`.
- Solver options: P12. Read rtol/atol/max_it from `solver_opts`. Detect NaN residuals as failure (SNES reason `DIVERGED_FNORM_NAN` or a non-finite `u.x.array`).
- Frame outputs: P13 into float32 arrays. Unconverged frames are all NaN, including u_nodes.
- `log(dict)` is called once per substep attempt with the substep record.

Analytic tests in `tests/test_fem_solver.py`. Use small meshes so the module runs in under 2 min. Tolerances are as stated.
- **T1a (2D free eigenstretch, stress-free)**: 6×4 px all one label, custom props with stretch (1.2, 1.1, 1.0), E = 1000, ν = 0.3, `lateral="left"`, orientation bottom, frames [0, 1]. At frame 1: J = 1.32 (rel 1e-8) in every cell, and max |σ| < 1e-6 MPa.
- **T1b (Si homogeneous, plane strain)**: same mesh. `props_fn(s)` gives every label Si properties at u = 0.8·s, so frame 1 has u = 0.8 (λ = 3.24^(1/3), E = 52 000, ν = 0.258). Closed form: in-plane σ = 0 ⇒ μ(f²−1) + λ_L ln(f²/λ_y) = 0 for the in-plane elastic stretch f. Solve with `scipy.optimize.brentq` on f ∈ (0.5, 2). Expected: J = λ²·f², sxx = szz = 0 (abs 1e-6·E), syy = (μ(1/λ²−1) + λ_L ln Je)/Je with Je = f²/λ (rel 1e-6).
- **T1c (graphite anisotropic, plane strain)**: same as T1b with graphite at y = 0.91. Here fe = in-plane elastic stretch with μ(fe²−1) + λ_L ln(fe²/λy) = 0, J = λx λz fe² (rel 1e-6), and sxx = szz = 0.
- **T2 (constrained bilayer, small-strain limit)**: 8 px wide × 10 px tall. Image rows 0-4 (top, layer B) carry isotropic eigenstretch 1+ε* with ε* = 1e-4. Rows 5-9 (layer A) have stretch 1. Both layers E = 1000, ν = 0.3, `lateral="both"`, orientation bottom, free top. Expected in B: sxx = syy = −E ε*/(1−ν) (rel 1e-3), szz ≈ 0 (abs 1e-3·|sxx|). In A: all |σ| < 1e-6·|sxx_B|. Top-edge mean uz = 5·h·ε*(1+ν)/(1−ν) (rel 1e-3). A repeat with ε* = 1e-2 asserts the same signs, with magnitudes within rel 3e-2.
- **T3 (orientation)**: T2 with orientation top. B is now adjacent to the collector; the free edge is the bottom node row. Expect u_nodes[1, H, :, 1] mean = −5·h·ε*(1+ν)/(1−ν) (rel 1e-3), the same stress in B, and zero stress in A.
- **T4 (pixel mapping)**: a 7×9 label map with a single SI pixel at (row 1, col 6), the rest BINDER. `props_fn = lambda s: phase_properties(s, load_params("si"))`, frames [0, 0.2], lateral both, orientation bottom. At frame 1, the pixel with the largest vm is (1, 6), and `result.labels` equals the input.
- **T5 (failure path)**: T1b geometry with solver_opts snes_max_it = 1, ds_min = 0.05, frames linspace(0,1,11). Expect `failed_at_s` finite and ≤ 0.1, `converged[0]` True, every later frame False with NaN fields.
- **T6 (soft pores compress without aborting)**: 12×12 px. A 4×4 SI block in the centre, a 1-px PORE ring around it, BINDER outside. Default props (`phase_properties(s, load_params("si"))`), lateral both, orientation bottom, free top, frames [0, 0.25]. Expect `converged[1]` True and min J over PORE cells at frame 1 < 0.9. The closure flag itself (J < 0.1) is computed only in `features` (Step 5).

Verification (Stage 1). Run `conda run -n pmdb-fem pytest -q tests/test_fem_solver.py`; all 7 tests must pass. Also run `pytest -q tests/test_fem_solver.py` in the default env: the module is skipped, not errored.

**Gate G1**: any T1-T5 failure is a user gate. Stop and report the failing assertion values. T6 failure: also stop and report.

### Step 5 — Features (pure numpy/pandas)

Target: `pmdb/fem/features.py`; tests in `tests/test_fem_unit.py`.

```python
def fem_tile_slices(width_px: int, px_um: float, n_tiles: int = 6, edge_um: float = 20.0) -> list[slice]   # P15
def region_metrics(r: SimResult, frame: int, cols: slice, orientation: str, p: dict) -> dict[str, float]
def run_curves(r: SimResult, orientation: str, p: dict, window: bool) -> tuple[list[dict], list[dict]]
    # (site_rows, tile_rows); one row per (region, frame); keys: region ("site" or tile index 0..5), frame, s, converged,
    # tile_x0_um, tile_x1_um, first_pore_closure_s, failed_at_s, + all metrics. window=True -> site region = all cols, no tiles.
def symmetrise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame   # P17; returns df + orientation="sym" rows
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

Verification: `pytest -q tests/test_fem_unit.py -k features`. Tests must pass on synthetic SimResults with no dolfinx:
- (a) `fem_tile_slices(1747, 0.1)` gives 6 slices with first start 200, last stop 1547, and contiguous edges.
- (b) An all-GRAPHITE 20×600 grid with uz = 0.1·z (bottom), J = 1.1 and sxx = −5 gives swelling 0.1 (rel 1e-6), surface_rough 0, sxx_mean −5, J_gr_mean 1.1, J_si_mean NaN.
- (c) The same with orientation top and uz = −0.1·(H_um − z) gives swelling 0.1.
- (d) A PORE pixel with J = 0.05 at frame 3 gives pore_closed_frac > 0 and first_pore_closure_s = s[3].
- (e) `symmetrise` on two rows with swelling 0.1 and 0.2 gives sym 0.15, and NaN propagates.
- (f) The number of metric keys = 17 + 60 + 4 = 81.

### Step 6 — GIF rendering

Target: `pmdb/fem/gif.py`; tests in `tests/test_fem_unit.py`.

```python
def render_site_gif(bse_coarse: np.ndarray, r: SimResult, title_prefix: str, swelling: Sequence[float],
                    cfg: dict) -> tuple[bytes, dict]   # (gif bytes, {"width_px", "colors", "bytes"}); raises GifTooLarge after ladder
```

Implementation per P19. matplotlib Agg; each frame drawn to an RGB array. Each frame is quantised with Pillow `quantize(colors=c, method=Image.Quantize.MEDIANCUT)`. Saved with `save_all=True, append_images=..., duration=frame_ms, loop=0, optimize=True`. Downsample for drawing with stride f = ceil(W/width_px): corners `u_nodes[:, ::f, ::f]` with the last row/col included, cell values block-mean over f×f.

Verification: `pytest -q tests/test_fem_unit.py -k gif`. On a synthetic 580×1748 case (Gaussian-smoothed σ = 1 noise BSE, linear uz, log-uniform vm in [1, 1e4], last 2 frames unconverged), the output has `PIL.Image.open(...).n_frames == 11`, ≤ 2 000 000 bytes, and width ∈ {900, 720, 600}.

### Step 7 — Case runner, local CLI, Stages 2-4 → STOP POINT A

Targets: `pmdb/fem/run.py`, `scripts/fem_run_local.py`.

```python
def run_case(batch: str, site: str, config: str, orientation: Literal["bottom", "top"], *,
             cache_root: str | Path | None = None, res_nm: float | None = None, crop_um: float | None = None,
             fields_path: Path | None = None, render_gif: bool = False, log=print) -> dict
```

Pipeline:
1. `p = load_params(config)`.
2. `site = load_site(batch, site, resolution="half", normalise="none", cache_root=cache_root)`.
3. `masks = segment(site)`.
4. labels = `labels_from_masks(masks)`, cropped to `central_cols` if `crop_um`, then coarsened with factor = round((res_nm or p["mesh"]["res_nm"]) / site.nm_per_px). The BSE (`site.image[..., 0]`) is cropped and coarsened the same way.
5. `simulate(..., props_fn=lambda s: phase_properties(s, p), bc=BCSpec(orientation, p["bc"]["top"], p["bc"]["pressure_MPa"]), frames=linspace(0,1,11), extra_targets=(p["soc"]["s_star"],))`.
6. `save_npz` if `fields_path`.
7. `run_curves(window=crop_um is not None)`.
8. Render the GIF if `render_gif`, catching `GifTooLarge` into `gif_error`.

Returns `{"meta": {...P21/P22 fields, batch, site, heldout: batch == "Batch_heldout", config, orientation, H, W, px_um, n_cells, failed_at_s, first_pore_closure_s, n_substeps, newton_its_total, substeps, wall_s, peak_rss_mb (resource.getrusage ru_maxrss), versions, params, params_hash, gif_error}, "site_rows": [...], "tile_rows": [...], "gif": bytes | None}`. Every row gets batch, site, heldout, config and orientation.

`scripts/fem_run_local.py`: argparse `--site Batch/id` (repeatable), `--config` (default si), `--orientation bottom|top|both`, `--crop-um`, `--res-nm`, `--cache-root`, `--out` (default `outputs/modal/fem_local/<tag>`; gitignored). It writes `result.json` (without gif bytes), `fields.npz` and the gif per case, and prints wall time, substeps and Newton iterations per case plus the Tier-1 metrics at s = 0.5 and 1.0.

Stages, all in `pmdb-fem`, all on site `Batch_3/vc2whyaq` (the Batch 3 sweep site, typical K01):
- **Stage 2, resolution check (gate G2)**. Crop 20 µm, config si, orientation bottom, at `--res-nm 100` and `--res-nm 50`. Compare site-region Tier-1 at s = 0.5 and 1.0. Pass if the relative difference is ≤ 10% for `swelling`, `sxx_mean_MPa`, `porosity_change` and (J_si_mean − 1), and ≤ 20% for `vm_si_p50_MPa` and `vm_si_p95_MPa`. Relative difference = |a−b|/max(|b|, 1e-12), with b = the 50 nm value.
- **Stage 3, end-to-end window with timings**. Crop 40 µm, config si, `--orientation both`, res 100, then the same for config siox. Record the wall time per case, cells, substeps, Newton iterations, seconds per Newton iteration and peak RSS. Extrapolate to a full site using measured seconds per Newton iteration × (n_cells_site/n_cells_window)^1.5 × the measured iteration count. Open the GIF from the si/bottom case and check frames 0, 5 and 10 visually: the warp goes upward and the colour bar is fixed.
- **Stage 4, validation on the window (gate G4w)**. For si: swelling_sym(s = 1) = mean of the bottom and top `swelling`. Pass if it is in [0.09, 0.39] (P23). Report, without gating: sxx_mean_MPa(s = 1)/(−10) (VT2), porosity_change and porosity_rel_change at s = 1 (VT4), J_si_mean(s = 1) against the plane-strain expectation (VT5), si_yield_frac, pore_closed_frac, first_pore_closure_s and failed_at_s. Report the same for siox.

Verification:
- `conda run -n pmdb-fem pytest -q tests/test_fem_unit.py tests/test_fem_solver.py` passes.
- `conda run -n pmdb-fem pytest -q -m data tests/test_fem_data.py` passes.
- Stage outputs exist under `outputs/modal/fem_local/`.

**STOP POINT A (mandatory; no Modal spend yet).** Report back to the dispatcher with:
- the G1 test summary;
- the G2 table (both resolutions, both frames, relative differences, pass/fail);
- the Stage-3 timings and full-site extrapolation;
- the G4w numbers and the VT2/VT4/VT5 report;
- any Expected-surprise responses used.

Do not continue until the dispatcher replies. If G2 or G4w fails, say so first.

### Step 8 — Modal app, held-out upload, benchmark (Stage 5) → STOP POINT B

Targets: `modal_fem.py` (new), `scripts/fem_collect.py` (new).

`modal_fem.py`:
- Image, volumes and defaults: P20, with the conda package list identical to `environment-fem.yml` minus pytest and modal.
- `@app.function(**FN_KW) def run_case_remote(batch, site, config, orientation, tag, threads: int, render_gif: bool) -> dict`:
  - set `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` = str(threads) before importing `pmdb.fem.run`;
  - cache_root = `/data/heldout` if batch == "Batch_heldout" else `/data`;
  - call `run_case(..., fields_path=Path(f"/out/fields/{tag}/{config}/{orientation}/{batch}__{site}.npz"), render_gif=render_gif)`;
  - write `result.json` (gif bytes removed) to `/out/results/{tag}/{config}/{orientation}/{batch}__{site}.json` and the GIF to `/out/gifs/{tag}/{batch}__{site}.gif`;
  - `volume.commit()`;
  - add cpu, memory_mb and cost_usd (P21) to meta;
  - return the result dict (gif included);
  - any exception → return `{"meta": {..., "error": "Type: msg"}, "site_rows": [], "tile_rows": [], "gif": None}` (modal_app.py D7 pattern).
- `@app.local_entrypoint() def main(mode: str, config: str = "si", cpu: float = 4.0, memory: int = 16384, timeout: int = 21600, sites: str = "")`. `mode ∈ {bench, full, sweeps}`.
  - **bench**: three cases on `Batch_3/vc2whyaq`: (si, bottom, cpu 4), (si, top, cpu 4), (si, bottom, cpu 8 via `run_case_remote.with_options(cpu=8.0, memory=16384)`, tag `bench_cpu8`). GIF on for the first.
  - **full**: all 31 labelled sites (from `cache/half/manifest.csv`) + 3 held-out (from `cache_heldout/half/manifest.csv`) × {bottom, top} for `--config`, via `.with_options(cpu=cpu, memory=memory, timeout=timeout).map(..., return_exceptions=True)`. GIF only for si/bottom.
  - **sweeps**: `sweep_sites()` × `list_sweep_configs()` × {bottom}, tag `sweeps`.
  - Each returned result is written locally to `outputs/modal/fem/results/{tag}/{config}/{orientation}/{batch}__{site}.json` and GIFs to `outputs/modal/fem/gifs/{tag}/`.
  - Print a summary: n ok, n error, n failed_at_s < 1, total and mean wall, cost.
  - `--sites` (comma-separated Batch/id) restricts the case list; it is used for reruns.
- `scripts/fem_collect.py --results outputs/modal/fem/results`: reads all JSONs and writes the committed tables to `outputs/fem/`:
  - `site_curves.csv` and `tile_curves.csv` (tags full; configs si and siox; orientations bottom, top and sym via `symmetrise`);
  - `sweep_site_curves.csv` and `sweep_tile_curves.csv` (tag sweeps, plus the si/bottom default rows of the 6 sites relabelled config `si`);
  - `robustness.csv` (P18: columns metric, frame, s, variant, rho; plus `robust_summary.csv` with metric, frame, min_rho, robust);
  - `validation.csv` (per site × config: swelling bottom/top/sym at s = 1, sxx_ratio_vt2, porosity_change, porosity_rel_change, J_si_mean, si_yield_frac, failed_at_s, first_pore_closure_s);
  - `run_log.json` (P22 provenance + per-case meta without the substeps list; substep logs go to the gitignored `outputs/modal/fem/substeps/`);
  - copies `gifs/full/*.gif` to `outputs/fem/gifs/`.

  It prints the gate summary (P23): the median swelling_sym for si and siox with band verdicts, the count of failed sites for si, and cumulative cost. Floats are written with `float_format="%.6g"`. It never overwrites an existing table with fewer sites than it already has: it writes `<name>_partial.csv` instead and exits 1, the same rule as modal_app.py.

Before launching:
- `modal volume ls pmdb-data /half` must list 31 npz + manifest; if `/half` is missing, run `modal volume put pmdb-data cache/half /half`.
- `modal volume put pmdb-data cache_heldout/half /heldout/half` (reads only `cache_heldout/`, never `data_heldout/`).
- Check the P21 rates on modal.com/pricing.
- Then `modal run modal_fem.py --mode bench`.

Verification: 3 results with no `error`. A `fields.npz` loads with `load_npz` and the shapes are (11, H/2, W/2) and (11, H/2+1, W/2+1, 2). The benchmark GIF is ≤ 2 MB.

**STOP POINT B (mandatory).** Report:
- per case: wall_s, n_substeps, newton_its_total, peak_rss_mb, cost_usd, failed_at_s, swelling_sym(s = 1) with the G4 band verdict, and VT2/VT4/VT5;
- the cpu 4 vs cpu 8 cost per case and the chosen cpu (the lower cost_usd; if the two are within 10%, choose the lower wall time);
- memory for production = max(4096, ceil(1.5·peak_rss_mb/1024)·1024) MB;
- timeout = min(86400, max(7200, 3·max bench wall_s rounded up to the hour));
- the projection: per-case cost × (68 si + 68 siox + 72 sweep cases), scaled per site by n_cells/n_cells_bench, + cumulative spend, compared with the $180 cap;
- whether the vm range [1, 1e4] MPa covered the benchmark (see Expected surprises).

Do not launch `full` until the dispatcher replies.

### Step 9 — Full Si run → STOP POINT C

Run `modal run modal_fem.py --mode full --config si --cpu <chosen> --memory <chosen> --timeout <chosen>`. Rerun errored cases (exceptions, not Newton failures) once with `--sites`. Then run `python scripts/fem_collect.py --results outputs/modal/fem/results`.

Verification: 68 si result JSONs with no `error`, and `outputs/fem/validation.csv` has 34 si rows. Gate check (P23):
- G4 = median swelling_sym(s = 1) over 34 sites in [0.09, 0.39];
- G5 = sites with failed_at_s < 1 in either orientation ≤ 3.

**STOP POINT C (mandatory).** Report:
- G4/G5 values and verdicts;
- the swelling distribution by batch;
- VT2/VT4 medians;
- the list of failed sites with failed_at_s and first_pore_closure_s;
- actual spend so far;
- the projection for siox + sweeps against the $180 cap.

Do not launch siox or sweeps until the dispatcher replies.

### Step 10 — SiOx full run, sweeps, collection, docs, commit

Run `modal run modal_fem.py --mode full --config siox ...` and `modal run modal_fem.py --mode sweeps ...` with the Step-8 settings. Rerun errored cases once. Run `scripts/fem_collect.py`.

Write `docs/fem/README.md`, which describes:
- the commands (local stages, Modal modes, collect);
- the output tables and their columns (metric definitions copied from Step 5);
- the orientation `sym` rule (P17);
- the volume layout (P14);
- the robustness flag (P18);
- the gate results with numbers.

Add a "FEM simulation" pointer line to the top-level `README.md` under the Modal section.

Commit to branch `fem-lit-review` and push. Never merge.

Verification:
- `outputs/fem/site_curves.csv` has 34×2×3×11 = 2244 rows and `tile_curves.csv` has 13 464 rows.
- `sweep_site_curves.csv` has (12+1)×6×11 = 858 rows.
- `robustness.csv` has 81 metrics × 10 frames × 12 variants = 9720 rows.
- `outputs/fem/gifs/` holds 34 files, each ≤ 2 000 000 bytes.
- `pytest -q -m "not data"` and `pytest -q -m data` pass in the default env (FEM solver tests skipped there).
- `conda run -n pmdb-fem pytest -q tests/test_fem_unit.py tests/test_fem_solver.py` passes.
- Report G4/G5 for siox (reported; the siox swelling band uses the same P23 range, informational only), the number of robust (metric, frame) pairs per tier, and final cumulative spend.

## 5. Expected surprises

1. **dolfinx version.** If `pmdb-fem` has dolfinx ≥ 0.10, pin that exact version in `environment-fem.yml` and the Modal image, and adapt mechanically to its API (`interpolation_points` property vs method, `NonlinearProblem` kwargs such as `petsc_options_prefix`, `dofmap.list` shape). If it has < 0.10 (no SNES-based `fem.petsc.NonlinearProblem`), run `conda install -n pmdb-fem -c conda-forge "fenics-dolfinx>=0.10"`. If that cannot resolve on this Mac, stop and escalate.
2. **pmdb imports fail under numpy 2 / pandas in `pmdb-fem`.** Do not edit `pmdb/io.py` or `segment.py`. Pre-authorized: pin `numpy<2` in `environment-fem.yml` if conda resolves it with dolfinx. Otherwise stop and escalate.
3. **Newton stagnates.** If the logs show quadratic convergence that stalls between 1e-8 and 1e-6 relative residual, causing max_it failures, set `snes_rtol: 1.0e-6` in `fem.yaml` (all configs), record it in the Revision-log hand-back, and rerun Stage 1-3. Any other recurring Newton failure before s = 1 on the window: stop at STOP POINT A and report the substep log.
4. **Window swelling lands just below 9%.** This is likely: a back-of-envelope estimate gives about 76% graphite × 9.4% c-axis strain ≈ 7%, plus about 1-3% from Si, and the lit review predicts under-prediction by a dense 2D model. Not pre-authorized: it is a user gate (G4w). Report the value and the per-phase J means.
5. **Stage 2 resolution gate fails.** Not pre-authorized, because running at 50 nm changes cost by about 8×. Stop and report.
6. **Micromamba image fails to build on Modal.** Pre-authorized fallback: `modal.Image.from_registry("dolfinx/dolfinx:v<V>")` (the same version <V>; add `add_python` only if Modal rejects the image for lacking python), then `pip_install` the non-dolfinx packages. Record which image was used in run_log.
7. **`with_options` rejects a parameter.** Define a second decorated function `run_case_remote_cpu8` with identical body for the benchmark, and pass the Step-8 settings through `FN_KW` constants edited after STOP POINT B.
8. **The benchmark vm range is not covered.** If the si/bottom benchmark has p99.9 of solid vm at s = 1 > 1e4 MPa, or p1 of solid vm at s = 0.1 < 1 MPa, set `gif.vm_range_MPa` to [10^floor(log10 p1), 10^ceil(log10 p99.9)] from the benchmark before the full run. Report it at STOP POINT B.
9. **GIF exceeds 2 MB after the ladder for some site.** Record `gif_error`, continue, and list the sites at the end. Do not change the ladder.
10. **Benchmark OOM at 16 GiB, or projection > $180.** Stop at STOP POINT B and report.
11. **Pre-existing test failures in the default env** that also fail on `main`. Report them; do not fix them.
12. **The 1 MPa stack-pressure variant or the confined variant fails early in the sweep.** This is an expected outcome. It is recorded as `failed_at_s` and makes the affected (metric, frame) ρ NaN, so those features become not robust. No rerun.

## 6. Done criteria

- G1 (analytic tests), G2 (resolution), G4w, G4 and G5 all pass, or each failure was reported at its stop point and the dispatcher instructed continuation.
- Committed on `fem-lit-review`:
  - code (`pmdb/fem/*`, `configs/fem/fem.yaml`, `scripts/fem_run_local.py`, `scripts/fem_collect.py`, `modal_fem.py`, `environment-fem.yml`);
  - tests (`tests/test_fem_unit.py`, `tests/test_fem_solver.py`, `tests/test_fem_data.py`);
  - `outputs/fem/{site_curves,tile_curves,sweep_site_curves,sweep_tile_curves,robustness,robust_summary,validation}.csv`, `outputs/fem/run_log.json`, 34 GIFs ≤ 2 MB;
  - `docs/fem/README.md`.
- Row counts as in Step 10.
- Cumulative Modal spend in `run_log.json` ≤ $180.
- Both pytest invocations (default env, `-m "not data"` and `-m data`) pass. The `pmdb-fem` FEM tests pass.
- No file under `data/` or `data_heldout/` changed (`git status --porcelain data data_heldout` is empty). Held-out rows are all `heldout=True` and absent from the sweep tables.

## 7. Revision log

(empty)
