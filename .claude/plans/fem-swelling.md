# FEM lithiation-swelling simulation — decisions so far (pre-spec)

Status: scoping. Literature review complete: `docs/fem/literature-review.md` (parameter table, SOC→Si/graphite mapping, validation bands, accepted risks; evidence in `docs/fem/evidence/`). Full spec to follow.

## Decided by user
- D1 Free/open-source Python: FEniCSx (finite strain needs a nonlinear solver; scikit-fem dropped).
- D2 2D, mechanics only (no electrochemistry); SEM cross-sections (x = width, z = thickness).
- D3 Start at 0% SOC everywhere, charge to 100%, snapshot every 10% -> 11-step sequence per site, used as features for downstream prediction.
- D4 Input: existing segmentation `pmdb.segment.segment(site)` on `cache/half/` (+ held-out via `cache_heldout/half/`).
- D5 Run on Modal, CPU (no GPU: conda-forge dolfinx has no GPU path; MUMPS direct solver is CPU-only).
- D6 Visualisation: one animated GIF per site (all 31 labelled + 3 held-out = 34), 11 frames (SOC 0..100%).
- D7 Shah, de Vasconcelos & Zhao 2022 (J. Appl. Mech. 89, 081005) is a methodological reference, not a replication target.

## Proposed (pending lit review / spec)
- GIF frame: BSE warped by displacement, von Mises / pressure overlay, SOC label, colour scale fixed across frames and sites.
- Fields stored on the reference pixel grid; per-site summary-curve CSV for downstream features.
- Held-out sites simulated for features only, never trained on (AGENTS.md).
- D8 (proposed, user wants rich time-series features) Feature extraction is tiered, all as 11-step SOC series, full fields kept so tiers can be re-extracted:
  1. site curves (~15 metrics); 2. tile curves on the existing KPI tile grid (primary classifier input, more samples);
  3. per-phase quantile series (p5..p99 of stress, J); 4. through-thickness band profiles (magnitude only, like K12);
  5. per-Si-particle trajectories aggregated to distributions.
  Batch 1/2/3 classifier: train on tiles, leave-one-site-out CV, aggregate tile probabilities per site.
  Models in order: regularised logistic / RF on flattened curves -> curve-shape descriptors -> ROCKET / small 1D CNN.
  Ablation required: FEM features + existing KPIs vs KPIs alone.
- D9 (user agreed: tile-based features AND tile-based prediction) Tiles: simulate whole site, then tile the fields.
  ~22 µm wide (≈ one graphite flake length), full height, non-overlapping, within the central ~135 µm (20 µm lateral edge
  bands excluded for BC influence) -> ~6 tiles/site, ~200 labelled tiles. Existing KPI grid is 4 full-height tiles
  (~44 µm, spec 002 D-012); recompute KPIs on the new grid for the joint model / ablation.
- D10 (user inclined to agree; open) One vote per tile from a classifier over the tile's full 11-step series (not per
  tile×step, which are strongly correlated); calibration checked under leave-one-site-out. Per-step classification kept
  only as explanation (at which SOC batches separate).
  OPEN RISK (user-raised): batch signal may be localised to one "witness" tile; mean pooling would hide it
  (multiple-instance learning). Pooling rule NOT fixed to mean. Proposed default: hybrid pooling, site features =
  [mean, max, spread] of tile scores; max-type threshold calibrated on Batch 3's own LOSO tile spread; fixed tile count
  per site. Checks before fixing: (1) per-site tile-prediction spread under LOSO, (2) within- vs between-site variance
  (ICC) per feature, (3) compare <=4 pooling rules under LOSO, (4) synthetic single-tile injection into Batch 3 sites
  (tests/synthetic_patterns.py). Checks 1-2 can run now on outputs/kpis/tile_kpis.csv.

## Resolved from the literature review (user delegated these to Claude's judgment, 2026-10-03)
Guiding principle: features serve batch discrimination, so between-site ranking must be robust to uncertain
parameters; weakly sourced choices become measured robustness sweeps rather than bets.
- D11 Pores/artefact: ersatz E = 1e-4·E_binder, ν 0.3, Fλ = I. Pore element det F < 0.1 -> flag, do NOT stop; record
  SOC of first closure and closed-pore fraction per step. NaN only for steps after a Newton failure (record failure SOC).
  Basis: no precedent incl. Shah (lit review C17) -> numerical choice, swept in D14.
- D12 Separator-side edge traction-free by default; confined (Shah-like) and 1 MPa stack-pressure runs only in D14.
  Basis: no separator in images (lit review l.182); stack 0.1-1 MPa << GPa eigenstresses (C19, item 8).
- D13 Run BOTH Si and SiOx full configurations on all 34 sites. Pure Si = default (primary-sourced params, matches Yao
  split) and drives the GIFs. SiOx: Fλ = (1+1.6u)^(1/3) I, E 34 GPa, ν 0.17. Classifier ablation decides utility.
  Basis: BSE cannot separate Si/SiOx; particle ECD d50 ~0.45 µm, max ~10 µm (>150 nm Si fracture size) hints SiOx/Si-C,
  but SiOx params are weak (C10, C11); second full run costs ~$10-20.
- D14 Sweeps: 6 labelled sites (2 per batch, nearest batch-median K01 Si fraction) × 12 one-at-a-time variants:
  Si utilisation {0.6, 0.95}; binder E {0.05, 2 GPa}; pore stiffness {1e-6, 1e-2}×E_binder; isotropic graphite strain;
  SOC breakpoint 0.36; proportional Si/Gr split; confined top; 1 MPa stack; Shah E(C) for Si.
  Robustness criterion: per feature, Spearman rank correlation of site values between default and each variant >= 0.8;
  features failing are dropped or flagged before classification.
- D15 Outputs: field maps on a Modal Volume (not git). Committed under outputs/fem/: tile_curves.csv, site_curves.csv,
  run log (params, solver convergence per step), 34 GIFs (<=2 MB, ~900 px wide, 11 frames, fixed colour scale).
- Budget: ~$40-70 Modal total (2 full runs + sweeps).

## Pre-approval for unattended execution (user, 2026-10-03)
User pre-approved proceeding without intervention through: plan review, implementation, code review, local tests,
one Modal benchmark site, the 2 full runs (Si + SiOx, 34 sites), the D14 sweeps, features, GIFs, pooling checks 3-4,
committing and pushing to branch fem-sim (PR #18 merged 2026-10-03; new PR from fem-sim).
- HARD CAP: total Modal spend <= $180 (user-set). Before every Modal launch, project cost from the measured
  benchmark; stop and ask if cumulative actual + projected would exceed $180.
- Stop and wait for the user only if a gate fails: synthetic analytic tests fail; simulated electrode swelling falls
  outside the literature bands (docs/fem/literature-review.md validation targets); solver non-convergence on many
  sites (> 3 of 34 before 100% SOC in the default run); or the cap would be exceeded.
- Never: merge the PR, write to data/ or data_heldout/, train on held-out sites.
