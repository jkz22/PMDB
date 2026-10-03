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
