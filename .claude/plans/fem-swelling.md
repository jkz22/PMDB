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
  CHECKS 1-2 RESULT (KPI tiles, 4/site, scripts/tile_signal_checks.py -> outputs/pooling_checks/):
  - Mean pooling of tile probabilities beats max pooling in every variant (site acc mean/max: 3-class 0.58/0.55,
    B1vB3 0.75/0.42, B2vB3 0.67/0.54). Max pooling of probabilities is ruled out: Batch_3 false-alarm rate at p>=0.7
    is 0.29 (vs B1) and 0.41 (vs B2).
  - Classifier verdict "diffuse" for both B1vB3 and B2vB3, but feature-level tails matter: B2 has 3 tail-driven
    features (K04_agglom_frac, K14_empty_p50_um, K15_si_graphite_contact_frac), B1 has 1 (K04_agglom_frac).
  - Most KPIs have low ICC (within-site tile variance dominates), so tiles are noisy individually.
  DECISION D10 (updated): pool at the FEATURE level, not the probability level: site features = [mean, max, spread]
  of tile features (captures diffuse shifts and tail-driven features), plus mean-pooled tile probabilities as a
  baseline. No max pooling of probabilities. Checks 3-4 still run on FEM features.
  CHECK 3 RESULT (scripts/pooling_compare.py -> outputs/pooling_checks/check3_*; LOSO, balanced acc with site-bootstrap
  95% CI): R1 mean_prob 0.49/0.66/0.68 (multiclass / B1vB3 / B2vB3); R4 site_kpis 0.49/0.69/0.67; R2 feat_mean
  0.39/0.57/0.64; R3 feat_mean_max_std 0.41/0.47/0.57. CIs ~±0.2, so R1≈R4; R3 (45 feats on 31 sites) overfits.
  DECISION D10 (final, supersedes the feature-level choice above): site prediction = class-weighted MEAN of tile
  probabilities (R1), one vote per tile over its full feature vector. Feature-level [mean,max,spread] pooling dropped.
  Baseline for the FEM ablation = R1 on KPI tiles and R4 site KPIs. Check 4 (injection) dropped under D17 scope cut.

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

- D17 (user, 2026-10-03: "this is a hackathon, not a scientific paper") SCOPE CUT: D13 SiOx full run and D14 robustness
  sweeps are REMOVED. Only the default pure-Si configuration runs (34 sites × 2 foil orientations). D13/D14 above are
  superseded; no SiOx gate (G4x), no sweep config/mode, no Spearman robustness filter. Prefer pragmatic scope.
- D18 (user, 2026-10-03) CLASSIFIER ARCHITECTURE:
  - Input: curated summaries, ~10-20 physically meaningful scalars per tile from tile_curves (e.g. swelling @50%/100%
    SOC, pore area left @100%, first pore-closure SOC, p95 Si von Mises @100%, interface stress, curve slopes 0-25% and
    25-100%), orientation `sym`. No full-curve PCA, no time-series or image models.
  - Task structure: two-stage vs the Batch_3 baseline. Stage 1: Batch_3 vs not-Batch_3. Stage 2 (non-B3 only):
    Batch_1 vs Batch_2. Each stage pools tiles by mean tile probability (D10). Site confidence = product of stage probs.
  - Model: random forest (fixed hyperparameters, class-balanced, no tuning on 31 sites), one per stage.
  - Explanation vs Batch_3: RF feature importance + per-feature z-scores of the site vs the Batch_3 distribution.
  - Ablation (LOSO, bootstrap CIs): KPI-tiles-only vs FEM-only vs KPI+FEM, same two-stage RF; check-3 logistic R1/R4
    numbers kept as external reference.
- D19 (user, 2026-10-03) DOCUMENTATION FOR A PITCH DECK (pitch is 2026-10-04): everything in the simulation and
  classifier process must be documented well enough to one-shot a deck from the docs alone. Required:
  - docs/fem/method.md: pipeline (image -> segmentation -> mesh -> lithiation -> fields -> tile features), physics
    assumptions + parameter table with citations (link literature-review.md), BCs, SOC mapping, gates, runtime/cost,
    limitations. A mermaid pipeline diagram.
  - docs/fem/results.md: validation numbers vs literature bands, convergence summary, cost actually spent, example
    figures + GIF paths (outputs/fem/gifs/), notable per-batch differences.
  - docs/classifier/method.md + results.md: D18 architecture (diagram), feature list with physical meaning, LOSO
    ablation table with CIs, confusion matrix figure, held-out predictions (batch, confidence, explanation vs Batch_3),
    limitations.
  - Every number in docs is generated by a script (written into the doc or a CSV it cites), never typed by hand;
    figures saved as PNG under outputs/ and referenced by relative path.
  - Final step: docs/pitch-brief.md — one-page deck source (slide-by-slide key message, the number/figure for each,
    paths), written by the writer lane from the docs above once results exist.
- D16 (user, "modal-only", 2026-10-03) No local FEniCSx. All dolfinx code (unit tests, resolution check, smoke window,
  benchmark, production) runs on Modal in the official dolfinx Docker image (version pinned), with pure-Python deps
  added via `uv pip install`. Reason: local conda solve on this Intel (osx-64) Mac stalled; no Docker locally;
  Modal-only also guarantees test env == production env. Pure-numpy parts may still be tested locally.

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
- FULL AUTONOMY (user, 2026-10-03, supersedes the stop-and-wait gates above): run everything unattended until the
  classifier results for the 3 held-out sites exist ("I don't want to wake up and see the simulation was aborted").
  The ONLY hard stop is the $180 Modal cap. No stop point requires the user; the classifier plan does not need user
  sign-off (architecture fixed by D18). Every former gate becomes fix-and-continue, logged in the run report:
  * Analytic/unit test failure -> it is a bug: fresh implementer fixes it (code-reviewer if non-trivial), re-run.
  * Swelling outside [3%, 39%] -> treat as a bug: investigate (sign, eigenstretch applied, BC), fix, re-run window.
    If no bug is found after one investigation round, proceed and flag the result in docs/fem/results.md.
  * Solver non-convergence -> remediation ladder, applied globally and logged: (1) finer adaptive substeps,
    (2) pore ersatz stiffness 1e-4 -> 1e-3 × E_binder, (3) line search basic <-> bt, (4) accept NaN frames after the
    failure SOC (classifier imputes in-fold). Never abort the run for convergence; >3 failures is reported, not a stop.
  * Modal infra errors / timeouts -> retry once per case with 1.5× timeout; then record as failed and continue.
  * Cost: re-project before each chunk; if the projection exceeds the cap, drop orientation `top` (bottom-only), then
    the 50 nm resolution check, before stopping. Stop only if even that exceeds $180.
  * Long runs use `modal run --detach`; progress polled from the volume.
- SWELLING GATE (user-amended "ok gate", 2026-10-03): the literature band [9%, 39%] is REPORTED as a comparison only.
  The STOP condition is swelling < 3% or > 39% (bug-level: eigenstrain not applied / wrong sign / runaway).
  Reason: a dense 2D elastic model cannot capture particle rearrangement, binder deformation and SEI growth that
  make measured electrode swelling exceed particle-scale swelling; planner estimate ~7%. The analytic
  free-expansion unit test (J = Jλ, zero stress) must still pass exactly.
