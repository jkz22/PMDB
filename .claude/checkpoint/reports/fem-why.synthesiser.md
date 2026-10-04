## Answer
The FEM arm did not fail because of a bug. The integrity checks are clean [R1:E1.integrity]. It failed because of what it measures. The swelling features are almost a linear rescaling of Si area fraction (Spearman rho about 0.9 with K01; OLS R^2 = 0.971; batch KW p on the residual = 0.699). Si fraction itself does not separate the batches (p = 0.377) [R1:E3][R1:E2]. The stress features are saturated or unphysical under the linear D20 mechanics: si_yield_frac = 1 everywhere, Si von Mises is about 24 GPa against a 637 MPa yield, and the spread between sites is smaller than the spread within a site [R1:E4]. Both problems come on top of a framing weakness that the KPI arm shares: scalar tile features, a two-stage tile-level RF, and stage 2 with n = 14. Every variant of our model lands in the same noise band (end-to-end balanced accuracy 0.37-0.47, CI 0.25-0.51) [R1:E5]. Leo's batch signal comes from site-level spatial curves (depth profile, pair-correlation), not from scalars, and he finds no scalar KPI that separates the batches either [R2:E-features][R2:E-eval].

The BCs are right in direction but wrong where they are applied. Constraining in-plane and leaving the thickness free is sensible at the macro scale. But the image edges are coating interior: no foil, no free surface [R3]. So a roller at one edge and a traction-free opposite edge add artificial boundary layers. The features most affected are surface roughness, depth-band stress deviation and edge-adjacent quantiles. Fixing the BCs will not rescue the swelling or stress-magnitude features.

For today: add a small number of FEM features to Leo's site-level naive Bayes, choose them inside each LOO fold, and test with a nested permutation test against a decision rule stated in advance (about 2-3 h). If FEM adds nothing, pitch that as an informative negative: "bulk swelling is set by Si content, which is the same across batches; the batch difference is in spatial arrangement."

## Sources
- R1: /Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/reports/fem-why.explorer-diag.md — why FEM features did not improve batch classification (integrity, univariate signal, redundancy, saturation, model sensitivity)
- R2: /Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/reports/fem-why.explorer-leo.md — Leo's KPI/fingerprint batch classifier on origin/main
- R3: /Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/reports/fem-why.explorer-edges.md — do image edges look like a free surface or the foil?

R1 and R2 have no E numbering, so citations use section labels: R1:E1 = section 1 Integrity, E2 = Univariate, E3 = Redundancy, E4 = Saturation, E5 = Model sensitivity. R2 sections are Where / Features / Model / Eval / Heldout / Ours. R3 is a single block.

## Claims

### 1. Diagnosis, causes ranked by evidence

- **C1** (CONFIRMED) **Rank 1: the FEM features repeat Si fraction, and Si fraction carries no batch signal.** Site-level swell_50, swell_100 and slope_early have rho of 0.903, 0.891 and 0.909 with K01_si_frac_adm. fem_sxx_mean has rho = -0.926 and vm_gr_p95 has rho = 0.825 with K01. Regressing swell_100 on K01 gives R^2 = 0.971, and the residual has KW p = 0.699 (rank version 0.336). K01 itself has KW p = 0.377. [R1:E3][R1:E2]
- **C2** (INFERRED) The mechanism behind C1 is that a linear small-strain model with eigenstrain only in Si gives macroscopic swelling close to a rule of mixtures in Si volume fraction. The simulation therefore mostly re-derives K01. This is a reasoning step from D20 linearity [R1:E4 sigmaY note; D20 per the spawn prompt] plus the R^2 of 0.971 [R1:E3]. The excerpts do not show it directly.
- **C3** (CONFIRMED) **Rank 2: under D20 the stress features are saturated or unphysical.**
  - fem_si_yield_frac_100 equals 1.0 on every labelled tile and every frame from 1 to 10.
  - Site-level Si von Mises p95 is 23.4-27.4 GPa, against sigmaY = 637.5 MPa.
  - Site-level CV is 0.039 for vm_si_p95, 0.034 for p_si_mean and 0.010 for J_si_mean.
  - For vm_si_p95, the mean SD between tiles of one site (1181 MPa) is larger than the SD between sites (959 MPa).
  [R1:E4][R1:E1]
- **C4** (CONFIRMED) Removing the stress group does not rescue the arm. Non-stress RF has end-to-end balanced accuracy 0.437 vs 0.370 for all 16 features. Every variant lies between 0.370 and 0.473, each from a single seed, and the bootstrap CI of the FEM arm's end-to-end score is 0.250-0.513. [R1:E5]
- **C5** (CONFIRMED) **Rank 3: model and framing.** The problem is not FEM-specific.
  - The repo's KPI arm also scores low: end-to-end balanced accuracy 0.437, site accuracy 0.581, against a majority baseline of 0.548. Our scores are balanced accuracy for the two-stage end-to-end model. Leo's 0.677 is plain accuracy; his balanced accuracy is about 0.636. [R1:E5][R2:E-ours][R2:E-eval]
  - Stage 2 has only 14 sites. [R1:E5]
  - Leo's model is single-stage, site-level and has no fitted weights, and it reaches LOO accuracy 0.677 (permutation p 0.0025). [R2:E-model][R2:E-eval]
- **C6** (CONFIRMED) Scalar features are a dead end for both arms. Our scan finds 0 of 42 KPIs at p < 0.05, best K12 at p = 0.065. Leo also reports none of 51 scalar KPIs separating the batches, best p = 0.065, with LOO below the majority baseline. His signal comes from depth-band Si curves, x/z pair-correlation bins and the spread of K15 across tiles. [R1:E2][R2:E-features]
- **C7** (CONFIRMED) **Rank 4: feature curation probably picked the wrong statistics, but the evidence carries selection bias.**
  - The raw FEM metrics (160 tests) give 34 at p < 0.05 and 8 at p < 0.01, against about 8 and 1.6 expected by chance.
  - The curated set gives 2 at p < 0.05 out of 15.
  - The top raw metrics are lower quantiles of Si stress (q5 and q25 of vm_si, Cliff's delta B1-B3 and B2-B3 of about +0.7 to +0.8), plus q25 of p_si and q75 of J_si.
  - The curated set uses p95 tails and means instead.
  - Caveats: no multiplicity correction was applied, many columns are duplicates or near-duplicates, and the same 31 sites were used throughout. [R1:E2]
- **C8** (INFERRED) The lower Si-stress quantiles measure the least-constrained Si, meaning Si with room to expand into nearby pores or binder. B3 has lower values, i.e. more of its Si sits in unconstrained surroundings. That is a mechanically interpretable form of the Si-graphite/pore contact signal. Supporting evidence: the curated J_si and p_si correlate best with K15_si_graphite_contact_frac (|rho| = 0.632) [R1:E3], and Leo uses the spread of K15 across tiles [R2:E-features]. The physical reading is my inference.
- **C9** (INFERRED) **Rank 5: the boundary conditions** (see C12-C14). They are physically wrong at the image edges [R3], but no evidence links them to the classification failure. The features they affect are not the ones that carry, or fail to carry, the K01 effect.
- **C10** (CONFIRMED) **Rank 6: bug, for which there is no evidence.**
  - No NaNs and no duplicate rows.
  - All runs converged and none failed before s = 1.
  - The KPI-FEM join is exact (186/186), and the tile grid matches to within 0.2 um when the half-resolution width is used.
  - sym equals the mean of bottom and top to within 7e-7.
  - The repo metrics are reproduced exactly (0.626 / 0.357 / 0.370).
  - Oddities: si_yield_frac is constant, first_closure_s takes only 2 values, and fem_sxx_mean_100 is constant within each site. [R1:E1][R1:E5]
- **C11** (INFERRED) fem_sxx_mean_100 being constant within each site is expected physics, not a reduction bug. sxx_mean_MPa is a plain mean of the sxx field over the reduction region [V2][V3]. With σxz = 0 on both the roller edge and the traction-free edge, x-equilibrium (∂σxx/∂x + ∂σxz/∂z = 0) makes ∫σxx dz independent of x. Any full-height vertical strip therefore has the same mean σxx. The derivation is mine. So this feature carries one value per site and is about 93% Si fraction (rho -0.926 with K01) [R1:E3].

### 2. Boundary-condition verdict

- **C12** (CONFIRMED) The production BCs (docs/fem/method.md) are:
  - collector edge: u_z = 0 (rollers)
  - lateral edges: u_x = 0
  - opposite edge: traction-free
  - plane strain
  - run with the collector at the bottom and again at the top, then averaged into "sym".
  [R3][V1]
- **C13** (CONFIRMED) Neither horizontal edge is a free surface or the foil:
  - Median pore fraction is 0.016 at the top, 0.045 at the bottom and 0.041 in the middle. No edge comes close to 1 (maximum 0.285).
  - Segmentation finds no Cu phase.
  - The docs say no foil is visible.
  - Image heights are 40.3-57.9 um.
  The images are windows inside the coating. [R3]
- **C14** (INFERRED) Verdict:
  - **What is right:** constraining in-plane while leaving the thickness free matches the real macro state of a coating on a foil. "Free to expand upward" is the right macroscopic idea.
  - **What is wrong:** that state is imposed as local edge conditions on an interior window.
    1. The traction-free top edge invents a free surface. Particles at the top edge can bulge into material that actually continues above them.
    2. The roller edge invents a rigid, flat, frictionless foil.
    3. u_x = 0 on the lateral edges invents rigid walls where the coating actually continues.
  - **Likely effect on features:**
    - fem_surface_rough_100 is measured on an artificial free edge, so it is mostly an artefact.
    - fem_band_vm_maxdev_100 (stress deviation across depth bands) is dominated by boundary layers at both ends.
    - Edge-adjacent quantiles of stress and J, and pore closure near the edges, are biased.
    - fem_sxx_mean_100 is fixed by the lateral constraint (C11).
    - Global swelling is the least affected, because it is a whole-window average. Bottom and top orientations agree at r = 0.878, with a maximum |difference| of 0.0286 [R1:E1].
  - **Better BCs:** periodic in x with zero mean εxx, and in z either periodic with zero mean σzz or a "window" condition. That needs a solver change and a re-run of all 68 cases, which is not feasible before the pitch.
  - **Cheap approximation:** keep the current runs and leave out the top and bottom edge bands (and optionally lateral margins) when reducing the fields. By Saint-Venant, boundary-layer effects decay within a few microstructure lengths. The decay argument is inference; the band width has to be fixed in advance, not tuned.

### 3. What to borrow from Leo

- **C15** (CONFIRMED) **Pooling level:** site level, one row per site, 31 rows, built from curves rather than tile scalars. [R2:E-features]
- **C16** (CONFIRMED) **Model:** robust per-batch naive Bayes, numpy only, no fitted weights.
  - Per batch and feature: median plus MAD × 1.4826, with the scale shrunk 50% toward the pooled scale.
  - Pooled robust standardisation, winsorised at |z| = 10.
  - Score: mean Laplace negative log-likelihood, assign by argmin.
  - Mondrian conformal p-values give credibility and confidence, and an OOD flag.
  - Assigning by max-p instead drops LOO accuracy to 0.355.
  [R2:E-model]
- **C17** (CONFIRMED) **Evaluation:** LOO over 31 sites with a refit in every fold, and a label-permutation test of the whole LOO (500 and 10k permutations; null mean 0.373, null max 0.742). He also pre-registered a test of an extended feature set, and it was rejected (26 features: LOO 0.613, p 0.0185). [R2:E-eval]
- **C18** (INFERRED) The rejected extended set implies that piling features into an equal-weight mean-NLL naive Bayes dilutes the good ones. Any FEM addition should therefore be small (k ≤ 3). This inference comes from the 26-feature result [R2:E-eval] and the mean-NLL score [R2:E-model].
- **C19** (CONFIRMED) **Framing:** single-stage, 3-class, plus an OOD flag, with no hierarchy. BSE black level is excluded on purpose as an acquisition artefact. [R2:E-model][R2:E-features]
- **C20** (CONFIRMED) Leo's feature families were chosen by exploring the labelled data, and his docs say so. His permutation p therefore does not account for that selection, and his 0.677 is optimistic too. [R2:E-features]

### 4. Held-out agreement (useful for the pitch)

- **C21** (CONFIRMED) Our FEM arms call xrv9xvzb Batch_2, the same as Leo's fingerprint (B2, credibility 1.0). Our KPI arm calls it B3 (0.586). 3e122cbj is B1 and fn0mhxef is B3 in both systems. [R2:E-heldout][R2:E-ours]

## Contradictions
- **xrv9xvzb:** it has an elevated BSE black level (p1 = 6, a Batch-3-like acquisition artefact), yet both Leo's material fingerprint and our FEM arm call it B2. Our KPI arm calls it B3 [R2:E-heldout][R2:E-ours]. I weight B2: the two systems that see only geometry agree, and black level is an acquisition artefact [R2:E-features]. Our KPI arm's B3 call has low confidence (0.586).
- **Metrics do not compare directly:** ours are two-stage end-to-end balanced accuracy, Leo's is plain accuracy [R1:E5][R2:E-eval]. Like-for-like on site accuracy: ours is at best 0.581 (both the KPI arm and the RF variants) vs Leo's 0.677. In balanced accuracy it is about 0.47 vs about 0.636.
- **Raw-FEM signal vs no improvement in the classifier:** 34/160 raw metrics hit p < 0.05 [R1:E2], yet no model variant beats about 0.47 balanced accuracy [R1:E5]. These are not truly in conflict. The curated set excluded the top raw metrics (C7), and the hit count is inflated by duplicate columns and the lack of correction [R1:E2].

## Gaps
- **Correlation between the lower Si-stress quantiles (q5/q25 vm_si) and K01, K15 and Leo's 16 features was not computed.** If q5_vm_si is just Si fraction or K15 in another form, C8 and recommendation step 2 lose most of their value [R1: Not checked].
- **No evidence on how far the boundary layers reach** (bottom vs top differences are only available as a global swelling summary). This decides the width of the excluded edge band (C14). The recommendation fixes the width in advance instead of measuring it.
- **The variant runs used a single seed and have no CIs** [R1:E5]. Differences of 0.05 between variants are uninterpretable.
- **Whether site_curves.csv holds per-depth-band or per-tile quantiles was not checked.** Without them, edge-band exclusion needs the Modal re-reduction.
- **The ~24 GPa Si stresses were not checked against the material parameters** beyond sigmaY [R1: Not checked]. This could be a units or parameter problem as well as saturation. It would not change the ranking, because the features are uninformative either way.

## Gather next
- (Optional; only if step 2 below goes ahead.) In /Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data, compute site-level Spearman rho between q5_vm_si@s1.0, q25_vm_si@s0.5 and q25_vm_si@s1.0 (outputs/fem/site_curves.csv, sym, labelled) and each of K01_si_frac_adm and K15_si_graphite_contact_frac (outputs/kpis/site_kpis.csv) plus Leo's 16 features (origin/main outputs/fingerprint/features.csv). This settles whether the lower-quantile stress features hold information beyond Si fraction and contact (Gap 1, claim C8).

## Verification
- **V1** `/Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/docs/fem/method.md:42-43`. Tests C12 (the production BCs); confirmed.
  ```
  | Current-collector edge | u_z = 0; run both orientations because the foil side is unknown | R5:E18 (Shah: u·n = 0 at z = 0) | verbatim (analogue) |
  | Lateral edges | u_x = 0 | R5:E18 (Shah: u·n = 0 at r = R0); P:V3 | verbatim (analogue) |
  ```
- **V2** `/Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/pmdb/classify/features.py:146`. Tests C11 (the curated sxx is passed straight through from the tile metric, with no site-level aggregation); confirmed.
  ```
              "fem_sxx_mean_100": m["sxx_mean_MPa"][10],
  ```
- **V3** `/Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/pmdb/fem/features.py:67`. Tests C11 (sxx_mean is a plain field mean over the reduction region); confirmed.
  ```
      out["sxx_mean_MPa"] = _mean(F["sxx"])
  ```

## For the planner

### Recommendation for TODAY (time-boxed, about 2-3 h of agent time, with an optional extra 1.5 h)

**Step 0: write the decision rule down before running anything** (commit it, e.g. docs/classifier/fem-nb-prereg.md).
- Primary comparison: Leo's naive Bayes on his 16 features (baseline: LOO accuracy 0.677 = 21/31 [R2:E-eval]) vs Leo's 16 plus FEM features, using the same LOO.
- FEM counts as "adding value" only if both of these hold:
  - (a) combined LOO accuracy ≥ 23/31 (at least +2 sites; +1 site is noise at n = 31);
  - (b) a nested permutation test (≥ 1000 permutations, with the feature selection repeated inside every permuted LOO) gives p ≤ 0.05 for the combined model.
- Secondary: a FEM-only naive Bayes against the 0.548 majority baseline, with its own nested permutation p.
- Report balanced accuracy alongside plain accuracy.
- Otherwise the result is reported as "no added value".

**Step 1: add pre-registered physics features** (about 1 h, using the existing outputs/fem/site_curves.csv, no re-reduction).
- Add at most 2 features chosen on physical grounds:
  - the swelling residual after Si fraction (swelling@s1.0 regressed on K01, with the regression fitted inside each training fold);
  - Si stress heterogeneity normalised by its median, e.g. (q75 − q25)/q50 of vm_si at s = 0.5.
- Exclude by rule: si_yield_frac (constant), the vm_si/p_si/J_si magnitudes (saturated), fem_sxx_mean (one value per site, equal to K01; C11) and surface_rough (BC artefact; C14).
- Expectation from the evidence: the residual is likely null (p = 0.699) [R1:E3]. Run it anyway, because it was declared in advance.

**Step 2: select features inside each fold** (about 1 h).
- In each LOO fold, on the 30 training sites only: rank the 160 raw FEM metrics (site_curves.csv, sym, frames 5 and 10) by KW p.
- Greedily drop any metric with |rho| > 0.9 to one already picked, and keep the top k = 2 (k fixed in advance).
- Append them to Leo's 16, fit the naive Bayes, predict the left-out site.
- The permutation test must rerun the same selection under each permuted labelling. That cost is tiny (numpy).
- This is the only honest way to use the q5/q25 vm_si signal, because those metrics were found by a 160-test scan on these same 31 sites [R1:E2].

**Step 3 (optional, about 1.5 h, only if time allows): edge-band sensitivity.**
- Re-reduce the fields on Modal (volume pmdb-fem-out; reduce on Modal instead of downloading 68 × ~260 MB).
- Exclude a top and a bottom band of width fixed in advance, e.g. 5 um (about 10% of the 40-58 um image height [R3]).
- Recompute only the step 1-2 features and rerun with the same decision rule. Report it as BC sensitivity, not as a second shot at the result.
- Periodic BCs need a solver change and 68 re-runs: post-pitch.

**Implementation facts:**
- Leo's code: origin/main pmdb/fingerprint.py
  - build_features: lines 98-134
  - model: lines 64-91 and 202-203
  - LOO: lines 402-419
  - permutation test: lines 440-473
  - script: scripts/run_fingerprint.py
  - constants: SCALE_SHRINK = 0.5, MIN_MAD_TO_SD_RATIO = 0.1, MAX_Z = 10
  - assignment: argmin mean Laplace NLL, not max-p
  [R2:E-where][R2:E-model]
- To add features, extend build_features output or join on site. Merge PR #19 code into the fem-sim branch, or cherry-pick pmdb/fingerprint.py. Work goes via PR, not straight to main (user memory).
- FEM raw metrics: outputs/fem/site_curves.csv, orientation == 'sym', heldout excluded, metric columns from index 9 on, frames 5 and 10 [R1: t2.py].
- Known duplicate columns: vm_si_p50_MPa = q50_vm_si; J_si_mean and p_si_mean give identical p [R1:E2]. Deduplicate before selection.
- Held-out sites: score them only after the decision rule has been applied, and report credibility and confidence (track spec: batch + confidence + how each differs from B3).

### What to say in the pitch if FEM still does not beat Leo
1. "We simulated swelling for every site with FEM. Predicted swelling is set by Si content (R² = 0.97 against Si area fraction), and Si content does not differ between batches (p = 0.38)" [R1:E3][R1:E2]. "So the batches do not differ in how much they swell. They differ in how the material is arranged, which is what the fingerprint picks up" [R2:E-features].
2. Present this as a mechanistic negative result that rules out a hypothesis, not as a failed model. The FEM also gives a per-site swelling estimate, which is useful for process QC whatever the batch.
3. "On the hardest held-out site the physics agrees with the fingerprint. xrv9xvzb looks like Batch 3 on grey levels, but both FEM and the fingerprint call it Batch 2" [R2:E-heldout][R2:E-ours].
4. Limitations, stated openly:
   - The mechanics are linear (stresses are well above yield), so stress magnitudes are not quantitative [R1:E4].
   - The images are interior windows, and the BCs approximate the coating's macro constraint; periodic BCs come next [R3].
   - n = 31; every claim was permutation-tested and the selection was done inside each fold.
