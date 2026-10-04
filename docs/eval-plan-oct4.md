# Pre-registered evaluation plan: extended fingerprint features (2026-10-04)

Committed **before** any evaluation of the extended feature set is run. This is
the single evaluation decision for the overnight data in `outputs/overnight/`
(see AGENTS discipline: tonight produces data, tomorrow has exactly one
evaluation decision).

## Candidate feature set (fixed in advance)

The 16 features of the merged model (`pmdb/fingerprint.py::build_features`),
unchanged, **plus** exactly these 10, all derived from
`outputs/overnight/features/` and all motivated by the batch physics already
established on the labelled data *before* the overnight run (Si depth
arrangement; medium-range order; within-site homogeneity). None were screened
against labels.

From the 15-band relative Si depth profile (normalised by the site mean, as in
the merged model):

1. `si15_top3_rel` — mean of relative bands 0–2 (top fifth of the coating).
2. `si15_bottom3_rel` — mean of relative bands 12–14.
3. `si15_mid5_rel` — mean of relative bands 5–9 (the mid-depth region where
   Batch 2 is depleted and Batch 3 is flat).
4. `si15_roughness` — std of successive differences of the relative profile.

Arrangement length scales (`site_scalars.csv`):

5. `acl_depth_si_um`
6. `acl_lateral_si_um`

Other-phase depth arrangement (15-band profiles, relative to phase site mean):

7. `graphite15_slope` — bottom3 mean − top3 mean of the relative graphite profile.
8. `pore15_slope` — same for the pore profile.

Within-site heterogeneity at 16 tiles (`tile_kpis_rich.csv`, `n_tiles == 16`;
nan-tolerant std requiring ≥ 8 finite tiles, else NaN):

9. `k15_contact_tilestd16` — std of `K15_si_graphite_contact_frac`.
10. `k01_si_frac_tilestd16` — std of `K01_si_frac_adm`.

Total: 26 features, 31 labelled sites. The model, its constants and the
assignment rule are the merged defaults; only the feature table changes.

## Decision rule (fixed in advance)

Run **once**, seed 0:

- `fp.loo_evaluate` on the 26-feature table (31 labelled sites).
- `fp.permutation_test` with `n_perm = 2000`, seed 0.

**Adopt** the extended set iff BOTH:

- LOO accuracy ≥ **0.742** (≥ 23/31 correct, i.e. at least +2 sites over the
  merged model's 21/31 = 0.677), AND
- permutation p ≤ **0.005**.

Otherwise **keep the merged 16-feature model** and present it unchanged. No
feature subsets, no re-weighting, no constant tuning, no second evaluation
regardless of how close the result is. Held-out predictions of the extended
model are computed for the record in either case but are only presented if the
set is adopted.

## Outputs

`outputs/overnight/eval_extended/`: extended feature tables (labelled +
held-out), LOO predictions, permutation result, held-out predictions, and an
`evaluation.json` recording the decision against the rule above.
