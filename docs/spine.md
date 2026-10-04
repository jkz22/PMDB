# Pretrained spine → swelling targets: is a vision embedding the right model for the lithiation geometry?

Jihan asked for a model on the "expansion" (swelling) data with a pretrained backbone as the spine, as in
the v2 representation session — *if that is optimal*. This page is the test of that condition.

## 1. Set-up (`scripts/run_swelling_spine.py`)

Same protocol as `outputs/v2` (so the numbers are comparable to the KPI-alignment leaderboard there):
256 px crops at 50 nm/px (12.8 µm), stride 128, BSE only, percentile-normalised per field and replicated
to three channels, two frozen off-the-shelf spines —

- **DINOv2 ViT-S/14** (`torch.hub`, CLS + mean patch token, 768-d), the best generic self-supervised spine;
- **MicroNet EfficientNet-B4** (Stuckner et al., micrograph-pretrained), multi-stage mean+std pooling
  (1328-d) — the configuration the v2 session found best for a frozen model on BSE (KPI R² 0.66).

Head: standardised ridge (α by inner CV), evaluated **leave-one-site-out** over the 31 labelled fields.
Targets are the functional swelling test (`pmdb.functional.swelling_test`, SOC 1, 280 % vol) computed on
each crop's own masks: **constrained share** (growth landing on graphite) and **pore loss**; the Si
fraction is run as a sanity target to tie back to the v2 leaderboard. Reported at crop level (pooled R²)
and at site level (mean crop prediction vs mean crop target, R² and Spearman).

Must-beat baseline: ridge on five numbers read off the *same crop's mask* — Si, graphite and pore
fraction, Si–graphite contact (K15) and Si object count. The swelling targets are deterministic
functions of the mask, so a spine is only "optimal" if it carries information those five numbers do not
(sub-crop arrangement, object shape), and it has to show that on fields it never saw.

Held-out fields (`data_heldout`) are embedded and scored only after the LOSO comparison, with heads fit on
the 31 labelled fields; they never enter a fit.

## 2. Results (`outputs/spine/results.json`, `figures/spine_vs_baseline.png`)

5,434 labelled crops (31 fields) + 572 held-out crops; 4.4 % of crops have no Si and no constrained-share
target. Leave-one-site-out, R² at crop level / site level:

| target | features | crop R² | site R² | site ρ |
|---|---|---|---|---|
| constrained share | **5 mask fractions (baseline)** | **0.75** | **0.91** | 0.94 |
| | MicroNet frozen | 0.24 | 0.61 | 0.62 |
| | DINOv2-S frozen | 0.13 | 0.44 | 0.47 |
| | MicroNet + masks | 0.75 | 0.92 | 0.94 |
| | DINOv2 + masks | 0.75 | 0.88 | 0.92 |
| pore loss | 5 mask fractions (baseline) | 0.58 | 0.84 | 0.87 |
| | MicroNet frozen | 0.64 | 0.88 | 0.82 |
| | DINOv2-S frozen | 0.51 | 0.68 | 0.74 |
| | **MicroNet + masks** | **0.65** | **0.90** | 0.83 |
| | DINOv2 + masks | 0.59 | 0.84 | 0.81 |
| Si fraction (sanity) | MicroNet frozen | 0.94 | 0.96 | 0.88 |
| | DINOv2-S frozen | 0.79 | 0.85 | 0.83 |

Paired per-site |error| vs the baseline (Wilcoxon, 31 fields): constrained share, MicroNet alone is
*worse* (median error 0.013 vs 0.006, p < 0.001) and MicroNet + masks is indistinguishable (p = 0.66);
pore loss, MicroNet + masks is not significantly better (0.0075 vs 0.0087, p = 0.42).

Held-out fields, heads fit on all 31 labelled fields (`heldout_predictions.csv`): every head puts
3e122cbj at constrained share ≈ 0.77–0.80 (measured 0.78) and pore loss ≈ 0.18–0.22 (measured 0.23),
and fn0mhxef / xrv9xvzb at 0.87–0.88 / 0.09–0.10 (measured 0.87, 0.86 / 0.09, 0.10). The joint-outlier
status of 3e122cbj (`docs/acceptance.md` §4) is reproduced by a model that never saw it, from either the
masks or the image alone.

## 3. Verdict: a pretrained spine is not the optimal model for the swelling data

- **Constrained share** is read almost completely off five mask numbers (site R² 0.91). A frozen spine
  alone recovers 0.44–0.61 and adds nothing on top of the masks. The target is a function of the
  segmented geometry, and the segmentation is the model — a deep embedding only re-estimates the
  Si/graphite/contact fractions from grey levels, with error.
- **Pore loss** is the one place the image carries a little that the five fractions do not (where the
  pores sit relative to the Si); MicroNet alone matches the baseline and the combination gains ≈ 0.08
  crop R², ≈ 0.05 site R² — real but not significant on 31 fields. If a spine is used at all, this is
  the target that justifies it.
- **MicroNet beats DINOv2 on every target** (crop R² 0.94 vs 0.79 even for Si fraction), the same ordering
  the v2 session found for a frozen BSE model; micrograph pre-training matters more than model size.
- Fine-tuning a spine on 31 fields was not attempted on purpose: the v2 classification diagnostics showed
  that end-to-end training on this data learns BSE noise and sharpness (the Batch 3 imaging signature)
  before it learns microstructure, and the swelling targets are already recoverable from the masks.

Recommended model, therefore: `swelling_test` on the segmenter's masks (exact), summarised by the five
mask fractions when a scalar regression is wanted; MicroNet frozen features as an optional,
segmentation-free surrogate for pore loss with a documented ≈ 0.88 site R². The KPI-aligned VAE-C of the
v2 session remains the right *representation*; it was not re-fit here because its value is in the
embedding, not in predicting a quantity that the masks give exactly.

Reproduce (CPU, ≈ 7 min extraction + 1 min evaluation):
`python scripts/run_swelling_spine.py extract && python scripts/run_swelling_spine.py evaluate`.
Features are cached in `outputs/spine/features_*.npy` (gitignored if large); `crop_targets.csv` and
`site_predictions.csv` are committed.
