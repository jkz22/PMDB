# Physical clean pipeline — results (`clean-v1`, seed 20261003)

Second, physics-based route alongside the LUT harmonisation (`docs/harmonisation.md`). Method and
usage: [`docs/clean.md`](../../docs/clean.md). Per-site numbers: `summary.csv`, `<Batch>/<site>/params.json`;
acceptance checks: `eval/acceptance.{md,json}`, `eval/*.csv`, `eval/fig_*.png`. Arrays are rebuilt
locally (`python scripts/build_clean.py --workers 4`, ≈ 12 min, deterministic); they are not committed.

## What the data showed (raw, before any correction)

| group | sites | BSE dark level `D` (DN) | BSE graphite level `G` (DN) | grey step | 0-clipping |
|---|---|---|---|---|---|
| strong Batch 3 | `71vgq3fw` `kbdh4tri` `tuy3zymq` `x7u69zsw` | 31.3 – 32.7 | 28.4 – 30.1 | 1 DN | none |
| reference Batch 3 (unflagged) | `ptg8lmto` `utfgcjfa` `vc2whyaq` `xgj4xftb` | 12.2 – 18.9 | 37.5 – 42.5 | 1–2 DN | little |
| everything else | 23 sites | −0.9 – 18.9 | 37.5 – 56.6 | 2–3 DN | up to 2.7 % of pixels |
| held-out | `3e122cbj` / `fn0mhxef` / `xrv9xvzb` | 2.2 / 0.8 / 18.5 | 53.9 / 54.6 / 44.4 | 2 / 2 / 3 DN | — |

The Batch 3 "black level" is therefore a detector offset *and* a ≈ 0.6× gain on the strong sites, and
`9luzk4jm`, `hzumfsms`, `ufdvpb81` (and held-out `xrv9xvzb`) were re-quantised after capture (3-DN
steps, 4 on Inlens). Both are imaging fingerprints, not material.

## What the pipeline does with it

`z = (I − D) / G(x, y)` per detector with `D` from eroded deep pores (Tobit when pores clip at 0;
Inlens: lower edge of the pore distribution) and `G(x, y)` a robust quadratic over 64-px graphite
block modes. Unreliable pixels are masked (uint16 bit field), never filled; cracks / pore bands are
site flags, pixels untouched. Resolution is harmonised downward (Gaussian blur to the 75th-percentile
reference edge width) and noise upward (Poisson–Gaussian fit on graphite/Si patches, deterministic
noise to the reference target), never the other way.

## Acceptance (31 labelled sites; `eval/acceptance.md`)

Passed
- strong sites: `D` ≥ 15 DN and no 0-clipping; `ufdvpb81` marker columns masked; reference sites unflagged.
- pore median `z` within ±0.05 of 0 on every site (max |·| = 0.031, SD across sites 0.009); the
  strong-minus-clean pore gap goes from 0.96 (relative dark level) to −0.007.
- `hzumfsms` crack flagged (165 µm² network, 47 µm span; next largest on any site 88 µm²), 0 pixels patched.
- pore-rich bands flagged on `0grcilhi`, `ufdvpb81` (site flags only).
- scan bands: 33 corrected (multiplicative, ≤ 8 rows or seen by ≥ 2 detectors), 1 masked; median
  corrected-row fraction 0.
- charging: local on 11 sites, broad on 5 (`rxax5ozo`, `0grcilhi`, `hawkfj64`, `mgxahqnk`,
  `x77cy643`), ≤ 2.7 % of the image masked beyond the border on any detector.
- harmonisation: edge width within target on all but the listed outliers (sharpening is never
  applied, so sites already blurrier than target stay as they are); noise variance at graphite within
  10 % of the reference target on all sites and detectors (CV of σ_graphite 0.095 → 0.007 on BSE).
- material: fixed-threshold porosity agrees with per-site Otsu (r = 0.97); Si/graphite ratio stays
  free (2.0 – 2.6 by group; it is not a harmonisation target); porosity and Si fraction change by
  < 0.027 between `norm` and `harm`.
- held-out: run with the labelled `targets.json` / `material_thresholds.json` only; `xrv9xvzb`
  (D = 18.5 DN, 3-DN steps) lands in the mild group; nothing flagged.

Not met, left as is (documented in `docs/clean.md`)
- graphite half-sample mode within ±2 % of 1: max deviation 11 % (SD 0.032). Medians are within
  ±4 %; the mode of a textured graphite distribution is not a stable 2 % quantity, and forcing it
  would mean a histogram operation this route excludes.
- 64-px block flatness < 3 %: median 8.9 %, max 13 % residual scatter of the graphite block modes
  about the fitted `G(x, y)`. The fitted surface itself removes gradients of up to 49 % (BSE) and
  136 % (Inlens, `G_ptp_rel` in `summary.csv`) across the field; what is left is block-to-block
  graphite texture at the 1.6 µm block scale, similar on strong and clean sites.
- imaging-fingerprint batch classifier (LOSO, permutation null): acquisition-only features drop from
  0.61 (p = 0.03) to 0.42 = chance after cleaning; intensity-percentile features remain at 0.71
  (p = 0.005) before and after, and strong-vs-rest stays separable (0.94 vs chance 0.87). The remaining
  signal is the material intensity distribution (Si brightness / phase fractions), which this route
  deliberately does not touch — the LUT `histmatch` route in `docs/harmonisation.md` shows what it
  costs to remove it.

## Use in modelling

```python
from pmdb.io import load_clean
from pmdb import clean
z, mask = load_clean("Batch_3", "71vgq3fw", "BSE", kind="harm", resolution="half")
valid = clean.valid_for_kpis(mask)
```

`kind="norm"` is the physically normalised image (per-site corrections only); `kind="harm"` adds the
resolution/noise harmonisation to the reference. Always mask with `valid_for_kpis` (KPIs) or
`valid_for_stats` (fits/thresholds, additionally drops clipped pixels). Ablations vs the LUT route:
`load_site(..., harmonise="hybrid")` on the same sites.
