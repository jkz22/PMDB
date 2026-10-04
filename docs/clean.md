# Physical cleaning and harmonisation of the SEM images (`pmdb.clean`)

Second, physics-based preprocessing route, alongside the per-site grey-level LUTs of
[`harmonisation.md`](harmonisation.md). Instead of mapping grey levels to a reference histogram,
every image is corrected for the acquisition effects we can measure on it and then expressed in
physical units: `z = (I − D) / G(x, y)` with `D` the detector dark level and `G(x, y)` the
graphite level (so pores ≈ 0, graphite = 1, Si-rich particles ≈ 1.7–2.7 and free to vary).
Unreliable pixels are *masked, never filled*; real microstructure is flagged, never patched.

```
python scripts/build_clean.py --workers 4                      # 31 labelled sites -> outputs/clean/
python scripts/build_clean.py --data-root data_heldout --out outputs/clean_heldout \
       --targets outputs/clean/targets.json --workers 3        # held-out, labelled-only targets
python scripts/eval_clean.py --out outputs/clean --heldout outputs/clean_heldout   # acceptance checks
```

```python
from pmdb.io import load_clean
from pmdb import clean

z, mask = load_clean("Batch_3", "71vgq3fw", "BSE", kind="harm", resolution="half")
valid = clean.valid_for_kpis(mask)          # bool; False on border/markers/FOV intrusions/bad bands/charging
# resolution="half": mask-aware 2×2 mean; a half pixel is valid iff one of its four parents is, and
# carries the flags of the contributing parents (a fully masked block keeps the OR of all four)
porosity = ((z < 0.5) & valid).sum() / valid.sum()
```

`outputs/` is gitignored for the arrays (≈ 9 GB at full resolution); the committed part is the
per-site `params.json`, `summary.csv`, `targets.json`, `material_thresholds.json` and the
evaluation under `outputs/clean/eval/`. A rebuild is deterministic (fixed seed `20261003`) and takes
≈ 12 min on 4 CPU workers. Raw TIFFs under `data/` are never touched.

`params.json` stores the raw-TIFF `paths` relative to the repository root (absolute only for files
outside it), so the committed parameter files and `scripts/eval_clean.py` work from any checkout.

## Pipeline (per site, full resolution 25 nm/px)

| stage | what is done | evidence used | output |
|---|---|---|---|
| sanitise | R channel of the RGB TIFF; columns where R≠G≠B are colour markers (`ufdvpb81`); 8-px border | per image | mask bits `BORDER` |
| field of view | copper collector band (saturated BSE rows at an edge) + attached delamination gap; coating free surface at the top (porosity/Inlens change-point over column strips, ≥ 3 adjacent strips); both dilated by 20 px (0.5 µm) | BSE + Inlens | `COLLECTOR`, `FREE_SURFACE` |
| dequantise | `U(−s/2, s/2)` on unclipped pixels, `s` = populated grey step (1, 2, 3 or 4 DN) | per image | float32 |
| dark level `D` | BSE / SE: Gaussian MLE on eroded deep-pore pixels, left-censored at `s/2` when pores clip at 0 (Tobit); Inlens: 1 % quantile of pore pixels (Inlens pores are not dark, see below) | per image | `anchors.<det>.D`, method, s.e. |
| graphite level `G(x, y)` | half-sample mode of `I − D` over eroded graphite in 64-px blocks (≥ 20 % coverage), robust (Huber) weighted 2-D quadratic | per image | `G_coef`, `G_level`, `G_ptp_rel` |
| scan-line bands | per-detector row profile = interquartile mean of provisional `z` over graphite, detrended by a 151-row running median, robust z with a statistical-error floor; runs ≥ 3 rows with |z| > 5. Acted on only when seen by ≥ 2 detectors or ≤ 8 rows wide (wider single-detector runs are flake-scale Inlens/SE topography, recorded but untouched); corrected multiplicatively about `D`, taper ± 2 rows; residual |z| > 3 → masked | per image | `BAND_CORRECTED`, `BAND_BAD`, `bands.events[].actionable` |
| charging | local: saturated, compact Inlens/SE components ≥ 1 µm² whose BSE is not Si-bright; broad: the lower-decile Inlens *floor* over graphite, per 64-px block, lifted > 75 % above the site-wide floor over ≥ 20 contiguous blocks (Inlens masked only, BSE kept) | Inlens, SE, BSE | `CHARGE_LOCAL`, `CHARGE_BROAD` |
| clipping | 255 → `CLIP_HIGH`; 0 → `CLIP_LOW` when `D ≤ 0`. Kept for segmentation, excluded from statistics | per image | bits |
| material flags | largest pore network area/span (crack, `hzumfsms`), top/bottom-eighth porosity (pore-rich bands) vs robust thresholds from the labelled set; site-level flags only, pixels unchanged | labelled set | `flag_crack`, `flag_pore_band_*` |
| normalise | `z = (I − D) / G(x, y)`, stored as `uint16` fixed point `z × 10⁴` | — | `<det>_norm.tif`, `<det>_mask.tif` |
| harmonise down | ESF edge width σₑ from ≥ 10 000 pore/graphite edges (erf fits, R² ≥ 0.9); mask-aware Gaussian blur (normalised convolution over KPI-valid pixels, so masked markers/bands/charging cannot bleed into their neighbours) with `σₖ = √(σₜ² − σₑ²)` to the 75th percentile σₜ of the four accepted references; Poisson–Gaussian fit `var(z) = αz + β` on flat 7×7 graphite/Si patches (pore patches are censored by the 0-clip); zero-mean noise added up to the noisiest reference when the gap exceeds 10 % of the target variance (≈ 5 % in σ). Never sharpened or denoised | references | `<det>_harm.tif`, `harmonisation.<det>` |

The four accepted references are the unflagged Batch 3 sites `ptg8lmto`, `utfgcjfa`, `vc2whyaq`,
`xgj4xftb`; the targets (`targets.json`) are fitted on the labelled set only and re-used for the
held-out sites.

### Mask bits (`uint16`, one word per detector)

```
BORDER 1  COLLECTOR 2  FREE_SURFACE 4  BAND_CORRECTED 8  BAND_BAD 16  CHARGE_LOCAL 32
CHARGE_BROAD 64  CLIP_HIGH 128  CLIP_LOW 256  CRACK 512  PORE_BAND 1024
valid_for_kpis  = none of BORDER|COLLECTOR|FREE_SURFACE|BAND_BAD|CHARGE_LOCAL|CHARGE_BROAD
valid_for_stats = valid_for_kpis and not CLIP_HIGH|CLIP_LOW
```

## Deviations from the brief and why

* **Inlens dark level.** The brief's dark anchor (Gaussian / Tobit on deep-pore pixels) assumes pore
  interiors are dark. On Inlens they are not: pore pixels spread almost uniformly from the black
  level to the graphite level (SE escaping from pore walls), so neither the mean nor the mode is a
  physical anchor. Inlens uses the lower edge (1 % quantile) of the pore pixels, reported as
  `quantile`, or `quantile-clipped` (D = 0, true level below range) when ≥ 1 % of them are 0.
* **Grey-level step.** Several sites were re-quantised after capture — the three "mild" sites
  `9luzk4jm`, `hzumfsms`, `ufdvpb81` populate only every 3rd DN (4th on Inlens), most others every
  2nd, the four strong sites every DN with no clipping. Dequantisation uses the measured step and
  the Tobit censor point is `s/2`. The step is recorded (`grey_step`) as an imaging fingerprint.
* **Row statistic.** The row *median* of a 3-DN-quantised image jumps between levels and produced
  dozens of fake 2–5 % "bands" per site; the interquartile mean with a statistical-error floor on
  the robust scale is used instead. Remaining detections are ≤ 1 per site on most images.
* **Broad charging.** The brief's rule (Inlens graphite block medians > 25 % off the flat-field)
  flagged 10–70 % of most images: Inlens graphite brightness is topographic — flake edges and small
  particles are 2–3× brighter than flat flake interiors — so block medians track the microstructure,
  not charging (two of the four accepted references would have been masked). A charging glow lifts
  the *floor* of every block it covers, which topography does not, so the detector compares the
  lower-decile Inlens level over graphite with the site-wide floor (> +75 %, ≥ 20 contiguous blocks).
* **Scan-band action filter.** Detected runs that are wider than 8 rows and seen by a single detector
  are Inlens/SE brightness structure (e.g. a 30-row dark region at the top of `3806gxp0`), not
  scan-line faults; they are recorded with `actionable=false` and the pixels are left alone.
* **Crack rule.** The largest pore network must be an area outlier on the labelled distribution *and*
  long: either longer than the labelled outlier span or ≥ 75 % of the image height (`hzumfsms`:
  165 µm², 90 %; the next largest network on any site is 88 µm²). Length alone is not used: on the
  short `xgj4xftb` field a 26 µm wide shallow pore cluster already spans 85 % of the height.
* **Block statistic** for `G(x, y)` is the half-sample mode rather than the median: Inlens/SE
  graphite has a bright topographic tail that pulls the median up.
* Everything else (thresholds, windows, 20 px dilation, 75th-percentile σₜ, noisiest reference)
  follows the brief. Nothing is fitted on the held-out sites.
