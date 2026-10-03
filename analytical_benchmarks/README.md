# Analytical benchmarks: baseline-free batch QC

An analytical/statistical QC pipeline for the Polaron SEM batches. There is no machine learning and no training labels. **No batch is used as an approved baseline**: each batch is compared with all the other sites pooled.

Run everything (about 2 min for 31 sites on 8 cores):

```bash
./analytical_benchmarks/run_all.sh
```

To test a new (surprise) batch, copy it to `data/Batch_N/` and run the same command. Nothing is retrained. `scripts/build_cache.py` adds the new sites to `cache/half/`, and the new batch is compared with all other sites.

## Pipeline

| step | file | what it does |
|---|---|---|
| 1. Segmentation | `seg.py` | Raw BSE (half res, 50 nm/px). Each image is rescaled with its own levels: 0 = pore black (0.5th percentile), 1 = graphite peak (histogram mode). Pore = a < 0.5; Si = above the histogram valley before the bright peak (peak search a ≥ 1.4; fallback 1.6), then opening r=2 px and objects smaller than 40 px removed. Everything else is graphite/binder. The per-image scaling absorbs the Batch_3 black-level offset. |
| 2. Tile KPIs | `kpis.py` | 64 px (3.2 µm) tiles: porosity, Si fraction, graphite fraction, phase-boundary density (µm/µm²), Inlens fine texture (std of high-pass σ=2 px, divided by the image's own 5–95% intensity range). |
| 3. Per-site GP | `kpis.py` | One GP per KPI per site. X = tile centre (row, col) in µm, y = tile KPI. Kernel C·RBF + White, fitted by maximum marginal likelihood. Outputs: GP mean, correlated standard error `1/sqrt(1ᵀK⁻¹1)`, naive `std/sqrt(n)` for comparison, lengthscale in µm (patch size). |
| 4. Particle KPIs | `kpis.py` | Connected Si regions ≥1 µm: size distribution d10/d50/d90 and span (d90−d10)/d50, count per 1000 µm², median aspect ratio (major/minor axis), circularity 4π·A/P², solidity. Clustering: mean nearest-neighbour distance of centroids, Clark–Evans ratio (mean NN / 0.5·λ^-½; below 1 = clustered), cluster size (particles closer than 1 µm grouped). Cracked/irregular = internal gaps ≥2% of filled area or solidity <0.8; anomalous = >8 µm or aspect ratio >3. Inlens edge strength (gradient magnitude, relative) in graphite/binder pixels. |
| 5. Batch comparison | `compare.py` | The site is the unit. Leave-one-batch-out and pairwise comparisons: permutation p, bootstrap 95% CI, Cohen's d, Holm correction over all 20 KPIs, spread test, robust-z site flags (>3.5), confound checks (BSE black level, SE vs ETD, Si/graphite contrast). |
| 6. Feature importance | `importance.py` | Model-free ranking: η² (share of site-to-site variance explained by batch, on ranks), Kruskal–Wallis p with Holm correction, largest leave-one-batch-out Cohen's d. Robust-z heatmap of every site × KPI. Cross-check: random forest predicting batch, 5×10-fold CV balanced accuracy vs a label-shuffle null. |
| 6. Report | `figs.py`, `overlays.py`, `report.py` | Figures and a standalone `qc_report.html`. |

Verdicts: **outlier** = Holm p < 0.01 and \|d\| ≥ 0.8. **investigate** = Holm p < 0.05, or significantly larger spread, or 2 or more sites flagged on at least 2 KPIs each (single-KPI flags are expected by chance with 20 KPIs and are only listed). **consistent** otherwise. Without a specification, "outlier" means *different*, not *defective*.

## Current results (31 sites, 20 KPIs)

| batch | sites | verdict | driver |
|---|---|---|---|
| Batch_1 | 7 | investigate | 4ih2ggld and 5n1q8atc are unusual on 4–5 KPIs each: Si area fraction 0.10–0.14 vs ≈0.05, lower solidity and circularity, more cracked/irregular particles (0.31–0.33 vs ≈0.13), and larger Si clusters (5n1q8atc) |
| Batch_2 | 7 | consistent | one single-KPI flag (3806gxp0, low aspect ratio) |
| Batch_3 | 17 | consistent | single-KPI flags only (e.g. hzumfsms pore patch size, vc2whyaq Clark–Evans 0.84 = clustered Si) |

No pairwise batch difference is significant after Holm correction. Batch averages are very close for size distribution, aspect ratio, clustering, anomalous counts and Inlens texture.

### Feature importance

| rank | KPI | η² | p (Holm) | batch that differs most, d |
|---|---|---|---|---|
| 1 | Si size span (d90−d10)/d50 | 0.20 | 1.0 | Batch_1, +0.97 |
| 2 | Si d90 | 0.19 | 1.0 | Batch_1, +1.06 |
| 3–4 | Inlens edge strength / fine texture | 0.14 | 1.0 | Batch_2, −0.85 |
| 5 | Si Clark–Evans ratio | 0.13 | 1.0 | Batch_3, −0.55 |

No KPI separates the batches after Holm correction. The random forest reaches 0.42 balanced accuracy vs 0.33 chance (shuffled-label 95th percentile 0.41, p = 0.06), so the batches are not separable as a whole and model-based importances are not reported. Site level: 4ih2ggld is driven by Si fraction (z +6.1), solidity (−4.5), cracked fraction (+3.8) and circularity (−3.6); 5n1q8atc by Si fraction (+10.3), solidity (−6.8), circularity (−4.8) and cluster size (+4.3). See `fig_importance.png`.

Caveats:
- The two flagged Batch_1 sites get the lowest Si thresholds (a ≈ 1.50 vs 1.6–1.9 elsewhere). In `overlays.png` and `fig_particles.png`, some of their "Si" is on bright graphite-flake edges or mid-grey particles. Their higher Si fraction and their cracked/irregular counts may partly come from segmentation; check before acting on the Batch_1 flag.
- The cracked/irregular rule mostly finds irregular or concave particles and mis-segmented regions, not only true cracks (see `fig_particles.png`). Treat it as a screening count.
- d10 sits close to the 1 µm size cutoff, so it mostly reflects the cutoff. Circularity from pixel perimeters is biased low; use it only for relative comparisons.
- Inlens texture and edge strength are higher on the 4 Batch_3 images with the BSE black-level offset (p = 0.003 and p < 0.001), so they are probably partly an imaging-session effect.
- Porosity correlates with the BSE black-level offset and with Si/graphite contrast; boundary density correlates with contrast. Bright = Si is inferred from BSE contrast.
- Nearest-neighbour statistics have no edge correction.

Outputs: GP mean, correlated standard error `1/sqrt(1ᵀK⁻¹1)`, naive `std/sqrt(n)` for comparison, lengthscale in µm (patch size). |
| 4. Particle KPIs | `kpis.py` | Connected Si regions ≥1 µm: size distribution d10/d50/d90 and span (d90−d10)/d50, count per 1000 µm², median aspect ratio (major/minor axis), circularity 4π·A/P², solidity. Clustering: mean nearest-neighbour distance of centroids, Clark–Evans ratio (mean NN / 0.5·λ^-½; below 1 = clustered), cluster size (particles closer than 1 µm grouped). Cracked/irregular = internal gaps ≥2% of filled area or solidity <0.8; anomalous = >8 µm or aspect ratio >3. Inlens edge strength (gradient magnitude, relative) in graphite/binder pixels. |
| 5. Batch comparison | `compare.py` | The site is the unit. Leave-one-batch-out and pairwise comparisons: permutation p, bootstrap 95% CI, Cohen's d, Holm correction over all 20 KPIs, spread test, robust-z site flags (>3.5), confound checks (BSE black level, SE vs ETD, Si/graphite contrast). |
| 6. Report | `figs.py`, `overlays.py`, `report.py` | Figures and a standalone `qc_report.html`. |

Verdicts: **outlier** = Holm p < 0.01 and \|d\| ≥ 0.8. **investigate** = Holm p < 0.05, or significantly larger spread, or 2 or more sites flagged on at least 2 KPIs each (single-KPI flags are expected by chance with 20 KPIs and are only listed). **consistent** otherwise. Without a specification, "outlier" means *different*, not *defective*.

## Current results (31 sites)

| batch | sites | verdict | driver |
|---|---|---|---|
| Batch_1 | 7 | investigate | 4ih2ggld and 5n1q8atc: Si area fraction 0.10–0.14 vs ≈0.05, lower Si solidity, larger d90 |
| Batch_2 | 7 | consistent | none |
| Batch_3 | 17 | consistent | hzumfsms has large pore patches (GP lengthscale 2.8 vs ≈2.0 µm) |

Caveats: the two flagged Batch_1 sites get the lowest Si thresholds (a ≈ 1.50 vs 1.6–1.9 elsewhere). In `overlays.png`, part of their orange area falls on bright graphite-flake edges, so their higher Si fraction may partly be a segmentation artefact; check this before acting on the Batch_1 flag. Also, porosity correlates with the Batch_3 BSE black-level offset and with Si/graphite contrast, and boundary density correlates with contrast. Phase identity (bright = Si) is inferred from BSE contrast.

Outputs: `site_kpis.csv`, `compare.json`, `verdicts.txt`, `fig_kpis.png` (all KPIs per site), `fig_psd.png` (size/shape distributions), `fig_particles.png` (flagged particle crops), `fig_importance.png`, `importance.csv`/`importance.json` (feature importance), `fig_gpmaps.png`, `overlays.png`, `qc_report.html`.
