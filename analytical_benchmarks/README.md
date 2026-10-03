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
| 2. Tile KPIs | `kpis.py` | 64 px (3.2 µm) tiles: porosity, Si fraction, graphite fraction, phase-boundary density (µm/µm²). |
| 3. Per-site GP | `kpis.py` | One GP per KPI per site. X = tile centre (row, col) in µm, y = tile KPI. Kernel C·RBF + White, fitted by maximum marginal likelihood. Outputs: GP mean, correlated standard error `1/sqrt(1ᵀK⁻¹1)`, naive `std/sqrt(n)` for comparison, lengthscale in µm (patch size). |
| 4. Particle KPIs | `kpis.py` | Connected Si regions: d50, d90 (µm), count per 1000 µm², median solidity. |
| 5. Batch comparison | `compare.py` | The site is the unit. Leave-one-batch-out and pairwise comparisons: permutation p, bootstrap 95% CI, Cohen's d, Holm correction over 9 KPIs, spread test, robust-z site flags (>3.5), confound checks (BSE black level, SE vs ETD, Si/graphite contrast). |
| 6. Report | `figs.py`, `overlays.py`, `report.py` | Figures and a standalone `qc_report.html`. |

Verdicts: **outlier** = Holm p < 0.01 and \|d\| ≥ 0.8. **investigate** = Holm p < 0.05, or significantly larger spread, or 2 or more flagged sites. **consistent** otherwise. Without a specification, "outlier" means *different*, not *defective*.

## Current results (31 sites)

| batch | sites | verdict | driver |
|---|---|---|---|
| Batch_1 | 7 | investigate | 4ih2ggld and 5n1q8atc: Si area fraction 0.10–0.14 vs ≈0.05, lower Si solidity, larger d90 |
| Batch_2 | 7 | consistent | none |
| Batch_3 | 17 | consistent | hzumfsms has large pore patches (GP lengthscale 2.8 vs ≈2.0 µm) |

Caveats: the two flagged Batch_1 sites get the lowest Si thresholds (a ≈ 1.50 vs 1.6–1.9 elsewhere). In `overlays.png`, part of their orange area falls on bright graphite-flake edges, so their higher Si fraction may partly be a segmentation artefact; check this before acting on the Batch_1 flag. Also, porosity correlates with the Batch_3 BSE black-level offset and with Si/graphite contrast, and boundary density correlates with contrast. Phase identity (bright = Si) is inferred from BSE contrast.

Outputs: `site_kpis.csv`, `compare.json`, `verdicts.txt`, `fig_kpis.png`, `fig_gpmaps.png`, `overlays.png`, `qc_report.html`.
