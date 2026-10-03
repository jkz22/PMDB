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
| 3. Per-site GP | `kpis.py` | One GP per KPI per site. X = tile centre (row, col) in µm, y = tile KPI. Kernel C·RBF + White, fitted by maximum marginal likelihood. ### Physics estimates

| batch | capacity (mAh/g) | swelling (vol. fraction) | swelling ÷ porosity | diffusion time of d90 (relative) |
|---|---|---|---|---|
| Batch_1 | 624 (median 578) | 0.29 | 5.3 | 1.32 |
| Batch_2 | 554 | 0.24 | 3.8 | 1.08 |
| Batch_3 | 553 | 0.24 | 3.7 | 1.00 |

Batch means (Batch_1 median in brackets, because 2 sites pull its mean up). 4ih2ggld and 5n1q8atc: 729 and 834 mAh/g (+30–50%), swelling 0.37 and 0.47, swelling ÷ porosity 4.9 and 10.6, diffusion time 1.5× and 1.9×. If their bright particles are SiOx instead of Si, capacity drops to 505 and 545 mAh/g. At every site, swelling is 2–10× the pore volume, so the electrode must thicken on charging. None of the batch differences is significant after Holm correction.

Caveats: these are textbook-constant estimates, not measurements. They are almost linear in the Si fraction, so they carry the same segmentation uncertainty and add no independent evidence; their value is translating the Si difference into battery terms. Binder/carbon counted as graphite (capacity slightly high). 2-D area fractions are used as volume fractions (stereology, unbiased for random sections).

Outputs: `site_kpis.csv`, `physics.csv`/`physics.json`/`fig_physics.png` (physics estimates), `compare.json`, `verdicts.txt`, `fig_kpis.png` (all KPIs per site), `fig_psd.png` (size/shape distributions), `fig_particles.png` (flagged particle crops), `fig_importance.png`, `importance.csv`/`importance.json` (feature importance), `fig_gpmaps.png`, `overlays.png`, `qc_report.html`.
