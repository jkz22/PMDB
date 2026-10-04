# Held-back sites: batch assignment with the analytical pipeline

Sites: `3e122cbj`, `fn0mhxef`, `xrv9xvzb` (`data_heldout/` on `main`). They were never used for fitting.

## Reproduce

```bash
export PMDB_HELDOUT=<checkout of main with data_heldout/, cache_heldout/, outputs/clean*/>
export PYTHONPATH=$PWD:$PWD/..
python3 heldout_kpis.py          # same segmentation, KPIs and per-KPI GPs as kpis.py -> heldout_kpis.csv (~1 min)
python3 overlap_check.py $PMDB_HELDOUT   # is any held-back field a re-image of a labelled one? -> heldout_overlap.csv
python3 classify_heldout.py $PMDB_HELDOUT/outputs/clean/summary.csv $PMDB_HELDOUT/outputs/clean_heldout/summary.csv
python3 fig_heldout.py           # -> fig_heldout.png
```

## Method

- **Material evidence:** the 20 analytical KPIs from `site_kpis.csv`. Per KPI, the model uses a Gaussian per batch (the batch mean and the pooled within-batch SD), with equal priors. The top-k KPIs by Kruskal–Wallis are chosen on the training sites only.
  - Validation: leave-one-site-out on the 31 labelled sites, with a 200× label-shuffle null and temperature calibration of the posteriors.
- **Imaging fingerprint (acquisition, not material):** the same model on the clean-pipeline imaging parameters, scored separately. The parameters are dark level, gain, Si/graphite contrast, grey-level step, noise and frame height.
- **Combined:** the product of the two posteriors. This assumes the material and imaging evidence are independent.
- **Overlap check:** the peak NCC at 200 nm/px against every labelled site. All peaks are ≤ 0.07; an exact re-image scores 1.0 and a noisy re-image scores 0.7. So no held-back field is a re-image of a labelled one.

## Validation (LOSO, 31 sites)

| evidence | k | accuracy | balanced | null 95% | p |
|---|---|---|---|---|---|
| material KPIs | 10 | 0.48 | 0.49 | 0.49 | 0.065 |
| imaging fingerprint | 3 | 0.71 | 0.71 | 0.51 | 0.005 |

At site level, the material KPIs alone do **not** reliably separate the three batches. Only the Batch_1 sub-population (`4ih2ggld`, `5n1q8atc`) is distinctive.

## Assignments

| site | assigned | confidence | material posterior B1/B2/B3 | imaging posterior B1/B2/B3 | nearest labelled sites (material; imaging) |
|---|---|---|---|---|---|
| 3e122cbj | **Batch_1** | high (0.98 combined) | 0.60 / 0.14 / 0.26 | 0.92 / 0.08 / 0.00 | 4ih2ggld, 5n1q8atc (both B1); 4ih2ggld, 5n1q8atc |
| fn0mhxef | **Batch_1** (vs Batch_2) | low (0.54; not Batch_3 0.99) | 0.28 / 0.37 / 0.34 | 0.61 / 0.38 / 0.01 | fzrt2k6r (B1), f1vzngrs (B1); fzrt2k6r (B1), avn74qx1 (B2) |
| xrv9xvzb | **Batch_3** | moderate (0.48 combined; 3-NN imaging 3/3 B3) | 0.21 / 0.42 / 0.38 | 0.17 / 0.36 / 0.47 | hzumfsms (B3); 9luzk4jm, hzumfsms (B3) |

- **3e122cbj:** it has the Batch_1 "odd-site" signature, which no Batch_2 or Batch_3 site has.
  - Si 8.2% vs 5.1–5.2% median in B2/B3.
  - Si solidity 0.86 vs 0.905 (robust z −5.9 vs all 31 sites).
  - Circularity 0.46 (z −4.5).
  - 31% cracked/irregular Si vs 13%.
  - 11.6 Si particles per 1000 µm² vs 7–8.
  - A dimmer Si BSE peak (1.65× graphite vs 2.0–2.3×).
  - Its frame (2316 × 6996 px) and detector settings match the two odd Batch_1 sites.
- **fn0mhxef:** every material KPI is within |z| ≤ 1.3 of the labelled population, so the material evidence cannot place it.
  - The imaging evidence (dark level 0.8 DN, gain 54.6, grey step 2) excludes the Batch_3 sessions.
  - It only weakly favours Batch_1 over Batch_2. Its frame height of 2048 px is shared only with Batch_2 `3806gxp0` and `avn74qx1`.
- **xrv9xvzb:** its material KPIs are typical (nearest site: Batch_3 `hzumfsms`).
  - It has slightly higher porosity (7.6%) and rounder, more solid Si than the batch medians.
  - Its acquisition fingerprint matches the Batch_3 session `9luzk4jm` / `hzumfsms` / `ufdvpb81`: every 3rd DN populated on BSE (every 4th on Inlens), dark level ≈ 18 DN, noise σ_g ≈ 0.17, frame height 2088. No other labelled site has this fingerprint.

The material KPI differences relative to Batch_3 are in `heldout_assign.json` (`z_Batch_3`) and in the table above.
