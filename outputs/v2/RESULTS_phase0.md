## Phase 0: data, imaging statistics, normalisation, KPIs

### Observed
- **Data.** There are 31 fields (Batch_1 7, Batch_2 7, Batch_3 17) and 124 TIFFs: 31 BSE, 31 Inlens, 27 ETD, 4 SE.
  - The TIFFs contain no microscope metadata, only resolution tags (25 nm/px), so detector and field come from the file names (`metadata/`).
  - Detector co-registration was checked by phase correlation on left, centre and right windows. The offset is at most 0.11 native px for every detector pair (`metadata/coregistration_summary.csv`).
- **Raw imaging statistics.** Computed per image and per evaluation crop (`imaging_stats/`).
  - Batch_3 has a raised black level (mean p1) on every detector, compared with Batch_1:

    | Detector | Batch_1 p1 | Batch_3 p1 |
    |---|---:|---:|
    | BSE | 0 | 7.2 |
    | Inlens | 1.7 | 15.9 |
    | SE/ETD | 0 | 4.7 |

  - Batch_3 also has lower normalised sharpness. For BSE it is 0.109 against 0.151 for Batch_1.
  - Batch_3 Inlens images have a larger dark mixture component: weight w0 = 0.39 against 0.22.
- **Teammate normalisation** (`pmdb.io.load_site`, p0.5→0, p99.5→1, clipped).
  - It is an exact affine map of the raw data: linear R² = 1 and lookup spread = 0, with gain = 1/(p99.5−p0.5).
  - Its parameters are set per image and per channel, not per batch or globally. Gain coefficient of variation: BSE 0.12, Inlens 0.09, SE/ETD 0.11.
- **KPI code.** Used from Kevin's `geometric-kpis` branch at d23a116c3d07, wrapped without modification (`src/v2/kpi_adapter.py`).
  - G1 synthetic phantom: passes. Si fraction is 4.85% against 4.87% truth, and D50 is 0.751 µm against 0.751 µm truth.
  - G2 1.2× upscale: fails for K03 D50. The ratio is 0.77–1.02 (mean 0.89), against the expected 1.20. Phase fractions move by at most 0.5 percentage points.
  - G3 native 25 nm vs 50 nm per field (n = 31). Values are half-res minus native, with the Spearman correlation between resolutions:

    | KPI | Mean relative difference | Spearman |
    |---|---:|---:|
    | Si fraction | −0.8% | 0.996 |
    | K01 | −0.8% | 0.994 |
    | K04 agglomerate fraction | +1.3% | 0.996 |
    | Pore fraction | −8.3% | 0.978 |
    | K02 | −20% | 0.974 |
    | K03 D50 | +26% | 0.924 |
    | K04 cluster density | −10% | 0.929 |

- **KPI set used** (decided with Jihan): `frac_si`, `frac_graphite`, `frac_pore`, `K01_si_frac_adm`, `K04_agglom_frac`.
  - K02, K03 and K04 cluster density are excluded until G2 passes.
  - K04 agglomerate fraction is undefined on 3.8% of evaluation crops and 4.2% of training crops, because those crops contain too few Si objects. These values are kept as NaN. Their loss is masked in VAE-C. In VAE-B they are passed as a missing-value indicator.

### Interpretation
- The Batch_3 black-level and sharpness differences are imaging differences between batches. No metadata field records the cause, so nothing causal is claimed.
- Batch identity is therefore partly confounded with imaging condition for any representation trained on raw counts. The latent audit in Phase 6 tests this directly.
- Size and count KPIs change with resolution and do not scale under a pure geometric enlargement. This is most likely because the segmentation uses smoothing and minimum-area steps defined in fixed pixels; that explanation is not verified.
