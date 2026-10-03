# Materials KPIs vs our KPIs: consolidation

**Regenerate** (from repo root):

```bash
python scripts/materials_kpis_to_csv.py
python -m pmdb.screen --kpis outputs/kpis/site_kpis.csv outputs/kpis/materials_site_kpis.csv \
  --site-manifest cache/half/manifest.csv \
  --replicates outputs/kpis/tile_kpis.csv \
  --sensitivity outputs/kpis/sensitivity.csv --sensitivity-params d_um d_star_um \
  --covariates outputs/kpis/screen/covariates_bse.csv \
  --sentinel K08_pcf_rpeak_x_um=-1 K08_pcf_rpeak_z_um=-1 \
  --out-dir outputs/kpis/screen_combined
python scripts/consolidation_tables.py
```

**Screen output:** `kept 31/51 KPIs; drops by gate: {'reliability': 6, 'robustness': 1, 'redundant_with': 3, 'degeneracy': 8, 'artefact': 2}`

## 1. Result

| run | KPIs | kept | dropped | degeneracy | artefact | reliability | robustness | redundant_with |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 43 | 26 | 17 | 8 | 0 | 6 | 1 | 2 |
| combined | 51 | 31 | 20 | 8 | 2 | 6 | 1 | 3 |

New KPIs kept: 5/8.

_No existing KPI changed decision._

- Baseline = `outputs/kpis/screen/`, combined = `outputs/kpis/screen_combined/`; same flags and thresholds.

## 2. Every new KPI

| new KPI | decision | deciding gate | untested gates | closest existing KPI | rho | cluster rep |
| --- | --- | --- | --- | --- | --- | --- |
| mat_porosity | drop | artefact:BSE_p1 | reliability;robustness | D04_porosity_mean | +0.574 | - |
| mat_bright_fraction | drop | redundant_with:K01_si_frac_adm | reliability;robustness | K01_si_frac_adm | +0.915 | K01_si_frac_adm |
| mat_active_fraction | keep | - | reliability;robustness | D05_pore_size_d50_um | -0.592 | mat_active_fraction |
| mat_graphite_d10_um | keep | - | reliability;robustness | K09_mst_m_norm | +0.531 | mat_graphite_d10_um |
| mat_graphite_d50_um | keep | - | reliability;robustness | K11_lacey_w10 | +0.580 | mat_graphite_d50_um |
| mat_graphite_d90_um | keep | - | reliability;robustness | K10_cv_w10 | +0.512 | mat_graphite_d90_um |
| mat_crack_fraction | keep | - | reliability;robustness | D04_porosity_mean | +0.555 | mat_crack_fraction |
| mat_rim_coverage | drop | artefact:BSE_p50 | reliability;robustness | K08_pcf_rpeak_x_um | +0.797 | - |

- rho = Spearman over 31 sites vs the most similar of our 43 KPIs.

## 3. Near-duplicates: who was kept and why

- Rule (screen gate 4): keep the member with the highest replicate ICC; if no member has an ICC, keep the medoid (highest mean |rho| to the others).
- mat_ KPIs have no replicates, so they lose any tie against one of our KPIs that has an ICC.

| cluster | KPI | decision | ICC | mean abs rho to other members | rule |
| --- | --- | --- | --- | --- | --- |
| 1 | K01_si_frac_adm | keep | 0.50 | 0.915 | highest ICC |
| 1 | mat_bright_fraction | drop | n/a | 0.915 | - |

## 4. Same name, different answer (investigate, NOT redundancy)

| new KPI | existing KPI | rho | verdict | definition difference |
| --- | --- | --- | --- | --- |
| mat_porosity | D04_porosity_mean | +0.574 | disagree: check segmentation | theirs: pore px / all px, whole image; ours: mean pore fraction over 5 depth bands |
| mat_graphite_d50_um | D03_graphite_ecd_d50_um | +0.410 | disagree: check segmentation | theirs: watershed instances, border-touching excluded; ours: graphite object ECD D50 |
| mat_bright_fraction | K01_si_frac_adm | +0.915 | agree | theirs: opened brightest Otsu class / all px; ours: Si area / non-artefact area |

- 'disagree' = two segmentations of the same quantity rank the sites differently. This is a segmentation question, not a duplicate.
- Theirs: full-res BSE (25 nm/px), Gaussian + 4-class multi-Otsu (`src/segment.py`). Ours: half-res 3-channel (50 nm/px), `pmdb/segment.py`.
- Next step: overlay both pore / graphite masks on the sites with the largest rank gaps.

## 5. Caveats

- No replicates for mat_ KPIs -> reliability untested (see `untested_gates`). No sweep -> robustness untested too.
- `src/kpis_geometry.py` and `src/kpis_physics.py` are not in the repo -> `mat_orientation_anisotropy`, `geo_*` and `phys_*` KPIs are absent.
- mat_graphite_d*_um come out of `src/kpis_materials.py` already in micrometres (`pixel_size_um` from `config.yaml`); the converter passes them through and rejects a legacy pixel-unit parquet.
- n = 31 sites (7/7/17): borderline calls are fragile. See [screening.md](screening.md) section 6.
