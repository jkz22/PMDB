# KPI reliability and sectioning noise

Two sampling questions the modelling sessions assume away: (1) how much of a site-level KPI is the
*site* rather than *where in the field you looked* (`scripts/run_kpi_reliability.py` →
`outputs/reliability/`); (2) how much field-to-field spread a single 2D slice of a 3D electrode produces
on its own (`scripts/run_sectioning_sim.py` → `outputs/sectioning/`). Both are label-free; held-out
fields are excluded.

## 1. Within-field reliability, ICC(1) from 16 tiles x 31 fields

`outputs/overnight/features/tile_kpis_rich.csv` (16 full-height tiles of ≈ 3.2 µm per field). ICC(1) is
the share of tile-KPI variance that is between fields; the Spearman–Brown step gives the reliability of
the 16-tile mean, i.e. roughly of the site-level value.

| KPI | ICC one tile | ICC 16-tile mean | sampling SD of the site mean | batch-median range / sampling SD |
|---|---|---|---|---|
| K02 Si objects per 1000 µm² | **0.82** | 0.99 | 2.6 | 0.8 |
| K15 Si–graphite contact | 0.31 | 0.88 | 0.018 | **1.8** |
| K05 Voronoi σ (13 fields) | 0.29 | 0.87 | 0.08 | 1.3 |
| K01 Si fraction | 0.28 | 0.86 | 0.010 | 0.7 |
| K09 MST mean | 0.23 | 0.83 | 0.12 | 0.5 |
| K14 empty-space p95 | 0.23 | 0.83 | 0.8 µm | 0.9 |
| K03 ECD d50 / d90 / max | 0.20 / 0.18 / 0.20 | 0.80 / 0.78 / 0.80 | 0.08 / 0.21 / 0.37 µm | 1.5 / 1.2 / 0.2 |
| K04 clusters per 1000 µm² | 0.19 | 0.79 | 23 | 0.6 |
| K09 MST σ | 0.16 | 0.76 | 0.16 | 0.1 |
| K14 empty-space p50 | 0.07 | 0.56 | 0.3 µm | 0.0 |
| K04 agglomerate fraction | 0.07 | 0.54 | 0.075 | 1.0 |
| **K07 R (CSR) / R (RL)** | **0.03 / 0.02** | **0.30 / 0.26** | 0.05 / 0.06 | 1.0 / 1.4 |
| K16 Si–graphite distance median | — (constant 0) | — | — | — |

Reading:

- One tile tells you the field only for **K02** (Si number density; it is also what separates the two
  high-Si Batch 1 fields and 3e122cbj). Every other KPI is dominated by sampling at the tile scale
  (ICC 0.03–0.31): a 3 µm-wide strip is not a measurement of the electrode.
- At the full field, most KPIs reach reliability 0.76–0.99 — the field size was chosen well — **except
  K07 (Clark–Evans R vs CSR / vs random labelling), K04 agglomerate fraction and K14 p50**, whose site
  values are 45–75 % sampling noise. K07's two columns are nearly pure noise (16-tile reliability 0.26–0.30):
  their batch-median range is about one sampling SD, so any batch effect they appear to carry is at the
  level of the noise. These columns should not be in a fingerprint or a classifier at one field per site;
  that they are among the top "explanations" the RF classifier offers for fn0mhxef (`docs/classifier/results.md`,
  K07_R_csr z = +1.4) is consistent with that.
- The column with the largest batch range relative to its own sampling noise is **K15 Si–graphite
  contact** (1.8 sampling SDs) — the same quantity the fingerprint and the functional analysis point to.
  K03 d50 is next (1.5). Both are small: the batch-median differences are 1–2 sampling SDs of a single
  field, which is the power problem of `outputs/functional/power.csv` seen from the other side.
- Practical rule for the next imaging campaign: reliability ≥ 0.9 for K01/K15/K03 needs ~2× the current
  field area per site (Spearman–Brown), or two fields per electrode; nothing short of that will make K07
  usable (reliability 0.5 would need ~4× the area).

## 2. Sectioning noise: identical 3D volumes, different planes

`scripts/run_sectioning_sim.py`: four synthetic 50 × 16 × 175 µm volumes (0.1 µm voxels) with *identical*
generating parameters — in-plane oblate graphite flakes to 61 %, lognormal Si spheres (median d 0.6 µm,
σ 0.7) to 6 %, 0.4–1 µm spherical pores to 3.5 % — sliced on 8 planes each (2 µm apart); every slice is
the size of a real half-res field and goes through the real KPI and swelling code. Slice-to-slice SD
with the volume fixed is the sectioning floor; compared with the spread across the 17 Batch 3 fields:

| quantity | sectioning CV (same volume) | observed CV, Batch 3 fields | sectioning share of Batch 3 variance |
|---|---|---|---|
| K01 Si fraction | 0.11 | 0.16 | **≈ 0.44** |
| K02 Si objects / 1000 µm² | 0.05 | 0.37 | 0.05 |
| K03 ECD d50 / d90 | 0.04 / 0.06 | 0.31 / 0.16 | 0.02 / 0.08 |
| F02 pore loss (SOC 1) | 0.17 | 0.27 | 0.14 |
| K15 Si–graphite contact, F02 constrained share | 0.11 / 0.11 | 0.04 / 0.02 | not interpretable (see below) |

Reading:

- **Roughly half of the Si-fraction spread inside Batch 3 is the slice.** A 175 × 50 µm section of a
  6 % Si electrode with particles up to several µm has a sectioning CV of ≈ 0.11 on its own; the
  observed CV is 0.16. Field-to-field K01 differences of < 0.01 inside a batch are within sectioning
  noise; the two high-Si Batch 1 fields (0.14, 0.16 vs 0.06) are ~8 sectioning SDs away and are not.
- Number density, particle size and pore loss are **not** sectioning-limited (2–14 % of the variance):
  their between-field spread is real field-to-field variation (or segmentation), which is what the
  fingerprint and the GP maps are entitled to use.
- For **K15 and the constrained share the synthetic null is the wrong geometry**: random flakes give a
  sectioning CV of 0.11 while the real fields vary by only 0.04 / 0.02 across 17 Batch 3 sections. Real
  Si–graphite contact is far more *consistent* than any random arrangement of the same phases — the
  slice-level reading of the relocation-null result (`docs/functional.md` §2.6). The share is therefore
  reported as not interpretable rather than clipped to 1.
- The synthetic volume is a sampling model, not a reconstruction (flake shapes, Si size distribution and
  the Si-on-graphite preference are guessed; parameters in `outputs/sectioning/params.json`), so the
  shares are order-of-magnitude statements. The qualitative split — composition sectioning-limited,
  arrangement/size not — is stable under the two geometries tried (box size, Si placement rule).

## 3. Segmenter agreement: do the functional conclusions survive a different segmenter?

`scripts/run_segmenter_agreement.py` runs the analytical-benchmarks session's per-image anchored
segmenter (`scripts/_ab_seg_ref.py`, copied verbatim; pore / Si / everything-else, no binder class) next
to `pmdb.segment` v0r1 on the same 31 BSE fields (`outputs/segagree/`):

| quantity | Spearman across 31 fields | batch KW p, v0r1 → anchored |
|---|---|---|
| Si IoU median 0.88 (min 0.68), pore IoU median 0.62 | | |
| K01 Si fraction | 0.86 | 0.38 → 0.14 |
| F02 pore loss (SOC 1) | **0.97** | 0.14 → 0.15 |
| F02 share of growth not landing on pore | 0.84 | 0.47 → 0.73 |
| F02 share landing on graphite (constrained share) | **0.38** | **0.015 → 0.73** |

Reading:

- Si fraction, the two high-Si Batch 1 outliers, pore loss and the pore-avoidance share are
  segmenter-independent. The pore-closure story and the FEM crosswalk (`docs/crosswalk.md` §2) stand.
- **The "Batch 3 is the most constrained" result does not transfer.** Under v0r1 the batch effect is in
  how much growth lands on *graphite rather than binder/carbon*; the share that lands on anything solid
  (1 − into pore) has no batch effect under either segmenter. The anchored segmenter has no binder class,
  so it cannot see a graphite/binder split — and the v0r1 split is a grey-level threshold on BSE in the
  range where carbon black, binder and graphite overlap. The earlier robustness check (±0.05 threshold
  perturbation, `docs/functional.md` §2.3) moved the threshold but kept the class structure; this is the
  stronger test and the result is: the constrained-share batch difference is a statement about the v0r1
  graphite/binder boundary, not demonstrably about the electrode. It is downgraded accordingly in
  `docs/functional.md` and `docs/story.md`; the depth-independence of the constrained share
  (`docs/crosswalk.md` §3) is unaffected because it holds within every segmenter.
