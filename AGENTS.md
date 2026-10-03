# Agent Instructions & Data Reference

This document guides AI coding agents and automated workflows working on PMDB.

## Aligned Multi-Channel Dataset (`cache/half/`)

**All 31 microscopy sites are already pre-processed and committed to `cache/half/`.**
Agents do not need to downsample raw TIFFs from scratch or run cache build scripts unless specifically modifying the loader pipeline.

### Quick Data Loading

```python
from pmdb.io import list_sites, load_site

# 1. Inspect manifest of available sites (31 sites across Batch_1, Batch_2, Batch_3)
manifest = list_sites()

# 2. Load aligned 3-channel site (recommended default: half resolution, 50 nm/px)
site = load_site("Batch_1", "4ih2ggld", resolution="half")
# site.image: float32 ndarray of shape (H, W, 3), scaled [0, 1] via percentile normalisation
# site.nm_per_px: 50.0
# site.se_detector: 'ETD' or 'SE'
# site.channel_names: ['BSE', 'Inlens', 'ETD'] (or ['BSE', 'Inlens', 'SE'])

# Channel slicing:
bse = site.image[..., 0]      # Backscattered electron (compositional/phase contrast)
inlens = site.image[..., 1]   # In-lens secondary electron (topography/morphology)
se_type = site.image[..., 2]  # ETD or standard SE

# 3. Load unnormalised uint8 array (raw camera counts)
site_raw = load_site("Batch_1", "4ih2ggld", resolution="half", normalise="none")

# 4. Load with imaging-artefact harmonisation (per-site grey-level LUT, see docs/harmonisation.md).
#    Fixes the Batch 3 black-level / gain offset; raw data and site.raw_stats are untouched.
#    normalise="fixed" keeps every site on one common grey scale (grey / 255) instead of per-image percentiles.
site_h = load_site("Batch_3", "71vgq3fw", resolution="half", normalise="fixed", harmonise="hybrid")
# methods: "none" | "offset" | "affine2" | "affine3" | "histmatch" | "hybrid" (recommended)
# site_h.harmonised_stats: per-channel intensity stats after the LUT (site_h.raw_stats = before)
```

### Dataset Specifications

- **Detector Alignment**:
  - Channel 0: `BSE`
  - Channel 1: `Inlens`
  - Channel 2: `SE_type` (27 sites use `ETD`, 4 sites use `SE`)
- **Border Cropping**: 4-pixel margins from both left and right edges (`arr[:, 4:W-4]`) have been cropped to eliminate microscope border artefacts. The remaining pixels satisfy $R == G == B$ for all detectors.
- **Physical Scale**:
  - Full resolution (`data/`): $25.000 \pm 0.001$ nm/px
  - Half resolution (`cache/half/`): $50.0$ nm/px (2×2 local mean downsampling)
- **Known Confounds & QC Data**:
  - **Batch 3 BSE Brightness Offset**: Batch 3 BSE images have an elevated black level (`p1` averages ~7.18 vs 0.00 in Batches 1 and 2). Refer to [`outputs/raw_intensity_stats.csv`](outputs/raw_intensity_stats.csv).
    It is a per-site **affine** imaging artefact (offset + gain) on all three detectors: 4 Batch 3 sites (`71vgq3fw`, `kbdh4tri`, `tuy3zymq`, `x7u69zsw`) have black level ≈ +19–23 and gain ≈ 0.78×, 5 more (`9luzk4jm`, `hzumfsms`, `ptg8lmto`, `ufdvpb81`, `xgj4xftb`) have +3–7; the other 8 match Batches 1/2. Use `load_site(..., harmonise="hybrid")` for modelling so a model cannot read the batch off the grey levels. LUTs live in `cache/harmonised/<method>/` (rebuild: `python scripts/build_harmonised.py`; evaluation: `python scripts/eval_harmonisation.py` → `outputs/harmonisation/`). See [`docs/harmonisation.md`](docs/harmonisation.md).
  - Visual QC overviews are available in [`outputs/qc_contact_sheet.png`](outputs/qc_contact_sheet.png) and [`outputs/raw_stats_by_batch.png`](outputs/raw_stats_by_batch.png).

## Rules for Agents

1. **`data/` is Read-Only**: Never edit, move, delete, or write files to `data/`.
2. **Use Pre-processed Cache**: For segmentation, feature extraction, and KPI calculations (e.g. Spec 002), load from `resolution="half"`.
3. **Reproducibility**: Always run tests before completing tasks (`pytest -q -m "not data"` and `pytest -q -m data`).

## Held-back Test Sites (`data_heldout/`)

3 unlabelled sites (`3e122cbj`, `fn0mhxef`, `xrv9xvzb`) released by the organisers for scoring. Raw TIFFs in `data_heldout/Batch_heldout/`, processed half-res cache in `cache_heldout/half/` (load with `load_site("Batch_heldout", site, resolution="half", data_root="data_heldout", cache_root="cache_heldout")`). Read-only, never train on them. Batch 3 is the supplier baseline; every held-back site must be assigned to a batch with a confidence and an explanation of how it differs from Batch 3. See [`data_heldout/README.md`](data_heldout/README.md).
