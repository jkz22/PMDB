# PMDB (Porous Material Database)

Microscopy analysis and automated KPI pipeline for porous silicon/graphite composite materials.

## Dataset & Pre-processed Aligned Data

The repository contains raw microscopy TIFFs in `data/` and **pre-processed, aligned multi-channel arrays** in `cache/half/`:

- **31 sites** across 3 batches (`Batch_1`: 7, `Batch_2`: 7, `Batch_3`: 17).
- **Channels**: Fixed 3-channel stack `[BSE, Inlens, SE_type]`:
  - Channel 0 (`BSE`): Backscattered Electron detector (compositional/phase contrast).
  - Channel 1 (`Inlens`): In-lens detector (high-resolution surface morphology).
  - Channel 2 (`SE_type`): Secondary Electron detector (`ETD` for 27 sites, `SE` for 4 sites; check `site.se_detector`).
- **Pre-processing**:
  - Fixed 4-px border artefact columns cropped from both left and right edges (`arr[:, 4:W-4]`).
  - Downsampled 2×2 block mean to 50 nm/px (full resolution is 25 nm/px).
  - Stored as compressed numpy archives in `cache/half/<batch>__<site>.npz`.
  - How processed arrays differ from the raw TIFFs, what normalisation costs, and what is *not* done: see [`docs/data-processing.md`](docs/data-processing.md).

> **Note for Agents & Analysis Pipelines**:
> All 31 sites are pre-processed and tracked directly in `cache/half/`.
> Use `resolution="half"` for fast, out-of-the-box loading without re-processing raw TIFFs. See [`AGENTS.md`](AGENTS.md) for detailed guidelines.

## Quickstart: Data Loading

```python
from pmdb.io import list_sites, load_site

# 1. Discover all 31 sites and metadata
manifest = list_sites()

# 2. Load aligned site at half resolution (50 nm/px, float32 normalized to [0, 1])
site = load_site("Batch_1", "4ih2ggld", resolution="half")
print(site.image.shape)      # (1158, 3494, 3) -> [H, W, 3]
print(site.nm_per_px)        # 50.0 nm/px
print(site.channel_names)    # ['BSE', 'Inlens', 'ETD']
print(site.se_detector)      # 'ETD'

# Slice channels directly:
bse, inlens, se_type = site.image[..., 0], site.image[..., 1], site.image[..., 2]

# 3. Load raw uint8 counts without percentile normalization:
site_raw = load_site("Batch_1", "4ih2ggld", resolution="half", normalise="none")

# 4. Load with imaging-artefact harmonisation (per-site grey-level LUT, see docs/harmonisation.md).
#    Fixes the Batch 3 black-level / gain offset; raw data and site.raw_stats are untouched.
#    normalise="fixed" keeps every site on one common grey scale (grey / 255) instead of per-image percentiles.
site_h = load_site("Batch_3", "71vgq3fw", resolution="half", normalise="fixed", harmonise="hybrid")
# methods: "none" | "offset" | "affine2" | "affine3" | "histmatch" | "hybrid" (recommended)
# site_h.harmonised_stats: per-channel intensity stats after the LUT (site_h.raw_stats = before)
```

## Key Files & Outputs

- **`cache/half/manifest.csv`**: Pre-computed manifest of all 31 sites with bounding shapes and detector types.
- **`outputs/raw_intensity_stats.csv`**: Baseline intensity percentiles (`p0_5`, `p1`, `p50`, `p99`, `p99_5`, `mean`, `std`) across all 93 detector channels. (Note: Batch 3 exhibits a known BSE brightness offset where `p1` averages ~7.18 vs 0 in Batches 1 & 2).
- **`outputs/qc_contact_sheet.png`**: Contact sheet visualization of all 31 sites across all 3 detectors.
- **`outputs/raw_stats_by_batch.png`**: QC strip plot of intensity percentiles across batches.
- **`docs/kpis/screening.md`**: KPI screening tool (`python -m pmdb.screen`) and the KPI submission contract.

## Testing & Pipeline Execution

```bash
# Run unit tests (synthetic data, no real data required)
PMDB_DATA=/nonexistent pytest -q -m "not data"

# Run integration tests against real microscopy data
pytest -q -m data

# Build cache or regenerate QC figures
python scripts/build_cache.py
python scripts/qc_overview.py
```

## Running on Modal

Each person uses their own Modal account; no credentials live in the repo. One-time login (stores a token in `~/.modal.toml`):

```bash
pip install modal
modal token new
```

One-time upload of the half-resolution cache to a Modal Volume (only `cache/half` is uploaded, never `data/`):

```bash
modal volume create pmdb-data
modal volume put pmdb-data cache/half /half
modal volume ls pmdb-data /half
```

To refresh after a cache rebuild: `modal volume put --force pmdb-data cache/half /half`.

Run (local `pmdb/` edits ship automatically on each run):

```bash
modal run modal_app.py --smoke   # 1 site
modal run modal_app.py           # all 31 sites
# option: --name <name>
```

Results land in `outputs/modal/<name>.csv` (gitignored), one row per site with the KPI columns from `docs/kpis/kpi_catalogue.csv` v1 plus `runner`, `elapsed_s` and `error`. If any site errors, rows go to `<name>_failed.csv` instead (an existing `<name>.csv` is left untouched) and the command exits 1.
