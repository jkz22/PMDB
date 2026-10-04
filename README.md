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
#    normalise defaults to "fixed" (grey / 255, one common scale) when harmonise is set; "percentile" would cancel the LUT.
site_h = load_site("Batch_3", "71vgq3fw", resolution="half", normalise="fixed", harmonise="hybrid")
# methods: "none" | "offset" | "affine2" | "affine3" | "histmatch" | "hybrid" (recommended)
# site_h.harmonised_stats: per-channel intensity stats after the LUT (site_h.raw_stats = before)

# 5. Physically cleaned arrays (second, physics-based route, see docs/clean.md): z = (I - D) / G(x, y)
#    with D the measured dark level and G the graphite flat-field, so pores ~ 0 and graphite = 1 on every
#    site; mask = uint16 bit field (border/markers, Cu collector, free surface, scan bands, charging, clipping).
#    Arrays are rebuilt locally (gitignored): python scripts/build_clean.py --workers 4   (~12 min)
from pmdb.io import load_clean
from pmdb import clean
z, mask = load_clean("Batch_3", "71vgq3fw", "BSE", kind="harm", resolution="half")  # kind: "norm" | "harm"
valid = clean.valid_for_kpis(mask)  # never compute statistics/KPIs on masked pixels

# 6. Imported, literature-standard pipelines (third route, see docs/harmonisation_ext.md): "nyul" = N4ITK bias-field
#    + Nyul-Udupa histogram standardisation, "basic" = BaSiC flat-/dark-field + per-image baseline.
#    Arrays are rebuilt locally (gitignored): python scripts/build_harmonise_ext.py   (~6 min, after build_clean.py)
from pmdb.harmonise_ext import load_ext
img, mask3 = load_ext("Batch_3", "71vgq3fw", method="nyul")  # img (H, W, 3) uint8 [BSE, Inlens, SE_type], mask3 (H, W, 3) uint16
```

## Key Files & Outputs

- **`cache/half/manifest.csv`**: Pre-computed manifest of all 31 sites with bounding shapes and detector types.
- **`outputs/raw_intensity_stats.csv`**: Baseline intensity percentiles (`p0_5`, `p1`, `p50`, `p99`, `p99_5`, `mean`, `std`) across all 93 detector channels. (Note: Batch 3 exhibits a known BSE brightness offset where `p1` averages ~7.18 vs 0 in Batches 1 & 2).
- **`outputs/qc_contact_sheet.png`**: Contact sheet visualization of all 31 sites across all 3 detectors.
- **`outputs/raw_stats_by_batch.png`**: QC strip plot of intensity percentiles across batches.
- **`docs/kpis/screening.md`**: KPI screening tool (`python -m pmdb.screen`) and the KPI submission contract.
- **`docs/story.md`**: the connecting narrative across every analysis layer (harmonisation → KPIs → fingerprint → GP maps → representation learning → functional morphology → FEM), with pointers to the numbers.
- **`docs/kpis/stretch.md`** / `outputs/stretch/`: catalogue v2 stretch KPIs S01–S04 (`pmdb/kpis/stretch.py`; `python scripts/run_stretch.py && python scripts/analyse_stretch.py`), implemented and shown redundant with v1 — do not add them to the fingerprint.
- **`docs/acceptance.md`** / `outputs/acceptance/`: the one-class question (is a field inside the Batch 3 baseline?) — best single-column AUC 0.74 = best-of-84 chance level (perm p = 0.50); fingerprint B3-vs-rest AUC 0.56. `python scripts/run_acceptance.py`; pre-specified joint family score (functional AUC 0.71, p ≈ 0.08 corrected; held-out percentiles) `python scripts/run_acceptance_joint.py`; per-field swelling QC sheets `python scripts/swelling_cards.py`.
- **`docs/spine.md`** / `outputs/spine/`: frozen pretrained spine (DINOv2-S, MicroNet) vs five mask fractions → crop-level swelling targets, LOSO. Constrained share: masks alone site R² 0.91, spine adds nothing; pore loss: MicroNet + masks 0.90 vs 0.84 (n.s.). Verdict: not the optimal model for the swelling data. `python scripts/run_swelling_spine.py extract|evaluate`.
- **`docs/functional.md`** / `outputs/functional/`: functional morphology (`pmdb.functional`): pore access, the Si swelling stress test (where lithiation growth lands: graphite / binder / pore), and a sites-per-batch power analysis. `python scripts/run_functional.py && python scripts/analyse_functional.py`; arrangement null `python scripts/run_swelling_null.py`; held-out explanation `python scripts/explain_heldout_functional.py`.

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
