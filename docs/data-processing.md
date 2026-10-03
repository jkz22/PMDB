# Raw vs processed data: what the site loader changes

**Applies to:** `pmdb/io.py` (spec [`001-site-loader.md`](specs/001-site-loader.md), revision 2)
**Audience:** anyone reading `cache/half/` or calling `load_site()` who needs to know how far the arrays are from the original TIFFs.

## Summary

Processing is mostly *packing*: the three detector TIFFs for one cross-section become one 3-channel array. Around that, the loader does three more things. It **cleans** the data (removes redundant colour channels and border artefacts), **validates** it (inventory, shapes, scale), and adds **metadata** (scale, detector flag, raw intensity statistics). It also offers two **optional transforms**: downsampling, for the cache, and normalisation, applied at load time.

Only normalisation changes what pixel values *mean*. Everything else is lossless or close to it.

## Step by step, for one site

| # | Step | Raw (`data/`) | Processed | Why |
|---|---|---|---|---|
| 1 | Collapse fake colour | Each TIFF is stored as RGB, but R, G and B are copies of one greyscale image. One site is 3 files × 3 channels = 9 planes, 6 of them redundant | One greyscale plane per detector | Saves memory. Asserting R == G == B also catches corrupted or unexpected files |
| 2 | Crop border artefacts | In 39 of the 93 files (13 sites × 3 detectors), G carries non-image values in full-height edge columns: left columns 0–3 and/or the last two columns | A fixed 4 px cropped from the left and right of every image (`arr[:, 4:W-4]`), about 100 nm per side | Stops artefact values from distorting intensity statistics and edge-sensitive features. A fixed crop keeps the geometry identical across sites |
| 3 | Stack detectors | 3 files per site: BSE, Inlens, and ETD or SE | One `(H, W, 3)` array in the fixed order `[BSE, Inlens, SE_type]` | The packing. Valid because the detectors are already aligned to within 0.5 px (phase cross-correlation, all 62 pairs) |
| 4 | Validate | Nothing is checked | Site counts per batch, detector set per site, ETD vs SE flag, equal shapes across detectors, scale of 25 ± 0.01 nm/px | A malformed future batch fails loudly rather than quietly producing wrong KPIs |
| 5 | Attach metadata | Filenames only | `batch`, `site`, `se_detector`, `nm_per_px`, `resolution`, plus per-channel raw intensity statistics | KPIs can be reported in µm rather than pixels, and evidence of imaging confounds is kept |
| 6 | Downsample (cache only) | 25 nm/px | 50 nm/px: 2×2 block mean, rounded to `uint8` | About 4× less data, so development iterations take seconds |
| 7 | Normalise (at load time, never stored) | 0–255 grey levels whose meaning depends on the microscope settings | `float32` in [0, 1]. Each image and each channel is scaled so its 0.5th percentile maps to 0 and its 99.5th to 1, then clipped | Removes brightness and contrast differences between imaging sessions, e.g. the Batch_3 BSE offset |

Two consequences of where the steps happen:

- **`cache/half/*.npz` holds un-normalised grey levels**: steps 1–6 only. Normalisation (step 7) runs inside `load_site()`. You can therefore change the normalisation method without rebuilding the cache.
- **`load_site(..., normalise="none")` returns those `uint8` values.** They are *not* raw camera counts. The original TIFFs had already been contrast-stretched before we received them (the BSE histogram has a comb pattern of empty grey levels), and steps 2 and 6 have also been applied.

## What each transform costs

### Normalisation can hide real material differences

Percentile scaling responds to what is *in* the image, not only to the microscope settings. A site with more bright silicon has a higher 99.5th percentile, so after scaling its silicon looks dimmer than identical silicon in a site with less of it. Likewise, a site with more pore area shifts the 0.5th percentile.

Normalisation can therefore partly remove a genuine material difference along with the imaging difference. Three safeguards exist for this reason:

1. The raw per-file statistics are always written to `outputs/raw_intensity_stats.csv`, and `Site.raw_stats` carries them per site.
2. Spec 001 acceptance criterion 6 requires the Batch_3 offset to stay visible in those statistics. Mean BSE p1 is about 7 in Batch_3 against 0 in Batches 1 and 2.
3. Planned improvement: anchor normalisation on regions that should look the same in every batch, namely pore black and the graphite interior, rather than on whole-image percentiles.

### Downsampling blurs fine texture

At 50 nm/px, the fine grain texture inside Si particles and the thinnest carbon-binder strands blur. That does not matter for detecting, counting and locating Si particles a few µm across (they still span 40+ px). It does matter for texture-based features, which should use `resolution="full"`.

### The border crop removes a little width

Eight pixels of width (0.2 µm) are lost per image. Edge-touching objects are already excluded by most KPIs, so this is negligible.

## What processing does *not* do

- **No registration.** The detectors are already aligned. Only the central 1024 × 2048 px region was checked, though; small distortions at the far ends of the 175 µm field have not been ruled out.
- **No denoising.**
- **No removal of the vertical milling streaks (curtaining)** visible mainly in ETD. Lateral-uniformity KPIs must be read alongside a curtaining index.
- **No segmentation, and no correction of orientation.** No Cu foil is visible, so which edge is the top surface and which is the foil remains unknown.

The processed site is still an image of the same specimen. It is cleaned, checked and in physical units, but nothing in it has been interpreted yet.
