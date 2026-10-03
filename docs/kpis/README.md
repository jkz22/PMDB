# Si dispersion KPIs: what we measure and why

**Status:** v1 KPI set frozen for implementation (spec [`docs/specs/002-si-kpis.md`](../specs/002-si-kpis.md)).
**Machine-readable list:** [`kpi_catalogue.csv`](kpi_catalogue.csv). This file is the single source of truth for KPI IDs, definitions and output column names.
**Background reading:** [`literature-review.md`](literature-review.md) covers the geometry and topology methods behind these KPIs, with references.

---

## 1. The question the KPIs answer

The data provider describes the electrode as a graphite anode with silicon as the secondary active material. All images are SEM cross-sections. The provider's only stated quality criterion is that **silicon must not cluster and must not be localised**.

A QC verdict therefore has to answer two questions about each batch, relative to the approved baseline:

1. **Clustering:** is Si grouped into agglomerates or crowded patches, rather than dispersed as separate particles?
2. **Localisation:** is Si unevenly distributed through the coating thickness, across its width, or into Si-rich and Si-poor regions at some length scale?

Average Si content is not the question. A batch can have exactly the baseline Si loading and still fail because that Si is clumped or sits near one face.

## 2. Facts about the data that shaped the design

| Fact (verified on the data) | Consequence for the KPIs |
|---|---|
| 31 sites in 3 batches (7 / 7 / 17); each site is one ~175 × 40–58 µm cross-section at 25 nm/px, with BSE, Inlens and ETD or SE detectors aligned to within 0.5 px | One "slice" = one site. KPIs are computed per site and per tile within a site |
| In BSE, Si appears as compact, brighter, finely grained particles a few µm across, and graphite as large flakes 10–30 µm long aligned with the coating plane | Si can be resolved as objects, so object and point-pattern statistics are usable. The alignment means statistics must be split by direction (x = along the width, z = through the thickness) |
| The BSE histogram has a single peak; Si is a bright tail, not a separate mode | No global threshold. v0 uses an adaptive bright-tail rule; a trained pixel classifier replaces it later |
| Batch_3 was imaged with a different brightness offset | Normalisation happens per image. Thresholds are relative to each image's own statistics, never tuned per batch |
| No Cu current collector is visible in any frame | The through-thickness direction has no known sign. Gradients are reported as magnitude only, so "Si moved toward the top" cannot be told apart from "Si settled toward the foil" |
| ETD shows vertical curtaining streaks from ion milling | Lateral-uniformity KPIs are read alongside a curtaining index, so milling streaks are not reported as Si localisation |
| Which batch is the approved baseline is not confirmed | KPI computation is baseline-agnostic. Thresholds and verdicts belong to a later decision layer |

## 3. Design principles (and the alternatives we rejected)

1. **The verdict is about Si only.** Verdict KPIs (K01–K16) measure Si or Si relative to graphite.
   - Graphite and pore metrics (D01–D06) and artefact flags (A01–A03) are diagnostic. They explain why a batch differs (raw-material lot, calendering, drying, sample prep) and never fail a batch on their own.
   - *Rejected:* giving all three phases the same full KPI suite. That would triple the number of comparisons against the baseline, and therefore the false alarms, without answering the provider's question.
2. **Use both object metrics and field metrics.**
   - Object metrics (size, Voronoi, nearest neighbour, MST) are easy to interpret, but they depend on separating individual particles.
   - Field metrics (window variance, depth and lateral profiles, empty space) work directly on the Si pixel map and do not need particles to be split.
   - Each family covers a failure mode of the other.
3. **Compare against the right randomness.** Si can only sit between graphite flakes, never inside them.
   - Testing Si positions against complete spatial randomness over the whole image would call a perfectly mixed electrode "clustered", because the space available to Si is itself patchy.
   - Every point-pattern statistic is therefore compared with **random labelling in the admissible space**: the same number of points placed uniformly over non-graphite, non-artefact pixels.
   - Classical Clark–Evans R is also logged, only for comparison with the literature.
4. **Split by direction.** The graphite microstructure is anisotropic, so g(r), depth profiles and lateral profiles are computed separately along x and z, never averaged.
5. **Mask artefacts before measuring.** Isolated large voids (particle pull-out, cracks) are flagged and removed from the admissible space. Otherwise a hole left by pulled-out Si would read as "Si depletion".
6. **Keep the sampling hierarchy.**
   - Scalar KPIs are logged per tile (4 tiles per slice) as well as per slice.
   - The decision layer then keeps the tile, site and batch variance levels separate rather than pooling pixels.
   - With 7 sites in a batch, pooling is the easiest way to overstate confidence.
7. **No thresholds at this stage.** KPIs are measurements.
   - Acceptance limits come later from the baseline distribution, with a stated false-alarm rate.
   - The only fixed parameters are the cluster distance d, the agglomerate size d*, the window sizes and the band counts. Their v0 values are frozen in spec 002, with a sensitivity sweep logged for d and d*.

## 4. The KPIs

### 4.1 Verdict KPIs: clustering

| ID | KPI | What it measures | What failure looks like |
|---|---|---|---|
| K01 | Si fraction of admissible area | How much of the space Si can occupy is Si. Context for every other KPI | A shift points to Si loading or a segmentation change |
| K02 | Si number density | Si objects per 1000 µm² of admissible area | Fewer objects at the same K01 means particles have merged |
| K03 | Si object size (D50 / D90 / max) | Equivalent-circle diameter of Si objects. These are 2D section sizes, so use them only relative to the baseline | D90 or max rises while D50 holds: a tail of agglomerates |
| K04 | Agglomerate fraction | Groups Si objects that touch or lie within d of each other, and reports the share of Si area in groups larger than d* | Rises directly with clustering |
| K05 | Voronoi cell-area spread | How unequal the "free area" around each Si particle is. A random pattern gives ≈ 0.53 for point Voronoi; we compare against the admissible-space null | Above the null means clustered; below means regular |
| K06 | Voronoi cluster and void regions | Admissible area covered by unusually small cells (crowded) and unusually large cells (Si-free) | Both rise together when Si is clumped |
| K07 | Nearest-neighbour index R | Mean nearest-neighbour distance relative to random placement in admissible space | R < 1 means clustered |
| K08 | Pair-correlation excess (x and z) | Excess of Si pairs at distance r over the random envelope, and the distance at which it peaks | A positive peak gives both the presence and the size of clusters, separately in-plane and through-thickness |
| K09 | Minimum spanning tree (m, σ) | Mean and spread of edge lengths in the tree connecting all Si particles | Low m with high σ marks the clustered region of the m–σ plane (Dussert et al. 1986) |

### 4.2 Verdict KPIs: localisation

| ID | KPI | What it measures | What failure looks like |
|---|---|---|---|
| K10 | Scale-of-segregation curve (**headline**) | CV of local Si fraction across windows of 1–20 µm | For random dispersion the CV falls quickly with window size. A flat or slowly falling curve means Si-rich and Si-poor regions about that size |
| K11 | Lacey mixing index (10 µm) | Observed variance placed on a 0–1 scale between fully segregated (0) and random (1) | Well below 1 |
| K12 | Through-thickness profile | Si fraction in 5 depth bands: largest deviation and slope magnitude | Si migrated toward one face (direction unknown, see §2) |
| K13 | Lateral uniformity | CV of Si fraction across 10 µm column bins | Lateral Si streaks, provided the curtaining index A02 is low |
| K14 | Empty space | How far an admissible point can be from the nearest Si (P50, P95) | Si-free pockets |

### 4.3 Verdict KPIs: Si relative to graphite

| ID | KPI | What it measures | What failure looks like |
|---|---|---|---|
| K15 | Si–graphite contact fraction | Share of the Si boundary touching graphite | Si sitting in pores or binder, poorly connected |
| K16 | Si-to-graphite distance | Median distance from Si particles to the nearest graphite | Si detached from the graphite network |

### 4.4 Diagnostic and artefact KPIs (never fail a batch on their own)

| ID | KPI | Used for |
|---|---|---|
| D01 | Graphite fraction by depth band | Calendering or settling signature |
| D02 | Graphite orientation spread | Calendering alignment |
| D03 | Graphite size and aspect ratio | Raw-material lot change |
| D04 | Porosity by depth band | Calendering or drying signature |
| D05 | Pore size proxy | Densification |
| D06 | Pore connectivity (Euler number) | Pore-network change |
| A01 | Large-void fraction | Pull-out or cracks; high values make Si KPIs unreliable |
| A02 | Curtaining index | Milling streaks that could fake lateral localisation |
| A03 | Slice height | Coating-thickness proxy |

### 4.5 Stretch KPIs (v2, only after v1 is logged)

| ID | KPI | Purpose |
|---|---|---|
| S01 | Two-point cluster function C2(r) length | Cluster connectivity without splitting particles |
| S02 | Euler-number dilation merge radius | Scale at which Si merges under dilation |
| S03 | Persistent-homology H0 lifetimes | Multi-scale crowding |
| S04 | Minkowski-tensor anisotropy of Si | Si streak alignment |

## 5. How the KPIs feed the verdict (later specs)

```
site images → segmentation (Si / graphite / pore / artefact)
            → KPIs per tile and per site            ← spec 002 (this document)
            → baseline distribution per KPI (tile < site < batch)
            → per-KPI deviation + batch-level p-value with a stated false-alarm rate
            → accept / investigate / reject, with per-KPI attribution
            → diagnostic KPIs rank the likely process cause
```

Process-cause fingerprints used for attribution are hypotheses, not proofs, because no process-labelled data exists:
- clustering with no depth trend points to mixing;
- a depth gradient in Si points to drying or settling (sign unknown);
- a porosity gradient together with tighter graphite alignment points to calendering;
- a particle-size shift alone points to a raw-material lot;
- curtaining or pull-out points to sample prep.

## 6. Known limitations

- **v0 segmentation is provisional.** It is an adaptive bright-tail rule on BSE, not chemically confirmed, and there is no EDS ground truth. KPI values are conditional on it. Spec 002 keeps segmentation behind a single function, so a better segmenter can replace it without touching the KPI code.
- **2D sections.** Section diameters underestimate 3D particle sizes, so sizes are meaningful only relative to the baseline.
- **Unknown depth sign** (no foil in frame).
- **Unknown baseline batch.** The parameters d and d* use v0 defaults and must be re-frozen from the baseline once it is identified, before any incoming batch is scored.
- **Resolution floor.** At 50 nm/px, Si below about 0.2 µm drops out of the object layer.

## 7. Key references

Full list with retrieval status: [`literature-review.md`](literature-review.md).

- Clark & Evans (1954), Ecology. Nearest-neighbour index.
- Ripley (1977). K-function. Diggle (2003). Random labelling and case/control D-function.
- Duyckaerts & Godefroy (2000), J. Chem. Neuroanat. Voronoi cell-area CV for regular, random and clustered patterns.
- Monchaux et al. Voronoi σ ≈ 0.53 Poisson reference in 2D.
- Dussert et al. (1986), Phys. Rev. B 34, 3528. Minimum spanning tree m–σ plane.
- Danckwerts (1952). Intensity and scale of segregation. Lacey (1954). Mixing index.
- Torquato (2002), *Random Heterogeneous Materials*. Jiao, Stillinger & Torquato (2009), PNAS. Two-point and cluster functions.
- Schröder-Turk et al. (2010), J. Microsc. 238, 57. Minkowski tensors.
- Hiraoka et al. (2016), PNAS 113, 7035. Pritchard et al. (2023), Sci. Rep. Persistent homology on microscopy images.
- Müller et al. (2018), J. Electrochem. Soc. 165, A339. Electrode inhomogeneity across sub-volumes, split in-plane and through-plane.
- Vorauer et al. (2020), Commun. Chem. Region-based Si segmentation in BSE cross-sections of a Si-composite anode.
