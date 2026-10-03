# Geometry and topology for Si-dispersion KPIs in graphite/Si anode cross-sections — literature review

**Scope.** Mathematical descriptors that turn a segmented SEM cross-section (Si / graphite / pore) into scalar or curve-valued KPIs for the provider's two criteria: Si must **not cluster** and must **not be localised**. Recommended pipeline: segment Si → (a) object layer: labelled Si particles, (b) field layer: binary Si indicator map. Every KPI below acts on one of these two layers.

**Retrieval status.** References marked [V] were retrieved and checked in this session. Those marked [B] come from background knowledge and are standard textbook sources, but their bibliographic details were **not** re-verified here. Check them before citing in a manuscript.

---

## 1. Reasons to use more than one family

Each descriptor family has a different failure mode:

| Layer | Family | Needs resolved particles? | Sensitive to | Blind to |
|---|---|---|---|---|
| Object | Size/shape distribution | yes | agglomerate size | where agglomerates are |
| Object | Point-pattern (NN, Ripley, MST, Voronoi) | yes (centroids) | clustering vs regularity across scales | particle size and shape (point reduction) |
| Field | Two-point / cluster / lineal-path functions | no | correlation length, connectivity, anisotropy | the location of a defect within the image |
| Field | Minkowski functionals / tensors | no | amount, boundary, connectivity, orientation | spatial arrangement beyond connectivity |
| Field | Persistent homology | no | multi-scale merging of Si into clusters | absolute position |
| Field | Mixing indices / window variance | no | localisation and gradients, scale of segregation | morphology within a window |

Because nano-Si may sit partly below SEM resolution, use the field-layer metrics as the primary evidence and the object-layer metrics as interpretable companions. The field-layer metrics do not need individual particles to be separated.

---

## 2. Object-level geometry ("Si blocks": size, number, shape)

- **Equivalent-circle diameter distribution, number density, area fraction** from connected-component labelling (`skimage.measure.regionprops`). KPIs: P95 and maximum agglomerate size; the share of Si area held in objects larger than d* (the "agglomerate fraction").
- **Shape factors**: circularity 4πA/P², solidity (A / convex-hull area) and aspect ratio. A compact Si particle has high solidity, while a touching agglomerate has low solidity and lobes.
- **Morphological granulometry** [B: Matheron 1975; Serra 1982]: open the Si mask with discs of increasing radius and record the remaining area against radius. The derivative of this curve is a size distribution that does not depend on particle splitting, so it is robust to touching particles.
- **Stereological caveat** [B: Wicksell 1925]: a 2D section through a 3D particle cuts it below its equator, so section diameters underestimate true size. Use section sizes only for **relative** comparison against baseline unless you deconvolve them.
- **Battery precedent** [V]: Vorauer et al. (Commun. Chem. 2020) segmented a Si-composite anode in an ion-sliced BSE cross-section by grey value. They reported the Si-domain extent (max 6.4 µm, mean ≈0.90 µm) and the distribution of the number of Si particles per unit area. Because pixel thresholds were error-prone, they used region-based segmentation (Felzenszwalb superpixels and per-region grey statistics). This matches the decision already logged for this project to classify per particle or per region rather than per pixel.

## 3. Point-pattern statistics on Si centroids

Reduce each Si object to its centroid and compare against a null model.

1. **Clark–Evans nearest-neighbour index R** [V via secondary; original Clark & Evans 1954]: R = observed mean NN distance / expected under complete spatial randomness (CSR). R < 1 means clustered, R ≈ 1 random, R > 1 regular. Gives a single number but responds only to the first-neighbour scale. Edge correction is required.
2. **Ripley's K / Besag's L, pair-correlation g(r)** [V via secondary; Ripley 1976/77]: clustering as a function of scale r. L(r) − r > 0 means clustered at scale r. g(r) is the non-cumulative form and is easier to read at larger r. Use the isotropic or translation edge correction, and report the r of maximum L(r) − r as a "cluster scale". Companions: G (NN distribution), F (empty space) and J = (1 − G)/(1 − F).
3. **Voronoi tessellation** (the method you recalled) [V]:
   - Assign each Si centroid its Voronoi cell. Cell area is the "free area" around a particle, and 1/area is a **local number density**.
   - **KPI: CV of normalised cell areas, σ_V.** For a 2D Poisson (random) pattern σ_V ≈ 0.53. Higher values mean clustering and lower values mean regularity. This benchmark comes from the turbulence preferential-concentration literature (Monchaux et al.; Obligado/Mora et al. arXiv:1907.07607).
   - The CV is scale-free and does not change if the section is stretched or shrunk (Duyckaerts & Godefroy 2000, J. Chem. Neuroanat.). This makes it robust to magnification or calibration differences between batches.
   - **Cluster detection:** contiguous runs of small Voronoi cells mark clusters, and runs of large cells mark Si-depleted voids (Duyckaerts & Godefroy). Threshold at the Poisson-predicted PDF crossovers, and report the cluster area fraction and the void area fraction.
   - Recent materials use: CV of Voronoi areas as a dimensionless inter-sample dispersion metric for nanoparticles (cryo-TEM, ACS 2026).
   - **Set Voronoi / SKIZ** [B]: for non-point particles, build the tessellation from particle boundaries (skeleton by influence zones, or a watershed on the distance transform of the Si mask). Then *local Si area fraction* = particle area / its cell area, which gives a per-particle local-concentration map.
   - Drop cells that touch the image border, or clip them to the window.
4. **Delaunay graph and minimum spanning tree (MST)** [V]: Dussert et al. (PRB 1986) characterise point sets by the mean m and standard deviation σ of MST edge lengths. The (m, σ) plane separates regular, random and clustered sets. Cutting MST edges longer than a distance d gives a **cluster definition** ("Si particles within d of each other") that depends on a single parameter. This agrees with the cluster definition already recorded for this project, and the MST makes it computable.

### Choice of null model (needed for valid inference here)
In a calendered graphite electrode, Si **cannot** sit inside graphite flakes. It occupies the interstitial (binder + pore) space, which is itself anisotropic and banded. CSR over the full image will therefore report "clustering" even for a perfectly mixed electrode, because the available space is clustered. Use one of the following instead:
- **Random labelling**: compare the Si pattern with random draws of the same number of locations from the *non-graphite* mask (Monte Carlo envelopes, for example `spatstat::envelope`).
- **Inhomogeneous K / g** with intensity estimated from the admissible-space mask.
- **Baseline-relative KPIs**: report each statistic as a z-score or ratio against the approved-baseline distribution, so that structure shared by all batches cancels.
Diggle's D(t) = K_cases − K_controls (random labelling) was designed for this kind of case/control situation [V via arXiv:0808.1409].

## 4. Field-level correlation functions (no particle splitting needed)

Torquato's n-point framework [V: Torquato 2002; Jiao, Stillinger & Torquato PRE 2007/2008]:
- **S2(r)**: probability that two points separated by r both lie in Si. Its decay length is a correlation length. Compute it **separately along the width (x) and through the thickness (z)**, because graphite flakes are aligned.
- **Two-point cluster function C2(r)**: probability that both points lie in the **same Si cluster**. It is short-ranged when Si is dispersed and lengthens as clusters grow, so it is the most direct field-level clustering KPI. Jiao et al. (PNAS 2009) show that C2 carries appreciably more information than S2, lineal-path or chord-length functions.
- **Lineal-path L(r) and chord-length distribution p(r)** (Lu & Torquato 1992; Torquato & Lu 1993) give connected Si run lengths along x and z. A long-tailed p(r) indicates streaks or agglomerates.
- Implementation: `porespy.metrics` (two-point correlation, chord-length, lineal path), or an FFT autocorrelation of the indicator.

## 5. Integral geometry: Minkowski functionals and tensors

- **Minkowski functionals** [V]: by Hadwiger's theorem, a small set of additive, motion-invariant measures fully characterises morphology. In 2D these are **area, perimeter and Euler characteristic χ** (number of Si components minus number of holes).
- **Dilation (parallel-body) profile** [B: Mecke; Schröder-Turk]: dilate the Si mask by radius ε and record A(ε), P(ε) and χ(ε). The ε at which χ collapses is the scale at which neighbouring Si objects merge. It is a field-based equivalent of the MST cut distance, and the χ(ε) curve is a candidate headline KPI curve.
- **Minkowski tensors** [V: Schröder-Turk et al., J. Microsc. 2010; NJP 2013; Adv. Mater. 2011]: tensor-valued generalisations that measure **anisotropy and orientation**. Their eigenvalue ratio β gives the degree of Si-streak alignment. Applying them to the graphite mask also gives the flake-orientation (calendering) KPI. Tools: `karambola`, `papaya2`, `quantimpy`.

## 6. Topology: persistent homology

- [V: Hiraoka et al. PNAS 2016; Obayashi, Nakamura & Hiraoka JPSJ 2022 (HomCloud); Pritchard et al. Sci. Rep. 2023]
- Filter the Si mask with the **signed Euclidean distance transform**:
  - **H0** pairs (birth, death) record when separate Si components merge as they grow. Long-lived H0 features are well-separated particles (good dispersion). Many short-lived H0 features are crowded clusters.
  - **H1** features are loops: Si rings enclosing graphite or pores. These would show Si forming shells or networks around flakes.
- Pritchard et al. used exactly this filtration on binary microscopy patches to quantify "the number, size, distribution, and crowding" of features. Their statistics map to interpretable quantities (feature radius, inter-feature distance).
- Vectorise the diagrams (persistence images or landscapes, or summary statistics) to obtain a fixed-length KPI vector. This vector can also feed the anomaly-detector backstop. Tools: `HomCloud`, `gudhi`, `ripser`/`cubical ripser`.

## 7. Mixing and heterogeneity indices (the "localisation" criterion)

- **Danckwerts (1952)** [V via secondary]: mixedness needs two numbers, the **intensity of segregation** (concentration variance across samples, normalised) and the **scale of segregation** (integral of the concentration correlogram). Both depend on the **scale of scrutiny** (window size).
- **Lacey index** [V via secondary; Lacey 1954]: M = (σ0² − σ²)/(σ0² − σR²). It compares the observed variance of local Si fraction with the fully segregated case (σ0²) and the random case (σR² = c(1 − c)/N).
- **Variance against window size (the scale-of-segregation curve)**: tile the image with windows of side w, compute the CV of the Si area fraction, and plot it against w. For a random mixture the variance falls as roughly w⁻². A plateau, or a slower decay, at some w gives the size of the segregated regions. This is the proposed headline figure, and it relates directly to representative-elementary-area analysis.
- **Battery precedent** [V]: Müller et al. (J. Electrochem. Soc. 2018, 165, A339) quantified inhomogeneity in commercial graphite anodes. They evaluated parameter distributions across 243 sub-volumes per electrode and split the results into in-plane and through-plane directions. That approach applies directly to 2D cross-sections.
- **Localisation profiles**: Si area fraction in depth bands from the Cu foil to the surface, and lateral CV along the width. Compare each band against the baseline envelope. A slope fitted through thickness gives a single "migration" KPI, since binder and fines migration during drying can cause through-thickness gradients.

---

## 8. Recommended KPI shortlist

| Criterion | KPI | Family | Output |
|---|---|---|---|
| No clustering | P95 / max Si object size; agglomerate area fraction (> d*) | object | scalar |
| No clustering | Voronoi area CV σ_V (vs 0.53 Poisson and vs baseline) | point pattern | scalar |
| No clustering | g(r) or L(r) − r under **random labelling within non-graphite space**, along x and z | point pattern | curve plus max-deviation scalar |
| No clustering | MST (m, σ) and cluster count at cut distance d | graph | scalar pair |
| No clustering | Two-point cluster function C2(r) correlation length | field | curve plus scalar |
| No clustering | Euler characteristic χ(ε) merging radius / persistent-homology H0 lifetime stats | topology | curve plus scalar |
| No localisation | Through-thickness Si band profile and slope | heterogeneity | curve plus scalar |
| No localisation | Lateral CV of Si fraction | heterogeneity | scalar |
| No localisation | CV of local Si fraction against window size (scale of segregation), Lacey M at a fixed scrutiny scale | mixing | **headline curve** |
| No localisation | Voronoi-based void (Si-depleted) area fraction | point pattern | scalar |
| Context | Graphite flake orientation (Minkowski-tensor β) | integral geometry | scalar |

Decision rule: express every KPI as a deviation from the baseline-batch distribution, for example a z-score or a position relative to the baseline 5–95 % envelope. This keeps the verdict free of assumptions about absolute thresholds that the provider has not supplied.

## 9. Practical pitfalls
1. **Edge effects**: all NN, K and Voronoi statistics need a border correction or guard zone, which matters most in a thin coating where the foil and surface are hard boundaries in z.
2. **Anisotropy**: do not average x and z. Graphite alignment makes isotropic statistics misleading.
3. **Resolution floor**: Si smaller than about 2–3 pixels drops out of the object layer. Report the pixel size and treat object metrics as conditional on it.
4. **Segmentation sensitivity**: perturb the Si classifier's probability threshold by ±δ and check that KPI rankings between batches are stable before trusting differences.
5. **2D vs 3D**: one cross-section is a single plane. Use several sections or fields per sample and report the between-field variance so a reader can see whether batch differences exceed it.

## References
[V] = retrieved this session; [B] = background, verify before citing.

- [V] Clark, P. J. & Evans, F. C. (1954). Distance to nearest neighbor as a measure of spatial relationships in populations. *Ecology* (cited via secondary sources).
- [V] Ripley, B. D. (1976, 1977) K-function papers (cited via secondary sources); Diggle (2003) D-function (via arXiv:0808.1409).
- [V] Duyckaerts, C. & Godefroy, G. (2000). Voronoi tessellation to study the numerical density and the spatial distribution of neurones. *J. Chem. Neuroanat.* (ScienceDirect S0891061800000648).
- [V] Monchaux et al. (Voronoi preferential concentration), via Obligado/Mora et al. arXiv:1907.07607 and arXiv:1906.09896 (σ_RPP ≈ 0.53 in 2D).
- [V] Cryo-TEM silica dispersion with Voronoi CV, ACS (2026), PMC13087927.
- [V] Dussert, C., Rasigni, G., Rasigni, M., Palmari, J. & Llebaria, A. (1986). Minimal spanning tree: a new approach for studying order and disorder. *Phys. Rev. B* 34, 3528.
- [V] Torquato, S. (2002). *Random Heterogeneous Materials*. Springer.
- [V] Jiao, Y., Stillinger, F. H. & Torquato, S. (2007, 2008). Modeling heterogeneous materials via two-point correlation functions I, II. *Phys. Rev. E* 76, 031110; 77, 031135.
- [V] Jiao, Stillinger & Torquato (2009). A superior descriptor of random textures and its predictive capacity. *PNAS*, doi:10.1073/pnas.0905919106.
- [V] Lu, B. & Torquato, S. (1992). Lineal-path function. *Phys. Rev. A* 45, 922; Torquato & Lu (1993) Chord-length distribution. *Phys. Rev. E* 47, 2950.
- [V] Schröder-Turk, G. E. et al. (2010). Tensorial Minkowski functionals and anisotropy measures for planar patterns. *J. Microsc.* 238, 57–74.
- [V] Schröder-Turk et al. (2011). Minkowski tensor shape analysis of cellular, granular and porous structures. *Adv. Mater.*; (2013) Minkowski tensors of anisotropic spatial structure. *New J. Phys.* 15, 083028.
- [V] Armstrong et al. (2018/19). Porous media characterization using Minkowski functionals (review); McClure et al. arXiv:1805.11032.
- [V] Hiraoka, Y. et al. (2016). Hierarchical structures of amorphous solids characterized by persistent homology. *PNAS* 113, 7035, doi:10.1073/pnas.1520877113.
- [V] Obayashi, I., Nakamura, T. & Hiraoka, Y. (2022). Persistent homology analysis for materials research and persistent homology software: HomCloud. *J. Phys. Soc. Jpn.* 91, 091013.
- [V] Pritchard, Y. et al. (2023). Persistent homology analysis distinguishes pathological bone microstructure in non-linear microscopy images. *Sci. Rep.*, doi:10.1038/s41598-023-28985-3.
- [V] Danckwerts (1952) scale and intensity of segregation; Lacey (1954) mixing index; Kukukova et al. (2011) — via secondary sources.
- [V] Müller, S., Eller, J., Ebner, M., Burns, C., Dahn, J. & Wood, V. (2018). Quantifying inhomogeneity of lithium ion battery electrodes and its influence on electrochemical performance. *J. Electrochem. Soc.* 165, A339.
- [V] Vorauer, T. et al. (2020). Multi-scale quantification and modeling of aged nanostructured silicon-based composite anodes. *Commun. Chem.*, s42004-020-00386-x.
- [B] Matheron, G. (1975). *Random Sets and Integral Geometry*; Serra, J. (1982). *Image Analysis and Mathematical Morphology*.
- [B] Wicksell, S. D. (1925). The corpuscle problem. *Biometrika*.
- [B] Baddeley, A., Rubak, E. & Turner, R. (2015). *Spatial Point Patterns: Methodology and Applications with R* (spatstat).
