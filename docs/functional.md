# Functional morphology: pore access and the Si swelling stress test (`pmdb.functional`, `f1`)

The KPI families (`docs/kpis/`), the batch fingerprint (`docs/fingerprint.md`) and the GP tile maps
all describe **where the Si sits**. This module asks the next question a cell engineer would put to
the same masks: **does that arrangement still work as an electrode, and what does lithiation do to
it?** It is deliberately simple, runs on CPU in 13 s for all 31 sites, and uses the same `v0r1` masks
as the KPIs, so it inherits their affine (offset/gain) invariance and needs no harmonised input.

Reproduce: `python scripts/run_functional.py --jobs 8` (sites + 4 x-tiles), `--heldout`,
`--perturb ±0.05` (segmenter robustness), then `python scripts/analyse_functional.py`. Outputs in
`outputs/functional/` (`site_functional.csv`, `tile_functional.csv`, `batch_comparison.csv`,
`redundancy.csv`, `robustness.csv`, `power.csv`, `loso.json`, `heldout_predictions.csv`, `figures/`).
Tests: `tests/test_functional.py` (known answers on synthetic masks, no data needed).

## 1. What is computed

| Block | Column(s) | Meaning | Literature anchor |
|---|---|---|---|
| F01 pore access | `F01_pore_spans_z/x` | does the pore phase percolate across the field (4-connectivity, the graph a diffusion solve uses)? | TauFactor construction, Cooper et al. 2016 (SoftwareX, 10.1016/j.softx.2016.09.002) |
| | `tortuosity_factor()` | steady-state diffusion solve on the pore pixels; `D_eff/D_0 = eps/tau`. Implemented and tested, but **not reported as a KPI** (see §2.1) | Cooper et al. 2016; Tjaden et al. 2016 (10.1016/j.coche.2016.02.006) |
| | `F01_pore_chord_{x,z}_p50/p90_um`, `_anisotropy` | pore chord lengths in-plane vs through-thickness | chord-length / lineal-path functions, Lu & Torquato 1992 (10.1103/physreva.45.922) |
| | `F01_pore_clusters_per_1000um2`, `F01_pore_largest_frac` | fragmentation of the pore phase | — |
| | `F01_si_pore_dist_p50/p90_um`, `F01_si_pore_access_frac` | distance from each Si pixel to the nearest pore (electrolyte access length); share of Si within 1 µm of a pore | — (mirror of K16, Si-to-graphite) |
| | `F01_si_isolated_frac` | Si not connected to the main solid body (pore-islanded) | — |
| F02 swelling stress test | `F02_soc{025,050,100}_*` | grow every Si object isotropically to its lithiated area, `A(soc) = A_0 (1 + 2.8·soc)^(2/3)`; record where the growth lands: `into_graphite` (constrained, stress proxy), `into_binder`, `into_pore`; `pore_loss` (share of pore consumed), `si_frac` and `pore_frac` after, `si_objects_ratio` (objects after / before = particle merging) | ~280 % volume expansion of Si to Li15Si4: Obrovac & Christensen 2004 (10.1149/1.1652421); Beaulieu et al. 2001 (10.1149/1.1388178) |

The swelling model moves no material (it is a morphological stress test, not mechanics) and a
single section under-reports connectivity (Cooper et al. 2014, 10.1016/j.jpowsour.2013.04.156), so
all numbers are for comparing fields and batches under one rule. The FEM review
(`docs/fem/literature-review.md`) is where the mechanics would go; F02 tells it *which* fields
have the most Si pressing on graphite.

## 2. What the 31 labelled fields show

### 2.1 Network transport is not measurable from one section (negative result, recorded)
Pore fraction in the v0 masks is 3–5 % of the field. On **0 / 31 sites** does the pore phase span the
field in either direction, and the same holds for the whole non-graphite space (Si + pore + binder,
10–25 %). This is expected — 2D site percolation needs ≈ 59 % — so a tortuosity factor from one
cross-section would be infinite everywhere and is not reported. The electrode is a 3D network seen
in a 2D slice; what *is* measurable in the slice is access distance and local geometry.

### 2.2 Pore access does not differ between batches
Median Si-to-pore distance is 1.35 / 1.35 / 1.41 µm (B1 / B2 / B3; Kruskal–Wallis p = 0.20), 28–32 % of
Si pixels lie within 1 µm of a pore, pore chords are ≈ 0.3–0.35 µm (p50) with no x/z anisotropy
(ratio 1.00 in every batch), and no Si is islanded in pore on any site. Pore cluster count is the
D06 Euler number restated (Spearman 0.98) and pore chords track D05 pore size (0.67–0.89): these
columns are kept for completeness, they are not new information.

### 2.3 The swelling budget is the new result

> **Caveat added after the segmenter-agreement test (`docs/reliability.md` §3).** The batch difference in
> the constrained share (B1 < B2 < B3) exists under `pmdb.segment` v0r1 but not under the independently
> written anchored segmenter of the analytical-benchmarks session (KW p 0.015 → 0.73; Spearman between
> segmenters only 0.38). The effect lives in the graphite-vs-binder allocation of the growth, a BSE
> grey-level split the alternative segmenter does not make; the pore-avoidance share and pore loss are
> segmenter-independent (Spearman 0.84 / 0.97). The v0r1 "binder" class is ≈ 1 % of the field and has no
> intensity identity in BSE, Inlens or SE (`docs/reliability.md` §3), so it cannot be validated with this
> data: **do not report "Batch 3 is the most constrained" as a material result.** The remaining content of
> this section (where growth lands, pore loss, held-out comparison) stands.
At every SOC, **most of the Si growth collides with graphite**: batch medians 0.77 / 0.81 / 0.83 at
SOC 0.25 and 0.86 / 0.87 / 0.88 at SOC 1 (`figures/swelling_budget.png`). Only 5–6 % of the growth
finds pore; the rest of the free space is binder / carbon black.

* `F02_soc050_into_graphite` separates the batches: Kruskal–Wallis **p = 0.009**, Cliff's δ = −0.63
  (B1 vs B2) and −0.55 (B1+B2 vs B3); at SOC 1 p = 0.015. Batch 3 (the supplier baseline) is the
  most constrained, Batch 1 the least. This is the lithiation reading of K15 (Si–graphite contact,
  Spearman 0.79–0.89; K15 alone: p = 0.07): more contact means more of the swelling is taken up by
  the stiff phase. The ordering is unchanged when the segmenter's Si and pore anchors are moved by
  ±0.05 (site-ranking Spearman 0.96–0.98, `robustness.csv`).
  Sensitivity to the two high-Si Batch 1 fields: without `4ih2ggld` and `5n1q8atc` the B1-vs-B2
  difference vanishes (Mann–Whitney p = 0.20–0.34) but B1+B2 vs B3 holds (p = 0.03 at both SOCs;
  Kruskal–Wallis p = 0.05–0.08). The robust statement is "Batch 3 is the most constrained", not a
  three-way ordering.
* `pore_loss` at SOC 1 is 13–25 % and is almost entirely a function of Si fraction (Spearman 0.79
  with K01, partial ρ ≈ 0.1 once K01 is controlled; `figures/pore_loss_vs_si_fraction.png`). The
  two high-Si Batch 1 fields (`4ih2ggld`, `5n1q8atc`, K01 = 0.14–0.16) would lose 22–25 % of their
  pore; every other field 7–20 %. Pore closure risk here is a composition story, not an arrangement
  story.
* `si_objects_ratio` falls to 0.80–0.82 at SOC 1 in all batches: about one Si object in five merges
  into a neighbour on full lithiation. No batch difference (p = 0.87), and this column is the least
  robust to the segmenter (Spearman 0.69–0.76), so treat it as indicative.

### 2.4 Interpretation, not discrimination
Used as a fingerprint on their own, seven non-redundant F columns reach leave-one-site-out
accuracy 0.45 (majority baseline 0.55, permutation p = 0.18): they do not identify batches. Added
to the 16 fingerprint features the accuracy is 0.71 vs 0.68 alone (one site; same permutation
p = 0.003), i.e. no measurable gain. On the held-out sites the F-only model gives 3e122cbj → Batch 1
(agreeing with the fingerprint), fn0mhxef → Batch 2 and xrv9xvzb → Batch 3 (disagreeing, with
confidence 0.0); these are not to be used for assignment. The value of F02 is that it states a
*consequence* of the arrangement the fingerprint already detects: Batch 3 Si is more boxed in by
graphite, so if anything in these batches cracks first under swelling, the geometry says Batch 3.

### 2.5 Held-out sites against the Batch 3 baseline (`heldout_explain.csv`, `figures/heldout_swelling_cards.png`)
Robust z (MAD units) of each held-out field against the 17 Batch 3 fields, from
`python scripts/explain_heldout_functional.py` (read-only use of the held-out data):

| site | constrained share (SOC 1) | pore loss | Si objects after/before | H0 lifetime p50 | reading |
|---|---|---|---|---|---|
| `3e122cbj` | 0.80, **z = −3.3** (below every B3 field) | 0.22, **z = +2.4** (above every B3 field) | 0.64, **z = −6.8** | 0.40 µm, z = −2.6 | K01 = 0.137, 107 objects / 1000 µm²: twice the Si of any Batch 3 field, crowded (Euler merge radius 0.07 µm, H0 lifetimes a third of B3), one object in three merges on lithiation and the growth spills past the graphite into binder and pore. Functionally it is the twin of the two high-Si Batch 1 fields (`4ih2ggld` 0.78 / 0.22 / 0.66; `5n1q8atc` 0.83 / 0.25 / 0.68); its z against Batch 1 is −2.9 / +0.6 / −2.7, i.e. still extreme but on Batch 1's side. Supports the fingerprint's Batch 1 call and explains *why* it is not Batch 3: too much Si, too close together, swelling into free space rather than against graphite. |
| `fn0mhxef` | 0.86, z = −0.8 | 0.12, z = −0.2 | 0.81, z = −0.8 | 1.42 µm, z = +1.0 | inside the Batch 3 range on every F and S column (|z| ≤ 1.8; the only mild oddity is the shortest median Si-to-pore distance of any site, 1.20 µm, z = −1.8). Functionally indistinguishable from the baseline, consistent with the fingerprint's Batch 3 assignment. |
| `xrv9xvzb` | 0.87, z = −0.7 | 0.10, z = −0.7 | 0.81, z = −0.4 | 1.24 µm, z = +0.2 | inside the Batch 3 range on every column (|z| ≤ 0.7). The functional layer sees nothing that the fingerprint's Batch 2 call (mid-depth Si dip) could be checked against — depth distribution is not a functional quantity here. |

### 2.6 Is "Si against graphite" arrangement or composition? (`null_relocation.csv`, `null_relocation_summary.json`)
Null model in the spirit of D-013: every 8-connected Si object is moved, shape intact, to a random
non-overlapping position inside the admissible space (anything that is not graphite or artefact),
so the null keeps the objects, the Si fraction and the graphite skeleton and randomises only the
arrangement (`pmdb.functional.relocate_si`, 20 draws per field, seeded per site;
`python scripts/run_swelling_null.py`).

| | Batch 1 | Batch 2 | Batch 3 |
|---|---|---|---|
| K15 contact: observed / null median | 0.70 / 0.45 | 0.71 / 0.48 | 0.73 / 0.48 |
| constrained share at SOC 1: observed / null | 0.86 / 0.81 | 0.87 / 0.82 | 0.88 / 0.83 |
| pore loss at SOC 1: observed / null | 0.17 / 0.29 | 0.13 / 0.24 | 0.13 / 0.27 |

- Every one of the 31 fields sits far outside its own null: contact excess +0.16 to +0.36
  (z ≥ 14), constrained share +0.02 to +0.12 (|z| ≥ 6), pore loss *below* null in all 31
  (z ≤ −8). The Si really is wetted onto the graphite surfaces and keeps out of the pore
  pockets; a random arrangement of the same objects would close about twice as much pore.
  This is an arrangement fact, not a consequence of how much Si there is.
- The **excess over null does not differ by batch** (Kruskal–Wallis p = 0.56 for constrained
  share, 0.67 for contact, 0.33 for pore loss; B1+B2 vs B3 p = 0.37). The adhesion effect is
  the same size in every batch. The batch ordering of the constrained share seen in §2.3 is
  therefore not "Batch 3 arranges its Si against graphite more than chance"; it is a modest
  shift of both observed and null together (null means 0.81 / 0.82 / 0.83, p = 0.21), i.e. it
  comes from the geometry the null preserves — object size, Si fraction and how finely the
  graphite is divided — not from an extra placement preference.
- Caveat: 11–43 % of objects (median ≈ 25 %, more where objects are large, Spearman 0.78 with
  K03) find no admissible pocket of their size in 400 draws and stay where they are, so the
  null is conservative (closer to the observation than a perfect relocation would be). All
  z-values above are therefore lower bounds on the effect.

### 2.7 How many fields would it take? (`power.csv`)
Plug-in bootstrap of a Mann–Whitney test at the observed effect sizes: with the present 7 / 7 / 17
fields the power to detect even the largest batch effects is 0.3–0.7 (`K06_void_region_frac`
B1 vs B2: 0.60; `F02_soc050_into_graphite`: 0.51 / 0.38). Reaching 80 % power needs ≈ 10–15 fields
per batch for the strongest KPIs and > 50 for most, and 49 of 62 B1+B2-vs-B3 contrasts are not
reachable at 50. This puts a number on the v2 modelling conclusion "more fields, not more models".
The estimate is optimistic (it assumes the observed separation is real).

### 2.8 QC sheets: swelling cards for all 31 labelled fields (`figures/swelling_cards_Batch_*.png`)

`scripts/swelling_cards.py` renders one card per field (BSE crop + SOC-1 growth map: yellow Si, red
growth landing on graphite, blue into pore, green into binder/CB), sorted by constrained share within
each batch, with the three F02 headline numbers in the title. Use them to check any F02 value against
the picture before it is quoted: the two high-Si Batch 1 fields (`4ih2ggld`, `5n1q8atc`) are
immediately recognisable as the pore-closure cases, and a field whose growth lands mostly on green is a
segmentation question, not a lithiation result. The joint pre-specified acceptance score built on this
family is in `docs/acceptance.md` §4.

## 3. Limits
* 2D proxies of 3D quantities; binder / carbon black are one "unassigned solid" class in the v0 masks.
* Isotropic, redistribution-free swelling with a single 280 % volume figure; real Si expands
  anisotropically and the electrode thickens. Only the *ranking* between fields is used.
* n = 7 per batch for Batches 1 and 2 (§2.5); the Batch 1 result leans on two high-Si fields that the
  classifier work (`outputs/v2/RESULTS.md`) already flags as possible outliers.
