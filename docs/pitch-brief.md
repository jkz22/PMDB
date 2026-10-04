# Pitch brief: Si/graphite anode batch QC from SEM cross-sections

Deck source for the 2026-10-04 pitch (decision D19, `.claude/plans/fem-swelling.md`). Every number below is copied from the file cited next to it. Paths are relative to the repository root. The "Numbers checklist" at the end lists each number once, for fact-checking.

## Headline

1. We turned all 34 SEM sites into 68 2D lithiation simulations on Modal (0 failures, $2.608 total), and the median simulated electrode swelling, 0.127, sits inside the literature band [0.09, 0.39] (`docs/fem/results.md`).
2. A two-stage random forest against the Batch_3 supplier baseline assigns the held-out sites: 3e122cbj to Batch_1 (0.855), fn0mhxef to Batch_3 (0.562), xrv9xvzb to Batch_3 (0.586), and the arms disagree on xrv9xvzb (`docs/classifier/results.md`).
3. With 31 labelled sites, FEM features did not beat image KPIs end to end (balanced accuracy 0.370 vs 0.437), so the rule fixed in advance picked the KPI arm (`docs/classifier/results.md`).

---

## Slide 1. The problem: is this batch what the supplier promised?

**Key message:** A Si/graphite anode is accepted or rejected by batch, and the only stated quality rule is about where the silicon sits.

- The electrode is a graphite anode with silicon as the secondary active material. The provider's only stated quality criterion: "silicon must not cluster and must not be localised" (`docs/kpis/README.md`).
- Batch_3 is the baseline, "what's been 'promised' by the supplier". Batch_1 and Batch_2 arrived later. They are not better or worse, they show the variation a model must detect (`data_heldout/README.md`).
- Deliverable for each held-back site: always a batch, a confidence, and how it differs from Batch_3. Abstaining is not allowed (`data_heldout/README.md`, `AGENTS.md`).
- Why it matters (organiser): a model that can do this can flag an unknown batch N as in or out of distribution, which supports accept/reject decisions (`data_heldout/README.md`).

**Figure:** none required. Optional: one raw BSE frame, e.g. frame 1 of `outputs/fem/figures/example_frames.png`.

**Speaker note:** Silicon holds about 3400 mAh/g against about 340 mAh/g for graphite in the reference half cell (`docs/fem/literature-review.md`, C2), and it swells far more. The supplier promised Batch_3. Our job is to say whether a new sample matches that promise and, if not, how it differs.

---

## Slide 2. The data: 31 labelled sites, 3 held out

**Key message:** We have 31 labelled SEM cross-sections in three unequal batches and 3 unlabelled sites to classify.

- Labelled sites: Batch_1 7, Batch_2 7, Batch_3 17 (row counts of `outputs/fem/validation.csv`; the 17 Batch_3 and 14 Batch_1/Batch_2 sites are also named in `docs/classifier/method.md`).
- Held out: `3e122cbj`, `fn0mhxef`, `xrv9xvzb`, unlabelled, never trained on (`AGENTS.md`, `data_heldout/README.md`).
- Three aligned detectors per site: BSE (composition), Inlens (topography), and ETD or SE. 27 sites use ETD, 4 use SE (`AGENTS.md`). All 3 held-out sites are BSE, Inlens, ETD (`data_heldout/README.md`).
- Scale: 25 nm/px raw, 50 nm/px in the half-resolution cache we use (`AGENTS.md`). Known confound: an imaging offset on some Batch_3 sites (black level ≈ +19–23, gain ≈ 0.69× on 4 sites), so grey levels alone could leak the batch (`AGENTS.md`).

**Figure:** `outputs/fem/figures/example_frames.png` (left column: BSE at SOC 0 for one site per batch).

**Speaker note:** 31 sites is small. Every metric later carries a bootstrap interval, and the site, not the tile, is the unit of evidence. We kept the held-out sites out of every training fold.

---

## Slide 3. Approach: two feature paths, one classifier

**Key message:** Segment each site once, then describe it two ways, as image KPIs and as a simulated charge, and classify against Batch_3 in two stages.

- Path A, image KPIs: segmentation, then 15 KPI features per tile (Si loading, particle size, clustering, Si-free pockets, Si-graphite contact) (`docs/classifier/method.md`, `docs/classifier/results.md`).
- Path B, physics: segmentation, then a 2D finite-element lithiation from 0% to 100% SOC in 11 frames, then 16 FEM features per tile (`.claude/plans/fem-swelling.md` D3; `docs/classifier/method.md`).
- Tiles: 6 tiles of 22 µm per site, full height, on the same grid for both paths (`docs/classifier/method.md`).
- Classifier: random forest, fixed hyperparameters, never tuned. Stage 1: Batch_3 vs not. Stage 2: Batch_1 vs Batch_2. Site probability = mean of tile probabilities; confidence = product along the chosen branch (`docs/classifier/method.md`). Three arms: KPI (15 features), FEM (16), KPI+FEM (31) (`docs/classifier/results.md`).

**Figure:** redraw the mermaid flowchart in `docs/classifier/method.md` section 2 (FEM side: `docs/fem/method.md` section 1).

**Speaker note:** The two-stage design mirrors the question. First, is this the supplier baseline? Only if not, which of the later batches does it resemble? The explanation for each site comes from stage-1 importance times the site's z-score against Batch_3.

---

## Slide 4. Physics in one slide: Si growth meets pore space

**Key message:** Silicon roughly triples its volume on charge while graphite swells about 10% through the thickness, so the outcome depends on how much pore space sits next to the silicon.

- Si: volume ratio J_Si = 3.240 at 100% SOC in our model (`docs/fem/method.md`, SOC table). Literature full-lithiation expansion: 263% (Qi), 277% (Obrovac), 280% (Beaulieu); we use β = 2.8 with utilisation u_max = 0.8, because full cells do not reach Li15Si4 (`docs/fem/literature-review.md`).
- Graphite: c-axis (through-thickness) strain 0.094 at 100% SOC in our model (`docs/fem/method.md`, SOC table); literature end point about 10.3% at LiC6 (`docs/fem/literature-review.md`, C12). In-plane strain is about 1% (`docs/fem/literature-review.md`, C14).
- Charge order (Yao 2019): over the first 25% of SOC, Si takes 96% of the charge; above that, Si 58% and graphite 42% (`docs/fem/literature-review.md`).
- Why simulate: in the image-based model of Shah et al. 2022, Si expansion is absorbed by "(i) reduction in porosity, (ii) compaction of CBD, and (iii) expansion into the separator" (`docs/fem/literature-review.md`, C17). A KPI counts Si. A simulation asks whether its neighbourhood can absorb it.

**Figure:** `outputs/fem/figures/swelling_vs_soc.png` (site swelling vs SOC, all 34 sites, with the stop window and literature band).

**Speaker note:** All parameters come from a literature review with evidence grades, listed in `docs/fem/literature-review.md`. Where nothing was published, for example pore stiffness, we say so: it is a numerical choice.

---

## Slide 5. Engineering: 68 simulations, $2.608

**Key message:** The full simulation campaign ran on Modal CPUs for a few dollars, with every case converged.

- 34 sites × 2 collector orientations = 68 cases (`outputs/fem/run_log.json`, `cases`; `docs/fem/results.md`, Convergence). Two orientations because the foil side is not visible; features use their mean (`sym`) (`docs/fem/method.md` section 4).
- Each case meshes one cell per pixel at 100 nm: 1011513 cells for Batch_1/4ih2ggld (`outputs/fem/run_log.json`). FEniCSx/dolfinx 0.10.0 in a pinned Docker image, 4 CPUs, 10240 MB (`outputs/fem/run_log.json`).
- Wall time per case: median 140 s, max 191 s (bottom); median 130 s, max 175 s (top). 0 of 34 sites failed before 100% SOC in either orientation (`docs/fem/results.md`).
- Cost: ledger total $2.608 of a $180 cap; production runs $1.31; per case median $0.019, max $0.023 (`docs/fem/results.md`). Checks: 49 unit tests passed; 100 vs 50 nm resolution check passed, max relative swelling difference 0.00643 (`docs/fem/results.md`).

**Figure:** GIF wall, one per batch, chosen as the site with the highest `swelling_sym` in its batch (`outputs/fem/validation.csv`):
- Batch_1: `outputs/fem/gifs/Batch_1__5n1q8atc.gif` (swelling_sym 0.183591)
- Batch_2: `outputs/fem/gifs/Batch_2__b3esycq1.gif` (swelling_sym 0.137581)
- Batch_3: `outputs/fem/gifs/Batch_3__x7u69zsw.gif` (swelling_sym 0.139353)
- Held out: `outputs/fem/gifs/Batch_heldout__3e122cbj.gif` (swelling_sym 0.168261)

**Speaker note:** Each GIF is 11 frames, 0 to 100% SOC: the BSE image warped by the computed displacement with von Mises stress on top. The colour scale is fixed across sites. The GIFs show the bottom orientation.

---

## Slide 6. The honest model story: finite strain did not converge

**Key message:** The literature calls for finite-strain mechanics, but on real microstructures it failed within the first few percent of charge, so production uses linear small-strain elasticity with a logarithmic eigenstrain (D20).

- Finite-strain neo-Hookean failed at s of about 0.02 to 0.08 for every remediation rung. Si growth crushed thin pores to J of order 1e-3. A pore compaction barrier made it worse (failed at s = 0.023 and 0.016) (`docs/fem/results.md`, Limitations; `.claude/plans/fem-swelling.md` D20).
- Contact mechanics, the proper fix, was out of hackathon scope (`.claude/plans/fem-swelling.md` D20). The finite-strain code stays in the repo as a tested reference, not used in production (`docs/fem/results.md`).
- Cost of the switch: absolute stresses are not physical. Si von Mises saturates at about 20 to 26 GPa and `si_yield_frac` is 1.0 everywhere (`docs/fem/results.md`). Linear J can drop to or below 0 in pores ("over-closure"), caught by the pore-closure flag (`docs/fem/results.md`).
- What survives: swelling, porosity change and geometry-driven features, which are less affected by the stress saturation (`docs/fem/results.md`).

**Figure:** none, or the three-line remediation log from `docs/fem/results.md`, Flags.

**Speaker note:** We tried the textbook model first and it broke on real images, for a physical reason: pores close and the model has no contact. We chose a model that runs on every site and treat its stresses as relative descriptors, not predictions.

---

## Slide 7. Swelling validation against the literature

**Key message:** Simulated electrode swelling at full charge sits inside the literature band for every site, at its low end, as expected for a 2D elastic model.

- Median `swelling_sym` at s = 1 over 34 sites: 0.127. Stop window [0.03, 0.39], G4 pass. Literature band [0.09, 0.39], `lit_band_ok = True` (`docs/fem/results.md`). Sites outside the stop window: none (`docs/fem/results.md`).
- Median by batch: Batch_1 0.132, Batch_2 0.125, Batch_3 0.125, Batch_heldout 0.127 (`docs/fem/results.md`).
- Literature reference points: graphite-only 9% (Michael), 19% (Prado), 33% (Kirner); 15 wt% Si 22% at a 100 mV cut-off and 39% at 10 mV (`docs/fem/literature-review.md`, VT1).
- Low end expected: no particle rearrangement, binder creep or SEI growth in the model (`docs/fem/method.md` section 9). Batch_1's two highest sites (0.183591, 0.179983) and held-out 3e122cbj (0.168261) stand out above the rest (`outputs/fem/validation.csv`).

**Figure:** `outputs/fem/figures/swelling_by_batch.png` (gold: literature band).

**Speaker note:** This is a sanity check, not a calibration. Batch medians are close, so swelling alone does not separate batches. The outliers in Batch_1 and the held-out site are worth a look on the next slide pair.

---

## Slide 8. Ablation: did physics help the classifier?

**Key message:** FEM features scored highest on the Batch_3-vs-rest stage but not end to end, Batch_1 vs Batch_2 was at or below chance for every arm, and the rule fixed before the FEM results picked the KPI arm.

| arm | level | n_sites | balanced_acc | 95% CI | brier |
|---|---|---|---|---|---|
| KPI | stage1 | 31 | 0.548 | 0.418–0.702 | 0.233 |
| FEM | stage1 | 31 | 0.626 | 0.468–0.791 | 0.248 |
| KPI+FEM | stage1 | 31 | 0.561 | 0.407–0.717 | 0.240 |
| KPI | stage2 | 14 | 0.357 | 0.125–0.625 | 0.277 |
| FEM | stage2 | 14 | 0.357 | 0.125–0.637 | 0.341 |
| KPI+FEM | stage2 | 14 | 0.286 | 0.056–0.525 | 0.322 |
| KPI | end_to_end | 31 | 0.437 | 0.292–0.630 | 0.191 |
| FEM | end_to_end | 31 | 0.370 | 0.250–0.513 | 0.208 |
| KPI+FEM | end_to_end | 31 | 0.350 | 0.224–0.497 | 0.201 |

Source: `docs/classifier/results.md` (full precision in `outputs/classifier/metrics.csv`).

- Selection rule, fixed in advance: highest end-to-end balanced accuracy; arms within 0.05 are tied; ties go to the lowest Brier. Applied: KPI (`docs/classifier/results.md`).
- Stage 2 (Batch_1 vs Batch_2) balanced accuracy is 0.286 to 0.357 across arms, below the 0.5 of a coin flip for two classes. End-to-end KPI confusion: 5 of 7 Batch_1 and 6 of 7 Batch_2 sites are called Batch_3; 15 of 17 Batch_3 sites are correct (`docs/classifier/results.md`).
- CIs are wide: with 31 sites, differences below ~0.1 balanced accuracy are within the bootstrap CI (`docs/classifier/results.md`). The FEM stage-1 lead is inside that noise.
- External reference (flat logistic, different task structure): 0.493, CI 0.316–0.691 (`outputs/pooling_checks/check3_results.md`, R1 multiclass).

**Figures:** `outputs/classifier/fig_confusion.png`, `outputs/classifier/fig_importance.png` (top stage-1 feature: `K15_si_graphite_contact_frac`, importance 0.117, `docs/classifier/results.md`).

**Speaker note:** We wrote the selection rule before we saw the FEM results and we kept it. Physics carried some signal on the baseline question. It did not carry it through to the three-way answer, and nothing we tried separates Batch_1 from Batch_2 on 14 sites.

---

## Slide 9. Held-out predictions

**Key message:** One site is clearly not Batch_3, one looks like Batch_3, and one is a genuine disagreement between the image and physics views.

| site | KPI (final) | FEM | KPI+FEM |
|---|---|---|---|
| 3e122cbj | Batch_1 (0.85) | Batch_1 (0.82) | Batch_1 (0.84) |
| fn0mhxef | Batch_3 (0.56) | Batch_3 (0.61) | Batch_3 (0.61) |
| xrv9xvzb | Batch_3 (0.59) | Batch_2 (0.45) | Batch_2 (0.40) |

Source: `docs/classifier/results.md`, "All arms". Final-arm confidences at three decimals: 0.855, 0.562, 0.586 (`docs/classifier/results.md`).

How each site differs from Batch_3 (final arm, `outputs/classifier/heldout_predictions.csv`):

- **3e122cbj → Batch_1, confidence 0.855.** 6/6 tiles vote not-Batch_3. Much more Si than the baseline: Si objects per 1000 µm² 116 vs Batch_3 27.7 ± 10.5 (z = +8.4); Si area fraction 0.139 vs 0.0623 ± 0.00955 (z = +8.0); less Si boundary touching graphite, 0.566 vs 0.738 ± 0.0294 (z = −5.9). It also has the highest simulated swelling of the held-out sites, 0.168261 (`outputs/fem/validation.csv`).
- **fn0mhxef → Batch_3, confidence 0.562.** 1/6 tiles vote not-Batch_3. Its top deviations are small: nearest-neighbour index vs CSR 1.06 vs 0.976 ± 0.0582 (z = +1.4); Si ECD d90 3.22 vs 2.83 ± 0.403 µm (z = +1.0); normalised mean MST edge 2.28 vs 2.14 ± 0.141 (z = +1.0). Read: within the baseline's spread.
- **xrv9xvzb → Batch_3, confidence 0.586 (disagreement case).** KPI arm: 1/6 tiles vote not-Batch_3. Slightly less Si (0.0518 vs 0.0623 ± 0.00955, z = −1.1), smaller large particles (ECD d90 2.47 vs 2.83 ± 0.403, z = −0.9), larger Si-free pockets (p95 distance 7.31 vs 5.61 ± 1.61 µm, z = +1.1). The FEM arm says not-Batch_3 on 6/6 tiles (P(Batch_3) 0.272361) and picks Batch_2 (`outputs/classifier/heldout_predictions.csv`). It has the lowest simulated swelling of all 34 sites, 0.117689 (`outputs/fem/validation.csv`).

**Figures:** `outputs/fem/gifs/Batch_heldout__3e122cbj.gif`; optional `outputs/fem/gifs/Batch_heldout__xrv9xvzb.gif` for the disagreement.

**Speaker note:** We report the final arm's answer and show the others. For xrv9xvzb treat our call as low confidence: the image view says baseline, the physics view says not baseline. For 3e122cbj, a z of +8 means it is far from Batch_3. The classifier must still pick one of three batches, so a large z is our only warning that a site may come from somewhere new.

---

## Slide 10. Limitations

**Key message:** The results are honest but thin: small data, a 2D linear model and an unseen-supplier blind spot.

- 31 labelled sites: bootstrap CIs are about ±0.2 and tiles of one site are not independent evidence (`docs/classifier/method.md` section 7).
- Calibration: 28 of 31 end-to-end confidences fall in [0.5, 0.7), with accuracy 0.607; none above 0.7 (`docs/classifier/results.md`, Calibration).
- FEM: 2D plane strain, elastic only, no electrochemistry, pure Si (no SiOx run), no parameter robustness sweeps, unsourced pore stiffness (`docs/fem/method.md` section 9; D17). Stresses are relative only (D20). Every site has its first pore closure at s = 0.1, so that feature does not separate sites at site level (`outputs/fem/validation.csv`, `first_pore_closure_s`).
- The classifier always assigns one of 3 batches; it has no "new supplier" output (`docs/classifier/method.md` section 7).

**Figure:** none.

**Speaker note:** We would rather show the wide intervals than hide them. The method is built so that more sites drop straight in.

---

## Slide 11. Next steps

**Key message:** The pipeline is in place; the gains now come from better physics and more data.

- 3D microstructures, for example SliceGAN-style reconstruction from the 2D sections, to remove the plane-strain assumption (proposed; not yet in the repo docs).
- Finite strain with pore contact, the missing piece that blocked the finite-strain run (`.claude/plans/fem-swelling.md` D20).
- More labelled sites, so stage 2 (Batch_1 vs Batch_2) has a chance and CIs narrow (`docs/classifier/method.md` section 7).
- Calibrated confidences and an explicit out-of-distribution score built on the z-scores against Batch_3 (organiser goal in `data_heldout/README.md`).

**Figure:** none.

**Speaker note:** The SiOx variant and the robustness sweeps were planned and cut for time (D17). They are cheap to add back: the whole production campaign cost $1.31 (`docs/fem/results.md`).

---

## Slide 12. Close

**Key message:** From SEM images to a simulated charge to a batch call with a reason, for every site, reproducibly, for under $3 of compute.

- 34 sites simulated, 68 cases, 0 failures, $2.608 total (`docs/fem/results.md`).
- 3 held-out calls with confidence and a Batch_3 comparison: Batch_1 (0.855), Batch_3 (0.562), Batch_3 (0.586) (`docs/classifier/results.md`).
- Every number in the docs is script-generated; two classifier runs are byte-identical (`.claude/plans/fem-swelling.md` D19; `docs/classifier/method.md` section 6).

**Figure:** `outputs/fem/gifs/Batch_heldout__3e122cbj.gif` looping.

**Speaker note:** Close on the held-out GIF. Thank the organisers. Invite questions on xrv9xvzb.

---

## Numbers checklist

| number | value | source path |
|---|---|---|
| Labelled sites per batch | Batch_1 7, Batch_2 7, Batch_3 17 | `outputs/fem/validation.csv` (row counts) |
| Held-out sites | 3 | `AGENTS.md`, `data_heldout/README.md` |
| Sites with ETD / SE third detector | 27 / 4 | `AGENTS.md` |
| Pixel size raw / half cache | 25 nm/px / 50.0 nm/px | `AGENTS.md` |
| Batch_3 imaging offset sites | 4 sites, black level ≈ +19–23, gain ≈ 0.69× | `AGENTS.md` |
| SOC frames | 11 (0% to 100%) | `.claude/plans/fem-swelling.md` D3 |
| Tiles per site, width | 6, 22 µm | `docs/classifier/method.md` |
| Feature counts KPI / FEM / KPI+FEM | 15 / 16 / 31 | `docs/classifier/results.md` |
| Si volume ratio at 100% SOC (model) | J_Si 3.240 | `docs/fem/method.md` (SOC table) |
| Si eigenstretch β, utilisation u_max | 2.8, 0.8 | `docs/fem/method.md` (Parameters) |
| Literature Si expansion | 263% (Qi), 277% (Obrovac), 280% (Beaulieu) | `docs/fem/literature-review.md` |
| Graphite c-axis strain at 100% SOC (model) | 0.094 | `docs/fem/method.md` (SOC table) |
| Graphite c-axis strain at LiC6 (literature) | about 10.3% | `docs/fem/literature-review.md` (C12) |
| Graphite in-plane strain | about 1% | `docs/fem/literature-review.md` (C14) |
| Si / graphite capacity (half cell, 0.01 V) | about 3400 / about 340 mAh/g | `docs/fem/literature-review.md` (C2) |
| Si/graphite charge split | 96% Si below s = 0.25; 58% / 42% above | `docs/fem/literature-review.md` |
| Simulation cases | 68 (34 bottom + 34 top) | `outputs/fem/run_log.json`, `docs/fem/results.md` |
| Cells, Batch_1/4ih2ggld bottom | 1011513 | `outputs/fem/run_log.json` |
| dolfinx version | 0.10.0 | `outputs/fem/run_log.json` |
| CPUs / memory per case | 4.0 / 10240 MB | `outputs/fem/run_log.json` |
| Wall time per case (bottom) | median 140 s, max 191 s | `docs/fem/results.md` |
| Wall time per case (top) | median 130 s, max 175 s | `docs/fem/results.md` |
| Failed sites | 0 in either orientation | `docs/fem/results.md` |
| Modal ledger total | $2.608 of $180 cap (2.6081 in run log) | `docs/fem/results.md`, `outputs/fem/run_log.json` |
| Production (full) mode cost | $1.31 (1.3122 in run log) | `docs/fem/results.md`, `outputs/fem/run_log.json` |
| Per-case cost | median $0.019, max $0.023 | `docs/fem/results.md` |
| Unit tests (G1) | 49 passed | `docs/fem/results.md` |
| Resolution check, swelling max rel. difference | 0.00643 | `docs/fem/results.md` |
| Finite-strain failure range | s of about 0.02 to 0.08 | `docs/fem/results.md`, D20 |
| Compaction-barrier failures | s = 0.023 and 0.016 | `docs/fem/results.md` |
| Si von Mises under linear model | about 20 to 26 GPa (saturated) | `docs/fem/results.md` |
| si_yield_frac | 1.0 everywhere | `docs/fem/results.md` |
| Median swelling_sym, 34 sites | 0.127 | `docs/fem/results.md` |
| Stop window / literature band | [0.03, 0.39] / [0.09, 0.39] | `docs/fem/results.md` |
| Median swelling by batch | B1 0.132, B2 0.125, B3 0.125, held-out 0.127 | `docs/fem/results.md` |
| Literature electrode swelling | 9%, 19%, 33% (graphite); 22% at 100 mV, 39% at 10 mV (15 wt% Si) | `docs/fem/literature-review.md` (VT1) |
| swelling_sym, GIF sites | 5n1q8atc 0.183591; b3esycq1 0.137581; x7u69zsw 0.139353; 3e122cbj 0.168261 | `outputs/fem/validation.csv` |
| swelling_sym, 4ih2ggld | 0.179983 | `outputs/fem/validation.csv` |
| swelling_sym, xrv9xvzb (lowest of 34) | 0.117689 | `outputs/fem/validation.csv` |
| first_pore_closure_s | 0.1 at all 34 sites | `outputs/fem/validation.csv` |
| Stage-1 balanced acc KPI / FEM / KPI+FEM | 0.548 / 0.626 / 0.561 | `docs/classifier/results.md` |
| Stage-1 CI KPI / FEM / KPI+FEM | 0.418–0.702 / 0.468–0.791 / 0.407–0.717 | `docs/classifier/results.md` |
| Stage-2 balanced acc KPI / FEM / KPI+FEM | 0.357 / 0.357 / 0.286 | `docs/classifier/results.md` |
| End-to-end balanced acc KPI / FEM / KPI+FEM | 0.437 / 0.370 / 0.350 | `docs/classifier/results.md` |
| End-to-end CI KPI / FEM / KPI+FEM | 0.292–0.630 / 0.250–0.513 / 0.224–0.497 | `docs/classifier/results.md` |
| End-to-end Brier KPI / FEM / KPI+FEM | 0.191 / 0.208 / 0.201 | `docs/classifier/results.md` |
| Arm tie margin | 0.05 | `docs/classifier/results.md` |
| Noise level stated | differences below ~0.1 balanced accuracy | `docs/classifier/results.md` |
| KPI confusion, correct per batch | B1 2/7, B2 1/7, B3 15/17 | `docs/classifier/results.md` |
| Calibration, end-to-end [0.5, 0.7) bin | n 28, accuracy 0.607 | `docs/classifier/results.md` |
| Top stage-1 feature | K15_si_graphite_contact_frac, 0.117 | `docs/classifier/results.md` |
| Check-3 R1 multiclass balanced acc | 0.493 (0.316–0.691) | `outputs/pooling_checks/check3_results.md` |
| Held-out final confidence | 3e122cbj 0.855, fn0mhxef 0.562, xrv9xvzb 0.586 | `docs/classifier/results.md` |
| Held-out all-arm confidences | see Slide 9 table | `docs/classifier/results.md` |
| xrv9xvzb FEM-arm P(Batch_3), tiles not-B3 | 0.272361, 6/6 | `outputs/classifier/heldout_predictions.csv` |
| Held-out z-scores and site values | as quoted on Slide 9 | `outputs/classifier/heldout_predictions.csv` |
| Bootstrap CI width | about ±0.2 | `docs/classifier/method.md` |
| Chance level, two-class balanced accuracy | 0.5 | definitional, not from a file |
