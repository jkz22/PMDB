# Pitch brief: Si/graphite anode batch QC from SEM cross-sections

> Paths under `.claude/` cited here (plans, agent reports) were removed from the tree in the release cleanup (#169). Read them from history: `git show ed0a840:<path>`.

Deck source for the 2026-10-04 pitch (decision D19, `.claude/plans/fem-swelling.md`). Every number below is copied from the file cited next to it. Paths are relative to the repository root. The "Numbers checklist" at the end lists each number once, for fact-checking.

Revision 2 (2026-10-04): the story now follows the FEM diagnosis (`.claude/checkpoint/reports/fem-why.synthesiser.md`) and the pre-registered test D21 (`.claude/plans/fem-swelling.md`). Leo's spatial fingerprint model (`pmdb/fingerprint.py`, `docs/fingerprint.md`) is the classifier of record. Our two-stage random forest, whose rule had picked the KPI arm, is now a cross-check only.

## Headline

1. We simulated lithiation inside all 34 real SEM microstructures: 68 of 68 cases ran to full charge for $2.608, and the median simulated swelling, 0.127, sits inside the literature band [0.09, 0.39] (`docs/fem/results.md`).
2. The simulation answered the question with a clear negative. Swelling is set by Si content (R² = 0.971 against Si area fraction), and Si content does not differ between batches (Kruskal-Wallis p = 0.377). The batches differ in how the Si is arranged, not in how much there is (`.claude/checkpoint/reports/fem-why.explorer-diag.md`).
3. Leo's spatial fingerprint reads that arrangement: 21 of 31 sites correct in leave-one-out, permutation p = 0.0060. Under a rule written before the run, adding FEM features did not help (20/31 and 17/31), so the fingerprint makes the held-out calls: 3e122cbj Batch_1, fn0mhxef Batch_3, xrv9xvzb Batch_2 (`outputs/fem_fingerprint/main/verdict.md`).

---

## Slide 1. The problem: is this batch what the supplier promised?

**Key message:** A Si/graphite anode is accepted or rejected by batch, and the only stated quality rule is about where the silicon sits.

- The electrode is a graphite anode with silicon as the secondary active material. The provider's only stated quality criterion: "silicon must not cluster and must not be localised" (`docs/kpis/README.md`).
- Batch_3 is the baseline, "what's been 'promised' by the supplier". Batch_1 and Batch_2 arrived later. They are not better or worse, they show the variation a model must detect (`data_heldout/README.md`).
- Deliverable for each held-back site: always a batch, a confidence, and how it differs from Batch_3. Abstaining is not allowed (`data_heldout/README.md`, `AGENTS.md`).
- Data: 31 labelled sites (Batch_1 7, Batch_2 7, Batch_3 17; row counts of `outputs/fem/validation.csv`) and 3 unlabelled held-out sites, `3e122cbj`, `fn0mhxef`, `xrv9xvzb`, never trained on (`AGENTS.md`). Three aligned detectors per site (BSE, Inlens, ETD or SE), 50 nm/px in the half-resolution cache (`AGENTS.md`).

**Figure:** one raw BSE frame, e.g. frame 1 of `outputs/fem/figures/example_frames.png`.

**Speaker note:** The supplier promised Batch_3. Our job is to say whether a new sample matches that promise and, if not, how it differs. One trap up front: some Batch_3 sites carry an imaging offset (black level ≈ +19–23, gain ≈ 0.69× on 4 sites, `AGENTS.md`), so grey levels alone could leak the batch. Every model in this deck works on segmented geometry, not on grey levels.

---

## Slide 2. The question: can a mechanics simulation of the real microstructure reveal batch differences?

**Key message:** Silicon swells far more than graphite, so whether a batch is acceptable could depend on how the surrounding microstructure absorbs that swelling. A KPI counts Si. A simulation asks whether its neighbourhood can absorb it.

- Si: volume ratio J_Si = 3.240 at 100% SOC in our model (`docs/fem/method.md`, SOC table). Literature full-lithiation expansion: 263% (Qi), 277% (Obrovac), 280% (Beaulieu) (`docs/fem/literature-review.md`).
- Graphite: c-axis (through-thickness) strain 0.094 at 100% SOC in our model (`docs/fem/method.md`, SOC table); literature end point about 10.3% at LiC6 (`docs/fem/literature-review.md`, C12).
- In the image-based model of Shah et al. 2022, Si expansion is absorbed by "(i) reduction in porosity, (ii) compaction of CBD, and (iii) expansion into the separator" (`docs/fem/literature-review.md`, C17).
- Hypothesis: batches with the same Si but a different neighbourhood (pores, binder, graphite contact) would swell and stress differently, and a classifier could read that.

**Figure:** none, or a two-panel sketch: Si particle next to a pore vs Si particle boxed in by graphite.

**Speaker note:** Silicon holds about 3400 mAh/g against about 340 mAh/g for graphite in the reference half cell (`docs/fem/literature-review.md`, C2). That capacity is why it is in the anode, and its swelling is why its placement matters. All parameters come from a literature review with evidence grades (`docs/fem/literature-review.md`).

---

## Slide 3. The simulation: 68 cases, 0 failures, $2.608

**Key message:** We turned every segmented SEM site into a 2D finite-element lithiation from 0 to 100% charge, and the whole campaign ran on Modal CPUs for a few dollars.

- Pipeline: image → segmentation → mesh at one cell per pixel at 100 nm → lithiation in 11 SOC frames → stress and strain fields → features (`docs/fem/method.md` section 1; `.claude/plans/fem-swelling.md` D3). 1011513 cells for Batch_1/4ih2ggld (`outputs/fem/run_log.json`).
- 34 sites × 2 collector orientations = 68 cases (`outputs/fem/run_log.json`; `docs/fem/results.md`, Convergence). Two orientations because the foil side is not visible; features use their mean (`sym`) (`docs/fem/method.md` section 4).
- 0 of 34 sites failed before 100% SOC in either orientation. Wall time per case: median 140 s (bottom), 130 s (top) (`docs/fem/results.md`).
- Cost: ledger total $2.608 of a $180 cap; production runs $1.31 (`docs/fem/results.md`). Checks: 49 unit tests passed; 100 vs 50 nm resolution check, max relative swelling difference 0.00643 (`docs/fem/results.md`).

**Figure:** GIF wall, one per batch, the site with the highest `swelling_sym` in its batch (`outputs/fem/validation.csv`):
- Batch_1: `outputs/fem/gifs/Batch_1__5n1q8atc.gif` (swelling_sym 0.183591)
- Batch_2: `outputs/fem/gifs/Batch_2__b3esycq1.gif` (swelling_sym 0.137581)
- Batch_3: `outputs/fem/gifs/Batch_3__x7u69zsw.gif` (swelling_sym 0.139353)
- Held out: `outputs/fem/gifs/Batch_heldout__3e122cbj.gif` (swelling_sym 0.168261)

**Speaker note:** Each GIF is 11 frames, 0 to 100% SOC: the BSE image warped by the computed displacement, with von Mises stress on top on a colour scale fixed across sites, bottom orientation. One honest model choice: the literature calls for finite-strain mechanics, but on real microstructures it failed at s of about 0.02 to 0.08, because Si crushes thin pores and the model has no contact. Production therefore uses linear small-strain elasticity with a logarithmic eigenstrain (`docs/fem/results.md`, Limitations; `.claude/plans/fem-swelling.md` D20).

---

## Slide 4. The simulation is physically plausible

**Key message:** Simulated electrode swelling at full charge falls inside the literature band at every site, at its low end, as expected for a 2D elastic model.

- Median `swelling_sym` at s = 1 over 34 sites: 0.127. Literature band [0.09, 0.39], `lit_band_ok = True`. Stop window [0.03, 0.39]; no site outside it (`docs/fem/results.md`).
- Median by batch: Batch_1 0.132, Batch_2 0.125, Batch_3 0.125, Batch_heldout 0.127 (`docs/fem/results.md`).
- Literature reference points: graphite-only 9% (Michael), 19% (Prado), 33% (Kirner); 15 wt% Si 22% at a 100 mV cut-off and 39% at 10 mV (`docs/fem/literature-review.md`, VT1).
- Low end expected: no particle rearrangement, binder creep or SEI growth in the model (`docs/fem/method.md` section 9).

**Figure:** `outputs/fem/figures/swelling_by_batch.png` (gold: literature band); optional `outputs/fem/figures/swelling_vs_soc.png`.

**Speaker note:** This is a sanity check, not a calibration. Look at the batch medians: 0.132, 0.125, 0.125. They are already close. The next slide explains why.

---

## Slide 5. The finding: batches differ in arrangement, not amount

**Key message:** The simulated swelling is set by how much Si a site holds, and the batches hold the same amount of Si. So the swelling cannot separate them. What differs between batches is where the Si sits.

- Swelling tracks Si content. Site swelling at s = 1 regressed on Si area fraction (`K01_si_frac_adm`): R² = 0.971. Spearman rho with K01: 0.891 for swelling at full charge, 0.903 at half charge (`.claude/checkpoint/reports/fem-why.explorer-diag.md`).
- Si content does not differ between batches: K01 Kruskal-Wallis p = 0.377. Swelling itself: p = 0.2166. What is left of swelling after Si content: p = 0.699 (`.claude/checkpoint/reports/fem-why.explorer-diag.md`).
- The stress features carry no usable signal under linear mechanics: `si_yield_frac` = 1 on every labelled site, and site-level Si von Mises p95 runs from 23414.9 to 27445.9 MPa against a yield stress of 637.5 MPa (`.claude/checkpoint/reports/fem-why.explorer-diag.md`).
- No single number separates the batches. 0 of 42 site-level image KPIs reach p < 0.05 in our scan (`.claude/checkpoint/reports/fem-why.explorer-diag.md`); Leo's scan finds none of 51, best p = 0.065 (`docs/fingerprint.md`). No bug: the integrity checks are clean and the repository metrics reproduce exactly (`.claude/checkpoint/reports/fem-why.synthesiser.md`, C10).

**Figure:** `outputs/fem/figures/swelling_vs_si.png` — site swelling at s = 1 (sym) vs K01, coloured by batch, held-out sites as stars, linear fit on labelled sites (R² = 0.97). Generated by `scripts/plot_swelling_vs_si.py` from `outputs/fem/site_curves.csv` and `outputs/kpis/site_kpis.csv` (+ `outputs/heldout/kpis/site_kpis.csv`).

**Speaker note:** This is a mechanistic negative result, and it rules out a hypothesis. In a linear model with swelling strain only in Si, the electrode swells roughly in proportion to its Si fraction (our reading of the R² of 0.971, not a separate measurement). Our first classifier, a two-stage random forest on tile features, showed the same thing from the other side: end-to-end balanced accuracy 0.437 for image KPIs, 0.370 for FEM, 0.350 for both (`docs/classifier/results.md`). Its pre-set rule picked the KPI arm, but every arm sat in the same noise band. That classifier is no longer our answer.

---

## Slide 6. Leo's spatial fingerprint reads the arrangement

**Key message:** Our teammate Leo built a classifier on spatial curves instead of scalars. It separates the batches better than chance with a permutation test behind it, and it is our classifier of record.

- Features, 16 per site: Si depth profile in five bands, normalised by the site mean so composition cancels, plus depth slope and mid-depth dip; pair-correlation level in four lag bins per axis; spread of Si-graphite contact (K15) across tiles (`docs/fingerprint.md`).
- What it sees: Batch_3 uniform through the coating; Batch_2 top-heavy with a depleted mid-depth (consistent with Si migration during drying); Batch_1 bottom-heavy (consistent with sedimentation) and more variable site to site. Mid-depth band Kruskal-Wallis p ≈ 0.001 (`docs/fingerprint.md`).
- Model: robust per-batch naive Bayes with no fitted weights, assignment by best likelihood score, plus class-conditional conformal p-values for credibility, confidence and an out-of-distribution flag (`docs/fingerprint.md`).
- Validation, leave-one-site-out: accuracy 0.677 (21/31) against a 0.548 majority baseline; label-permutation p = 0.002 over 500 permutations, null mean 0.369 (`outputs/fingerprint/evaluation.json`, `docs/fingerprint.md`). Per-batch recall: Batch_1 5/7, Batch_2 3/7, Batch_3 13/17 (`docs/fingerprint.md`).

**Figure:** `outputs/fingerprint/figures/depth_profiles.png` (per-batch Si depth profiles with the held-out sites against the batch medians).

**Speaker note:** Credit where due: the fingerprint is Leo's work (`pmdb/fingerprint.py`). It answers the provider's own rule, "silicon must not cluster and must not be localised", directly, because it measures where Si sits. Two caveats from Leo's own docs: the feature families were chosen by exploring the labelled data, and with n = 31 the permutation test, not the 0.677, is the claim (`docs/fingerprint.md`).

---

## Slide 7. Does FEM add anything to the fingerprint? A pre-registered test

**Key message:** We wrote the pass rule down before running anything. No FEM arm passed, so Leo's fingerprint alone (A0) stays the classifier of record.

- Rule (D21, fixed before any run): an FEM arm adds value only if leave-one-out gets at least 23/31 correct and a permutation test (1000 permutations, any feature selection rerun inside every permutation) gives p ≤ 0.05 (`.claude/plans/fem-swelling.md` D21; `outputs/fem_fingerprint/main/verdict.md`).
- Arms: A0 Leo's 16 features; A1 A0 plus 2 physics features (swelling left over after Si content, and Si stress spread); A2 A0 plus the top 2 of 160 raw FEM metrics, chosen inside each fold on training sites only (`.claude/plans/fem-swelling.md` D21).

| arm | correct | accuracy | balanced acc | perm p | passes |
|---|---|---|---|---|---|
| A0 (Leo) | 21/31 | 0.677 | 0.636 | 0.0060 | reference |
| A1 (+2 physics) | 20/31 | 0.645 | 0.616 | 0.0060 | False |
| A2 (+2 selected) | 17/31 | 0.548 | 0.557 | 0.0809 | False |
| A1, edge5 | 21/31 | 0.677 | 0.636 | 0.0060 | False |
| A2, edge5 | 18/31 | 0.581 | 0.521 | 0.0370 | False |

Sources: `outputs/fem_fingerprint/main/verdict.md`, `outputs/fem_fingerprint/main/metrics.csv`, `outputs/fem_fingerprint/edge5/verdict.md`, `outputs/fem_fingerprint/edge5/metrics.csv`.

- edge5 is a sensitivity run, never eligible as the final arm: the same arms with features re-reduced after dropping a 5 µm band at the top and bottom image edges (`outputs/fem_fingerprint/edge5/verdict.md`; `.claude/plans/fem-swelling.md` D21).
- A2 picked `q25_vm_binder@s0.5` in 31 of 31 folds (`outputs/fem_fingerprint/main/verdict.md`). The FEM signal that exists is not enough to move the fingerprint.

**Figure:** the table above. Optional: bar chart of correct/31 per arm with the 23/31 threshold line.

**Speaker note:** Adding physics made the fingerprint slightly worse or left it unchanged. With 31 sites, one site is noise, which is why the rule asked for two more correct sites and a permutation test. The held-out sites were scored only after the rule was applied (`.claude/plans/fem-swelling.md` D21).

---

## Slide 8. Held-out calls, and how to read the two numbers

**Key message:** Every held-out site gets a batch, a credibility and a confidence. One call is firm, one is weak, and one is a correctly reported toss-up.

| site | batch | credibility | confidence | p(B1) / p(B2) / p(B3) | out of distribution |
|---|---|---|---|---|---|
| 3e122cbj | Batch_1 | 0.875 | 0.0 | 0.875 / 0.875 / 1.0 | False |
| fn0mhxef | Batch_3 | 0.444 | 0.125 | 0.125 / 0.875 / 0.444 | False |
| xrv9xvzb | Batch_2 | 1.0 | 0.5 | 0.5 / 1.0 / 0.389 | False |

Source: `outputs/fingerprint/heldout_predictions.csv`; identical in `outputs/fem_fingerprint/main/heldout_predictions.csv`.

- **Credibility:** how typical the site is of the batch it was assigned to. 1.0 means it looks as ordinary for that batch as any labelled site of that batch (`docs/fingerprint.md`).
- **Confidence:** 1 minus the highest p among the other batches. It is high only when every other batch is ruled out; 0 means another batch fits at least as well (`docs/fingerprint.md`).
- The batch itself is chosen by the best likelihood score, not by the highest p. The p-values only say how typical a site is for each batch (`docs/fingerprint.md`).
- No held-out site is flagged out of distribution (`outputs/fingerprint/heldout_predictions.csv`, `ood` column).

**Figure:** `outputs/fingerprint/figures/card_3e122cbj.png`, `card_fn0mhxef.png`, `card_xrv9xvzb.png`.

**Speaker note:** The p-values move in coarse steps, 1/8 for a 7-site batch (`docs/fingerprint.md`), so read them as bins, not as precise probabilities. A confidence of 0 for 3e122cbj is the model saying "this one could be anyone", not a bug.

---

## Slide 9. How each held-out site differs from Batch_3

**Key message:** fn0mhxef matches the baseline's depth profile, xrv9xvzb has Batch_2's depleted mid-depth, and 3e122cbj matches the baseline's arrangement while holding far more Si.

- **xrv9xvzb → Batch_2 (credibility 1.0, confidence 0.5).** Mid-depth dip −0.723229, below every Batch_3 site (Batch_3 minimum −0.692301) and inside the Batch_2 range (−1.035662 to −0.400299) (`outputs/fingerprint/heldout_features.csv`, `outputs/fingerprint/features.csv`). Largest deviations from the Batch_3 centre: `gx_2.0_4.0` (dev 1.983792), `si_depth_rel_band3` (1.697059), `si_depth_mid_dip` (1.548426) (`outputs/fingerprint/heldout_explain.csv`). Read: less Si at mid-depth than the baseline, the Batch_2 pattern.
- **fn0mhxef → Batch_3 (credibility 0.444, confidence 0.125).** Mid-depth band Si 1.187464 of the site mean (`outputs/fingerprint/heldout_features.csv`). Batch_3 spans 0.482759 to 1.394753; Batch_1 reaches at most 1.100702 and Batch_2 at most 0.705665 (`outputs/fingerprint/features.csv`). Batch_1 is effectively excluded (p = 0.125). It is not a clean baseline match: its largest deviations from the Batch_3 centre are `si_depth_rel_band3` (dev 2.122747) and `gx_0.5_2.0` (1.872639) (`outputs/fingerprint/heldout_explain.csv`), and Batch_2 stays typical overall (p = 0.875).
- **3e122cbj → Batch_1 (credibility 0.875, confidence 0.0).** Its arrangement is typical of every batch: all 16 features lie within 1 scale unit of the Batch_3 centre, the largest being `gx_2.0_4.0` at 0.842804 (`outputs/fingerprint/heldout_explain.csv`), and p(B3) = 1.0. It wins Batch_1 on likelihood only (`docs/fingerprint.md`). Where it does differ from Batch_3 is amount, which the fingerprint cancels by design: Si area fraction 0.139 vs Batch_3 0.0623 ± 0.00955 (z = +8.0), Si objects per 1000 µm² 116 vs 27.7 ± 10.5 (z = +8.4) (`docs/classifier/results.md`). Among labelled sites, only Batch_1 reaches that loading: site K01 maximum Batch_1 0.164500, Batch_2 0.083739, Batch_3 0.089180 (`outputs/kpis/site_kpis.csv`). It also has the highest simulated swelling of the held-out sites, 0.168261 (`outputs/fem/validation.csv`), in line with its Si content.

**Figure:** `outputs/fingerprint/figures/depth_profiles.png` with the three held-out curves highlighted; `outputs/fem/gifs/Batch_heldout__3e122cbj.gif`.

**Speaker note:** xrv9xvzb has the elevated BSE black level (p1 = 6) seen otherwise only in Batch_3 acquisitions, yet its material fingerprint says Batch_2 (`docs/fingerprint.md`). Either it is a Batch_2 electrode imaged with Batch_3 settings, or the black level is not a batch marker. That is why no model here uses grey levels. For 3e122cbj, the Batch_1 call has two independent supports, likelihood and Si loading, but the conformal confidence is 0 and we report it as such.

---

## Slide 10. Cross-check: three models, one disagreement

**Key message:** Our earlier random forest agrees with the fingerprint on 3e122cbj and fn0mhxef in every arm. On xrv9xvzb the arms that include FEM agree with the fingerprint, and the image-KPI arm does not.

| site | fingerprint (record) | RF KPI | RF FEM | RF KPI+FEM |
|---|---|---|---|---|
| 3e122cbj | Batch_1 | Batch_1 (0.85) | Batch_1 (0.82) | Batch_1 (0.84) |
| fn0mhxef | Batch_3 | Batch_3 (0.56) | Batch_3 (0.61) | Batch_3 (0.61) |
| xrv9xvzb | Batch_2 | Batch_3 (0.59) | Batch_2 (0.45) | Batch_2 (0.40) |

Sources: `outputs/fingerprint/heldout_predictions.csv`; `docs/classifier/results.md`, "All arms".

- The RF numbers in brackets are products of stage probabilities, not conformal confidences, so the columns do not compare as numbers (`docs/classifier/method.md`; `docs/fingerprint.md`).
- xrv9xvzb under the RF KPI arm: 1 of 6 tiles votes not-Batch_3, slightly less Si than Batch_3 (0.0518 vs 0.0623 ± 0.00955, z = −1.1) (`docs/classifier/results.md`). The RF FEM arm votes not-Batch_3 on 6 of 6 tiles (P(Batch_3) 0.272361) (`outputs/classifier/heldout_predictions.csv`).

**Figure:** the table above.

**Speaker note:** We weight Batch_2 for xrv9xvzb: the two views that see only geometry and mechanics agree, and the KPI arm's Batch_3 call was its least certain (`.claude/checkpoint/reports/fem-why.synthesiser.md`, Contradictions). The RF stays in the repo as a cross-check, not as the answer.

---

## Slide 11. Limitations

**Key message:** The negative result is robust; the mechanics and the sample size are where the method is thin.

- Linear mechanics (D20): absolute stresses are not physical, Si von Mises saturates and `si_yield_frac` is 1.0 everywhere (`docs/fem/results.md`). 2D plane strain, elastic only, no electrochemistry, pure Si (no SiOx run), unsourced pore stiffness (`docs/fem/method.md` section 9).
- Interior windows: the images are 40.3–57.9 µm high and show neither the foil nor a free surface (median pore fraction 0.016 at the top edge, 0.045 at the bottom, 0.041 in the middle) (`.claude/checkpoint/reports/fem-why.explorer-edges.md`). Our boundary conditions are right in direction (in-plane constrained, thickness free) but place a roller "foil" on one edge and a free surface on the other (`.claude/checkpoint/reports/fem-why.synthesiser.md`, C14).
- Edge test: dropping 5 µm at the top and bottom edges leaves swelling nearly unchanged (Spearman 0.894 against the original, mean 0.1311 → 0.1297) but changes surface roughness (Spearman 0.639, mean 0.01088 → 0.00675), so roughness was mostly an edge artefact (`outputs/fem/edge5/site_curves.csv`; `.claude/reports/fem-edge5.implementer.md`). The D21 verdict did not change (Slide 7).
- n = 31 (7/7/17): coarse conformal steps, wide uncertainty on the 0.677, and fingerprint feature families chosen by exploring the same labelled data (`docs/fingerprint.md`). The fingerprint cancels composition, so it would not flag a shipment with normal arrangement but unusual Si loading; 3e122cbj is that case (Slide 9).

**Figure:** none.

**Speaker note:** We would rather show these than hide them. None of them rescues the swelling hypothesis: better boundary conditions would not change the fact that swelling follows Si content (`.claude/checkpoint/reports/fem-why.synthesiser.md`).

---

## Slide 12. Next steps and close

**Key message:** The pipeline from SEM image to simulated charge to a batch call with a reason runs on every site for a few dollars; better physics and more sites are the next gains.

- Periodic boundary conditions in x (and a window condition in z), so interior images are treated as interior; needs a solver change and a re-run of all 68 cases (`.claude/checkpoint/reports/fem-why.synthesiser.md`, C14).
- Finite strain with pore contact, the missing piece that blocked the finite-strain run (`.claude/plans/fem-swelling.md` D20).
- 3D microstructures, for example reconstructed from the 2D sections, to remove the plane-strain assumption (proposed; not in the repo docs).
- Pair the fingerprint with a composition check, so a Si-rich shipment like 3e122cbj is flagged even when its arrangement is typical (proposed, from Slide 9).
- Close: 34 sites simulated, 68 cases, 0 failures; the edge re-reduction added $0.377 (`docs/fem/results.md`; `.claude/reports/fem-edge5.implementer.md`). Held-out: 3e122cbj Batch_1, fn0mhxef Batch_3, xrv9xvzb Batch_2 (`outputs/fingerprint/heldout_predictions.csv`).

**Figure:** `outputs/fem/gifs/Batch_heldout__xrv9xvzb.gif` looping, or `scripts/demo_reject.py` (a shipment drifting from Batch_3 is first re-assigned, then rejected as out of distribution, `docs/fingerprint.md`).

**Speaker note:** The question was whether mechanics of the real microstructure reveals batch differences. The answer is that it reveals why they do not show up in bulk swelling: the amount of Si is the same, the arrangement is not. Thank Leo for the fingerprint and the organisers for the data. Invite questions on xrv9xvzb.

---

## Numbers checklist

| number | value | source path |
|---|---|---|
| Labelled sites per batch | Batch_1 7, Batch_2 7, Batch_3 17 | `outputs/fem/validation.csv` (row counts) |
| Held-out sites | 3 | `AGENTS.md`, `data_heldout/README.md` |
| Pixel size, half cache | 50.0 nm/px | `AGENTS.md` |
| Batch_3 imaging offset sites | 4 sites, black level ≈ +19–23, gain ≈ 0.69× | `AGENTS.md` |
| Si volume ratio at 100% SOC (model) | J_Si 3.240 | `docs/fem/method.md` (SOC table) |
| Literature Si expansion | 263% (Qi), 277% (Obrovac), 280% (Beaulieu) | `docs/fem/literature-review.md` |
| Graphite c-axis strain at 100% SOC (model) | 0.094 | `docs/fem/method.md` (SOC table) |
| Graphite c-axis strain at LiC6 (literature) | about 10.3% | `docs/fem/literature-review.md` (C12) |
| Si / graphite capacity (half cell) | about 3400 / about 340 mAh/g | `docs/fem/literature-review.md` (C2) |
| SOC frames | 11 (0% to 100%) | `.claude/plans/fem-swelling.md` D3 |
| Cells, Batch_1/4ih2ggld bottom | 1011513 | `outputs/fem/run_log.json` |
| Simulation cases | 68 (34 bottom + 34 top) | `outputs/fem/run_log.json`, `docs/fem/results.md` |
| Failed sites | 0 in either orientation | `docs/fem/results.md` |
| Wall time per case, median | 140 s bottom, 130 s top | `docs/fem/results.md` |
| Modal ledger total (production campaign) | $2.608 of $180 cap (2.6081 in run log) | `docs/fem/results.md`, `outputs/fem/run_log.json` |
| Production (full) mode cost | $1.31 (1.3122 in run log) | `docs/fem/results.md`, `outputs/fem/run_log.json` |
| edge5 re-reduction cost | $0.377 (ledger 2.608 → 2.985) | `.claude/reports/fem-edge5.implementer.md` |
| Unit tests | 49 passed | `docs/fem/results.md` |
| Resolution check, swelling max rel. difference | 0.00643 | `docs/fem/results.md` |
| Finite-strain failure range | s of about 0.02 to 0.08 | `docs/fem/results.md`, D20 |
| GIF sites swelling_sym | 5n1q8atc 0.183591; b3esycq1 0.137581; x7u69zsw 0.139353; 3e122cbj 0.168261 | `outputs/fem/validation.csv` |
| Median swelling_sym, 34 sites | 0.127 | `docs/fem/results.md` |
| Stop window / literature band | [0.03, 0.39] / [0.09, 0.39] | `docs/fem/results.md` |
| Median swelling by batch | B1 0.132, B2 0.125, B3 0.125, held-out 0.127 | `docs/fem/results.md` |
| Literature electrode swelling | 9%, 19%, 33% (graphite); 22% at 100 mV, 39% at 10 mV (15 wt% Si) | `docs/fem/literature-review.md` (VT1) |
| Swelling (s = 1) on K01, R² | 0.971 | `.claude/checkpoint/reports/fem-why.explorer-diag.md` |
| Spearman rho with K01 | swell_100 0.891, swell_50 0.903 | `.claude/checkpoint/reports/fem-why.explorer-diag.md` |
| KW p: K01 / swelling / swelling residual | 0.377 / 0.2166 / 0.699 | `.claude/checkpoint/reports/fem-why.explorer-diag.md` |
| Si von Mises p95 range (site, s = 1) vs yield | 23414.9–27445.9 MPa vs 637.5 MPa | `.claude/checkpoint/reports/fem-why.explorer-diag.md` |
| si_yield_frac | 1.0 everywhere | `docs/fem/results.md`; `.claude/checkpoint/reports/fem-why.explorer-diag.md` |
| Scalar KPIs at p < 0.05 (ours) | 0 of 42 | `.claude/checkpoint/reports/fem-why.explorer-diag.md` |
| Scalar KPIs separating batches (Leo) | none of 51, best p = 0.065 | `docs/fingerprint.md` |
| RF end-to-end balanced acc KPI / FEM / KPI+FEM | 0.437 / 0.370 / 0.350 | `docs/classifier/results.md` |
| Fingerprint feature count | 16 | `docs/fingerprint.md`, `outputs/fingerprint/evaluation.json` |
| Mid-depth band KW p | ≈ 0.001 | `docs/fingerprint.md` |
| Fingerprint LOO accuracy / majority baseline | 0.677 (21/31) / 0.548 | `outputs/fingerprint/evaluation.json` (0.6774193548387096 / 0.5483870967741935) |
| Fingerprint permutation p, null mean | 0.002 (500 perms), 0.369 | `outputs/fingerprint/evaluation.json` (0.001996007984031936, 0.3688387096774193) |
| Fingerprint recall | B1 5/7, B2 3/7, B3 13/17 | `docs/fingerprint.md`, `outputs/fingerprint/evaluation.json` |
| Conformal step, 7-site batch | 1/8 | `docs/fingerprint.md` |
| D21 rule | ≥ 23/31 and perm p ≤ 0.05, 1000 perms | `.claude/plans/fem-swelling.md` D21 |
| D21 main: correct / acc / bacc / perm p | A0 21/31, 0.677, 0.636, 0.0060; A1 20/31, 0.645, 0.616, 0.0060; A2 17/31, 0.548, 0.557, 0.0809 | `outputs/fem_fingerprint/main/verdict.md`, `metrics.csv` |
| D21 edge5: correct / acc / bacc / perm p | A1 21/31, 0.677, 0.636, 0.0060; A2 18/31, 0.581, 0.521, 0.0370 | `outputs/fem_fingerprint/edge5/verdict.md`, `metrics.csv` |
| A2 raw FEM pool, top selection count | 160 metrics; q25_vm_binder@s0.5 in 31/31 folds | `.claude/plans/fem-swelling.md` D21; `outputs/fem_fingerprint/main/verdict.md` |
| Held-out credibility / confidence | 3e122cbj 0.875 / 0.0; fn0mhxef 0.4444444444444444 / 0.125; xrv9xvzb 1.0 / 0.5 | `outputs/fingerprint/heldout_predictions.csv` |
| Held-out p(B1) / p(B2) / p(B3) | 3e122cbj 0.875 / 0.875 / 1.0; fn0mhxef 0.125 / 0.875 / 0.4444444444444444; xrv9xvzb 0.5 / 1.0 / 0.3888888888888889 | `outputs/fingerprint/heldout_predictions.csv` |
| xrv9xvzb mid-depth dip | −0.723229 | `outputs/fingerprint/heldout_features.csv` |
| Batch_3 min / Batch_2 range, mid-depth dip | −0.692301 / −1.035662 to −0.400299 | `outputs/fingerprint/features.csv` (min/max over batch rows) |
| fn0mhxef mid-depth band | 1.187464 | `outputs/fingerprint/heldout_features.csv` (`si_depth_rel_band2`) |
| Mid-depth band range B3; max B1, B2 | 0.482759–1.394753; 1.100702, 0.705665 | `outputs/fingerprint/features.csv` (min/max over batch rows) |
| Held-out deviations from Batch_3 centre (dev) | xrv9xvzb 1.983792, 1.697059, 1.548426; fn0mhxef 2.122747, 1.872639; 3e122cbj max 0.842804 | `outputs/fingerprint/heldout_explain.csv` (`dev_Batch_3`) |
| 3e122cbj KPI z-scores vs Batch_3 | K01 0.139 vs 0.0623 ± 0.00955 (z +8.0); K02 116 vs 27.7 ± 10.5 (z +8.4) | `docs/classifier/results.md` |
| Site K01 maximum per batch | B1 0.164500, B2 0.083739, B3 0.089180 | `outputs/kpis/site_kpis.csv` (max over batch rows) |
| xrv9xvzb BSE black level | p1 = 6 | `docs/fingerprint.md` |
| RF held-out calls, all arms | see Slide 10 table | `docs/classifier/results.md` |
| xrv9xvzb RF KPI arm | 1/6 tiles not-B3; K01 0.0518 vs 0.0623 ± 0.00955, z −1.1 | `docs/classifier/results.md` |
| xrv9xvzb RF FEM arm P(Batch_3), tiles not-B3 | 0.272361, 6/6 | `outputs/classifier/heldout_predictions.csv` |
| Image heights | 40.3–57.9 µm | `.claude/checkpoint/reports/fem-why.explorer-edges.md` |
| Edge pore fraction medians top / bottom / middle | 0.016 / 0.045 / 0.041 | `.claude/checkpoint/reports/fem-why.explorer-edges.md` |
| edge5 vs original, Spearman (sym, s = 1) | swelling 0.894, surface_rough 0.639 | `outputs/fem/edge5/site_curves.csv` vs `outputs/fem/site_curves.csv`; `.claude/reports/fem-edge5.implementer.md` |
| edge5 vs original, means (sym, s = 1) | swelling 0.1311 → 0.1297; surface_rough 0.01088 → 0.00675 | `.claude/reports/fem-edge5.implementer.md` |
| edge band width | 5 µm top and bottom | `.claude/plans/fem-swelling.md` D21 |
