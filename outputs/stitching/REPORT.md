# Do crops from one parent image abut? (seam test)

**Yes. 17 of the 34 same-parent pairs share a seam: they sit side by side (left|right) with no vertical offset (|dy| <= 5 px) and essentially no missing material.
Seven parents reassemble completely into one strip. No pair from different parents passes the test.**

Scripts: `scripts/stitch_seams.py` (Modal: `modal run scripts/stitch_seams.py`; full-res TIFFs in the new volume `pmdb-fullres`), `scripts/stitch_report.py` (local calibration, chains, previews).

## Method
- **Data:** full-res raw TIFFs (25 nm/px), 3 channels (BSE, Inlens, ETD|SE). On every edge we skip the artefact border before taking the strip: 3 columns on left/right, 1 row on top/bottom. 15 sites have a bright artefact column 0-1 px wide on one side, and a couple have up to 4-6 deviating columns (`sites.csv`: `art_left`, `art_right`). Because `SKIP_LR` = 3 (`stitch_seams.py:32`), some artefact columns stay inside the profile for x77cy643, ufdvpb81 and fzrt2k6r; no verdict changes. The smallest physical gap a true left/right seam can have is therefore 6 px.
- **Edge profile:** for each edge, take the mean of the 4 px nearest the edge per channel, then band-pass it (subtract a Gaussian with sigma = 25, smooth with sigma = 4). This keeps the particle-scale signal: particles cut by the boundary should continue on the other side.
- **Score `mp`:** the channel-mean Pearson r between two edge profiles, maximised over the shift along the edge (|shift| <= 64 px for L/R edges, any shift with >= 2000 px overlap for T/B edges). We tested the 4 unflipped placements (A|B, B|A, A over B, B over A) and, as a secondary check, all 12 flipped or reversed edge combinations.
- **Positive control:** each crop is split internally at 25/50/75 % of its width and height, with a known gap. The halves are scored with the exact same function and shift search.
- **Null:** (a) all 2108 different-parent unflipped placements (1054 per orientation; the 6324 flipped placements also all fail); (b) the 3 wrong placements of every same-parent pair (102). `z` = (mp - null mean) / null sd. The empirical p is computed against null (a).
- **Verdict ABUT:** z > 4, |shift| <= 8 px, and a left/right placement.

**Threshold robustness.** The empirical p sits at its floor (1/1055) for all 17 seams, and across 136 tests that floor alone cannot support significance (Bonferroni gives about 0.13). The verdict therefore rests on the margin between groups, not on p. Among different-parent left/right placements with |dy| <= 8, the highest z is 3.01. The smallest abutting z is 5.60, and the best non-abutting same-parent pair has z 1.64. Any z threshold between 3.02 and 5.60 gives identical verdicts, so Z_MIN = 4 is not a fine-tuned choice. Every seam's mp (0.389 to 0.691) is above the highest different-parent left/right null (0.348).

## Score levels
| | mp median (min) |
|---|---|
| Positive control, L/R split, gap 0 | 0.84 (0.65) |
| Positive control, L/R split, gap 6 (minimal real gap) | 0.58 (0.24) |
| Positive control, L/R split, gap 12 | 0.42 |
| Positive control, L/R split, gap 24 / 48 | 0.26 / 0.16 |
| Null, different parent, L/R | mean 0.127, sd 0.047, p99 0.262, max 0.348 |
| Null, different parent, T/B | mean 0.195, sd 0.032, max 0.343 |
| Null, same-parent wrong placements | mean 0.151, sd 0.052, max 0.253 |

Five different-parent placements reach z > 4, but none is a plausible seam: three left/right placements have |dy| >= 43, and two are top/bottom placements. None passes the |dy| <= 8 test. The 17 detected seams have mp 0.39-0.69. That matches the gap ~6-12 px controls, so the crops were cut directly next to each other: the only missing material is the artefact columns.

## Per-pair verdicts (same parent; full table in `same_parent_verdicts.csv`)
| parent | left crop (batch) | right crop (batch) | mp | dy | z | verdict |
|---|---|---|---|---|---|---|
| G2068SE | vc2whyaq (B3) | x77cy643 (B3) | 0.691 | -1 | 12.1 | ABUT |
| G2068SE | utfgcjfa (B3) | rxax5ozo (B2) | 0.645 | 1 | 11.1 | ABUT |
| G2068SE | x77cy643 (B3) | utfgcjfa (B3) | 0.501 | 1 | 8.0 | ABUT |
| G2088 | 9luzk4jm (B3) | hzumfsms (B3) | 0.636 | -2 | 10.9 | ABUT |
| G2088 | ufdvpb81 (B3) | **xrv9xvzb (H)** | 0.554 | 1 | 9.1 | ABUT |
| G2088 | **xrv9xvzb (H)** | 9luzk4jm (B3) | 0.536 | -1 | 8.7 | ABUT |
| G1904 | mgxahqnk (B3) | hawkfj64 (B3) | 0.634 | 2 | 10.9 | ABUT |
| G1904 | hawkfj64 (B3) | 0grcilhi (B3) | 0.389 | -4 | 5.6 | ABUT |
| G2060 | x7u69zsw (B3) | tuy3zymq (B3) | 0.596 | 1 | 10.0 | ABUT |
| G2060 | kbdh4tri (B3) | 71vgq3fw (B3) | 0.521 | 0 | 8.4 | ABUT |
| G2060 | tuy3zymq (B3) | kbdh4tri (B3) | 0.430 | -3 | 6.5 | ABUT |
| G2080 | r17byphk (B2) | ffwubibz (B1) | 0.573 | 0 | 9.5 | ABUT |
| G2080 | cfe5vt7s (B3) | r17byphk (B2) | 0.479 | -5 | 7.5 | ABUT |
| G2316 | **3e122cbj (H)** | 4ih2ggld (B1) | 0.543 | -2 | 8.9 | ABUT |
| G2316 | 5n1q8atc (B1) | **3e122cbj (H)** | 0.487 | 4 | 7.7 | ABUT |
| G2148 | f1vzngrs (B1) | epqdaau9 (B2) | 0.404 | 0 | 5.9 | ABUT |
| G2048 | **fn0mhxef (H)** | 3806gxp0 (B2) | 0.401 | -5 | 5.9 | ABUT |
| G2156 | fzrt2k6r (B1) | b3esycq1 (B2) | best z 1.6 | | | no seam |
| G2316 | 4ih2ggld | 5n1q8atc | best z 1.4 | | | no seam (3e122cbj sits between them) |

All other same-parent pairs have z < 1.7, including G1612 (ptg8lmto/xgj4xftb), G2272 (pl8uabbv/i9jiqjwl) and avn74qx1 within G2048. For every ABUT pair, the best wrong placement has z <= 1.9 and the best flipped combination has z <= 3.9. The images were therefore not flipped. Every seam is horizontal adjacency at dy ~ 0, which fits crops of a wide strip that keep the parent's full height. No top/bottom seam was found.

## Reassembled parents (`chains.csv`, `stitched_all_parents.png`, `seam_zooms.png`)
| parent | left -> right | batches |
|---|---|---|
| G2060 | x7u69zsw, tuy3zymq, kbdh4tri, 71vgq3fw | 3 3 3 3 |
| G2068SE | vc2whyaq, x77cy643, utfgcjfa, rxax5ozo | 3 3 3 2 |
| G2088 | ufdvpb81, **xrv9xvzb**, 9luzk4jm, hzumfsms | 3 H 3 3 |
| G1904 | mgxahqnk, hawkfj64, 0grcilhi | 3 3 3 |
| G2080 | cfe5vt7s, r17byphk, ffwubibz | 3 2 1 |
| G2316 | 5n1q8atc, **3e122cbj**, 4ih2ggld | 1 H 1 |
| G2148 | f1vzngrs, epqdaau9 | 1 2 |
| G2048 (partial) | **fn0mhxef**, 3806gxp0 ... avn74qx1 (not adjacent) | H 2 ... 2 |

`stitched_all_parents.png` shows the BSE channel from the half-res cache, stretched per crop and reduced 4x, one row per parent. In the 1:1 zooms (`seam_zooms.png`, half-res, seam at x = 400 in each panel) the seam cannot be seen: particles continue across it.

The bright artefact column fits the chain order. On chained crops there are 10 artefact sides; 8 sit on an open chain end (x7u69zsw L, 71vgq3fw R, vc2whyaq L, ufdvpb81 L, 0grcilhi R, ffwubibz R, f1vzngrs L, fn0mhxef L). The two exceptions are interior: mgxahqnk R and x77cy643 R (columns 2-5). The artefact therefore usually marks a parent border. This also fits the unlinked crops: avn74qx1 (R) is the right end of G2048, and the pairs that do not abut (xgj4xftb L / ptg8lmto R, i9jiqjwl L, fzrt2k6r L+R) look like the two ends of a parent whose middle crop(s) were not released.

## Implications
- **Grouping confirmed.** All 17 detected seams fall inside the proposed parent groups. 0 of the 2108 different-parent placements pass. The four non-adjacent groups (G1612, G2156, G2272, and avn74qx1 in G2048) are not refuted. Their grouping rests on fingerprints only.
- **Batches are not parent-coherent.** Of the 12 seams between two labelled crops (5 of the 17 involve a held-out crop), 4 cross a batch boundary (utfgcjfa B3 | rxax5ozo B2; cfe5vt7s B3 | r17byphk B2 | ffwubibz B1; f1vzngrs B1 | epqdaau9 B2). Physically contiguous material therefore carries different batch labels. The organisers' batches are an artificial regrouping, and adjacency alone does not determine the batch.
- **Held-out sites.** Each held-out crop is physically wedged between, or attached to, crops of a single labelled batch:
  - 3e122cbj lies between two Batch_1 crops (G2316 is all Batch_1).
  - xrv9xvzb lies between two Batch_3 crops (G2088 is all Batch_3).
  - fn0mhxef is the left neighbour of 3806gxp0 (Batch_2; G2048 is all Batch_2).

  If the organisers kept each parent's crops in one batch, as they did for 6 of 11 multi-crop parents (G1612, G1904, G2048, G2060, G2088, G2316), the best guesses are 3e122cbj -> Batch 1, xrv9xvzb -> Batch 3 and fn0mhxef -> Batch 2. The mixed parents (G2068SE, G2080, G2148, G2156, G2272) show that this is a prior, not proof. This analysis only reads the held-out sites' edges; nothing is trained on them.
- **Leakage warning.** Neighbouring crops are contiguous material, so any cross-validation that splits crops of one parent across train and test leaks information. Split by parent (`GROUPS` in `scripts/stitch_seams.py`).
