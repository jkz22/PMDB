# Sibling-anchored batch calls from the pore-floor dark level (2026-10-04, test day)

`scripts/run_sibling_anchor.py <feature>`; per-feature results in `SE_type_D/` and `BSE_D/`.

Hypothesis tested: the organiser's per-crop grouping feature tracks the pore-floor dark level
(`pmdb/clean.py::dark_level`), ordered B1 < B2 < B3 — the best single two-cutpoint rule over 246 scalars
(27/34, max-stat p 0.009) and the only predictor that held under leave-one-parent-out (docs/parents.md,
functional-morphology branch). Two predictors, 34 labelled sites (31 + 3 released truths), majority 0.53:

| predictor | SE_type_D | BSE_D |
|---|---|---|
| two-cut rule, LOPO, 34 sites | 0.647 | 0.676 |
| sibling-anchored (site placed relative to the labelled crops of its own parent image), 32 sites | 0.562 | 0.594 |
| sibling-anchored, the 19 sites in mixed-label parents | 0.368 | 0.368 |
| LOPO two-cut rule on those same 19 sites | 0.526 | 0.474 |

## Reading

- **The within-parent part of the hypothesis fails.** Anchoring a crop on its labelled siblings — the one
  construction in which acquisition cancels exactly — predicts the labels in mixed parents at 0.37, *below*
  majority and below the cross-parent cut rule on the same sites. Within one image the dark level does not
  order the labels once the ordinal rule is actually applied (the earlier 12/13 concordance was pairwise and
  on differences of 0.3–1 DN, about one within-pure-parent SD of 0.5 DN).
- **The cross-parent cut rule is the only piece that holds** (0.65–0.68 LOPO on 34, both detectors), and it
  is carried by the pure-Batch-3 parents with the Batch 3 black-level offset. It is a parent-robust predictor
  of the *label*, not evidence that the organiser's feature is a grey level.
- Test-site calls below therefore use the cross-parent two-cut rule (full fit on 34). Both detectors give the
  same six calls. Confidence is low wherever the site's own parent contradicts the rule (G2156: both labelled
  siblings sit above the B3 cut yet are B1/B2) or the site sits within one SD of a cut.

| site | parent | SE_type_D | BSE_D | call | confidence | why |
|---|---|---|---|---|---|---|
| 0eryguqq | G1612 | +4.8 | +16.2 | Batch_3 | high | pure-B3 parent; slightly below both B3 siblings on SE and BSE (not inside their range), still well above the B3 cut |
| fhwrjtet | G1612 | +5.9 | +18.3 | Batch_3 | high | pure-B3 parent, inside sibling range |
| fspqbkxl | G2148 | −11.2 | +0.7 | Batch_1 | high | below B1 cut on both detectors; darker than the B1 sibling |
| 4hq27w4c | G2148 | −10.3 | −1.6 | Batch_1 | high | below B1 cut on both detectors; darker than the B1 sibling |
| soo2ax3r | G2156 | +4.1 | +10.6 | Batch_3 | low | above B3 cut, but both siblings (B1, B2) are too; sibling-anchored says B2 |
| y59rxmxl | G1880 | +0.1 | +5.1 | Batch_2 | low | between cuts on both detectors; its B1 sibling is 0.2 DN away |

Use these as a one-feature cross-check against the menu ensemble (`outputs/menu/`), not as the submission:
LOPO 0.65–0.68 on 34 sites from 13 parents, and the rule is known to ride on the Batch 3 instrument offset.
