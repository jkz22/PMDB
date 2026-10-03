# KPI ownership across lanes

Both KPI lanes measure some of the same physical quantities through
different segmentations: the materials lane (`mat_`, multi-Otsu on
normalised full-resolution BSE, `src/`) and the geometric lane's
diagnostics (`D01–D06`, MAD-threshold segmentation on un-normalised
half-resolution BSE, `pmdb/`). This document records which value is
canonical where, so the joint verdict never counts the same physical
signal twice.

## Rule

**The verdict vector takes exactly one KPI per physical quantity.**
The tiering already in the catalogue decides which: `mat_` KPIs are
verdict-tier materials quantities; `D01–D06` are diagnostic-tier
(process attribution) and never enter the verdict vector. Both
implementations are kept — the redundancy is used as a cross-lane QC
check (below), not deleted.

No locked decision from specs 001/002 is touched by this rule.

## Overlapping quantities

| Physical quantity | Verdict (canonical) | Counterpart (not in verdict) | Comparable as |
|---|---|---|---|
| Porosity | `mat_porosity` | `D04_porosity_mean` | fraction vs fraction (direct) |
| Graphite size | `mat_graphite_d50_um` | `D03_graphite_ecd_d50_um` | µm vs µm; methods differ (watershed instances vs connected components), rank agreement expected, not equality |
| Graphite alignment | `mat_orientation_anisotropy` | `D02_graphite_orient_circsd_deg` | inversely related (order parameter up ⇔ circular SD down). Pending: needs the geometry module's `instance_orientations` |
| Si / bright loading | `K01_si_frac_adm` | `mat_bright_fraction` | fraction vs fraction; denominators differ (admissible vs whole frame) |

**The one tier-rule exception is Si/bright loading**: both lanes carry
it at verdict tier. The geometric lane owns it (`K01`) — its segmenter
is tuned for the Si tail, its denominator excludes artefacts, and every
K-KPI keys off K01, so demoting it would orphan the lane. In the joint
verdict `mat_bright_fraction` is therefore demoted to diagnostic.

`mat_active_fraction` has no usable counterpart: it is nearly constant
by construction (within-batch CV ≈ 0.01–0.02), so rank agreement with
`D01_graphite_frac_mean` is noise-dominated and is not checked.

Quantities without a counterpart stay owned by their lane unchanged
(e.g. `mat_crack_fraction`, `mat_rim_coverage`, `D05`, `D06`, remaining
K/A KPIs).

## Cross-lane QC check

`scripts/check_cross_lane.py` joins the two lanes per site and reports
Spearman rank correlation and median ratio for each pair above. The two
lanes segment independently, so sustained disagreement (correlation
below the floor in the script) is a segmentation-drift alarm that
neither lane can raise alone. The check is report-first: it exits
non-zero only when a pair falls below its floor, and the floors are
provisional until enough runs exist to calibrate them.

## Status

Proposed by the materials lane; needs sign-off from the geometric lane
(review of the PR introducing this file counts). The joint verdict
construction in spec 003 must cite this document.
