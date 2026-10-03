# Stretch KPIs S01–S04 (catalogue v2): implemented, and what they do *not* add

The catalogue listed four v2 "stretch" KPIs for cluster connectivity and crowding at every scale
(`docs/kpis/README.md` §4.5). They are now implemented in `pmdb/kpis/stretch.py` on the same
`KpiContext` / `v0r1` masks as v1, registered in `pmdb.kpis.STRETCH_REGISTRY` (not `REGISTRY`, so the v1 tables are untouched), and produced separately by
`python scripts/run_stretch.py` (60 s for 31 sites, `--heldout`, `--perturb ±0.05`) into
`outputs/stretch/`; `python scripts/analyse_stretch.py` compares them with v1, the F columns and
the batch labels. `compute_site_kpis` still runs only the locked v1 rows. Tests:
`tests/test_kpis_stretch.py` (known answers on bars, lattices and stripes).

| ID | Columns | Definition (Si mask only) |
|---|---|---|
| S01 | `S01_c2_length_{x,z}_um`, `S01_c2_anisotropy` | two-point cluster function C2(r) (same 8-connected object), correlation length ∫C2(r)/C2(0) dr to 10 µm, per direction |
| S02 | `S02_euler_merge_radius_um`, `S02_euler_0` | dilation radius at which the Euler number of the Si mask first halves; undilated Euler number |
| S03 | `S03_h0_life_{p50,iqr}_um`, `S03_h0_inradius_p50_um`, `S03_h0_censored_frac` | H0 persistence of the signed-distance filtration (birth = −inradius, death = merge into an older object, elder rule; deaths quantised to 1 px, censored at 5 µm) |
| S04 | `S04_si_beta`, `S04_si_streak_angle_deg` | Minkowski tensor W1^{0,2} of the Si boundary: β = λmin/λmax (1 isotropic) and streak direction (0° = x, 90° = z) |

## Results on the 31 labelled sites (`outputs/stretch/batch_comparison.csv`, `redundancy.csv`, `loso.json`)

* **No batch separation.** Kruskal–Wallis p = 0.13–0.88 for every column; the largest effect is the
  C2 anisotropy (B2 more x-elongated, Cliff's δ = −0.55 vs B1, p = 0.13). Si streaks are mildly
  anisotropic everywhere (β ≈ 0.82–0.86) and roughly in-plane.
* **Mostly restatements of v1.** Inradius median = K03 size (Spearman 0.97), undilated Euler
  number = K02 density (0.95), H0 lifetime median = K11 Lacey (0.89), lifetime IQR = K02 (−0.86),
  C2 lengths = K04 agglomeration (0.78–0.81; partial ρ given K01 still 0.72–0.76). Only S04 β
  (|ρ| ≤ 0.41 with any v1 KPI) and the Euler merge radius (0.66 with K05 local-CV) carry much
  independent information, and neither differs by batch.
* **They hurt the fingerprint.** Alone: leave-one-site-out accuracy 0.26 (majority 0.55,
  permutation p = 0.85). Added to the 16 fingerprint features: 0.39 (p = 0.55) versus 0.68 for
  the fingerprint alone — eight non-informative features dilute the weight-free naive-Bayes vote.
  The stretch KPIs must therefore **not** be appended to the fingerprint feature set.
* **Robustness.** Site rankings are stable under ±0.05 segmenter-anchor shifts for S01 (Spearman
  ≥ 0.98), S03 lifetimes (0.84–0.94) and S04 β (0.97); the Euler merge radius (0.76) and streak
  angle (0.81) are less so.
* Held-out, S-only model (read-only, not for assignment): 3e122cbj → Batch 1, fn0mhxef and
  xrv9xvzb → Batch 3 at confidence 0.125.

## Reading
The v1 catalogue already captures the multi-scale clustering content of these sections: when
object size, density, agglomeration and the Lacey/scale-of-segregation curve are logged, C2
lengths, dilation-merge topology and H0 persistence are re-expressions of the same facts. This is
the expected outcome for a dispersed-particle microstructure at 3–16 % Si (Torquato's C2 and the
lineal-path function diverge from S2-type statistics only when clusters percolate, which Si never
does here). The honest value of S01–S04 is as a *closure* of the clustering family: the spec's
open items are done, measured, and shown to be redundant, so attention can go to more fields
(see `docs/functional.md` §2.7) rather than more clustering metrics.
