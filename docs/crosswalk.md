# Cross-session crosswalk

Two things that only exist once the sessions are put next to each other (`scripts/crosswalk.py` →
`outputs/crosswalk/`).

## 1. Every held-out call, side by side (`heldout_consensus.csv`)

| method (session) | 3e122cbj | fn0mhxef | xrv9xvzb |
|---|---|---|---|
| fingerprint, 16 curve features, conformal NB (fingerprint) | **Batch 1** (conf 0.00, p_B3 1.00) | Batch 3 (0.13) | **Batch 2** (0.50) |
| two-stage RF, KPI arm — the selected arm (classifier/FEM) | Batch 1 (0.85) | Batch 3 (0.56) | Batch 3 (0.59) |
| two-stage RF, FEM arm | Batch 1 (0.82) | Batch 3 (0.61) | Batch 2 (0.45) |
| two-stage RF, KPI+FEM arm | Batch 1 (0.84) | Batch 3 (0.61) | Batch 2 (0.40) |
| fingerprint + stretch S01–S04 (functional-morphology) | Batch 1 (0.63) | Batch 3 (0.13) | Batch 3 (0.13) |
| fingerprint + functional F (functional-morphology) | Batch 1 (0.88) | Batch 2 (0.00) | Batch 3 (0.00) |
| joint functional score vs Batch 3, one-class (functional-morphology) | outside B3 (100th pct) | inside (29th) | inside (35th) |

Reading:

- **3e122cbj — unanimous Batch 1**, and the only field any method puts *outside* the Batch 3 baseline.
  Composition (K01 0.139 vs B3 0.062 ± 0.010, z ≈ +8), Si density (z ≈ +8) and swelling behaviour
  (constrained share 0.78, pore loss 0.23) all say so; the fingerprint's own typicality says it is also
  consistent with Batch 3 (conformal p 1.00) — arrangement is unremarkable, composition is not.
- **fn0mhxef — Batch 3 in six of seven**; the one dissent (fingerprint + functional, confidence 0.00) is
  the appended-feature variant that was already rejected in `docs/functional.md` §2.4 as not improving the
  fingerprint. Inside the Batch 3 cloud on every one-class family. Confidence is moderate everywhere.
- **xrv9xvzb — a 3 : 3 split, and it is the informative one.** Methods that read *depth arrangement*
  (fingerprint; RF arms that include the FEM curve features) say Batch 2; methods that read *scalar
  composition/geometry* (RF KPI arm, fingerprint + stretch, fingerprint + functional, every one-class
  score) say Batch 3 / inside Batch 3. That is exactly the storyline of `docs/story.md`: composition does
  not separate these batches, arrangement does. If the arrangement signal is real (permutation p 0.0025
  on the labelled set) the Batch 2 call stands; the scalar methods are not contradicting it, they simply
  cannot see it. The elevated black level on this site (`docs/presentation-strategy.md`) is an
  acquisition setting and is not used by any of the methods above.

## 2. The 13-second geometric swelling test tracks the FEM (`geometric_vs_fem.csv`, `figures/geometric_vs_fem.png`)

Same 34 sites (31 + 3 held-out), FEM production run at s = 1, `sym` orientation
(`outputs/fem/site_curves.csv`) against the mask-growth test of `pmdb.functional` at SOC 1:

| geometric (`pmdb.functional`) | FEM (`outputs/fem`) | Spearman | partial, given K01 |
|---|---|---|---|
| pore loss | pore cells closed | **+0.73** (p 1e-6) | +0.49 (p 0.003) |
| pore loss | relative porosity change | −0.61 (p 1.5e-4) | −0.45 (p 0.007) |
| K01 Si fraction | surface swelling | +0.91 (p 2e-13) | — |
| constrained share | surface roughness | −0.36 (p 0.04) | — |
| constrained share | porosity change | +0.27 (n.s.) | −0.02 (n.s.) |

Reading:

- FEM **surface swelling is the Si fraction** (ρ 0.91) — expected for a linear eigenstrain model, and a
  useful sanity check that both pipelines segment the same Si.
- FEM **pore closure is predicted by the geometric pore-loss number** at ρ 0.73, and the agreement is not
  just "more Si → more closure": after removing K01 from both, ρ is still 0.49. The two methods share the
  masks but nothing else (the FEM solves linear elasticity with soft pores and a clamped base; the
  geometric test grows Si isotropically and counts overlaps), so this is a genuine cross-validation of
  the cheap test as a surrogate for the expensive one on *pore-closure risk*. The two high-Si Batch 1
  fields and 3e122cbj are the top-right corner in both.
- The **constrained share** has no FEM counterpart in this run: the only correlate is a weak negative one
  with surface roughness (Si pressed against graphite ⇒ a smoother swollen surface), and the FEM's stress
  outputs are saturated (`docs/fem/results.md`, Limitations), so the mechanics reading of "constrained
  swelling" remains a prediction the FEM cannot yet test. A finite-strain run would be the test.

Practical consequence: pore-closure ranking can be produced in seconds per field with
`scripts/run_functional.py` and agrees with the FEM at ρ ≈ 0.7; keep the FEM for what it adds (surface
rise, roughness, curves with SOC), not for the ranking.

## 3. The lithiation consequence follows the depth fingerprint (`scripts/run_depth_swelling.py`, `outputs/functional/depth_swelling.csv`, `figures/depth_swelling.png`)

The fingerprint's result is that the batches differ in *where* the Si sits through the coating (Batch 2
top-heavy with a depleted mid-depth, Batch 1 bottom-heavy and variable, Batch 3 flat). Running the SOC-1
swelling test in five through-thickness bands (free surface → collector) on every field:

| | Batch 1 | Batch 2 | Batch 3 |
|---|---|---|---|
| pore loss, band 0 (surface) → 4 (collector), medians | 0.12 / 0.15 / 0.13 / 0.11 / **0.21** | 0.14 / 0.09 / **0.06** / 0.09 / 0.13 | 0.10 / 0.11 / 0.12 / 0.07 / 0.10 |
| mid-depth dip of pore loss (band 2 − mean of bands 0, 4) | −0.07 | **−0.09** | +0.01 |
| constrained share, bands 0–4 | 0.85–0.88 | 0.86–0.87 | 0.86–0.89 |

- Pore loss differs by batch at the surface (KW p 0.045), mid-depth (p 0.026) and collector (p 0.019)
  bands, and the **mid-depth dip separates the batches (KW p 0.003; Batch 2 vs rest p 0.019)**: the Si
  depletion the fingerprint reads at mid-depth in Batch 2 becomes a mid-depth *pore-loss minimum* on
  lithiation, while Batch 3 is flat in both. Within a field, pore loss tracks the local Si fraction band
  by band (median Spearman 0.70, positive in 30/31 fields): the depth arrangement is the depth profile of
  pore-closure risk.
- The constrained share is depth-independent in every batch (0.85–0.89 everywhere): "Si pressed against
  graphite" is a property of how the particles sit, not of where in the coating they are.
- Held-out: **xrv9xvzb** shows the Batch 2 pattern — pore loss 0.18 at the surface falling to 0.03–0.06
  at bands 2–3 — which is the swelling-side reading of the fingerprint's Batch 2 call (`§1`);
  **fn0mhxef** is irregular (0.19 / 0.07 / 0.12 / 0.03 / 0.16) and **3e122cbj** is high in every band
  (0.12–0.30), the bottom-heavy high-Si Batch 1 profile.

So the chain is closed in one direction: artefact → arrangement (fingerprint) → where the pores will
close (this section) → FEM pore closure (§2). The depth-resolved pore loss is not a new discriminator
(it inherits the fingerprint's), but it is what the fingerprint *means* for the electrode.
