# FEM literature synthesis: 2D finite-strain lithiation mechanics of Si/graphite SEM cross-sections

## Answer
The literature backs the overall approach: image-based multiphase models, finite-strain F = Fe·Fλ kinematics, quasi-static uniform lithiation within each phase, and a compressive stack pressure of 0.1-1 MPa [R3:E2][R3:E7][R3:E8][R3:E10][R3:E12]. Four of the proposed defaults need revising.
- **Graphite strain:** it follows staging rather than rising linearly with SOC. The c-axis strain is about +5.5% by stage II (about 25% graphite lithiation), stays flat until about 50%, then reaches about +10% at LiC6 [V2][V4].
- **SOC split:** one shared SOC for all phases is wrong for blends. Si starts lithiating first, and graphite only starts below 0.2 V [V2].
- **Meaning of 100% SOC for Si:** no source gives x in LixSi at 100% SOC for a commercial cell, and commercial Si is described as "partially lithiated" (tool summary only) [R1:E25]. Si utilisation should therefore be a swept parameter, not fixed at Li15Si4.
- **Graphite elastic constants:** the claimed conflict is a transcription error. In Qi 2014 Table II, graphite C11 is 1105 GPa, not 110; LiC6 C11 is 989 GPa; and the isotropic Reuss E rises from 32 to 109 GPa [V1].

Two inputs have only tool-summary evidence and should be swept widely: binder E and the SiOx expansion. The pore stiffness, plane-strain choice and pore-closure criterion have no literature support at all; they are numerical choices.

## Sources
- R1: /Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/reports/fem-lit.explorer-si.md — Si/LixSi/SiOx expansion, moduli, yield, commercial utilisation, fracture size
- R2: /Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/reports/fem-lit.explorer-gr-binder.md — graphite strain/elastic constants, binder E/ν, Si-vs-Gr lithiation sequence, electrode swelling
- R3: /Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/reports/fem-lit.explorer-method.md — image-based FE precedent, finite-strain formulations, BCs and stack pressure, uniform SOC, 2D vs 3D, FEniCSx
- V1-V5: my own reads of the local PDFs that R2 lists under "Not checked" (Qi 2014; Yao 2019; Nadimpalli/arXiv 1511.02445 in two reads; Tardif/arXiv 2005.04983).

## Parameter table
Evidence strength:
- **verbatim**: read from the primary text, either by me (V) or quoted by an explorer from a fetched page.
- **tool-summary**: a WebSearch/WebFetch paraphrase only.
- **unsupported**: no source; a modelling choice.
- **derived**: arithmetic on cited values (the step is shown in the Claims section).

| Assumption / parameter | Chosen default | Sweep range | Source(s) | Evidence strength |
|---|---|---|---|---|
| Kinematics | Finite strain, F = Fe·Fλ, Fλ applied per phase | — (linear elasticity is not valid for Si) | R3:E7 (needs ηc_max ≪ 1 for linear), R3:E8, R3:E2, R1:E21 | verbatim (R3:E7, E8); tool-summary (R1:E21) |
| Hyperelastic law | Compressive neo-Hookean ψ = μ/2(I_C−d) − μ lnJ + λ/2 (lnJ)² (use d = 2 for the 2D form) | Optional St Venant-Kirchhoff check | R3:E17, R3:E18 | verbatim (FEniCSx docs); no lithiation-specific comparison found |
| Si full-lithiation volume change | +280% (c-Si to Li15Si4); +263% is the a-Si DFT value | 160% (SiOx) to 300% | V1 (Qi Table I: Si→Li3.75Si 263%), R1:E2, R1:E3, R3:E7 ("four times") | verbatim (V1, R3:E7); tool-summary (R1:E2, E3) |
| Si eigenstretch versus utilisation u | λ_Si = (1 + β·u)^(1/3), β = 2.8, u = fraction of Li15Si4 capacity used; volume linear in x | β 1.6-3.0 | R1:E2 ("roughly linear with Li content") | tool-summary for linearity; derived form |
| Si utilisation at 100% cell SOC | u = 0.75 (x ≈ 2.8) | 0.5-1.0 | R1:E25 ("partially lithiated"), R1:E26 (72-80% utilisation, source unidentified), V2 (Li15Si4 forms below 0.05 V) | tool-summary (E25, E26); verbatim (V2) |
| Si E (lithiation-dependent) | a-Si 96 GPa → Li3.75Si 41 GPa, linear rule of mixtures in x | 41-160 GPa (160 is c-Si, not retrieved) | V1 (Qi Table II citing Shenoy 2010; text says "E follows the linear rule of mixtures"), R1:E12 (Berla 41 GPa), R1:E11 (biaxial 70→35 GPa) | verbatim (V1, R1:E11); tool-summary (R1:E12) |
| Si ν | 0.29 → 0.25 | 0.21-0.29 | V1 (Qi Table II), R1:E15 (0.21/0.23), R1:E14 (0.25) | verbatim (V1); tool-summary (others) |
| Si yield / flow stress | None in v1; flag elements with von Mises above 1 GPa | 0.5-1.5 GPa (for v2) | R1:E19, R1:E22, R1:E20 | tool-summary (1 GPa value); verbatim (Bucci abstract: flow stress depends on c) |
| SiOx alternative for the Si phase | Run as a scenario: +160% volume, same E as Si (no SiOx E found) | 118-200% | R1:E6 | tool-summary only; SiOx modulus unsupported |
| Graphite c-axis strain ε_zz(y), y = graphite lithiation fraction | Piecewise linear: 0 at y=0; 0.055 at y=0.25; 0.055 at y=0.50; 0.103 at y=1 | End point 0.09-0.11 | V2 (d = 3.54 Å held from about 24% to 50%; 3.70 Å LiC6 above 50%), V4 (d0 = 3.355 Å; stage 1 "dilated by about 10% along the c axis"; stage 2 d/d0 = 1.05, stage 3 1.03), R2:E1 | verbatim (V2, V4); tool-summary (R2:E1) |
| Graphite a-axis strain ε_xx | 0.01 at y = 1, scaled in proportion to the volume change | 0-0.013 | Derived from R2:E1 (13.2% volume) and V4 (10.3% c-axis); V1 Table I gives 10% volume | derived; inputs are tool-summary plus verbatim |
| Graphite c-axis orientation | c ∥ z (image rows) | Isotropic-strain control run; per-particle orientation from segmented shape | None in evidence | unsupported |
| Graphite elastic (isotropic default) | E 32 → 109 GPa, ν 0.32 → 0.24 (Reuss average, linear in y) | Voigt or transversely isotropic Cij | V1 (Qi Table II) | verbatim |
| Graphite Cij (if anisotropic) | Graphite: C11 1105, C12 204, C13 −2.5, C33 30.9, C44 5.6, C66 450 GPa. LiC6: C11 989, C12 171, C13 2.1, C33 78.0, C44 21.9, C66 409 GPa | — | V1 (Qi Table II, DFT-LDA, from Qi et al. 2010 ref 20) | verbatim |
| Binder + carbon black ("unassigned solid") E | 0.5 GPa | 0.05-2 GPa | R2:E5 (CMC/SBR ≈ 0.7 GPa; PVDF 993 MPa dry; wet ≈ one fifth of dry) | tool-summary only |
| Binder ν | 0.34 | 0.3-0.45 | R2:E5 (PVDF 0.34) | tool-summary only |
| Binder lithiation strain | 0 | — | Not addressed. Electrolyte swelling of binder does occur but before cycling (V3: "swelling of binder", −1 to −1.25 MPa) | unsupported for lithiation; verbatim for wetting |
| Pore / artefact | Ersatz material, E_pore = 1e-4 × E_binder, ν = 0.3 | 1e-6 to 1e-2 × E_binder | None (R3 Not found) | unsupported |
| Out-of-plane condition | Plane strain | Generalised plane strain | Indirect: coating bonded to foil gives in-plane constraint (V3: "A state of equi-biaxial stress is induced in a film … when the latter [substrate] constrains the in-plane volume change"). No electrode-FE source | unsupported directly; inferred |
| Bottom/top BC | u_z = 0 on one horizontal edge; run both orientations | — | None (foil edge unknown) | unsupported |
| Lateral BC | u_x = 0 on both lateral edges | Periodic | Same in-plane constraint argument (V3) | inferred |
| Free surface | Traction-free (separator side) | Stack pressure 0.1-1 MPa | R3:E10 ("typically in the range of 0.1-1 MPa"); V3 (about 1 MPa peak on the casing of a jelly-roll) | verbatim |
| SOC field | Uniform within each phase, but with separate Si and graphite fractions | Proportional versus Si-first split | R3:E12 (quasi-static, uniform μ), V2 (Si first; 58/42 split from 0.2 to 0.01 V), V4 (through-thickness heterogeneity at C/5) | verbatim |
| SOC steps | 11 steps (0-100% in 10% steps) with Newton substeps where needed | Refine near the graphite stage II→I plateau | R3:E17, R3:E18, S4 snippet | verbatim (solvers); tool-summary (load stepping) |
| Resolution | 100 nm/px | 50 nm/px convergence check | None | unsupported |
| Pore-closure stop | Flag when J of a pore element falls below a threshold or the mesh inverts | — | None | unsupported |
| Si fracture interpretation | Particles above about 150 nm are fracture-prone (crystalline Si, first lithiation) | — | R1:E27 | tool-summary |

## Claims

### Kinematics and constitutive choice
- **C1** (CONFIRMED) Finite-strain kinematics are needed for Si. Linearised elasticity is valid only when ηc_max ≪ 1, and lithiated Si expands to about four times its volume. [R3:E7][R3:E8]
- **C2** (CONFIRMED) Precedent for the approach exists: a microstructure-resolved, 3-phase model (active material, CBD, pore) built from an SEM image of a Si composite electrode, with finite strain and viscoplasticity. [R3:E2][R3:E3] Note that it used a 2D axisymmetric geometry, not plane strain. [R3:E2]
- **C3** (CONFIRMED) A compressible neo-Hookean energy with Newton/SNES solvers is the standard FEniCSx hyperelastic template. Neither demo includes eigenstrain or load stepping. [R3:E17][R3:E18]

### Si expansion and stiffness
- **C4** (CONFIRMED) The Si full-lithiation volume change is 263-300%, depending on definition. Qi Table I gives 263% for Si→Li3.75Si (amorphous, GGA) [V1]. Beaulieu gives 280% for c-Si→c-Li3.75Si [R1:E2, tool-summary]. Meca gives "about 300%" and "about 280%" [R1:E3]. The 370% value is likely Li22Si5 and is not relevant at room temperature [R1:E9][R1:E4].
- **C5** (INFERRED) Linear stretch at full lithiation is 54% for 263%, 56% for 280% and 59% for 300% (cube root of 1 + ΔV). The proposed 59% therefore sits at the top of the range.
  - Volume is reported as roughly linear in Li content [R1:E2, tool-summary only]. So Fλ should be built as J_λ = 1 + β·u with λ = J_λ^(1/3), not as a stretch that is linear in SOC.
  - At u = 0.75 and β = 2.8 the linear strain is about 46%.
- **C6** (CONFIRMED) The amorphous LixSi modulus falls from 96 GPa (ν 0.29) to 41 GPa (ν 0.25) at Li3.75Si. These are Qi Table II values citing Shenoy 2010, and the text states "E follows the linear rule of mixtures of pure amorphous Si and BCC Li". [V1] The nanoindentation value of 41 GPa for fully lithiated Si (Berla) agrees. [R1:E12, tool-summary]
- **C7** (CONFIRMED) Qi notes that crystalline Si "undergoes solid-state amorphization" on first lithiation and treats only amorphous LixSi. [V1, p. F3013] The 160 GPa c-Si value was never retrieved [R1:E16]. Amorphous values are therefore the appropriate default for a cycled electrode.
- **C8** (INFERRED) Without plasticity, constrained Si eigenstrain of order 50% with E ≈ 40-96 GPa will produce stresses of tens of GPa. That is far above the reported flow stress of about 1 GPa [R1:E19, tool-summary]. The reasoning step is order-of-magnitude σ ~ E·ε. So v1 stress magnitudes in Si are not physical. Use them as relative features only, or flag elements above 1 GPa. Two sources treat lithiated Si as elastic-plastic or viscoplastic [R1:E21, tool-summary][R3:E2, verbatim].

### Si utilisation and the Si/graphite split
- **C9** (CONFIRMED) No source states x in LixSi at 100% cell SOC [R1:E26 note][R1 Not found]. The only commercial statement is "partially lithiated and hence amorphous" for the LG M50 [R1:E25, tool-summary].
  - Li15Si4 crystallises only below about 0.05 V. [V2: "crystallization of the amorphous Si alloy to Li15Si4 at voltages below ~0.05 V_Li during lithiation"]
  - One unidentified source estimates 72-80% Si utilisation [R1:E26, tool-summary].
- **C10** (CONFIRMED) In a 15 wt% Si/graphite blend (half cell, about C/30), "The lithiation begins with Li alloying with Si; lithiation of Gr occurs at later stages when the potential dips below 0.2 V." [V2] Also: "In the 0.2–0.01 V range, the relative lithiation of Si and Gr is ~58% and 42%, respectively." [V2]
- **C11** (INFERRED) A single SOC applied identically to both phases is not supported. Si should reach a larger fraction of its own capacity earlier than graphite does. This follows from C10. No source in the evidence gives the full per-component fraction-versus-capacity curve.

### Graphite strain
- **C12** (CONFIRMED) Graphite c-axis expansion is about 10% at LiC6. Tardif: "stage 1 lithiation, with the graphite dilated by about 10% along the c axis", d0 = 3.355 Å [V4]. Yao: LiC6 peak at 3.70 Å [V2]. Qi Table I lists 10% volume for C→LiC6 [V1].
- **C13** (CONFIRMED) Graphite strain follows staging and is not linear in graphite SOC. In Yao's graphite electrode [V2, pp. 6-8]:
  - d-spacing rises from about 3.36 to 3.54 Å before G2 (about 24% capacity).
  - "Between the G2 and G3 points, a Bragg peak centered at 3.54 Å is observed; the position of this peak does not change".
  - Above about 50%, LiC12 and LiC6 (3.70 Å) coexist.

  Tardif gives stage 2 d/d0 = 1.05 and stage 3 d/d0 = 1.03 [V4]. Schweidler splits the 13.2% total volume into about 5.9% in the dilute stages and about 7.3% in the stage 2→1 transition [R2:E1, tool-summary]. Both agree with Yao.
- **C14** (INFERRED) A piecewise-linear ε_zz(y) of (0, 0), (0.25, 0.055), (0.50, 0.055), (1.0, 0.103) reproduces C13.
  - 3.54/3.355 − 1 = 0.055 and 3.70/3.355 − 1 = 0.103.
  - Linear interpolation between 0.5 and 1 assumes the strain is phase-fraction weighted across the two-phase LiC12/LiC6 region.
- **C15** (INFERRED) The in-plane a-axis strain is about 1.3%: (1.132/1.103)^(1/2) − 1, using the 13.2% volume [R2:E1, tool-summary] and the 10.3% c-axis value [V4]. The proposed 0.01 is reasonable. If Qi's 10% volume is used instead [V1], ε_xx ≈ 0, so the plausible range is 0-0.013.

### Graphite and binder elastic properties
- **C16** (CONFIRMED) Graphite elastic constants come from Qi Table II (LDA values from Qi et al. 2010 [ref 20]) [V1]:
  - Graphite: C11 1105, C12 204, C13 −2.5, C33 30.9, C44 5.6, C66 450 GPa; E_R = 32 GPa, ν_R = 0.32.
  - LiC6: C11 989, C12 171, C13 2.1, C33 78.0, C44 21.9, C66 409 GPa; E_R = 109 GPa, ν_R = 0.24.

  The "C11 = 110 GPa" in R2:E3 is a tool truncation of 1105. 1105 is consistent with the textbook value of about 1060 GPa, and the 989 GPa LiC6 value is correct.
- **C17** (INFERRED) For isotropic neo-Hookean phases, use the Reuss E/ν (32 → 109 GPa; 0.32 → 0.24), linear in graphite lithiation.
  - This assumes graphite stiffening is linear in Li content. Qi cites the 2010 "triples" result [V1, p. F3010], and R2:E3 cites "linear relationship" (tool-summary).
  - Reuss is a lower bound. Qi says it is the appropriate scheme for anisotropic layered materials [V1, p. F3016].
- **C18** (CONFIRMED, tool-summary only) Binder moduli are about 0.7 GPa for CMC/SBR and 993 MPa for dry PVDF. Wet PVDF is about one fifth of dry, so roughly 0.2 GPa. PVDF ν is 0.34. [R2:E5] None of these quotes is tied to an identified primary source [R2:E5]. The unassigned phase also contains carbon black, which no source in the evidence characterises.

### Boundary conditions and validation
- **C19** (CONFIRMED) Stack pressure is "typically in the range of 0.1-1 MPa" [R3:E10]. A modelled jelly-roll casing pressure peaks at about 1 MPa [V3, p. 8].
- **C20** (INFERRED) A 1 MPa stack pressure is negligible beside GPa-level eigenstress in Si and graphite (compare C8, and the −10 MPa coating-average stress in C21). It will not change the features materially, so a single sensitivity run is enough.
- **C21** (CONFIRMED) A substrate-bonded graphite/PVdF anode coating "is subjected to compressive stress which increases with capacity and reaches a peak value of – 10 MPa at the end of lithiation" (half cell, C/20). Plateaus in the stress curve correspond to staging. [V3, p. 6] This is a validation target for the thickness-averaged σ_xx of a graphite-rich region under u_x = 0.
- **C22** (CONFIRMED, tool-summary tabulation) Electrode thickness expansion at full lithiation is about 19% for graphite only, 39% at 15 wt% Si, 84% at 30 wt%, about 300% at 70 wt% and 520% at 80 wt% Si [R2:E8]. The thickness basis and formulation were not retrieved [R2:E8]. Moyassari confirms the trend qualitatively: "A higher silicon content led to a higher thickness change." [R2:E7]
- **C23** (CONFIRMED, tool-summary) Commercial Si contents are low: "up to 5 wt%" [R1:E24], about 3.5 wt% in the MJ1, 2.43 wt% in the M50T, and Si capacity fractions of 13-15% [R1:E25]. Cell-level expansion stays below 7.5% [R1:E8].

### Uniform SOC
- **C24** (CONFIRMED) Uniform lithiation is an idealisation that holds only at slow rates.
  - Roper justifies it by a time-scale argument (quasi-static; chemical potential uniform within each material) [R3:E12].
  - Tardif measured alternating homogeneous and heterogeneous Li distributions through an 80 µm graphite electrode at about C/5 [V4, pp. 1-3, 8].
  - Nadimpalli also assumes uniform lithiation and calls the result a first-order estimate [V3, p. 8].

## Verdicts on the proposed defaults
1. **Si isotropic, about 59% linear (≈300% volume) at full lithiation: REVISE.**
   - Isotropy is acceptable for amorphous LixSi [R1:E1 verbatim: "amorphous phases undergo reversible shape and volume changes … homogeneous expansion"].
   - The magnitude should be 263-280% volume (54-56% linear) for full Li3.75Si (C4, C5).
   - Build the eigenstrain as J linear in utilisation (C5).
   - Do not equate 100% cell SOC with Li15Si4. No source gives x (C9). Make u_Si a swept parameter, default 0.75 and range 0.5-1.0, and record it as a feature metadata dimension.
   - **Si vs SiOx:** BSE alone cannot separate Si from SiOx. Run a SiOx scenario with 160% volume [R1:E6, tool-summary]. For SiOx stiffness, reuse the Si E as a placeholder because no SiOx E was found [R1:E17]. Mark this as unsupported.
2. **Graphite anisotropic, ε_zz ≈ 0.10, ε_xx ≈ 0.01: SUPPORTED for the end-point, REVISE the SOC path.**
   - Use the staging-shaped ε_zz(y) from C14 rather than a linear ramp, scaling ε_xx with it.
   - The assumption that c ∥ z (flakes lying in the coating plane) is UNSUPPORTED by any evidence. Add a control run with isotropic graphite strain ((1.132)^(1/3) ≈ 4.2% linear). Also consider orientation per particle from the long axis of each segmented particle.
3. **Uniform SOC everywhere: REVISE.** Keep strain uniform within each phase, but drive Si and graphite with separate fractions (C10, C11). Bracket two scenarios:
   - (a) proportional: both phases at the cell SOC fraction;
   - (b) Si-first: Si reaches u_Si·SOC_Si(SOC) earlier and graphite lags. Shape this from Yao's per-component curve once it is gathered (see Gather next).
4. **Binder soft, no expansion: SUPPORTED with caveats.** The 0.2-1 GPa range is tool-summary only (C18). Default 0.5 GPa, sweep 0.05-2 GPa. Zero lithiation strain is a reasonable assumption but has no source. Electrolyte swelling of binder happens before cycling and should be excluded from the lithiation increments [V3].
5. **Pores as ersatz soft material: UNSUPPORTED (numerical choice).** No stiffness ratio was found in the battery literature [R3 Not found]. Run a sensitivity sweep (1e-6 to 1e-2 × E_binder) and check that features do not depend on it.
6. **Plane strain: UNSUPPORTED directly, plausible by inference.** The foil imposes in-plane (x, y) constraint on a bonded coating [V3]. Plane strain (ε_yy = 0) is consistent with that. The one image-based precedent used axisymmetric geometry [R3:E2], and no quantitative 2D-vs-3D mechanics error was found [R3:E16].
7. **u_z = 0 on one horizontal edge, both orientations, opposite edge traction-free, lateral u_x = 0: SUPPORTED as a modelling choice, not by literature.** The lateral u_x = 0 matches the substrate-constrained (biaxial) stress state measured in coatings [V3]. Running both orientations is the right response to the unknown foil side.
8. **Optional 1 MPa stack pressure: SUPPORTED** [R3:E10][V3]. It is mechanically negligible (C20).
9. **Compressible neo-Hookean: SUPPORTED as a template** [R3:E17][R3:E18]. No lithiation-specific comparison with other laws exists [R3 Not found].
10. **No plasticity in v1: ACCEPT for v1, but flag.** Stress magnitudes in Si will be unphysical (C8). Report the elements above 1 GPa and plan elastic-perfectly-plastic Si (about 1 GPa) for v2 [R1:E19][R1:E21][R3:E2].
11. **Stop or flag at pore closure: UNSUPPORTED (numerical).** Reasonable as a guard. Define it as det F of the pore element below about 0.05, or as element inversion.
12. **Small-strain validity: REJECTED for Si, BORDERLINE for graphite** (C1). Finite strain is required.
13. **Coarsening to 100 nm/px: UNSUPPORTED.** No evidence either way. Si particles are a few µm, but thin binder and carbon-black bridges may be lost. Run one 50 nm/px convergence check.

## Validation targets
- **V-T1 Electrode thickness swelling versus Si wt% at full lithiation:** about 19% (graphite only), 39% (15 wt%), 84% (30 wt%) [R2:E8, tool-summary]. Increases monotonically with Si content [R2:E7]. Compare the predicted top-edge u_z / H at 100% SOC.
  - Expect the 2D model to under-predict the graphite-only value. 19% is well above the about 10% c-axis strain [V4], and dense phase strain alone cannot reach it (inferred).
- **V-T2 Average in-plane coating stress, graphite/PVdF, substrate-constrained:** compressive, reaching about −10 MPa at end of lithiation, with staging plateaus [V3]. Compare the area-averaged σ_xx over the domain under u_x = 0.
- **V-T3 Graphite c-axis strain path:** 3%, 5% and 10% at stages 3, 2 and 1 [V4], with a flat d-spacing between about 25% and 50% [V2].
- **V-T4 Cell-level expansion below 7.5%** for a Si-C cell despite Si expanding by more than 300% [R1:E8, tool-summary]. This is a weak sanity bound only.
- **V-T5 Interpretation, not validation:** a critical size of about 150 nm for crystalline Si fracture on first lithiation [R1:E27, tool-summary]. A flow stress of about 1 GPa [R1:E19, tool-summary].

## Contradictions
- **Graphite C11, 110 GPa [R2:E3] vs 1105 GPa [V1]: resolved.** Qi Table II prints 1105. The tool dropped a digit. Weight V1, which is the primary table.
- **Qi Table II labelling (internal).** For graphite, the row labelled B_H = 29 / G_H = 12 reproduces E_R = 32 and ν_R = 0.32 through eqs. [1]-[2]. B_R = 161 / G_R = 120 would give E ≈ 290 GPa. LiC6 behaves the same way: B_H = 69 / G_H = 44 gives E_R = 109 and ν_R = 0.24. [V1] (INFERRED: my arithmetic.) The H and R labels look swapped. The E_R/ν_R values used here are self-consistent with that row. Qi's text also gives graphite C33 as 33.9 GPa while the table gives 30.9 [V1, pp. F3014, F3016]. Use the table.
- **End-point modulus of lithiated Si: 20 GPa vs "<40" vs 41 GPa** [R1:E10 tool summaries][R1:E12][V1]. Weight 41 GPa: it is verbatim in V1 and agrees with Berla. R1:E14's "E ratio = 49" looks garbled. Ignore it.
- **Order of lithiation in blends: "simultaneous lithiation" (R2:E6, search summary) vs "lithiation begins with Li alloying with Si; lithiation of Gr occurs at later stages" (V2, verbatim, same paper).** Weight V2.
- **Graphite volume change: 10% (V1 Table I; R3:E7) vs 13.2% (R2:E1, tool-summary).** These are different measures (DFT/approximate versus operando XRD including a-axis expansion). This only affects ε_xx (0-0.013).
- **Si volume: 263% / 280% / 300% / 370% / "four times"** [V1][R1:E2][R1:E3][R1:E9][R3:E7]. The differences come from definition and phase (amorphous vs crystalline end phase; Li22Si5). Use 263-280% as the range for Li3.75Si.
- **Nadimpalli wetting stress (internal):** the text gives about −1 MPa coating stress (p. 4); the conclusions give "about −0.15 MPa" (p. 9), which appears to conflate it with the 0.15 MPa casing pressure [V3]. This is minor and does not affect the model.
- **Uniform SOC vs measured heterogeneity:** R3:E12 assumes it is valid; V4 measures through-thickness heterogeneity at C/5. Keep the uniform assumption but state that it applies to slow rates only.

## Gaps
- **The full Si-vs-graphite lithiation fraction curve versus cell capacity** is missing. Only the 58/42 split from 0.2 to 0.01 V was read (V2, pp. 1-8). This changes C11 and verdict 3.
- **Si lithiation state x at 100% SOC in commercial full cells** has no source. This changes C9 and verdict 1. It is mitigated by the u_Si sweep.
- **SiOx properties:** the expansion (160%) is tool-summary only, and the modulus is missing [R1:E6][R1:E17]. This affects the SiOx scenario.
- **Binder/CBD E is not tied to a primary source,** and carbon-black properties are missing [R2:E5]. This affects C18. Nadimpalli's Table 2 (which likely lists the PVdF E used) is not in pp. 1-16 of the local PDF [V3, V5].
- **There is no evidence for preferred c-axis alignment** of graphite flakes along z in coatings. This affects verdict 2.
- **Pore ersatz stiffness, pore-closure handling, plane strain vs alternatives, and the 2D-vs-3D error** were all explicitly not found [R3 Not found]. These remain numerical choices.
- **The basis of the dilatometry swelling values** is unknown: formulation, porosity, and whether the values are first-cycle or reversible [R2:E8]. This weakens V-T1.
- **Graphite strain at intermediate SOC** is read off Yao's figure rather than a tabulated source (C14 breakpoints are approximate).

## Gather next
- Read pages 9-end of /Users/Kevin/.claude/projects/-Users-Kevin-Documents-GitHub-PMDB--claude-worktrees-heldout-data/f7044de1-2c11-44c1-b109-e683c9c4f5ce/tool-results/webfetch-1791052225504-ewz1th.pdf (Yao 2019, Adv. Energy Mater.). Extract the figure or table giving Si and graphite fractional lithiation (or Li content x) versus electrode capacity or potential for the 15 wt% Si-Gr electrode, including the Si capacity reached at 0.01 V and the experimental section (rate, cut-offs). This settles the Si/Gr SOC split (C11, verdict 3).
- Read pages 17-end of /Users/Kevin/.claude/projects/-Users-Kevin-Documents-GitHub-PMDB--claude-worktrees-heldout-data/f7044de1-2c11-44c1-b109-e683c9c4f5ce/tool-results/webfetch-1791052235431-bev6rh.pdf (Nadimpalli, arXiv 1511.02445) for Table 1 (anode formulation and porosity) and Table 2 (E and ν used for PVdF, graphite and coatings). Gives a primary binder E (C18).
- Find primary text for the Si utilisation in commercial Si/graphite full cells at 100% SOC: the anode potential at top of charge and the x in LixSi. Candidate sources are the LG M50/M50T parameterisation papers (OSTI 1491439; ACS Appl. Energy Mater. acsaem.2c02047), and the IOP paper "Towards Improving the Practical Energy Density…" (10.1149/2.0481701jes). Settles C9.
- Find primary sources for SiOx (x ≈ 1) volume expansion and Young's modulus: the J. Power Sources 2017 SiOx review S0378775317309655 and Nat. Commun. s41467-026-72434-4. Settles the SiOx scenario.
- Find the formulation, porosity and thickness basis for the JES 2021 dilatometry paper (10.1149/1945-7111/abd465): whether the swelling is relative to the coating or the whole electrode, and first-cycle or reversible. Settles V-T1.
- Search for measurements of graphite flake texture or orientation in calendered anodes (XRD (002)/(110) intensity ratio, "orientation index", c-axis alignment through thickness). Settles the c ∥ z assumption (verdict 2).
- Optional: an electrode-scale image-based FE paper stating the pore/void treatment and the plane-strain vs generalised-plane-strain choice (for example the full text of Shah et al. 2022, J. Appl. Mech. 89(8) 081005, beyond pp. 1-2).

## Verification
- **V1** Qi 2014 PDF, pp. F3010-F3016. Tests R2:E3 (graphite C11 = 110 GPa) and the Si moduli. **Graphite C11 refuted (1105 GPa); LiC6 C11 = 989 confirmed; E_R/ν_R confirmed.** Excerpts:
  ```
  Table II … Graphite[20]
  C: [1105 204 −2.5 0 0 0 / 1105 204 0 0 0 / 30.9 0 0 0 / 5.6 0 0 / 5.6 0 / 450]  BV=293 BH=29 BR=161  GV=228 GH=12 GR=120  ER = 32  νR = 0.32
  LiC6: [989 171 2.1 0 0 0 / 989 171 0 0 0 / 78.0 0 0 0 / 21.9 0 0 / 21.9 0 / 409]  BV=267 BH=69 BR=168  GV=215 GH=44 GR=129  ER = 109  νR = 0.24
  Amorphous Si[22]: Si … B = 90 G = 62 … E = 96  ν = 0.29 ; Li3.75Si … B = 30 G = 20 … E = 41  ν = 0.25
  Table I: C (graphite) → 1/6 LiC6, Layered, 10% ; Si → Li3.75Si, amorphous, 263%
  "For the alloy-forming electrode materials, such as amorphous Si, DFT calculations have shown that E follows the linear rule of mixtures of pure amorphous Si and BCC Li."
  ```
- **V2** Yao 2019 PDF, pp. 1-8. Tests R2:E6 (lithiation sequence) and the graphite staging path. **Refutes "simultaneous"; confirms staging non-linearity.** Excerpts:
  ```
  "The lithiation begins with Li alloying with Si; lithiation of Gr occurs at later stages when the potential dips below 0.2 V (all potentials are given vs. Li/Li+). In the 0.2 – 0.01 V range, the relative lithiation of Si and Gr is ~58% and 42%, respectively."
  "Plateaus Si2 and Si3 are attributed to crystallization of the amorphous Si alloy to Li15Si4 at voltages below ~0.05 VLi during lithiation"
  "(iii) Between the G2 and G3 points, a Bragg peak centered at 3.54 Å is observed; the position of this peak does not change between the plateaus. (iv) Past the G3 point, in addition to this 3.54 Å peak, a new peak at 3.70 Å emerges."
  "At ~20%, stage IV/III converts to stage IIL, and at ~24% stage II begins to grow … Above ~50%, the Gr electrode potential drops to plateau G3, where coexistence of LiC12 and LiC6 phases is observed."
  ```
- **V3** Nadimpalli et al. (arXiv 1511.02445) PDF, pp. 1-8. Tests the binder E claims in R2:E5 (not found in these pages), the coating-stress validation target, and the BC rationale. Excerpts:
  ```
  "A state of equi-biaxial stress is induced in a film (or coating) on a substrate when the latter constrains the in-plane volume change of the former during electrolyte wetting or electrochemical cycling."
  "During the lithiation process the anode coating is subjected to compressive stress which increases with capacity and reaches a peak value of – 10 MPa at the end of lithiation."
  "The pressure exerted by electrodes due to electrolyte soaking alone is 0.15 MPa, and it increases to a peak value of approximately 1 MPa at the end of charging."
  "Fig. 3: Typical stress evolution during the initial intake of electrolyte into the porous regions of cathode and swelling of binder."
  ```
- **V4** Tardif et al. (arXiv 2005.04983) PDF, pp. 1-8. Tests the graphite c-axis strain and uniform SOC. **Confirmed about 10%; heterogeneity observed.** Excerpts:
  ```
  "empty graphite (d0 = 3.355 Å) … This indicates a stage 1 lithiation, with the graphite dilated by about 10% along the c axis."
  "reflections on successive graphene planes are also seen for stage 2 and 3 … corresponding to d = Ic/2, i.e. d/d0 = 1.05, and d = Ic/3, i.e. d/d0 = 1.03"
  "The lithium concentration evolution across the electrode thickness follows a very particular pattern with alternating sequences of homogeneous and heterogeneous lithium distributions."
  ```
- **V5** Nadimpalli PDF, pp. 9-16. Searched for Table 2 (PVdF/coating E, ν). **Not found** in these pages (conclusions, references, Figs. 1-5 only). Excerpt:
  ```
  "The pressure exerted by electrodes due to electrolyte soaking is ~0.15 MPa and increases to a peak value of approximately 1 MPa at the end of the first charge."
  ```

## For the planner
- **Si eigenstretch:** Fλ_Si = λ·I with λ = (1 + β·u_Si·s_Si(SOC))^(1/3).
  - β default 2.8, range 1.6-3.0 (the low end covers SiOx).
  - u_Si default 0.75, range 0.5-1.0.
  - Sources: C4, C5, C9.
- **Si stiffness:** E_Si(c) = 96 − 55·c GPa and ν = 0.29 − 0.04·c, where c is the fraction of Li3.75Si reached (linear rule of mixtures) [V1].
- **Graphite eigenstretch:** Fλ_Gr = diag(1 + ε_xx(y), 1 + ε_zz(y)) in the (x, z) frame, plus 1 + ε_xx for the out-of-plane y direction in the plane-strain eigenstrain.
  - ε_zz breakpoints: (0, 0), (0.25, 0.055), (0.5, 0.055), (1, 0.103).
  - ε_xx = 0.01·(ε_zz/0.103).
  - Sources: C14, C15.
- **Graphite stiffness (isotropic):** E = 32 + 77·y GPa, ν = 0.32 − 0.08·y [V1]. If transversely isotropic, use the Cij in C16.
- **Binder/CBD:** E = 0.5 GPa (sweep 0.05-2), ν = 0.34, Fλ = I [R2:E5, tool-summary].
- **Pore:** E = 1e-4·E_binder, ν = 0.3, Fλ = I. Sensitivity sweep required (unsupported).
- **SOC driver:** a separate s_Si(SOC) and s_Gr(SOC). Implement the proportional scenario first (s_Si = s_Gr = SOC). Add the Si-first scenario once Yao's component curve is gathered [V2].
- **Energy:** compressible neo-Hookean in Fe = F·Fλ⁻¹, with ψ weighted by det Fλ per reference volume. Solve with a PETSc SNES/Newton solver [R3:E17][R3:E18]. Substep inside each 10% SOC step, especially at graphite y in 0-0.25 and 0.5-1.
- **Outputs to flag:** Si elements with von Mises above 1 GPa [R1:E19]; pore elements with det F below a threshold (closure).
- **Validation outputs:**
  - Top-edge u_z/H at 100% SOC against 19% / 39% / 84% for 0 / 15 / 30 wt% Si [R2:E8].
  - Area-average σ_xx in graphite-rich runs against about −10 MPa [V3].
- **Stack pressure:** 0 or 1 MPa traction on the free edge [R3:E10]. Expect a negligible effect (C20).
