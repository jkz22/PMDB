# FEM literature review (final): 2D plane-strain finite-strain lithiation mechanics of Si/graphite SEM cross-sections

> Paths under `.claude/` cited here (plans, agent reports) were removed from the tree in the release cleanup (#169). Read them from history: `git show ed0a840:<path>`.

> Citation keys map to evidence files in [`evidence/`](evidence/): R1 = `R1-si.md`, R2 = `R2-graphite-binder.md`, R3 = `R3-method.md`, R4 = `R4-gaps.md`, R5 = `R5-shah2022.md`, P / Vn = `P-round1-synthesis.md` (round-1 synthesis and its targeted PDF checks). Items tagged "tool summary" in the evidence files were not verified against the source page. Decisions log: `.claude/plans/fem-swelling.md`.

## Answer
The approach is supported by the literature, and round 2 makes it concrete in three ways.
- **Si/graphite split.** Use a two-region SOC function taken from Yao 2019. Over the first 25% of overall SOC, Si takes 96% of the charge. Above that, Si takes 58% and graphite 42%. At 100% SOC this gives Si 67.5% of the charge and graphite 32.5%, which matches Yao's own end-of-lithiation figures [R4:E1][R4:E2][R4:E5][V2].
- **Si utilisation at 100% SOC.** Set u_max = 0.8 (sweep 0.6-0.95), not Li15Si4. Half cells taken to 0.01 V reach about 0.95 [R4:E2]. Full cells, where the anode bottoms out near 0.1 V, do not reach Li3.75Si [R4:E10]. Shah's model reaches only 235% volume at a 0.1 V cut-off [R5:E24].
- **What to take from Shah 2022.** Adopt its kinematics, F_Li = (1+ΩC)^(1/3)·I [R5:E9], its neo-Hookean energy [R5:E13], and its concentration-dependent E and σY for Si [R5:E16][V1].

Our set-up differs from Shah in four deliberate ways: plane strain instead of axisymmetric, no viscoplasticity, no electrochemistry, and a traction-free separator-side edge [R5:E2][R5:E3][R5:E12][R5:E18].

Validation can only be done against bands, not single numbers:
- electrode thickness swelling: 9-33% for graphite-only, up to 107% for SiO-only [R4:E24][R4:E25][R4:E18];
- coating stress: about −10 MPa [P:V3];
- graphite staging strains [P:V4];
- porosity loss: 7.6-13.5% [R5:E5][R5:E27].

The main accepted risks:
- no primary binder or CBD modulus;
- tool-summary-only SiOx properties;
- a Si/graphite split measured on one 15 wt% half cell;
- an unquantified degree of graphite texture;
- a purely numerical pore treatment.

## Sources
Labels keep the round-1 numbering (R1-R3) so that carried-forward citations stay valid. This departs from strict spawn-prompt order.
- R1: .claude/reports/fem-lit.explorer-si.md — Si/LixSi/SiOx expansion, moduli, yield, commercial utilisation, fracture size
- R2: .claude/reports/fem-lit.explorer-gr-binder.md — graphite strain and elastic constants, binder E/ν, lithiation sequence, electrode swelling
- R3: .claude/reports/fem-lit.explorer-method.md — image-based FE precedent, finite strain, BCs and stack pressure, uniform SOC, 2D vs 3D, FEniCSx
- R4: .claude/reports/fem-lit.explorer-r2.md — round 2: Yao 2019 Si/Gr split, arXiv 1511.02445 Tables 1-2, Si utilisation in full cells, SiOx, dilatometry basis, graphite texture
- R5: .claude/reports/fem-lit.explorer-shah.md — full-text extraction of Shah, de Vasconcelos & Zhao 2022, J. Appl. Mech. 89, 081005
- P: .claude/reports/fem-lit.synthesiser.md — round-1 synthesis. `P:Cn` is a round-1 claim and `P:Vn` a round-1 verification read of a primary PDF (V1 Qi 2014, V2 Yao 2019 pp. 1-8, V3/V5 Nadimpalli arXiv 1511.02445, V4 Tardif arXiv 2005.04983).
- V1, V2: my round-2 checks (see Verification).
- User decisions D1-D7: .claude/plans/fem-swelling.md

## 1. Final parameter table
Evidence strength:
- **verbatim**: primary text, read from a PDF or quoted from the fetched page.
- **tool-summary**: a WebFetch/WebSearch paraphrase only.
- **unsupported**: a modelling choice with no source.

A derived value is labelled with the strength of its inputs plus "(derived)".

| Parameter | Default | Sweep range | Source | Evidence strength |
|---|---|---|---|---|
| Kinematics | F = Fe·Fλ (no Fvp in v1); Fλ_Si = J_λ^(1/3)·I | — | R5:E8, R5:E9 (Shah eqs. 3, 5, 6); P:C1 (R3:E7, E8) | verbatim |
| Elastic energy (all phases) | W = μ/2(I1(Ce) − 3) − μ ln Je + λ/2 (ln Je)², with I1 from the 3D Ce (plane strain: the out-of-plane total stretch is 1, so Ce33 = 1/λ_y²) | Optional St Venant-Kirchhoff check | R5:E13 (Shah eq. 13, inactive phases); P:C3 (R3:E17, E18 FEniCSx demos) | verbatim |
| Si volumetric eigenstretch | J_λ = 1 + β·u_Si, with u_Si = fraction of Li15Si4 capacity; β = 2.8 | β 2.63-3.0 (3.58 is an outlier implied by Yao's 100%/1000 mAh g⁻¹ rule) | P:V1 (Qi 263%); R4:E18 (Obrovac 277%, quoted by Kirner); R1:E2 (Beaulieu 280%); R5:E16 (Shah ΩCmax = 8.1872e-6 × 3.6643e5 = 3.00); R4:E2 | verbatim (263, 277, linear-in-C form); tool-summary (280) |
| Si utilisation at 100% overall SOC, u_max | 0.80 | 0.60-0.95 | R4:E2 (3400 mAh g⁻¹ at 0.01 V; 3400/3578 = 0.95, with 3578 = 2.920 mAh/0.816 mg from R4:E5); R4:E10 (full-cell anode minimum about 0.1 V, "Li3.75Si is hard to be reached"); R5:E24 (235% at a 0.1 V cut-off, which is u ≈ 0.84 at β = 2.8) | verbatim bounds; default derived |
| Graphite lithiation at 100% overall SOC, y_max | 0.91 | 0.80-1.0 | R4:E2 (340 of 372 mAh g⁻¹ at 0.01 V, half cell) | verbatim at 0.01 V; full-cell value unsupported |
| Overall SOC → per-phase normalised fraction | Yao two-region function, breakpoint s* = 0.25 (see below) | s* 0.25-0.36; proportional control (f_Si = f_Gr = s) | R4:E1, R4:E2, R4:E5; V2 | verbatim inputs; function derived |
| Si E(u) | 96 − 55·u GPa (amorphous, rule of mixtures) | Shah form 120 − 80·C/Cmax GPa; range 40-120 | P:V1 (Qi Table II, Shenoy); R5:E16 (Shah Table 1, citing de Vasconcelos 2020); V1 | verbatim |
| Si ν(u) | 0.29 − 0.04·u | 0.21-0.29 | P:V1 | verbatim |
| Si yield (flag only, v1 is elastic) | σY(x) = 3 − 3.15·x/(1+x) GPa, x = 3.75·u_Si (3 GPa down to 0.51 GPa at Li3.75Si) | Constant 1 GPa | R5:E16, V1 (Shah Table 1, citing [20]); R5:E29 ([77] gives 1 GPa) | verbatim (the definition of x is ambiguous, see Contradictions) |
| SiOx scenario: eigenstretch | J_λ = 1 + 1.6·u | 1.18-2.0 | R4:E18 (Kirner: "SiOx can expand by up to 160%", citing Jung 2016); R4:E20 (118%); R4:E19 (200%) | verbatim secondary (160); tool-summary (118, 200) |
| SiOx scenario: E, ν | 34 GPa, 0.17 | 34-108 GPa (lithiated or coated SiO) | R4:E21 (Nat. Commun. 2026 snippet, AFM) | tool-summary only |
| Graphite c-axis strain ε_zz(y) | Piecewise linear: (0, 0), (0.25, 0.055), (0.50, 0.055), (1.0, 0.103) | End point 0.09-0.11 | P:C14 (P:V2 Yao d-spacings; P:V4 Tardif d0 = 3.355 Å, about 10%) | verbatim inputs; breakpoints derived |
| Graphite a-axis strain ε_xx(y) | 0.01·ε_zz/0.103 | 0-0.013 | P:C15 (R2:E1 13.2% volume; P:V1 10%) | tool-summary plus verbatim (derived) |
| Graphite orientation | c ∥ z (flakes parallel to the collector) | Isotropic-strain control; per-particle orientation from the segmented long axis | R4:E25 (CT: "alignment of the platelet particles … caused by packing ordering and calendering"); R4:E27, R4:E28 (flakes lie preferentially parallel to the collector) | verbatim qualitative (E25); tool-summary (E27, E28); degree unsupported |
| Graphite E, ν (isotropic) | E = 32 + 77·y GPa, ν = 0.32 − 0.08·y | Transversely isotropic Cij (P:C16) | P:V1 (Qi Table II Reuss) | verbatim |
| Binder + CBD (unassigned solid) | E 0.5 GPa, ν 0.34, Fλ = I | E 0.05-2 GPa; ν 0.3-0.45 | P:C18 (R2:E5); R4:E7 (Nadimpalli Table 2 has no PVdF row) | tool-summary |
| Homogenised coating modulus (check only, not an input) | 6.9 GPa, ν 0.3 (graphite/PVdF coating) | — | R4:E7 | verbatim |
| Pore / artefact | Ersatz material E = 1e-4·E_binder, ν = 0.3, Fλ = I | 1e-6 to 1e-2 × E_binder | None. Shah states no pore stiffness either (R5 Not found) | unsupported |
| Out-of-plane | Plane strain (user D2) | Generalised plane strain (not planned) | P:V3 (substrate in-plane constraint); Shah used axisymmetric (R5:E2, R5:E3) | unsupported directly |
| Current-collector edge | u_z = 0; run both orientations because the foil side is unknown | — | R5:E18 (Shah: u·n = 0 at z = 0) | verbatim (analogue) |
| Lateral edges | u_x = 0 | Periodic | R5:E18 (Shah: u·n = 0 at r = R0); P:V3 | verbatim (analogue) |
| Separator-side edge | Traction-free | Stack pressure 0.1-1 MPa; confined u_z = 0 bracket (Shah-like) | R3:E10 via P:C19; R4:E9 (about 1 MPa peak); R5:E18 (Shah fixes z = L) | verbatim |
| SOC snapshots | s = 0, 0.1, …, 1.0 (11 frames, user D3), with Newton substeps | Refine near s* and near graphite y ≈ 0.25 and y > 0.5 | User D3; P:C3 | user decision |
| Resolution | 50 nm/px cache/half (user D4) | Round 1 proposed coarsening to 100 nm/px; optional | None | unsupported |
| Si fracture interpretation | Particles > ~150 nm fracture-prone (crystalline Si, first cycle) | — | R1:E27 via P table | tool-summary |

### Overall SOC → (Si utilisation, graphite fraction)
Let s ∈ [0, 1] be the overall SOC, defined as the fraction of the electrode capacity reached at the end of charge. The breakpoint s* = 0.25 marks the end of region 1 (1.0-0.2 V vs Li).

Normalised component fractions (each equals 1 at s = 1):
- **Si.** f_Si(s) = 0.96·s / 0.675 for s ≤ 0.25. For s > 0.25, f_Si(s) = [0.24 + 0.58·(s − 0.25)] / 0.675.
- **Graphite.** f_Gr(s) = 0.04·s / 0.325 for s ≤ 0.25. For s > 0.25, f_Gr(s) = [0.01 + 0.42·(s − 0.25)] / 0.325.

The model inputs are u_Si(s) = u_max·f_Si(s), with default u_max = 0.80, and y(s) = y_max·f_Gr(s), with default y_max = 0.91. Then x in LixSi = 3.75·u_Si.

| s | f_Si | f_Gr | u_Si (u_max 0.8) | y (y_max 0.91) | J_Si (β 2.8) | ε_zz,Gr |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.1 | 0.142 | 0.012 | 0.114 | 0.011 | 1.319 | 0.002 |
| 0.2 | 0.284 | 0.025 | 0.228 | 0.022 | 1.637 | 0.005 |
| 0.3 | 0.399 | 0.095 | 0.319 | 0.087 | 1.893 | 0.019 |
| 0.4 | 0.484 | 0.225 | 0.388 | 0.204 | 2.085 | 0.045 |
| 0.5 | 0.570 | 0.354 | 0.456 | 0.322 | 2.278 | 0.055 |
| 0.6 | 0.656 | 0.483 | 0.525 | 0.440 | 2.470 | 0.055 |
| 0.7 | 0.742 | 0.612 | 0.594 | 0.557 | 2.662 | 0.061 |
| 0.8 | 0.828 | 0.742 | 0.662 | 0.675 | 2.855 | 0.072 |
| 0.9 | 0.914 | 0.871 | 0.731 | 0.792 | 3.047 | 0.084 |
| 1.0 | 1.000 | 1.000 | 0.800 | 0.910 | 3.240 | 0.095 |

The ε_zz column comes from the P:C14 breakpoints evaluated at y. All table values are derived arithmetic, rounded.

Why s* = 0.25 (INFERRED, C3): it is the only breakpoint for which Yao's two ratios (0.96/0.04 and 0.58/0.42) reproduce Yao's end-of-lithiation component capacities. Those capacities are Si 3400 mAh g⁻¹ × 0.816 mg = 2.774 mAh and graphite 340 × 3.970 = 1.350 mAh, so Si holds 0.673 of the total [R4:E2][R4:E5]. The function gives 0.96·0.25 + 0.58·0.75 = 0.675.

## Claims

### SOC split and utilisation
- **C1** (CONFIRMED) In Yao's 15 wt% Si blend (C/30 half cell), Si takes 0.96 of the lithiation above 0.2 V and about 0.58 below it (0.01-0.2 V). On delithiation, graphite empties first. [R4:E1][P:V2][V2]
- **C2** (CONFIRMED) At 0.2 V the Si holds about 1350 mAh g⁻¹ and graphite holds a negligible amount. At 0.01 V, Si holds about 3400 and graphite about 340 mAh g⁻¹ of their own masses. The electrode reaches about 93% of its theoretical capacity. [R4:E1][R4:E2][V2]
- **C3** (INFERRED) The two-region function with s* = 0.25 is consistent with C1 and C2, by the arithmetic shown under the parameter table. Cross-check: (2.774 + 1.350)/4.396 = 0.938, which agrees with "~93%" [R4:E1][R4:E5]. One counter-indication exists: Yao's "1.5 mAh → 1.44 mAh Si" example suggests a later breakpoint (see Contradictions).
- **C4** (CONFIRMED) The split depends on blend composition and cut-offs, as Yao states [R4:E6]. Yao's blend has Si at 66% of the theoretical capacity (2.920/4.396) [R4:E5]. Commercial cells have Si at 13-15% of capacity [P:C23]. So the ratios are not directly transferable (see Gaps).
- **C5** (CONFIRMED) Full cells do not reach Li3.75Si. The anode minimum potential during constant-current charge is about 0.1 V, compared with 0.01 V in half cells, and "the phase of Li3.75Si is hard to be reached in the full cell operation condition" [R4:E10]. Shah's model reaches 235% Si volume at a 0.1 V cut-off because the particle "is not fully lithiated" [R5:E24].
- **C6** (INFERRED) u_max ≈ 0.8 is the evidence-centred default. The bounds are 0.95, the half cell at 0.01 V [R4:E2], and about 0.66-0.84, Shah's 235% converted with either Yao's 100%/1000 mAh g⁻¹ rule or β = 2.8 [R5:E24][R4:E2]. The step is: full cells sit near 0.1 V [R4:E10], so they fall below the half-cell value. No source gives x for an LG M50/MJ1 cell [R4 Not found].
- **C7** (CONFIRMED) Every modelling source treats Si volume as linear in Li content. Shah uses J_Li = 1 + ΩC [R5:E9] and Yao uses "a linear expansion rate of 100% per 1000 mAh g⁻¹" [R4:E2]. Both are model assumptions, not measurements.

### Si and SiOx properties
- **C8** (CONFIRMED) Shah's Ω·Cmax = 8.1872e-6 × 3.6643e5 = 3.00, i.e. 300% volume at Cmax [R5:E16][V1]. Shah's E is 120 − 80·C/Cmax GPa (120 → 40), against Qi's 96 → 41 GPa [P:V1]. Both end at about 40 GPa.
- **C9** (CONFIRMED) Shah's yield stress depends on concentration, σY = −3.15·(x/(1+x)) + 3 GPa, citing de Vasconcelos 2020. The concentration-independent alternative is 1 GPa. Si is modelled as viscoplastic with A = 0.0023 s⁻¹ and n = 2.94. [R5:E12][R5:E14][R5:E16][V1]
- **C10** (CONFIRMED, secondary) The Si→Li15Si4 expansion is 277% (Obrovac 2007, as quoted by Kirner). SiOx "can expand by up to 160%" (Jung 2016, as quoted by Kirner). [R4:E18] Other SiOx figures (118%, 200%) and the SiO modulus (34 GPa, ν 0.17) are tool-summary only [R4:E19][R4:E20][R4:E21].
- **C11** (CONFIRMED) The LG M50T negative electrode is "SiOx-blended graphite" [R4:E11]; the Chen 2020 abstract says "Graphite-SiOx" [R4:E14, tool-summary]. The SiOx scenario is therefore a realistic case for commercial-style electrodes, not a fringe case. That BSE cannot separate Si from SiOx is a round-1 inference [P verdict 1].

### Graphite
- **C12** (CONFIRMED) The graphite strain path and end point are unchanged from round 1: about 5.5% c-axis strain through stage II and about 10.3% at LiC6 [P:C12][P:C13][P:C14].
- **C13** (CONFIRMED, qualitative) Graphite platelets in calendered electrodes align with the collector plane. Michael 2021 attributes the in-plane-isotropic, through-plane-high tortuosity to "alignment of the platelet particles within the electrode, caused by packing ordering and calendering" [R4:E25]. Two tool-summary sources say flakes lie preferentially parallel to the collector and that the oriented fraction rises with calendering [R4:E27][R4:E28]. No numeric orientation fraction was retrieved [R4 Not found].
- **C14** (INFERRED) Under c ∥ z, the larger strain (c-axis) acts through the thickness and the about 1% a-axis strain acts in-plane. That direction is supported by C13. Per-particle tilt is not.

### Binder, coating and pores
- **C15** (CONFIRMED) Nadimpalli Table 2 has no PVdF row. The only anode entry is "Graphite based anode coating", E 6.9 GPa, ν 0.3, biaxial modulus 10 GPa, for a coating of 89.8% graphite, 6% PVDF and 4% Super P at 26% porosity and 40 µm thickness. [R4:E7][R4:E8] Binder E stays tool-summary only [P:C18].
- **C16** (INFERRED) The measured −10 MPa coating stress [P:V3] with a 10 GPa biaxial modulus [R4:E7] implies an effective constrained in-plane mismatch strain of only about 0.1% (σ/M). The free a-axis graphite strain is about 1% [P:C15]. So porosity and binder accommodate most of the in-plane expansion, and a 2D model with stiff, dense phases is likely to over-predict in-plane stress.
- **C17** (CONFIRMED) Shah gives no pore stiffness, pore meshing approach, contact treatment or pore-closure handling. Its expansion is accommodated by "(i) reduction in porosity, (ii) compaction of CBD, and (iii) expansion into the separator", and CBD compaction is "minimal". [R5:E5][R5:E7][R5 Not found]

### Kinematics and BCs
- **C18** (CONFIRMED) Shah uses F = Fel·FLi·Fvp with FLi = (1+ΩC)^(1/3)·I, P = FS, a reference-configuration equilibrium ∇·P + Bv = 0, and a neo-Hookean W for the inactive material in the same form as our energy [R5:E8][R5:E9][R5:E10][R5:E13]. Its Si elasticity uses S = C:E_el (Green-Lagrange, St Venant-Kirchhoff-type) [R5:E11].
- **C19** (CONFIRMED) Shah's mechanical BCs are zero normal displacement at r = R0, at the current collector (z = 0) and at the Li-metal face (z = L). The domain is fully confined, with a separator layer between the electrode and the fixed wall [R5:E18]. It is a 2D axisymmetric model of a Si-only electrode built from a BSE SEM image of Müller 2018, with initial porosity 65.3%, and no graphite phase [R5:E1][R5:E2][R5:E5][R5 Not found].
- **C20** (CONFIRMED) Shah does not report mesh, time stepping, convergence settings or stack pressure. The solver is COMSOL. [R5:E22][R5 Not found]

### Validation basis
- **C21** (CONFIRMED, tool-summary) Prado 2020 (DOI abd465) measures thickness change "based on the coating thickness of the dry, pristine anode" in Li half cells. Graphite-only is about 19% at full lithiation. The 15 wt% Si electrode is about 22% in cycles 1-2, cut at 100 mV, and 39% in cycle 3 at C/25 to 10 mV. Porosity is 34% for graphite and 56% for Si-Gr; LiPAA binder. [R4:E24]
- **C22** (CONFIRMED) Kirner, by micrometer on rinsed and dried fully lithiated electrodes at 0.01 V after formation, reports coating swell of 33, 55, 76, 90 and 107% for graphite:SiO ratios of 1:0, 7:3, 5:5, 3:7 and 0:1 [R4:E18]. Michael 2021 reports a graphite electrode thickness change "close to 9%" on the first cycle [R4:E25].
- **C23** (CONFIRMED) Shah's model loses about 13.5% porosity from 65.3% at the end of lithiation. Pietsch measured a 7.6% porosity drop in a Si composite electrode (cited by Shah). [R5:E5][R5:E27]

## 2. Verdicts per assumption (changes since round 1)
1. **Si isotropic expansion. Was REVISE; now FIXED as J_λ = 1 + β·u, β = 2.8, u_max = 0.8 (sweep 0.6-0.95).**
   - What changed: the utilisation default moves from 0.75 (range 0.5-1.0) to 0.8 (range 0.6-0.95).
   - Why: there is now verbatim evidence for both bounds: 0.95 for a half cell at 0.01 V, and less than Li3.75Si in full cells near 0.1 V (C5, C6).
   - Linearity in Li content is now verbatim as a modelling convention (C7).
   - Kirner's verbatim quote of Obrovac (277%) is added to the β range (C10).
2. **Graphite anisotropic strain. Was SUPPORTED end point, REVISE path; now path FIXED (staging) and c ∥ z upgraded from UNSUPPORTED to WEAKLY SUPPORTED.**
   - Why the upgrade: qualitative evidence of collector-parallel platelets (C13).
   - Keep the isotropic control run, because the degree of alignment is unquantified.
   - A new end point, y_max = 0.91, comes from Yao (C2).
3. **Single SOC for all phases. Was REVISE; now RESOLVED as uniform-per-phase with the Yao two-region split (C1-C3).**
   - This matches user decision D2/D3 (uniform per phase).
   - Keep the proportional split (f_Si = f_Gr = s) as a single sensitivity run. The split is blend-specific (C4).
4. **Binder soft, no expansion. Was SUPPORTED with caveats; UNCHANGED.**
   - The hoped-for primary PVdF modulus does not exist in Nadimpalli Table 2 (C15).
   - New: the 6.9 GPa homogenised coating modulus is a check, and C16 suggests porosity and binder absorb most in-plane strain.
5. **Pores as ersatz soft material. UNSUPPORTED, UNCHANGED.** Shah also gives no treatment (C17). A sensitivity sweep is mandatory.
6. **Plane strain. UNSUPPORTED directly; UNCHANGED (fixed by user D2).** The only image-based precedents, R3:E2 and Shah, are both axisymmetric (C19).
7. **BCs (u_z = 0 at the collector edge in both orientations, lateral u_x = 0, free top). SUPPORTED as a modelling choice; partly upgraded.**
   - Shah uses the same zero-normal-displacement conditions on the collector and lateral boundaries (C19).
   - Difference: Shah confines the top through a separator against a fixed Li face. Add an optional confined-top bracket run.
8. **Stack pressure up to 1 MPa. SUPPORTED, UNCHANGED** [P:C19][R4:E9]. Shah applies none (C20).
9. **Compressible neo-Hookean. Was SUPPORTED as template; upgraded to SUPPORTED by a lithiation precedent.** Shah uses the identical W for inactive material (C18).
10. **No plasticity in v1. ACCEPT; flag criterion REVISED.**
    - Flag Si elements with von Mises above σY(u) = 3 − 3.15·x/(1+x) GPa, with x = 3.75u, instead of a constant 1 GPa (C9).
    - Shah's Si is viscoplastic, so v1 Si stresses remain non-physical in magnitude [P:C8].
11. **Pore-closure flag. UNSUPPORTED (numerical), UNCHANGED** (C17, C20).
12. **Finite strain. REQUIRED; confirmed again by Shah** (C18).
13. **Resolution.** Round 1 proposed coarsening to 100 nm/px, which remains UNSUPPORTED. User D4 sets 50 nm/px cache/half input, so coarsening is optional and needs a convergence check either way.

## 3. Shah 2022: adopt vs deliberately differ
**Adopt**
- Si eigenstretch form Fλ = (1 + ΩC)^(1/3)·I, written as (1 + β·u)^(1/3)·I. Shah's ΩCmax = 3.0 sits inside our β sweep (C8, C18).
- Multiplicative split and P = FS with equilibrium in the reference configuration (C18).
- Neo-Hookean W for the non-Si phases, in the same form (C18).
- Concentration-dependent Si properties:
  - E(C) as an alternative to Qi in the E sweep (120 → 40 vs 96 → 41 GPa);
  - σY(x) as the plasticity flag threshold (C8, C9).
- Zero-normal-displacement BCs on the collector and lateral boundaries (C19).
- Reported outputs to emulate: von Mises in Si normalised by σY, porosity change versus SOC, and Si volumetric strain at end of charge [R5:E23].

**Deliberately differ**
- **Plane strain instead of axisymmetric.** The user fixed this (D2). The SEM sections are not rotationally symmetric.
- **No viscoplasticity or electrochemistry.** User D2 makes this mechanics only. Uniform-per-phase lithiation replaces Shah's diffusion-reaction field (C9, C18).
- **Graphite phase included** (Shah has none) (C19).
- **Free separator-side edge instead of confinement through a separator to a fixed wall.** There is no separator in our images. The confined bracket is optional (C19).
- **Si elasticity.** Shah uses St Venant-Kirchhoff-type S = C:E_el for Si [R5:E11]. We use neo-Hookean for all phases, which is more robust at large Je.
- **Lithiation endpoint.** The user's 0-100% SOC with u_max = 0.8 corresponds roughly to Shah's 0.1 V cut-off. Do not reuse Shah's Cmax as u = 1: Cmax may correspond to more than Li3.75Si (see Gaps).

## 4. Validation targets (with measurement basis)
- **VT1 Electrode thickness swelling, top-edge u_z/H at s = 1.** Compare against bands, not points:
  - Graphite-only: 9% (Michael, first cycle, synthetic graphite, verbatim), 19% (Prado, LiPAA, 34% porosity, half cell, tool-summary) and 33% (Kirner, LiPAA plus SWCNT, rinsed and dried, 0.01 V, verbatim) [R4:E25][R4:E24][R4:E18].
  - 15 wt% Si: 22% at a 100 mV cut-off, which is the most full-cell-like, and 39% at 10 mV [R4:E24].
  - SiOx scenario: 55-107% for 30-100% SiO [R4:E18].

  The basis is thickness relative to the pristine dry coating, not volume [R4:E24]. Expect the 2D dense-phase model to under-predict graphite-only swelling [P V-T1 note].
- **VT2 In-plane coating stress.** Compare area-averaged σ_xx under lateral u_x = 0 against about −10 MPa at end of lithiation, with staging plateaus [P:V3]. The coating basis is 89.8/6/4 graphite/PVDF/Super P, 26% porosity, 40 µm, homogenised E 6.9 GPa [R4:E7][R4:E8]. C16 predicts over-prediction. Report the ratio rather than expecting agreement.
- **VT3 Graphite c-axis strain path:** 3%, 5% and 10% at stages 3, 2 and 1, with a flat d-spacing between 25% and 50% graphite lithiation [P:V4][P:V2]. This is an input-consistency check.
- **VT4 Porosity loss.** Compare the pore-area fraction change at s = 1 against 7.6% (experimental, Si composite, via Shah) and about 13.5% (Shah model, Si-only, 65.3% initial) [R5:E5][R5:E27]. Whether these are relative or absolute changes is unstated.
- **VT5 Si volumetric strain at end of charge:** 235% at 0.1 V (Shah model) [R5:E24], and 135% at 0.2 V and 340% at 0.01 V (Yao, from the linear rule) [R4:E2]. This is a consistency check on u_max and β, not an independent validation.
- **VT6 Interpretation only:**
  - von Mises/σY(u) above 1 marks plastic flow [R5:E16];
  - Si above ~150 nm is fracture-prone [R1:E27];
  - cell-level expansion below 7.5% [P V-T4] is a weak bound.

## Contradictions
- **Yao internal, Si capacity at the region-1 boundary.** One passage says "electrode capacity … 1.5 mAh, the capacity of the Si and Gr components are ~1.44 mAh and ~0.06 mAh" [R4:E1][V2]. That is 1765 mAh g⁻¹ Si, not the "~1350 mAh g⁻¹Si" at 0.2 V [R4:E2][V2]; 1350 × 0.816 mg = 1.10 mAh. The 1.5 mAh example implies s* ≈ 0.36, which would give Si 0.72 of the total at end of charge, not the 0.673 that Yao's endpoint numbers give. I weight the Fig. 4b numbers (s* = 0.25) because they reconcile with the 93% total (C3). s* is swept 0.25-0.36.
- **Si expansion per capacity.** Yao's linear rule gives 100% per 1000 mAh g⁻¹, so 358% at 3578 mAh g⁻¹ [R4:E2]. That conflicts with 263% (Qi [P:V1]), 277% (Obrovac [R4:E18]), 280% (Beaulieu [R1:E2]) and 300% at Cmax (Shah [R5:E16]). I weight the 263-280% values for Li15Si4, which are definitional sources. Yao's rule is an approximation it adopts from its ref. [9].
- **Shah's σY definition.** "x is atomic fraction in LixSi" [V1], yet the formula uses x/(1+x), which is itself the atomic fraction if x is the stoichiometry. I read x as the stoichiometry, giving 3 GPa → 0.51 GPa at Li3.75Si. If x were already the atomic fraction, σY at full lithiation would be 3 − 3.15·(0.79/1.79) = 1.6 GPa. This affects only the flag threshold.
- **Shah's Cmax basis.** Cmax = 3.6643e5 mol m⁻³ with Shah's Si density of 2230 kg m⁻³ [R5:E16] implies x ≈ 4.6 at Cmax (INFERRED arithmetic using 28.09 g/mol, which is not in the evidence). So C/Cmax ≠ u. I do not adopt Cmax as the u = 1 reference.
- **LG M50T Si content.** 10 wt% (Bonkile citing Chen, verbatim [R4:E11]) against 2.43% (search snippet [R4:E15]; also round-1 [P:C23]). The definitions are unknown (Si vs SiOx mass, anode vs composite). This does not affect the model, because phases come from segmentation.
- **Graphite-only electrode swelling.** 9% [R4:E25], 19% [R4:E24] and 33% [R4:E18] come from different electrodes, binders, protocols and dryness states. Treat them as a band. Kirkaldy's "graphite expands by around 20%" [R4:E12] is probably an electrode-level figure. Lattice volume is 10-13.2% [P:C12].
- **Same formulation, different porosity.** Prado's 15 wt% Si electrode is at 56% porosity [R4:E24]; Yao's identical formulation is at 42.4% [R4:E4]. The explorer flags this as unreconciled. Prado's porosity is tool-summary.
- **Carried from round 1:** lithiation order, "simultaneous" vs Si-first, resolved for Si-first [P Contradictions]; Qi label issues [P].

## 5. Residual gaps (accepted risks for the spec)
- **Split transferability.** The only measured Si/graphite split is one 15 wt% nano-Si (50-70 nm) blend in a C/30 half cell, where Si is 66% of capacity [R4:E4][R4:E5]. Commercial-like sites (Si at 13-15% of capacity [P:C23]), and anodes in full cells near 0.1 V [R4:E10], will have a different f_Si/f_Gr. Risk: the SOC timing of Si versus graphite strain in the 11 frames is wrong by a breakpoint shift. Mitigation: the proportional control run and the s* sweep.
- **u_max and y_max for full cells are unmeasured** [R4 Not found]. Risk: absolute swelling scales with u_max (J_Si at s = 1 ranges 2.7-3.7 across the sweep). Mitigation: record u_max as feature metadata. Relative site rankings should be more robust (INFERRED).
- **The SiOx properties are not primary.** 160% is a secondary citation, and 34 GPa / ν 0.17 is a search snippet [R4:E18][R4:E21]. Si-in-SiOx lithiation timing is unknown, and Yao's split is for pure Si. Risk: if the sites contain SiOx, as in the M50T (C11), the default Si scenario over-predicts.
- **Binder/CBD modulus has no primary source** [P:C18][R4:E7]. Carbon black is uncharacterised. Risk: this controls how much Si expansion pores and binder absorb (C16, C17). Mitigation: a 40× sweep.
- **Pore ersatz stiffness, closure and contact have no precedent**, Shah included [R5 Not found]. Risk: post-closure frames (high s near large Si) are numerically defined, not physical. Mitigation: a closure flag and a stiffness sweep.
- **Plane strain versus 3D or axisymmetric error is unquantified** [P Gaps][R5:E4]. Risk: plane strain over-constrains the out-of-plane direction, so in-plane stresses are over-predicted.
- **The degree of graphite texture is unquantified** [R4 Not found]. Risk: anisotropic swelling direction errors for tilted flakes. Mitigation: the isotropic control.
- **No plasticity.** Si stress magnitudes are not physical [P:C8][R5:E14]. Use them only as flags or relative features.
- **Uniform-per-phase lithiation ignores rate-driven gradients** [P:C24]. Shah shows stress depends strongly on C-rate (1C vs 3C) [R5:E26]. Accepted by user decision D2/D3.
- **Some validation data is weak.** The Prado values are tool-summary [R4:E24]. The relative versus absolute basis of the porosity loss is unstated [R5:E5]. Agreement should be judged against bands only.
- **Graphite intermediate strain breakpoints are read from figures** [P Gaps].

## Gather next
None. Per the dispatcher, no further gathering rounds will run, and the residual gaps above are accepted as spec risks. If a round ever reopens, the highest-value item is the PyBaMM parameter files Chen2020_composite / OKane2022 from GitHub raw, for Si and graphite stoichiometry windows at 100% SOC in the LG M50 [R4 Not checked]. That would settle u_max and y_max and the low-Si split.

## Verification
- **V1** `105.pdf` p. 9 (Table 1). Tests C8/C9 (Shah E(C), σY(x), Ω, Cmax). Confirmed.
  ```
  Yield stress, σY  Independent of concentration C  1 GPa [77]
  Concentration dependent  −3.15(x/(1 + x)) + 3 GPa [20] where x is atomic fraction in LixSi
  Elastic modulus, E  Independent of concentration C  80 GPa [42]
  Concentration dependent  −80(C/Cmax) + 120 GPa [20]
  Density of silicon, ρ  2230 kg/m3
  Maximum theoretical concentration of Li in silicon, Cmax  3.6643 × 10^5 mol/m3 [79]
  Partial molar volume of Li in silicon, Ω  8.1872 × 10−6 m3/mol [79]
  ```
- **V2** `/tmp/yao9.txt:86-131` (the explorer's pdftotext of Yao 2019, pp. 9-end). Tests C1-C3 and the region-1 contradiction. Confirmed, and the internal inconsistency is present.
  ```
  Li mainly alloys with the Si; the lithiation ratio LiSi/LiGr of Si vs. Gr is 0.96/0.04 in this range.
  That is, when the electrode capacity during lithiation is 1.5 mAh, the capacity of the Si and Gr
  components are ~1.44 mAh and ~0.06 mAh, respectively.
  specific capacity in the Gr is negligible, whereas for the Si particles it is ~1350 mAh.g-1Si, which
  corresponds to a volume expansion of 135 % (assuming a linear expansion rate of 100% per
  Gr is ~340 mAh.g-1Gr, whereas for the Si particles it is ~3400 mAh.g-1Si, which corresponds to
  ```

## For the planner
- **SOC driver.** For each frame s ∈ {0, 0.1, …, 1}:
  - u_Si = u_max·f_Si(s) and y = y_max·f_Gr(s), using the piecewise functions and table above. Defaults: s* = 0.25, u_max = 0.8, y_max = 0.91 (C1-C6).
  - Sweeps: u_max 0.6-0.95; s* 0.25-0.36; plus one proportional run.
- **Si.**
  - Fλ = (1 + 2.8·u_Si)^(1/3)·I.
  - E = 96 − 55·u_Si GPa; ν = 0.29 − 0.04·u_Si.
  - Flag where von Mises > 3 − 3.15·x/(1+x) GPa, with x = 3.75·u_Si (C8-C10) [P:V1][R5:E16].
- **SiOx scenario.** Fλ = (1 + 1.6·u)^(1/3)·I; E 34 GPa; ν 0.17 [R4:E18][R4:E21].
- **Graphite.**
  - Fλ = diag(1 + ε_xx(y), 1 + ε_zz(y)) in (x, z), with c ∥ z.
  - The out-of-plane eigenstretch is 1 + ε_xx(y).
  - ε_zz breakpoints (0, 0), (0.25, 0.055), (0.5, 0.055), (1, 0.103); ε_xx = 0.01·ε_zz/0.103.
  - E = 32 + 77·y GPa; ν = 0.32 − 0.08·y.
  - Run an isotropic control [P:C14-C17][R4:E25].
- **Binder/CBD:** E 0.5 GPa (sweep 0.05-2), ν 0.34, Fλ = I. **Pore:** E = 1e-4·E_binder (sweep 1e-6 to 1e-2), ν 0.3 [P:C18].
- **Energy:**
  - W = μ/2(I1 − 3) − μ ln Je + λ/2 (ln Je)² in Fe = F·Fλ⁻¹, with I1 from the 3D Ce under plane strain.
  - Weight by det Fλ per reference volume.
  - Use a SNES Newton solver with substeps between frames, refined around s = 0.25 [R5:E13][P:C3].
- **BCs:**
  - u_z = 0 on the collector edge (run both orientations);
  - u_x = 0 on the lateral edges;
  - free top by default;
  - optional 1 MPa traction run and confined-top run [R5:E18][R4:E9].
- **Per-frame outputs** (they map to VT1-VT4):
  - top-edge u_z/H;
  - area-averaged σ_xx;
  - pore-area fraction change;
  - mean J_Si;
  - fraction of Si elements above σY(u);
  - pore elements with det F below the closure threshold.
