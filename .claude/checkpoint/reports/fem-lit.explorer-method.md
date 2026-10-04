## Question
Web literature evidence-gathering on methodology for a 2D, mechanics-only, image-based FEM of SEM cross-sections (50->100 nm/px) of Si/graphite anodes, in FEniCSx. Phases from segmentation: Si, graphite, binder/carbon (unassigned solid), pore. Lithiation as eigenstrain ramped 0->100% SOC. Gather (1) image-based FE precedent, (2) finite-strain lithiation formulations, (3) pore/electrolyte treatment, (4) BCs and stack pressure, (5) uniform-SOC vs diffusion, (6) 2D vs 3D / SliceGAN, (7) FEniCSx examples. Log searches and non-finds. No conclusions.

## Search log
- S1 WebSearch: image-based FE, FIB-SEM Si/graphite anode mechanical lithiation - 9 hits (summary only)
- S2 WebSearch: SliceGAN Kench Cooper 2021 - 9 hits
- S3 WebSearch: Zhao Suo Bower multiplicative decomposition lithiation - 9 hits
- S4 WebSearch: dolfinx hyperelasticity thermal expansion eigenstrain load stepping - 2 rounds, ~18 hits, none a dolfinx eigenstrain/growth example
- S5 WebSearch: Li diffusivity a-Si and graphite - 9 hits
- S6 WebSearch: stack pressure pouch cell MPa Si/graphite - 9 hits
- S7 WebFetch nature.com s42256-021-00322-1 (SliceGAN) - blocked (303 redirect to idp)
- S8 WebFetch newfrac FEniCSx finite elasticity I - OK
- S9 WebFetch nature.com s42004-020-00386-x - blocked (303)
- S10 WebFetch arxiv.org/pdf/1908.00390 - OK (PDF read via Read)
- S11 WebFetch newmaeweb.ucsd.edu Zhao 2011 PDF - SSL certificate error
- S12 WebFetch sciencedirect S037877531301197X - 403
- S13 WebFetch arxiv.org/abs/2102.07708 - OK
- S14 WebFetch PMC8397968 (graphite GITT) - reCAPTCHA, no content
- S15 WebFetch sciencedirect S0013468616323738 (LCO FIB-SEM) - 403
- S16 WebSearch: 2D vs 3D electrode microstructure simulation error - 9 hits (summary only; no 2D-vs-3D mechanics comparison located)
- S17 WebSearch: image-based FEM SEM cross-section plane strain pore ersatz soft void stiffness ratio - 9 hits, all irrelevant (tissue, elastomer)
- S18 WebFetch arxiv 1908.02175 - OK (abstract)
- S19 WebFetch docs.fenicsproject.org dolfinx demo_hyperelasticity - OK
- S20 WebFetch arxiv 2509.13947 - OK (abstract)
- S21 WebFetch purdue kjzhao/papers/105.pdf - OK (PDF read pp1-2)
- S22 WebSearch: Sethuraman/Mukhopadhyay Si-graphite pouch stack pressure - 9 hits (summary only)
- S23 WebFetch spikelab Princeton 103.pdf (Cannarella & Arnold) - OK (PDF read pp1-2)

## Evidence

### 1. Image-based / microstructure-resolved precedent
- **E1** <https://iopscience.iop.org/article/10.1149/MA2025-013373mtgabs/meta> (S1, search-result snippet only, page not fetched) - "Digital-Twin Simulation for a Chemo-Mechanical Analysis of SiOx/Graphite Composite Electrodes". Snippet: "The finite volume method has been used to analyze real-time structural changes and mechanical stresses during lithiation, with high-resolution FIB-SEM imaging providing accurate 3D reconstruction of SiOx/graphite composite electrode morphology. ... slower charging leads to more significant volumetric changes and localized stress concentrations, particularly around SiOx agglomerates and at SiOx/graphite interfaces." (snippet is the search tool's paraphrase, not verified against the source.)
- **E2** Shah, de Vasconcelos, Zhao, "Computational Modeling of Electrochemomechanics of High-Capacity Composite Electrodes in Li-Ion Batteries", J. Appl. Mech. 89(8), 081005 (2022), DOI 10.1115/1.4054759 (pp.1-2 read from PDF, S21):
  > "We use silicon as a model system and construct a microstructurally resolved porous composite electrode model."
  > "we model each of the three phases explicitly based on an SEM image of a pristine silicon composite electrode presented by Müller et al. [9]. The two-dimensional axisymmetric geometry generated from the image is shown in Fig. 2(a)"
  > "Porous composite electrodes generally consist of three components (Fig. 1): active particles, CBD, and liquid electrolyte-filled pore phase."
  > "Here, we present a fully coupled electrochemomechanical computational framework that integrates (i) finite strain kinematics, (ii) the rate-dependent viscoplastic constitutive behavior, ..."
- **E3** Same paper, p.2, related-work passage: "Various studies explicitly define the active particles and the surrounding porous nature to develop microstructure resolved models for NMC [47-49] and graphite/LiCoO2 (LCO) [50]. ... Srivastava et al. [54], and Ferraro et al. [55] explicitly define the three phases, active material, carbon-binder domain (CBD), and pores, in a porous NMC electrode." and "Gao et al. [57] and Liu et al. [58] presented multi-scale models for commercial silicon composite electrodes. The former studied the effect of silicon content, mechanical constraint, and charging rate on the performance of a Si-C composite electrode paired with an LCO cathode in full cell configuration."
- **E4** <https://www.nature.com/articles/s42004-020-00386-x> (S1 snippet only; fetch blocked S9) "Multi-scale quantification and modeling of aged nanostructured silicon-based composite anodes" (Communications Chemistry, 2020). Snippet: "Models based on measured, reconstructed, and image-analyzed FIB-SEM data show the pore, active material, and SEI/carbon/binder compound-domain evolution after cycling." Authors/DOI not verified.
- **E5** Titles only (S1 results, not fetched): "Three-Dimensional Finite Element Study on Li Diffusion Induced Stress in FIB-SEM Reconstructed LiCoO2 Half Cell" (Electrochimica Acta, ScienceDirect S0013468616323738); "A particle-resolved 3D finite element model to study the effect of cathode microstructure on the behavior of lithium ion batteries" (S0013468618323119); "Material parameters affecting Li plating in Si/graphite composite electrodes" (S0013468624012477); "Image Segmentation for FIB-SEM Serial Sectioning of a Si/C-Graphite Composite Anode Microstructure Based on Preprocessing and Global Thresholding" (PubMed 31387658); "Mechano-electrochemical analysis of lithiation-induced deformation of composite electrodes using carbon fibre as current collector" (S0266353825002192).

### 2. Finite-strain formulations
- **E6** Search-result summary (S3; tool paraphrase, not verbatim from source): "The theory uses a multiplicative decomposition of the deformation gradient (F=FeFpFs) that separates elastic distortion (Fe), plastic distortion (Fp), and swelling distortion (Fs)." "Both Zhao et al. (2011) and Bower et al. (2011) have proposed theories which couple large elastic-plastic deformations with large volumetric swelling due to lithium diffusion." Source listing: Zhao, Pharr, Cai, Vlassak, Suo, "Large Plastic Deformation in High-Capacity Lithium-Ion Batteries Caused by Charge and Discharge" (J. Am. Ceram. Soc., 2011; journal/year not verified at source). PDF at <http://newmaeweb.ucsd.edu/groups/cai_group/pdf/2011-03.pdf> failed SSL (S11). Bower et al. original paper not retrieved.
- **E7** Roper, Chapman, Please, "The effect of mechanical stress on lithium distribution and geometry optimisation for multi-material lithium-ion anodes", arXiv:1908.00390 (2019), pp.2,3,6 (S10):
  > "Linear elasticity models [8, 36, 57] have been used to model the mechanical response when strains are small, but more commonly finite strain models using geometrically nonlinear elasticity are used for silicon to capture the large strains that occur [4, 6, 9, 10, 15]."
  > "when silicon is lithiated, it expands to around four times its original size [6], much more than the ~10% volume increase observed in fully lithiated graphite."
  > "For simplicity, we assume the strains are small enough that we may use linear elasticity theory." Eq (1): E^e = 1/2[(grad u)^T + grad u] - eta_a V^m_a c*_a 1 in Omega_a.
  > "We note that the linear forms of the stress (4) and the chemical potential (13) are only justified in situations where eta_a c_a^max << 1 for a = 1...n, since then the linearisation of the elastic strain tensor in (1) is permitted."
  > BC: "We prescribe a traction free boundary condition on Gamma_e" (sigma.n = 0) and "continuity of normal stress and displacement between anode materials".
- **E8** Shah et al. 2022 (E2) p.2: "Finite deformation kinematics and mechanics theories describe Li insertion-induced volume change and the resulting stress fields in the active particles and the conducting matrix." and "The large volumetric changes upon (de)lithiation make it necessary to formulate the governing equations using finite deformation kinematics."
- **E9** Mesgarnejad & Karma, "Vulnerable Window of Yield Strength for Swelling-Driven Fracture of Phase-Transforming Battery Materials", arXiv:1908.02175 (S18) abstract: "a multi-physics phase-field approach to model self-consistently anisotropic phase transformation, elasto-plastic deformation, and crack initiation and propagation during lithiation of Si nanopillars."
- Neo-Hookean vs St Venant-Kirchhoff comparison for lithiation: no source found (see Not found).

### 3. Pores / electrolyte, contact, mesh distortion
- E2/E3 give three-phase explicit models (active/CBD/pore), with the pore as a separate phase carrying electrolyte for transport; no stiffness-ratio value for a soft pore phase was found. See Not found.

### 4. Boundary conditions and stack pressure
- **E10** Cannarella & Arnold, "Stress evolution and capacity fade in constrained lithium-ion pouch cells", J. Power Sources 245 (2014) 745-751, DOI 10.1016/j.jpowsour.2013.06.165 (pp.1-2 read, S23):
  > "Compressive stack pressure is present in all lithium-ion batteries and is used to maintain intimate contact between battery components as well as to prevent layer delamination and deformation during operation. This stress is applied during manufacturing when the electrode stack is placed into a rigid constraint and is typically in the range of 0.1-1 MPa. Examples of rigid constraints are the rigid housings placed over pouch cells in design applications or the canisters of cylinder or prismatic cells."
  Table 1 (stack pressures in MPa; Initial / Min / Max): Low 0.05 / 0 / 0.5; Medium 0.5 / 0.2 / 1.5; High 5 / 1 / 3. Cells: 500 mAh LCO/graphite pouch, 25 mm x 35 mm x 6.5 mm (no silicon).
- **E11** Search snippets (S6, S22; not verified at source): "Stack pressure-dependent performance shows a nonmonotonic trend, with moderate pressure (~80 MPa) delivering optimum capacity and lowest impedance." (source page not identified; context unverified). Titles seen: "Mechanical behavior of Silicon-Graphite pouch cells under external compressive load: Implications and opportunities for battery pack design" (J. Power Sources, S037877532030077X); "Effects of Mechanical Compression on the Aging and the Expansion Behavior of Si/C-Composite|NMC811 in Different Lithium-Ion Battery Cell Formats" (J. Electrochem. Soc., DOI 10.1149/2.1121915jes); "Stress Evolution in Lithium-Ion Composite Electrodes during Electrochemical Cycling and Resulting Internal Pressures on the Cell Casing" (JES, DOI 10.1149/2.0341514jes); "Si-C/G based anode swelling and porosity evolution in 18650 casing and in pouch cell" (S0378775321010508). Values from these not retrieved.
- Plane strain vs plane stress for electrode cross-sections; current-collector constraint; periodicity: no source retrieved (E2 uses a 2D axisymmetric geometry).

### 5. Uniform SOC vs diffusion
- **E12** Roper et al. (E7), pp.2-4: "We focus on a time-scale much slower than the diffusion of lithium through the nano-particle allowing us to use a quasi-static approximation, focusing on the stresses induced by the different expansions of the different materials in chemical and mechanical equilibrium." and eq (10) "c* grad mu* = 0 ... implying the chemical potential of the lithium is uniform in each anode material for all non-trivial concentrations". SOC closure: "we describe this as a proportion of the maximum amount possible using the state of charge parameter c0 in [0,1], where c0 = 0 denotes no lithium and c0 = 1 denotes a fully lithiated anode." Also: "silicon can accommodate 3.75 lithium atoms per silicon atom, forming the alloy Li3.75Si".
- **E13** Diffusivity values, search-result summaries (S5; not verified at source, tool paraphrase): "Reported Li diffusivity values [in a-Si] range from 10^-13 m2/s to 10^-19 m2/s."; "Li diffusivity was estimated at 2 x 10^-17 m2/s for Li-poor phase and about 2 x 10^-15 m2/s for Li-rich phase in amorphous silicon nanospheres."; "Graphite diffusivity spans from 2x10^-14 to 3.3x10^-9 m2/s"; "average diffusion coefficient of lithium-ions in graphite was determined as 8.7 (+/-0.4) x 10^-12 cm2/s in the direction perpendicular to graphene planes and 4.4 (+/-0.1) x 10^-6 cm2/s in the direction parallel to graphene planes." Source pages seen as titles: "Study of lithium diffusivity in amorphous silicon via finite element analysis" (J. Power Sources, S0378775315306947); "Lithium Diffusion in Graphitic Carbon" (arXiv:1108.0576); "Lithium-ion diffusion behaviour in silicon nanoparticle/graphite blended anodes" (J. Power Sources, S0378775325004598); GITT graphite (PMC8397968, blocked). Critical C-rate / particle-size criteria for uniform lithiation: no source retrieved.
- **E14** Sugunan, Jiang, Guo, Wang, Marinescu, Offer, "Modelling Carbon Coated Silicon Anodes for Lithium-Ion Batteries and the Influence of Contact Area on Rate Performance", arXiv:2509.13947 (S20) abstract (via fetch tool, paraphrase with quoted fragments): "substantial volume expansion (300-400%) during lithiation, leading to mechanical degradation and capacity fade"; model "incorporates lithium transport, interfacial kinetics, evolving contact area from silicon expansion, and cracking frameworks"; examines "effects of particle size, shell thickness, and charging protocols".

### 6. 2D vs 3D; SliceGAN
- **E15** Kench & Cooper, "Generating 3D structures from a 2D slice with GAN-based dimensionality expansion", arXiv:2102.07708 (Feb 2021); published as Nature Machine Intelligence (2021), DOI 10.1038/s42256-021-00322-1 (S13):
  > "this is especially relevant for the task of material microstructure generation, as a cross-sectional micrograph can contain sufficient information to statistically reconstruct 3D samples."
  > "generation time for a 10^8 voxel volume is on the order of a few seconds."
  Search snippet (S2, paraphrase): "The ability to statistically reconstruct anisotropic microstructures with a simple extension is also demonstrated." Follow-up seen: "Reusability report: Feature disentanglement in generating a three-dimensional structure from a two-dimensional slice with sliceGAN", Nat. Mach. Intell., DOI 10.1038/s42256-021-00400-4.
- **E16** S16 snippets (tool paraphrase, not verified): "Although 2D analysis has been widely used to investigate electrochemical behavior due to its computational simplicity, it is inherently limited in representing realistic electrode microstructures." (attributed to "A New Three-Dimensional Microstructure-Resolved Model to Assess Mechanical Stress in Solid-State Battery Electrodes", ResearchGate 396252357). Also seen by title only: "The electrode tortuosity factor: ..." (npj Comput. Mater., DOI 10.1038/s41524-020-00386-4) and "Tortuosity of Battery Electrodes: Validation of Impedance-Derived Values and Critical Comparison with 3D Tomography" (JES, DOI 10.1149/2.0231803jes). No quantitative 2D-vs-3D mechanics error located.

### 7. FEniCSx
- **E17** <https://docs.fenicsproject.org/dolfinx/main/cpp/demos/demo_hyperelasticity.html> (S19), fetch-tool output with quoted fragments:
  > "This demo illustrates how to: Solve a nonlinear elasticity problem with a Newton solver; Compute the Jacobian of a nonlinear form using automatic differentiation; Evaluate a derived quantity (the Cauchy stress) at points in the domain using `dolfinx::fem::Expression`"
  Energy as reported: psi(F) = (mu/2)(tr(C) - 3) - mu ln(J) + (lambda/2)(ln(J))^2 (compressible neo-Hookean, 3D, unit cube with one face rotated 60 degrees). Solver as reported: PETSc SNES with direct LU.
- **E18** <https://newfrac.gitlab.io/newfrac-fenicsx-training/02-finite-elasticity/finite-elasticity-I.html> (S8):
  > "psi(J, I_C) = (mu/2)(I_C - 2) - mu ln(J) + (lambda/2) ln(J)^2"  (the I_C - 2 form)
  > "DOLFINx has various Newton solvers already built-in... Here we will use the basic `dolfinx.cpp.nls.NewtonSolver`."
  > solver.atol = 1e-8; solver.rtol = 1e-8; solver.convergence_criterion = "incremental"
  Fetch tool reported the page has no load-stepping, thermal, eigenstrain or growth content.
- S4 snippet (tool paraphrase, source page not identified): "Load stepping improves convergence of the nonlinear solver by solving multiple sub-problems with increasing load steps, where each step i is solved with a scaled right-hand side, and the obtained solution in iteration i is used as the initial guess for the subsequent load step."

## Not found
- Neo-Hookean vs St Venant-Kirchhoff comparison for lithiation swelling, and a quantitative threshold at which small strain fails (S3, S10; only the qualitative "eta c_max << 1" condition in E7).
- Original Bower et al. 2011 and Gao et al. papers; Zhao 2011 full text (S11 SSL error).
- Void removal vs ersatz soft material, stiffness ratio values, pore closure/contact handling, mesh distortion limits in battery models (S17 irrelevant; S1, S3 no hits).
- Plane strain vs plane stress choice and periodic/lateral BCs for electrode cross-section models; current-collector constraint statements (S1, S10, S21 pp.1-2 only).
- Quantitative 2D-vs-3D comparison for electrode mechanics (S16).
- Uniform-lithiation criteria (C-rate, particle size, diffusion-time scaling) (S5).
- dolfinx example with eigenstrain/thermal-expansion/growth Fe*Fg and load stepping (S4, S8, S19).
- Verbatim text of: Nature Machine Intelligence SliceGAN page, Commun. Chem. FIB-SEM paper, graphite GITT paper, J. Power Sources pouch-cell pages (S7, S9, S12, S14, S15 blocked).

## Not checked
- Full text of Shah et al. 2022 beyond pp.1-2 (BCs, constitutive law, mesh) - depth cap.
- Mechanical behavior of Si-graphite pouch cells under compressive load (S037877532030077X) and Si/C pouch pressure papers - paywalled, not fetched.
- SliceGAN reusability report and any SliceGAN-based battery FE follow-ups.
- FEniCS Discourse / dolfinx_materials / Dokken tutorial searches for growth or thermal-strain hyperelasticity - only 2 search rounds run.
