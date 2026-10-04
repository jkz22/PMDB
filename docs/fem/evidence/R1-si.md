## Question
Web literature evidence-gathering for a 2D finite-strain, mechanics-only FEM of SEM cross-sections of a graphite anode containing silicon (Si appears in BSE as compact, finely grained particles a few µm across — could be crystalline Si, Si/C composite, or SiOx; gather evidence for all three). Lithiation is applied as an SOC-dependent eigenstrain (F = Fe·Fλ).

Gather, with quoted values, units, citation (authors, year, journal, DOI/URL) for each:
1. Volumetric/linear expansion of Si vs Li content (LixSi, x up to 3.75, Li15Si4): total value (~280-300%?), is it linear in x (partial molar volume), amorphous vs crystalline. Same for SiOx (x≈1) and Si/C composites.
2. Elastic modulus and Poisson ratio of Si vs Li content (crystalline ~160 GPa → lithiated ~?), measurements (nanoindentation, DFT). Same for SiOx if available.
3. Yield / flow stress of lithiated Si (e.g. Sethuraman et al. ~1 GPa), plasticity models used in FEM.
4. Practical Si utilisation in commercial Si/graphite anodes: what "100% SOC" corresponds to for the Si (full Li15Si4 or capped), typical Si weight fractions (5-15%?).
5. Fracture/critical size of Si particles (e.g. ~150 nm crystalline nanoparticles, Liu et al. 2012) — useful for interpreting stress outputs.

## Caveat on provenance
WebSearch returns tool-written summaries; WebFetch returns model-extracted text (several PDFs/publisher pages returned 403, binary, captcha or redirect). Items below in quotation marks are the text as returned by those tools, not read from the original page by me, unless noted. Where the text is a tool summary rather than the source's words it is marked "(tool summary)".

## Search log
- **S1** WebSearch — Beaulieu Dahn volume expansion LixSi in situ AFM — 10 results
- **S2** WebSearch — Shenoy Johari elastic softening lithiated a-Si DFT — 10 results
- **S3** WebSearch — Sethuraman Srinivasan yield stress lithiated a-Si 1 GPa — 10 results
- **S4** WebSearch — Liu 2012 size-dependent fracture Si nanoparticles 150 nm — 10 results
- **S5** WebSearch — commercial Si graphite anode Si fraction / SiOx blend — 9 results
- **S6** WebSearch — SiOx volume expansion / modulus / nanoindentation — 9 results
- **S7** WebFetch pubmed.ncbi.nlm.nih.gov/22217200 — cookie wall, no abstract
- **S8** WebFetch arxiv.org/pdf/1108.0372, 1108.0567, 1311.5844 — binary PDF, unreadable
- **S9** WebFetch sciencedirect S0378775314014827 (Berla) — 403
- **S10** WebFetch PMC9230857 (Jo 2022 SiOx-graphite) — content returned
- **S11** WebFetch iopscience 10.1149/1.1613668 (Beaulieu 2003) — abstract returned
- **S12** WebFetch arxiv.org/abs/1108.0372, 1108.0567, 1311.5844 — abstracts returned
- **S13** WebSearch — Bower/Guduru finite-strain elastic-plastic model — 10 results
- **S14** WebSearch — Berla/Hertzberg nanoindentation lithiated Si — 10 results
- **S15** WebSearch — Si/C composite expansion, Si utilisation at 100% SOC — 9 results
- **S16** WebFetch Stanford Lucas_Cui_Nix_JPS_2014.pdf — binary, unreadable
- **S17** WebFetch PMC12679607 — captcha; WebFetch doi 10.3390/batteries11110423 — redirect, then mdpi 403
- **S18** WebFetch PMC5046981 (Meca et al. 2016) — values returned
- **S19** WebSearch — Obrovac Li15Si4 / Li3.75Si — 10 results
- **S20** WebSearch — SiOx expansion 118/160%, Li4SiO4, modulus — 9 results
- **S21** WebFetch nature.com srep01615 and ncomms9417 — auth redirect, not fetched; sciencedirect S0378775317309655 — 403
- **S22** WebSearch — Si-graphite commercial (LG M50 etc.) Si fraction/utilisation — 9 results
- **S23** WebSearch — LixSi Poisson ratio first-principles — 10 results
- **S24** WebSearch — lithiated Si yield strength/modulus/FE composite — 9 results

## Evidence

### 1. Expansion
- **E1** <https://iopscience.iop.org/article/10.1149/1.1613668> — Beaulieu, Hatchard, Bonakdarpour, Fleischauer, Dahn, "Reaction of Li with Alloy Thin Films Studied by In Situ AFM", J. Electrochem. Soc. 150 (2003) A1457 (S11, S1)
  > "Although these materials all undergo large volume expansions, the amorphous phases undergo reversible shape and volume changes."
  > "The crystalline materials do not. We attribute this difference to the homogeneous expansion and contraction that occurs in the amorphous materials."
  > "Inhomogeneous expansion occurs in the crystalline materials due to the presence of coexisting phases with different Li concentrations."
- **E2** same paper via S1 (tool summary): "theoretical volume change of 280% from c-Si to c-Li3.75Si (equivalent to Li15Si4)"; "volume changes of up to 300% ... in silicon thin films with AFM ... roughly linear with Li content."
- **E3** <https://pmc.ncbi.nlm.nih.gov/articles/PMC5046981/> — Meca et al. 2016, Proc. R. Soc. A / Math. Phys. Eng. Sci. (S18, tool-extracted quotes)
  > "its volume increases by about 300% when fully lithiated"
  > "large volume changes, estimated to be about 280%"
- **E4** S19 (tool summary): "Obrovac and Christensen first reported that crystalline Li15Si4 forms when the potential is less than 0.05 V vs. Li"; "terminal stage of lithiation that can be attained at room temperature is a metastable crystalline phase, where x = 3.75 (c-Li3.75Si), typically at voltages below 50 mV vs Li."; "Li15Si4 formation was found to be coincidental with capacity fade and delamination of the Si film" (Iaboni & Obrovac, "Li15Si4 Formation in Silicon Thin Film Negative Electrodes", J. Electrochem. Soc. 2016; primary Obrovac & Christensen, Electrochem. Solid-State Lett. 7 (2004) A93 — citation from memory, not verified here).
- **E5** <https://www.nature.com/articles/srep01615> — "Anisotropic Compositional Expansion and Chemical Potential for Amorphous Lithiated Silicon under Stress Tensor", Sci. Rep. 2013 — appears in S2/S13 results; page not fetched (S21). Title only.
- **E6** SiOx (S6, S20, tool summaries): "SiOx-based anodes have a much smaller volume change of approximately 160% during lithiation/delithiation"; "the Li15Si4 phase shows initial volume expansion of approximately 280%, which is much larger than the 200% of Li4SiO4 or Li2O." Sources not individually identified; the 160% claim is attributed in results to a review "Recent advancement of SiOx based anodes for lithium-ion batteries", J. Power Sources (2017) <https://www.sciencedirect.com/science/article/abs/pii/S0378775317309655> (not fetched, 403).
- **E7** <https://pmc.ncbi.nlm.nih.gov/articles/PMC9230857/> — Jo et al. 2022, Nanomaterials, DOI 10.3390/nano12121956 (S10, tool-extracted)
  > "thickness of the SiOx blending electrode expanded from 75 μm to 77 μm (after 250 cycles) and 90 μm (after 500 cycles), representing 103% and 120% expansion rates"
  Si microparticle blend: 194% after 250 cycles and 198% after 500 cycles (electrode thickness).
- **E8** <https://doi.org/10.3390/batteries11110423> — "A Multi-Physics Coupled Model for Elucidating Expansion in Si–C Composite Anode Lithium-Ion Batteries", Batteries 2025 (S15; abstract-level tool summary only): "when silicon particles undergo volume changes exceeding 300%, the overall cell expansion remains below 7.5% due to structural dilution effects from other components." Also S15: "Silicon (de)lithiation causes severe volume expansion and contraction—often exceeding 300%"; and "Although silicon accounts for only a small fraction of anode mass, it can contribute 30% to the capacity of the cell".
- **E9** S6 (tool summary, source unidentified): "silicon undergoes large volume expansion (approximately 370%) upon complete lithiation" (likely Li22Si5; not verified).

### 2. Modulus / Poisson
- **E10** Shenoy, Johari, Qi, "Elastic softening of amorphous and crystalline Li-Si phases with increasing Li concentration: A first-principles study", J. Power Sources 195 (2010) 6825-6830 (S2, tool summary): "Young's modulus decreases approximately linearly from 90 GPa to 20 GPa as Li concentration increases from zero to full capacity." Other results (S6, S24, tool summaries): "Young's modulus of amorphous silicon decreasing from about 90 GPa for α-Si to less than 40 GPa for α-Li15Si4"; "The moduli for the most highly-lithiated phases, Li15Si4 and Li22Si4 [sic], are nearly an order of magnitude smaller than the corresponding values for Si". Note the two tool summaries give differing end values (20 vs <40 GPa); source not read directly.
- **E11** <https://arxiv.org/abs/1108.0567> — Sethuraman, Chon, Shimshak, Van Winkle, Guduru, "In situ Measurement of Biaxial Modulus of Si Anode for Li-ion Batteries" (S12; published Electrochem. Commun. 2010 — journal from memory)
  > "the biaxial modulus was seen to decrease from ca. 70 GPa for Li0.32Si to ca. 35 GPa for Li3.0Si."
- **E12** Berla, Lee, Cui, Nix, "Mechanical behavior of electrochemically lithiated silicon", J. Power Sources 273 (2015) 41-51 (S14, tool summary; primary page 403/binary): "Young's modulus and the hardness of lithiated silicon are found to decline with increasing lithium content"; "Young's modulus of fully lithiated silicon was measured to be 41 GPa"; "nanoindentation creep experiments demonstrated that lithiated silicon creeps readily ... power law creep with large stress exponents (>20)". URL <https://web.stanford.edu/group/cui_group/papers/Lucas_Cui_Nix_JPS_2014.pdf>.
- **E13** S24 (tool summaries): "Young's modulus for Li12Si7 was 52.0 ± 8.2 GPa" (nanoindentation, J. Power Sources 2012, <https://www.sciencedirect.com/science/article/abs/pii/S0378775312003874>); Li22Si5 polycrystal Young's modulus paper at <https://www.researchgate.net/publication/241095593_Young's_modulus_of_polycrystalline_Li_22Si_5> (value not retrieved).
- **E14** Meca et al. 2016 (S18, tool-extracted): "E_LixSi/E_Si = 49 for the ratio of the Young moduli" (citing Shenoy et al. 2010); "ν = 0.25 for Poisson's ratio".
- **E15** S23 (tool summary, source unidentified; probably Qi/Shenoy first-principles, IOP J. Electrochem. Soc. 161 (2014) F3010 "Lithium Concentration Dependent Elastic Properties of Battery Electrode Materials from First Principles Calculations", <https://iopscience.iop.org/article/10.1149/2.0031411jes>): "Pure silicon has a Poisson's ratio of 0.21, while fully lithiated Li4.4Si has a Poisson's ratio of 0.23."
- **E16** Crystalline Si ~160 GPa: not retrieved; Hopcroft et al. "What is the Young's Modulus of Silicon?", J. Microelectromech. Syst. 19(2), 2010 appeared in S23 results at <https://sites.engineering.ucsb.edu/~sumita/courses/Courses/ME141B/HopcroftJMEMS.pdf> (not fetched, no value quoted).
- **E17** SiOx modulus: a result "High-strength and high-modulus silicon monoxide for high-energy-density and fast-charging lithium-ion batteries", <https://www.nature.com/articles/s41467-026-72434-4> appeared in S24; not fetched; no values retrieved.
- **E18** Spring 2026 Springer result "Regulation Mechanisms of Component Parameters on Modulus in Silicon-Based Composite Anodes..." <https://link.springer.com/article/10.1007/s10338-026-00706-z> and "Effective modulus of Si electrodes considering Li concentration, volume expansion, pore, and Poisson's ratio" <https://link.springer.com/article/10.1007/s12206-021-0427-1> appeared in S2; not fetched.

### 3. Yield / flow stress and FE plasticity
- **E19** Sethuraman, Srinivasan, Bower, Guduru, "In Situ Measurements of Stress-Potential Coupling in Lithiated Silicon", J. Electrochem. Soc. 157(11) (2010) A1253, <https://iopscience.iop.org/article/10.1149/1.3489378> (abstract from S12): "the relation between stress change and electric-potential change is measured to be 100 - 120 mV/GPa". The "flow stress ≈ 1 GPa" statement is from S3 (tool summary): "Sethuraman et al. (2010) observed plastic flow in Si electrodes during lithiation and de-lithiation, concluding that the material has a flow stress of approximately 1 GPa." Original wording not read.
- **E20** <https://arxiv.org/abs/1311.5844> — Bucci, Nadimpalli, Sethuraman, Bower, Guduru, "Measurement and modeling of the mechanical and electrochemical response of amorphous Si thin film electrodes during cyclic lithiation", J. Mech. Phys. Solids 62 (2014) 276 (journal from memory) (S12)
  > "Parameters extracted from the experiment include the variation of elastic modulus and the flow stress as functions of Li concentration; the strain rate sensitivity; the diffusion coefficient for Li transport in the electrode..."
  Numeric flow stress vs concentration not retrieved (PDF unreadable, S8).
- **E21** Bower, Guduru, Sethuraman, "A finite strain model of stress, diffusion, plastic flow, and electrochemical reactions in a lithium-ion half-cell", J. Mech. Phys. Solids 59 (2011) 804, <https://arxiv.org/pdf/1107.6020> (S13, tool summary): "kinematics based on the multiplicative decomposition of the deformation gradient into elastic and compositional parts, where the compositional part represents both volume and shape changes"; "The amorphous lithiated silicon phase is modeled as a perfect elastic-plastic material, and plastic deformation is considered when stresses exceed the yield strength." Not read directly.
- **E22** Meca et al. 2016 (S18): "The maximum stress measured in the experiments does not reach the predicted yield stress for amorphous silicon"; "the value of the yield stress is significantly lower than the one expected".
- **E23** Other FE/plasticity-related titles surfaced (not fetched): "Vulnerable Window of Yield Strength for Swelling-Driven Fracture of Phase-Transforming Battery Materials" <https://arxiv.org/pdf/1908.02175>; "Intrinsic stress mitigation via elastic softening during lithiation" (Teng Li, JMPS 2016) <http://lit.umd.edu/publications/TengLi-Pub74-JMPS-2016.pdf>; "Mechanical-electrochemical modeling of silicon-graphite composite anode for lithium-ion batteries", J. Power Sources 2022 <https://www.sciencedirect.com/science/article/abs/pii/S0378775322001987>; "A composite electrode model for lithium-ion batteries with silicon/graphite negative electrodes" <https://www.sciencedirect.com/science/article/pii/S0378775322001604>; "A free volume-based analytical model for plastic flow in thin-walled silicon structures" <https://link.springer.com/article/10.1007/s00707-021-03121-2>.

### 4. Commercial utilisation / weight fractions
- **E24** Jo et al. 2022 (S10, tool-extracted): "currently, up to 5 wt% Si mixed with graphite anodes is utilized for commercial anodes"; own blend 6.9 wt% SiOx / 93.1 wt% natural graphite, ~3.5 mAh/cm2, 440 mAh/g; SiOx-graphite 519 mAh/g, ICE 90.0%, 83.4% retention at 0.2C/100 cycles (S5).
- **E25** S22 (tool summaries; sources: OSTI 1491439, ResearchGate LG M50 parameter tables, ACS Appl. Energy Mater. acsaem.2c02047, JES/IOP add112): "In the LG M50 anode, the Si phase is partially lithiated and hence amorphous, with approximately 5% by weight of Si mixed with graphite"; "the silicon content in the composite electrode of the LG M50T has been determined to be 2.43% using five complementary, cross-validated methods"; "relative composition in terms of capacity was found to be 0.85 Gr: 0.15 Si at beginning of life"; "silicon capacity fraction of 13.1% was found to match measured data best for the LG M50T"; "The LG MJ1 18650 cell has a silicon content of approximately 3.5 wt%."
- **E26** S5 (tool summaries): "Studies systematically evaluate Si content (5–20 wt %) in commercial graphite" (Relationship between Silicon Percentage in Graphite Anode..., ACS Appl. Mater. Interfaces 2024, <https://pubs.acs.org/doi/abs/10.1021/acsami.4c10178>); "for composites consisting of ~54-60% silicon by weight, the silicon contribution to first cycle discharge capacity is estimated to be ~2570-2850 mAh g⁻¹, suggesting ~72-80% utilization of theoretical silicon capacity" (source not identified); IOP "Towards Improving the Practical Energy Density of Li-Ion Batteries: Optimization and Evaluation of Silicon:Graphite Composites in Full Cells" <https://iopscience.iop.org/article/10.1149/2.0481701jes> (not fetched).
- No source retrieved that states explicitly the Si state (x in LixSi) at 100% SOC in a commercial cell beyond "partially lithiated" (E25).

### 5. Fracture / critical size
- **E27** Liu, Zhong, Huang, Mao, Zhu, Huang, "Size-dependent fracture of silicon nanoparticles during lithiation", ACS Nano 6(2) (2012) 1522-1531, DOI 10.1021/nn204476h, <https://pubmed.ncbi.nlm.nih.gov/22217200/> (S4, tool summary; PubMed page itself returned cookie wall, S7): "A critical particle diameter of approximately 150 nm was discovered, below which particles neither cracked nor fractured upon first lithiation, and above which particles initially formed surface cracks and then fractured"; "surface cracking resulted from the buildup of large tensile hoop stress in the surface layer ... lithiation in crystalline Si occurring through movement of a two-phase boundary between the inner pristine Si core and the outer amorphous Li-Si alloy shell."
- **E28** Other fracture-related titles surfaced, not fetched: "High damage tolerance of electrochemically lithiated silicon", Nat. Commun. 2015 <https://www.nature.com/articles/ncomms9417>; "Failure mechanisms of single-crystal silicon electrodes in lithium-ion batteries" <https://arxiv.org/pdf/1606.06283>; "In situ atomic-scale imaging of electrochemical lithiation in silicon", Nat. Nanotechnol. 2012 <https://www.nature.com/articles/nnano.2012.170>; "Fully coupled multiphysics modelling of fracture behaviour in silicon particles ... phase-field" <https://link.springer.com/article/10.1007/s00366-025-02233-w>; "Anisotropic Volume Expansion of Crystalline Silicon during Electrochemical Lithium Insertion" Nano Lett. <https://pubs.acs.org/doi/abs/10.1021/nl3027197>. No values retrieved.

## Not found
- Verbatim original-text quotes for Shenoy 2010, Berla 2015, Bucci 2014 numeric flow stress, Bower 2011, Obrovac 2004 (S2, S8, S9, S16, S21: 403/binary/redirect).
- Numeric flow/yield stress as function of x from Bucci et al. (S8, S12).
- Crystalline Si Young's modulus (~160 GPa) value quoted (S23 gave only a Hopcroft link).
- SiOx Young's modulus / lithiated SiOx elastic properties (S6, S20, S24 — only a title surfaced, E17).
- Explicit Si-phase lithiation state (x) at 100% SOC in commercial Si/graphite cells (S15, S22 — only "partially lithiated").
- Linear-in-x partial molar volume numeric value (Å3 per Li, or Ω) — only "roughly linear" (E2).
- Si/C composite (Si nanoparticle in carbon matrix) particle-level expansion values; only cell-level (E8).
- Primary source for the "~370%" and "160%" SiOx figures (E6, E9).

## Not checked
- Sethuraman 2010 original Electrochem. Commun./JES PDFs and Bucci 2014 full text (PDFs unreadable; saved binaries exist under <local scratch file> and could be read with a PDF reader).
- Hertzberg et al. nanoindentation of lithiated Si (named in S14 summary, no link).
- Li-Si-O silicate/Li2O mechanical properties; SiOx particle fracture studies.
- Lithiated Si Poisson's ratio measurements (only DFT, E15, and a model input, E14).
- Depth cap reached after two search rounds per topic.
