## Question
Round-2 evidence gathering for a literature review on a 2D finite-strain mechanics FEM of Si/graphite anode SEM cross-sections. Gather, with verbatim quotes, page/table/figure numbers, and citations:
1. Yao et al. 2019 (Adv. Energy Mater. 9, 1803380) PDF pages 9-end: per-component Si and graphite lithiation fraction vs capacity/SOC for the 15 wt% Si blend.
2. arXiv 1511.02445 PDF from page 17: Table 2 (PVdF E and nu, other phase properties).
3. Si lithiation state (x in LixSi or Si utilisation) at 100% SOC in commercial Si/graphite full cells (LG M50/MJ1 parameterisation papers: Chen 2020 JES, O'Kane, Ai, Schmitt, Richter), and blended-anode models.
4. Primary sources for SiOx (x~1) volume expansion on lithiation and SiOx elastic modulus.
5. Measurement basis (thickness vs volume, first lithiation vs cycled, electrode composition) of dilatometry paper DOI 10.1149/1945-7111/abd465.
6. Evidence on graphite flake c-axis texture/orientation in calendered anodes (XRD texture, I004/I110, EBSD).
Log searched-and-not-found. No conclusions or recommendations.

Provenance flag: items marked (tool summary) are WebFetch/WebSearch model outputs, not text I read directly from the source; items marked (pdftotext) were read directly from locally saved PDFs. Quotes in WebFetch outputs were returned in quotation marks by the tool but I could not check them against the page.

## Search log
- **S1** Bash pdftotext -layout -f 9 on webfetch-1791052225504-ewz1th.pdf (Yao 2019, local) -> /tmp/yao9.txt, 534 lines, OK
- **S2** Bash pdftotext -layout -f 17 on webfetch-1791052235431-bev6rh.pdf (arXiv 1511.02445, local) -> 148 lines, Table 1 and Table 2 found
- **S3** WebSearch Chen 2020 JES parameterization LG M50 - 9 hits (abstract only; confirms "NMC 811 positive electrode and bi-component Graphite-SiOx negative electrode")
- **S4** WebSearch Si lithiation fraction at 100% SOC LG M50 - 9 hits, no explicit Si lithiation number
- **S5** WebSearch SiOx volume expansion / modulus nanoindentation - 9 hits
- **S6** WebSearch DOI abd465 dilatometry - 9 hits
- **S7** WebSearch graphite calendering I004/I110 c-axis - 9 hits
- **S8** WebFetch osti.gov/pages/biblio/1755253 - abstract only
- **S9** WebFetch iopscience abd465/ampdf - redirect to bot-check (blocked)
- **S10** WebFetch arxiv.org/html/2607.11521 - OK (tool summary)
- **S11** WebFetch sciencedirect S0378775322001604 (composite electrode model) - 403
- **S12** WebFetch malvernpanalytical AN230120 - OK (tool summary)
- **S13** WebFetch osti.gov/servlets/purl/1755253 - 404; curl same - OSTI "Page Not Found"
- **S14** WebFetch research.birmingham.ac.uk Chen 2020 - OK but landing page only (no numbers)
- **S15** WebFetch discovery.ucl.ac.uk Michael 2021 PDF (this is DOI 10.1149/1945-7111/abd648, a different paper) - binary, saved; pdftotext OK
- **S16** curl pure-oai.bham.ac.uk Chen PDF, iopscience ab9050/pdf - returned HTML bot pages, not PDFs
- **S17** WebFetch iopscience abd465 (abstract page) x2 - OK (tool summary; gave Table I and results)
- **S18** WebFetch semanticscholar (abd465, Malifarge) - 403
- **S19** WebFetch iopscience ab9050 (Chen 2020), add112 (Is Silicon Replaceable) - redirected to bot-check (blocked)
- **S20** WebFetch sciencedirect S0378775324002076 (Bonkile "Is silicon worth it") - 403; curl of Imperial Spiral open copy OK -> /tmp/spiral.txt (778 lines)
- **S21** WebFetch researchgate Chen 2020 - 403; docs.pybamm.org parameter sets - 404
- **S22** WebFetch pubs.acs.org Kirkaldy 2022 - 403; curl PMC9709825 OK, grepped (no explicit Si lithiation at 100% SOC)
- **S23** WebFetch arXiv 2305.15599 (Hybrid-MPET) - binary, saved, pdftotext + grep
- **S24** curl arXiv 2508.06413 and 1901.01491, pdftotext + grep
- **S25** WebSearch SiO volume expansion 200%; WebSearch SiOx modulus; WebFetch OSTI 1632823 (Kirner, Argonne Gr-SiO) binary -> pdftotext OK; skku review 404; PMC12781111 (tool summary); nature s41467-026-72434-4 redirect to login (blocked); OSTI 1466681 abstract only; arXiv 2304.06817 fetched, returned irrelevant text
- **S26** WebSearch Steinhauer/Malifarge XRD orientation; WebSearch calendering I004/I110; WebSearch EBSD graphite anode - results listed; WebFetch acsami.1c19735, S2949821X26000864, researchgate 312645676 - all 403
- **S27** grep orient/c-axis/texture in local PDFs arXiv 2005.04983 and Qi 2014 - no graphite orientation hits (only LiFePO4 "anisotropic")

## Evidence

### Item 1: Yao 2019 (pdftotext of local PDF, SI included; the PDF page numbers below are the numbered pages printed in the file)
- **E1** Yao et al., Adv. Energy Mater. 2019, 9, 1803380 (local PDF webfetch-1791052225504-ewz1th.pdf, p.10-11) - text on Figure 3/Figure 4a, Gr vs Si capacity partition (15 wt% Si-Gr half-cell, C/30)
  ```
  lithiation of Gr becomes significant only when the potential decreases below 0.20 VLi.
  Furthermore, while the Gr electrode attains the theoretical capacity at the end of lithiation cycle,
  the Si-Gr electrode only attains ~93% of the theoretical capacity. This observation indicates the
  presence of Si particles that are not easily accessed during lithiation even at the slow C/30
  ...
  The relative contributions of the Gr and Si components to the net capacity are shown in
  Figure 4a. It is seen from this figure that during the initial lithiation (region 1, 1.0 – 0.20 VLi),
  Li mainly alloys with the Si; the lithiation ratio LiSi/LiGr of Si vs. Gr is 0.96/0.04 in this range.
  That is, when the electrode capacity during lithiation is 1.5 mAh, the capacity of the Si and Gr
  components are ~1.44 mAh and ~0.06 mAh, respectively. In region 2, the relative fraction of
  Gr lithiation increases as the electrode potential decreases to 0.20 VLi, below which most of the
  Gr lithiation is expected.[7, 19, 22] In this region (0.01 – 0.2 VLi), the average LiSi/LiGr is
  ~0.58/0.42; i.e., the lithiation of Si occurs preferentially over Gr even in this region. During the
  initial delithiation (region 3, 0.01 – 0.22 VLi), almost all Li+ ions are extracted from Gr, and the
  LiSi/LiGr ratio is zero. ... In region 4, almost all Li+ ions are extracted
  from Si, and in this region (0.22 – 1.0 VLi) the LiSi/LiGr ratio is ~ 0.97/0.03.
  ```
- **E2** same PDF, p.11-12 - Figure 4b text (specific capacity per component)
  ```
  Figure 4b shows the specific capacity in the individual components plotted as a
  function of the net specific capacity. When the potential reaches 0.2 VLi during lithiation, the
  specific capacity in the Gr is negligible, whereas for the Si particles it is ~1350 mAh.g-1Si, which
  corresponds to a volume expansion of 135 % (assuming a linear expansion rate of 100% per
  1000 mAh∙g-1Si). [9] When the potential is 0.01 VLi during lithiation, the specific capacity of the
  Gr is ~340 mAh.g-1Gr, whereas for the Si particles it is ~3400 mAh.g-1Si, which corresponds to
  a volume expansion of 340 %. On delithiation, the Si particles remain at ~340 % expansion
  until the electrode potential reaches 0.20 VLi since Li+ ions are extracted only from the Gr
  particles in the 0.01 – 0.2 VLi range.
  ```
- **E3** same PDF, p.10-11 - Figure 3d/delithiation staging, and Fig. 4 caption
  ```
  Stage I’ appears when the capacity decreases below 60% that corresponds to ~366 mAh∙g-
  Gr. ... Li extraction from the Si component occurs only when
  the capacity decreases below 60% at potentials exceeding 0.23 VLi.
  ...
  Figure 4. Capacity (a) and specific capacity (b) of the Gr and Si components in a Si-Gr
  electrode during lithiation and delithiation. ... The percentages shown in panel (a) are the estimated expansions of Si particles.
  ```
- **E4** same PDF, Experimental section (p.~14) - electrode composition and testing
  ```
  electrode was prepared from a mixture of 88 wt% Gr, 2 wt% carbon, and 10 wt%
  LiPAA; the Si-Gr composite electrode was prepared from a mixture of 15 wt% Si, 73 wt% Gr,
  2 wt% conductive carbon, and 10 wt% LiPAA. ...
  The Si-Gr electrode, with a
  material loading of 2.94 mg∙cm-2, was calendered to a porosity of 42.4% and 30 µm coating thickness.
  ...
  Each cell contained a lithium foil anode (99.9% purity, MTI Corp.), 40 µL of 1.2 M LiPF6 in ethyl carbonate/ethyl methyl carbonate (3:7 w/w ...),
  Celgard 2325 (PP/PE/PP) separator, and the working electrode (Gr or Si-Gr). For the Si-Gr cell,
  10 wt% fluoroethylene carbonate was used as an electrolyte additive ...
  Prior to the XRD experiment, the cells were cycled twice at a C/20 rate and 30 °C
  ```
  (Si particles: "silicon nanopowder (Nanostructured & Amorphous Materials, Inc., size = 50-70 nm)". Note: the quoted 30 um/42.4% in the file text; the Gr-only electrode is 48 um/34.1%.)
- **E5** same PDF, Supporting Information Table S1 and method for Si capacity
  ```
  Table S1. Composition and electrochemical characteristics of our electrodes
   Electrode Total active    Gr mass      Si mass      Carbon at        Si at        Theoretical
              mass [mg]        [mg]         [mg]          LiC6         Li15Si4        capacity
                                                        capacity      capacity         [mAh]
                                                         [mAh]         [mAh]
     Si/Gr         4.785           3.970       0.816         1.477        2.920         4.396
      Gr           8.833           8.833         0           3.286            0         3.286
  ...
  the amount of Li stored in the Si was
  estimated by subtracting the capacity stored in the Gr calculated from Eq. 1 (defined as x̄ ∙372
  mAh∙g-1Gr∙mGr, where mGr is the weight of Gr) from the total cell capacity that was determined
  using coulometry.
  ```
  (Si mass fraction of active material in Table S1: 0.816/4.785 mg by arithmetic of the table; not stated in the paper.)
- **E6** same PDF, p.13 - cutoff dependence statement
  ```
  Note, that the plots
  in Figure 4b will be affected by the Si content, which will alter capacity contributions of the Gr
  and Si components. The plots will also be affected by the lower and upper cutoff voltages
  ```
- Not in text: no per-component numeric table vs capacity or SOC beyond the figures (Fig. 4a/4b are plots; the text numbers above are the only numbers in the extracted text). Also: an accepted-manuscript copy of this paper is listed at <https://www.osti.gov/pages/servlets/purl/1496633> (search result title "Manuscript # aenm.201803380"); not opened.

### Item 2: arXiv 1511.02445 (Nadimpalli et al., local PDF, pdftotext, printed page 22 and 21)
- **E7** `/Users/Kevin/.claude/projects/-Users-Kevin-Documents-GitHub-PMDB--claude-worktrees-heldout-data/f7044de1-2c11-44c1-b109-e683c9c4f5ce/tool-results/webfetch-1791052235431-bev6rh.pdf` page 22 - Table 2
  ```
  Table 2: Mechanical properties and geometry of different layers of the specimen used in the study
  Si (111) wafer       E1 169 GPa; nu1 0.26; h1 450 µm; M1 228.3 GPa
  Epoxy layer          E2 4.3 GPa (Ref.[13]); nu2 0.36; h2 55 µm; M2 6.72 GPa
  Al current collector E3 70 GPa; nu3 0.334; h3 15 µm; M3 105 GPa
  Composite cathode coating  E4 40 GPa (Rule of mixtures); nu4 0.2 (Rule of mixtures); h4 35 µm; M4 50 GPa
  ...
  Graphite based anode coating
  E1  Young’s modulus  6.9 GPa  Ref. [13]
  nu1 Poisson’s ratio  0.3      Ref. [13]
  h1  Thickness        35 µm
  M1  Biaxial modulus  10 GPa   Calculated
  Copper current collector  E1 117 GPa; nu1 0.347; h1 15 µm; M1 179 GPa
  Celgard separator         E1 0.1 GPa (Ref. [28]); nu1 0.3; h1 20 µm (Measured); M1 0.14 GPa
  ```
  (Layout condensed from the pdftotext columns; values verbatim.) Table 2 contains NO separate row for PVdF; the only anode row is "Graphite based anode coating" (composite coating with PVDF binder). No explicit PVdF E or nu appears in this table.
- **E8** same PDF, page 21 - Table 1 (anode formulation)
  ```
  II. Anode
  ConocoPhillips: CGP-A12 graphite     89.8% wt.
  KF-9300 Kureha PVDF binder           6% wt.
  Timcal Super P                       4%wt.
  Oxalic Acid                          0.17% wt.
  Active-material loading density      5.61 mg/cm2
  Electrode porosity                   26%
  Thickness of the coating             40 µm
  Thickness of the Cu current collector 10 µm
  ```
- **E9** same PDF, Fig. 9 caption (p.20): "The pressure increases almost linearly with capacity during the first charge ... The peak pressure is ~1 MPa."

### Item 3: Si lithiation at 100% SOC
- **E10** Moon et al. (Samsung SAIT / NIMS), "Interplay of Inhomogeneous Electrochemical Reactions with Mechanical Responses in Silicon-Graphite Anode and its Impacts on Degradation", arXiv 1901.01491 (pdftotext of <https://arxiv.org/pdf/1901.01491>, main text lines ~198-203, ~424-446). Full cell, operando XRD, silicon-carbon composite (SSC) + graphite anode, 0.5C
  ```
  In the experiment of a half cell, we could not see internal redox couple. Lithiation in a working electrode is regulated by
  constant current until set potential, 0.01 V where lithium content in LixSi reaches x=3.75 [30]. However, lithiation in an anode
  of a full cell is regulated by the overall potentials difference of cathode and anode. The minimum potential of an anode (~0.1 V)
  during constant current (CC) charge measured by 3-electrode set up was higher than that measured by a half cell experiment ...
  ...
   So far, the anode electrode have designed on the assumption that the phase of LixSi would be x=3.75 at full lithiation. In the
  conventional design of electrode, the phase of Li3.75Si is hard to be reached in the full cell operation condition when the
  internal redox reaction dominates anode reaction at the end of charge as shown in figure 1.
  ...
  The revised anode electrode consisted of SSC (18 wt %), Graphite2 (79 wt %), and binder (3 wt %). With the revised design, the cross point of
  individual SOCs of SSC and that of graphite becomes higher from 72.9 % to 78.8 % SOC of anode ...
  By changing the design of electrode, the utilization of active materials in anode
  decreases from 91.5 % to 86.7 % ...
  ```
  and (line ~142-146) "3th position of a phase transition (red triangle at 65% SOC) of individual SOCs in Figure 1(b) corresponds to the phase transition from Li2.3Si to Li3.25Si [22, 23] that is located mostly on the surface of silicon". Also line 74: "commercialized silicon anode has been limited from 2 to 3 wt % of the total anode weight so far". Not LG M50 (the cell is a Samsung pouch/lab full cell with SSC 14.6-18 wt%).
- **E11** Bonkile et al., "Is silicon worth it? Modelling degradation in composite silicon-graphite lithium-ion battery electrodes", J. Power Sources 606 (2024) 234256 (pdftotext of Imperial Spiral open copy <https://spiral.imperial.ac.uk/server/api/core/bitstreams/4b04560d-ce32-473b-b224-28c7a6c7a71d/content>)
  ```
  The parameters are sourced from Ai et al. [33,53], based on a high-
  energy-density commercial battery (LG M50T). The LG M50T uses
  a SiOx-blended graphite negative electrode paired with an NMC811
  cathode, offering a nominal energy of 18.2 Wh, and capacity of 5
  Ah, as studied by Kirkaldy et al. [22].
  ...
  Chen et al. [34]
  reported that the LG M50T cell had a silicon loading of 10 wt%,
  ```
  (page 6/7 of that PDF, lines ~437-441, ~489-490). Also: "silicon only becoming active at low SoCs due to its OCP profile" (conclusion, line ~594 region) and "reached before silicon becomes fully utilised" (line ~602). A numerical Si state-of-lithiation at 100% SOC is in its Supplementary parameter tables/Fig S2, not in the main text read. 
- **E12** Kirkaldy et al., ACS Appl. Energy Mater. 2022, 5, 13367 (PMC9709825, HTML text grep). Excerpts: "Conversely, a greater proportion of the charge throughput in the 0-100% SoC range contributes toward (de)lithiating graphite, leading to higher levels of LAM-Gr in those cells"; "This leads to a significant volume expansion of over 300% in the fully lithiated state compared to graphite, which expands by around 20% upon lithiation."; "Performing the full-cell OCV-fitting using the method described here gave a similar graphite-to-silicon capacity ratio of 0." (sentence truncated at decimal in my regex, number not captured). No explicit x in LixSi at 100% SOC found.
- **E13** Hybrid-MPET, arXiv 2305.15599 (pdftotext, lines 701-741, 805-830, 1777): "we use an estimated 1% silicon mass fraction [101, 102]"; "most silicon only start to delithiate when the graphite has fully delithiated"; parameter table: "e Si capacity fraction of silicon - 0.086" (also 0.128 / 0.333 in another table, line 2020). Tool did not give a Si state-of-charge at end of charge in extracted lines.
- **E14** (tool summary, not verified at source) Chen et al. 2020 JES 167 080534, abstract via <https://research.birmingham.ac.uk/en/publications/development-of-experimental-techniques-for-parameterization-of-mu/> : "a NMC 811 positive electrode and bi-component Graphite-SiOx negative electrode"; "GITT in half cell and three-electrode full cell configurations". Birmingham page lists full PDF (CC BY-NC-ND) but direct download blocked. Zenodo data record: <https://zenodo.org/records/4032561> (not opened).
- **E15** (tool summary, search snippet) "The silicon content in the LG M50T composite electrode to be 2.43% using five complementary, cross-validated methods" - from the search result for <https://iopscience.iop.org/article/10.1149/1945-7111/add112> ("Is Silicon Replaceable? A Physical, Chemical, and Electrochemical Analysis of Different Commercial Lithium-Ion Battery Cells"); page itself blocked (S19).
- **E16** (tool summary) Search snippet for <https://www.sciencedirect.com/science/article/pii/S0378775322001604> ("A composite electrode model for lithium-ion batteries with silicon/graphite negative electrodes"): "silicon is normally limited to a small mass fraction ... approximately 10% for the LG M50 cells"; models use "X_Li[Gr] and X_Li[Si]". Page 403.
- **E17** Lory et al., J. Electrochem. Soc. 2020, 167, 120506? (DOI 10.1149/1945-7111/abaa69; volume/article number unverified) (tool summary of search result): blended anode "16 wt% of silicon carbon composite (SiC-C) at a gravimetric capacity of 1500 mAh.g-1 mixed with 84 wt% graphite". Not opened.

### Item 4: SiOx
- **E18** Kirner, Qin, Zhang, Janson, Lu (Argonne), "Optimization of Graphite-SiO Blend Electrodes for Lithium-Ion Batteries: Stable Cycling Enabled by Single-Walled Carbon Nanotube Conductive Additive" (OSTI 1632823; local PDF webfetch-1791052994180-8dixnl.pdf, pdftotext lines 60-90, 273-279, 652-666, 927)
  ```
  However, the 277% volume expansion
  calculated[1] for lithiation from Si to Li15Si4 ...      ([1] = Obrovac, Christensen, Le, Dahn, JES 154 (2007) A849)
  ...
  so, SiOx can expand by up to 160%,[8] which is likely too much for practical electrodes.
  [8] S.C. Jung, H.-J. Kim, J.-H. Kim, Y.-K. Han, J. Phys. Chem. C, 120 (2016) 886-892.
  ...
  SiO Electrode Volume Expansion. ... fully-lithiated cells were
  disassembled in a glove box, rinsed, dried, and removed to a humidity-controlled dry-room for
  measurement of electrode thicknesses by a digital micrometer gauge.
  ...
  Initial Coating Thickness (µm)   37  24  20  17  15
  Lithiated Coating Thickness (µm) 49  37  36  32  30
  Coating Swell %  (%)             33 ± 3  55 ± 3  76 ± 4  90 ± 3  107 ± 5
  Reversible Specific Capacity (mAh/g) 348 ± 1  713 ± 4  995 ± 3  1262 ± 8  1701 ± 29
  ```
  (columns = (Gr-SiO) 1-0, 7-3, 5-5, 3-7, 0-1; the last column is the graphite-free SiO electrode, coating swell 107 ± 5 % thickness at 0.01 V, after 3 formation cycles; SiO electrode = SiO-LiPAA-SWCNT 15-0.6 per table header.)
- **E19** (tool summary of search result) Statement "The volume expansion of SiO is about 200% upon electrochemical lithiation of about 2600 mAh g-1, comparing with silicon, which is 400% for lithiation of 4000 mAh g-1" - source snippet from one of the S25 results (candidate: <https://fml.skku.edu/Paper/2023/acs.energyfuels.3c00785.pdf> "Review on Improving the Performance of SiOx Anodes", 404 on fetch; or <https://link.springer.com/article/10.1007/s11664-021-09187-x>). Source attribution within the results not confirmed.
- **E20** <https://pmc.ncbi.nlm.nih.gov/articles/PMC12781111/> ("Phase Engineering of Lithium Silicate in SiOx Anodes for Fast-Charging Lithium-Ion Batteries") (tool summary): "the lithium silicate phase functions as a Li-ion conductor...resulting in a more moderate volume expansion (~118%) compared to pure Si (~300%)." No primary reference number returned.
- **E21** (tool summary of search snippet, source not opened) SiO modulus: "bare SiO (34.0 GPa), Li-SiO (51.8 GPa), and LiF-SiO (108.1 GPa)", AFM, "Poisson's ratio of 0.17"; source named by search as "High-strength and high-modulus silicon monoxide for high-energy-density and fast-charging lithium-ion batteries", Nat. Commun. 2026, <https://www.nature.com/articles/s41467-026-72434-4> (fetch redirected to login). Also snippet: Si-SiOx/C composite electrodes max modulus "924 MPa (Si-SiOx/C), 3.44 GPa ..." (<https://pmc.ncbi.nlm.nih.gov/articles/PMC11843606/>, not opened; these are electrode-level AFM values).
- **E22** (tool summary of search snippet) SiOx "around 50% volume expansion" for "1000-2000 mAh g-1" (source not identified); OSTI 1466681 abstract: "Major capacity loss occurs in the early cycles followed by less capacity fading during the following cycles." (no expansion number).
- **E23** Si reference moduli from search snippets (tool summary): "Young's modulus decreases from an initial value of 92 GPa for pure Si to 12 GPa at full lithium insertion (Li15Si4)" (source not identified in results; candidate <https://arxiv.org/pdf/1311.5844>); "Young's modulus of Li12Si7 ... 52.0 +/- 8.2 GPa" (<https://www.sciencedirect.com/science/article/abs/pii/S0378775312003874>). Not opened.

### Item 5: Dilatometry paper DOI 10.1149/1945-7111/abd465
- **E24** Prado, Rodrigues, Trask, Shaw, Abraham, "Electrochemical Dilatometry of Si-Bearing Electrodes: Dimensional Changes and Experiment Design", J. Electrochem. Soc. 167, 160551 (2020) (ADS bibcode 2020JElS..167p0551P) - via WebFetch of <https://iopscience.iop.org/article/10.1149/1945-7111/abd465> (tool summary; quotes returned by the tool in quotation marks)
  Abstract (also at <https://www.osti.gov/pages/biblio/1755253>): "The severe volumetric changes in Si particles during the Li (de)alloying process cause expansion and contraction of the electrodes"; "For silicon-rich anodes, the electrode dilation can be higher than 300%"; "By increasing the Si contribution to the electrode capacity, the swelling is aggravated upon lithiation".
  Measurement basis: "The charge-induced dimensional change of the working electrode is expressed by means of relative thickness variation (%), which is calculated based on the coating thickness of the dry, pristine anode." Sensor: "a capacitive displacement sensor with reported resolution of 5 nm".
  Table I (as returned by tool):
  ```
  Parameter           Gr   Si-Gr(15%)  Si-Gr(30%)  Si(70%)  Si(80%)
  Silicon (wt%)       —    15          30          70       80
  Graphite (wt%)      88   73          58          —        —
  C45 Carbon (wt%)    2    2           2           15       10
  LiPAA Binder (wt%)  10   10          10          15       10
  Porosity (%)        34   56          56          57       57
  Initial Thickness (µm) 49 39         18          9        12
  ```
  First vs later cycles: "the electrodes are charged to 100% and 95% of the nominal capacity at 100 mV (vs Li metal), respectively."; "the 15 wt% Si electrode expands ~22% during both first and second cycles, and 39% during the last one."; graphite-only "~19% total dilation at full lithiation"; 70 wt% Si "~300% dilation at full capacity"; 80 wt% "approximately 520% dilation (from Figure 6)" (the 520% parenthetical is the tool's phrasing).
  Protocol: "three galvanostatic cycles between 100 mV and 1 V, conducted at C/20 rate" (coin cells); "The currents are scaled to achieve the desired SOC upon lithiation in 15 hours"; "the final cycle...lithiated at a slower rate of C/25 to 10 mV vs Li/Li+." Delithiation to 1 V. Half cells vs Li (working electrode vs Li metal per the voltage references).
  Thickness vs theory: "Theoretical calculations indicate that full lithiation of a silicon particle will produce a volume expansion of ~300%." and "electrode-level expansion can be larger than the theoretical particle-scale swelling, even though the complex electrode system possesses mechanisms to partially accommodate the dilation."
  NOTE: the electrode set (88 wt% Gr, 15 wt% Si/73 Gr, LiPAA, MagE graphite, Timcal C45, same CAMP facility) matches the electrode family in Yao 2019 (E4) (same Argonne group); the abd465 text I received does not state porosity 56% vs Yao's 42.4% reconciliation.
- **E25** Michael et al., "A Dilatometric Study of Graphite Electrodes during Cycling with X-ray Computed Tomography", J. Electrochem. Soc. 168, 010507 (2021), DOI 10.1149/1945-7111/abd648 (pdftotext, local file webfetch-1791052725614-iod98e.pdf) - a different paper, graphite electrode dilatometry:
  ```
  During the first cycle, the graphite electrode underwent thickness
  changes close to 9% after lithiation and, moreover, it did not return to its initial thickness after subsequent delithiation.
  ...
  Expansion of graphite's structure can cause deformations as large as 10% of
  initial volume when C6 is fully lithiated to LiC6.6,7 However, the
  entire electrode can increase in volume by 13% if other contributory factors such as gas evolution8 are also taken into consideration.
  ...
  The electrode sheets are composed of synthetic graphite powder on copper sheets with 90% graphite; (NEI Corporation ready-made calendered sheets)
  ```
  Same paper, tomography: "TauFactor recorded a pore phase volume fraction of 42.0% in the pristine electrode"; "The z-direction represents the through plane orientation in the electrode and so the tortuosity is expected to be largest here due to the arrangement of the platelet shape particles. The isotropic recordings for the x- and y-directions are likely due to alignment of the platelet particles within the electrode, caused by packing ordering and calendering." (tortuosity factor 3.28 highest value, z-direction, cycled electrode).

### Item 6: Graphite c-axis texture
- **E26** Malvern Panalytical application note AN230120 "Graphitization degree and orientation index in graphite anode materials" <https://www.malvernpanalytical.com/en/learn/knowledge-center/application-notes/an230120graphitization> (tool summary; quotes in quotation marks)
  > "The orientation index (OI) is defined as the ratio of weight fraction oriented along, say, 110 (or any other direction orthogonal to 001) to that oriented along 001."
  > "The simplest way to estimate it is from the ratio of the 110 and 004 (fourth-order reflection of 001 planes) peak intensities (areas)."
  > "Theoretically, in a randomly oriented graphite material, f = I110/I004 = 0.63. A value less than 0.63 would mean that particles are preferentially oriented along 001. Sometimes, the intensity ratio, R = I004/I110 (=1/f) is measured instead. In a random orientation, R = 1.6 and values larger than 1.6 indicate a preferred orientation along 001."
  > "Orientations like 110, 100, etc., in which the c-axis of the graphite is in the plane of the current collector, offer much better electronic and ionic conductivity compared with the 001 orientation, in which the c-axis is out-of-plane from the current collector."
  (The tool reported the note contains no statement on calendering; the note concerns powders/coatings in general, no measured calendered-electrode value.)
- **E27** arXiv 2607.11521 "Capturing the calendering U-shape in lithium-ion electrode thermal conductivity" <https://arxiv.org/html/2607.11521> (tool summary; a modelling/closure paper, arXiv ID as returned by the search tool) :
  > "Graphite-electrode flakes are known to lie preferentially parallel to the collector, and a dedicated XRD method quantifies the oriented fraction and finds it rises with calendering, exactly the direction the mechanism requires" (Sec. 3.5)
  > "For a single flake whose c-axis (basal-plane normal) makes angle θ with the through-plane direction, the through-plane component of its conductivity tensor is λ_zz = λ_c cos²θ + λ_a sin²θ" (Sec. 3.5, Eq. 8)
  > "Flake alignment exposes the low-conductivity c-axis to the through-plane path, lowering the effective solid conductivity itself" (Sec. 2)
  The "dedicated XRD method" is not cited with a reference number in the tool output.
- **E28** Malifarge, Delobel, Delacourt, "Quantification of preferred orientation in graphite electrodes for Li-ion batteries with a novel X-ray-diffraction-based method", J. Power Sources 343 (2017) (publication details as in search result; volume/pages not verified), <https://www.sciencedirect.com/science/article/abs/pii/S0378775317300654> (page 403; abstract text is the tool's paraphrase from search results): "Graphite negative electrodes generally consist of anisotropic particles that exhibit a preferred orientation, with graphite particles tending to stack perpendicular to ionic pathways" ; method "derives a fraction of graphite particles oriented parallel to the electrode current collector within a tilt tolerance"; "applied ... on a set of graphite electrodes that underwent different calendering conditions." No numeric values retrieved. (Note: an earlier search result attributed this paper to "Steinhauer"; the Semantic Scholar entry names Malifarge.)
- **E29** Controlling the Crystallographic Orientation of Graphite Electrodes for Fast-Charging Li-Ion Batteries, ACS Appl. Mater. Interfaces <https://pubs.acs.org/doi/10.1021/acsami.1c19735> (search-result title only; 403 on fetch). Magnetic-alignment patent search results (USPTO 12322543) and Marangoni paper (<https://www.sciencedirect.com/science/article/pii/S2949821X26000864>) surfaced; snippet: "when graphite flakes are more parallelly oriented to the current collector, the basal plane (002) shows higher intensity at 26.5° 2θ, and as alignment becomes more vertical to the current collector, the (002) basal plane intensity decreases" (tool summary, source of snippet not confirmed).
- **E30** Michael 2021 (E25) is the only directly read primary text on platelet alignment in calendered graphite electrodes (CT tortuosity, not XRD/EBSD).

## Not found
- Per-component Si/graphite lithiation numbers for the 15 wt% blend as a table or by SOC: Yao 2019 gives only plots (Fig. 4a/4b) plus the regional ratios in E1/E2 (S1).
- PVdF-specific E and nu in arXiv 1511.02445 Table 2: Table 2 lists no PVdF row; only composite "Graphite based anode coating" E 6.9 GPa, nu 0.3 (S2). PVdF appears only in Table 1 as a binder wt%.
- Explicit numeric x in LixSi (or Si utilisation fraction) at 100% SOC for an LG M50/MJ1 cell: Chen 2020 JES full text blocked (S14, S16, S19, S21); Bonkile 2024 main text lacks the number (S20); Kirkaldy 2022 PMC lacks it (S22); Composite electrode model (S0378775322001604) and "Is Silicon Replaceable" blocked (S11, S19). O'Kane, Ai, Schmitt, Richter papers not fetched (not reached; see Not checked).
- Primary (non-review) experimental source for ~118%/~200%/up-to-160% SiOx volume expansion: only secondary statements found (E18 cites Jung et al. JPCC 2016, 120, 886; E19, E20 unattributed) (S25).
- Primary measurement of pristine SiOx (x~1) Young's modulus (e.g. nanoindentation of SiO particles/thin films): only the AFM snippet in E21 (not read at source) (S5, S25).
- Full text/experimental section of abd465 beyond the IOP abstract-page summary: PDF blocked (S9, S13); the first-cycle cut-off ("100% and 95% of the nominal capacity at 100 mV") text is the tool's quote; no statement found on whether abd465 converts thickness to volume (tool reported thickness only).
- Graphite c-axis texture numeric I004/I110 or EBSD values for calendered anodes: no EBSD study of graphite anode cross-sections returned (S26, S27); XRD-orientation papers (Malifarge 2017; ACS AMI 1c19735) blocked/not retrievable for numbers.

## Not checked
- Chen 2020 full PDF (open access) for Si wt%, stoichiometry windows at 100% SOC (blocked by bot-protection; try Zenodo record 4032561 / PyBaMM parameter files Chen2020_composite, OKane2022, Ai2020 on GitHub raw).
- O'Kane et al. 2022 PCCP, Ai et al. (JES 2020 "Electrochemical thermal-mechanical modelling of stress inhomogeneity in lithium-ion pouch cells"/2022 composite model), Schmitt, Richter papers: not fetched (round depth cap).
- Lory 2020 JES (abaa69), Heubner 2022 Batteries & Supercaps (batt.202100182), PMC9814897 (multi-scale quantification of aged Si composite anodes), arXiv 2508.06413 (operando nano-holo-tomography Si-Gr; grepped for lithiation terms, only graphite hits), Hybrid-MPET end-of-charge Si SOC figures.
- OSTI 1496633 (accepted manuscript of Yao 2019) not opened; supplementary Figures S3-S4 of Yao not described.
- Jung, Kim, Kim, Han JPCC 120 (2016) 886 (cited for "SiOx can expand by up to 160%") not retrieved; Yamada/Miyachi SiO mechanism papers not retrieved.
- Nature Commun. 2026 SiO modulus paper (s41467-026-72434-4) and PMC11843606 not read.
- Steinhauer/Malifarge JPS 2017 full text; Kehrwald/Wu/Habedank calendering-orientation papers not searched; ACS AMI 1c19735 not read.
