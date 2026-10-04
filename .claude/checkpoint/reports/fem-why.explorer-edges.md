# Evidence (dispatcher-run): do image edges look like a free surface or the foil?

Command: for all 31 labelled sites, `segment(load_site(b, s, resolution="half", normalise="none"))`; pore(+artefact) and Si area fraction in the top 20 px (1 µm), bottom 20 px, and middle 40 px rows.

Medians over 31 sites: pore top 0.016, bottom 0.045, middle 0.041; Si top 0.080, bottom 0.068, middle 0.045.
Per site: no edge shows a pore/resin fraction near 1 (max 0.285, ufdvpb81 bottom) and no Cu phase exists in the segmentation. Image heights 40.3–57.9 µm.

Docs: docs/data-processing.md:55 "No Cu foil is visible, so which edge is the top surface and which is the foil remains unknown."; docs/kpis/README.md:104 "Slice height | Coating-thickness proxy"; docs/specs/002-si-kpis.md:22 "No Cu foil is in frame".

FEM BCs in production (fem-build P10): collector edge u_z = 0 (rollers), lateral edges u_x = 0, opposite edge traction-free, plane strain; run with collector at bottom and at top, features averaged (`sym`).
