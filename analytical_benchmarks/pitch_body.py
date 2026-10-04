"""Body fragment for the held-back presentation report -> pitch/body.html (compile with the artifact kit)."""
import base64, html, json, numpy as np, pandas as pd
from compare import KPI_INFO

E = html.escape; B = ["Batch_1", "Batch_2", "Batch_3"]
A = json.load(open("heldout_assign.json")); R = json.load(open("compare.json"))
lab = pd.read_csv("site_kpis.csv"); hel = pd.read_csv("heldout_kpis.csv").set_index("site")
SITES = list(hel.index)
P = {k: {r["site"]: r for r in A[k]["sites"]} for k in ["material", "imaging", "combined"]}
TEAM = {"3e122cbj": ("Batch_1", 0.85), "fn0mhxef": ("Batch_3", 0.56), "xrv9xvzb": ("Batch_3", 0.59)}  # outputs/classifier (KPI arm, final)
TEAM_FP = {"3e122cbj": "Batch_1", "fn0mhxef": "Batch_3", "xrv9xvzb": "Batch_2"}  # outputs/fingerprint
CALL = {"3e122cbj": ("Batch_1", "High", "done", "0.98 combined; material alone 0.60"),
        "fn0mhxef": ("Batch_1", "Low", "at-risk", "0.54 vs Batch_2 0.45; not Batch_3 0.99"),
        "xrv9xvzb": ("Batch_3", "Moderate", "active", "0.48 combined; 3 of 3 nearest imaging sites are Batch_3")}
QC = {"3e122cbj": ("Investigate", "at-risk", "Batch_1-type sub-population: more Si (8.2%), irregular dim-grey Si particles (solidity 0.86), 31% cracked or irregular Si.",
                   "Run EDS on the dim-grey Si particles (Si vs SiOx / Si–C changes capacity and swelling). Image 2–4 more fields from the same sample."),
      "fn0mhxef": ("Consistent", "done", "All 20 material KPIs within |z| ≤ 1.3 of the 31 labelled sites.", "No material action. Batch origin unresolved (Batch_1 vs Batch_2); log as unknown session."),
      "xrv9xvzb": ("Consistent", "done", "Material typical; slightly more pore (7.6%) and rounder, more solid Si, all within normal site-to-site spread.", "No material action. Acquisition matches a known Batch_3 imaging session.")}


def img(p, alt):
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(open(p, "rb").read()).decode()}" alt="{E(alt)}" style="width:100%;height:auto;border-radius:6px">'


def pct(v): return f"{100 * v:.0f}%"
def kn(k): return KPI_INFO.get(k, (k,))[0]
def table(cap, head, rows, sticky=1, numeric_from=1):
    h = "".join(f'<th scope="col"{" data-numeric" if i >= numeric_from else ""}>{E(c)}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join((f'<th scope="row">{c}</th>' if i == 0 else f'<td{" data-numeric" if i >= numeric_from else ""}>{c}</td>')
                                 for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'<div class="a-table-scroll"><table class="a-table" data-a-sticky-columns="{sticky}"><caption class="a-visually-hidden">{E(cap)}</caption>'
            f"<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>")


o = []
# ---------- lead
o.append('<section class="a-section"><h2 class="a-section__title">Bottom line</h2><div class="a-grid">')
for s in SITES:
    b, c, st, why = CALL[s]
    o.append(f'<div class="a-metric"><div class="a-metric__label">{s}</div><div class="a-metric__value">{b}</div>'
             f'<div class="a-metric__delta"><span class="a-status" data-state="{st}">{c} confidence</span> {E(why)}</div></div>')
mat, ima = A["material"]["eval"][str(A["material"]["k"])], A["imaging"]["eval"][str(A["imaging"]["k"])]
o.append(f'<div class="a-metric"><div class="a-metric__label">Material KPIs, hide-one-site test</div><div class="a-metric__value">{pct(mat["bal_acc"])}</div>'
         f'<div class="a-metric__delta">balanced accuracy on 31 known sites; shuffled-label 95th percentile {pct(mat["null_bal_95"])}, chance 33%</div></div></div>')
o.append('<div class="a-callout a-callout--decision"><p class="a-prose"><strong>What we claim.</strong> Our physical KPIs show that Batch_2 and Batch_3 are the same material '
         'within uncertainty, and that Batch_1 contains a <em>sub-population</em> of sites with more, more irregular, dimmer Si. Held-back 3e122cbj carries that '
         'signature, so it is Batch_1 with high confidence. For fn0mhxef and xrv9xvzb the material is typical; their batch calls rest mainly on '
         'how the images were acquired, and we say so with low/moderate confidence instead of a fake 95%.</p></div></section>')

# ---------- 1 KPI quality
rows = []
for k, lbl, f in [("porosity", "Porosity", 100), ("si_frac", "Si area fraction", 100), ("interface_um_per_um2", "Phase-boundary density (µm/µm²)", 1)]:
    med = lab.groupby("batch")[k].median()
    r = [lbl] + [f"{hel.loc[s, k] * f:.2f} ± {1.96 * hel.loc[s, k + '__err'] * f:.2f}" for s in SITES] + [f"{med[b] * f:.2f}" for b in B]
    rows.append(r)
for k in ["si_d50_um", "si_d90_um", "si_solidity", "si_circularity", "si_cracked_frac", "si_count_per_1000um2", "si_clark_evans", "inlens_texture"]:
    med = lab.groupby("batch")[k].median()
    rows.append([kn(k)] + [f"{hel.loc[s, k]:.3g}" for s in SITES] + [f"{med[b]:.3g}" for b in B])
gp = "".join(f'<tr><th scope="row">{s}</th><td data-numeric>{100 * 1.96 * hel.loc[s, "si_frac__naive_err"]:.2f}%</td>'
             f'<td data-numeric>{100 * 1.96 * hel.loc[s, "si_frac__err"]:.2f}%</td></tr>' for s in SITES)
o.append('<section class="a-section"><h2 class="a-section__title">1 · Quality of the extracted material KPIs</h2>'
         '<p class="a-section__note">20 KPIs per image, all physical (µm, area fractions, shape ratios). The new sites ran through exactly the same code; nothing was refitted.</p>'
         f'<figure class="a-panel"><figcaption class="a-panel__title">Segmentation of the three held-back sites (central 30 × 20 µm crop)</figcaption>{img("pitch/seg_heldout.jpg", "BSE crops of the three held-back sites with pore and Si segmentation")}</figure>'
         '<div class="a-panel"><div class="a-panel__head"><h3 class="a-panel__title">Held-back KPI values vs batch medians (± = 95% GP interval)</h3></div>'
         + table("Held-back KPIs vs batch medians", ["KPI"] + SITES + [f"{b} median" for b in B], rows) + '</div>'
         '<div class="a-grid a-grid--halves">'
         '<figure class="a-panel" data-a-chart="bar" data-a-chart-unit="%"><figcaption class="a-panel__title">Si fraction uncertainty: naive vs spatial GP (95% half-width)</figcaption>'
         '<div class="a-table-scroll"><table class="a-table"><caption class="a-visually-hidden">Naive vs GP 95% half-width of Si fraction</caption>'
         f'<thead><tr><th scope="col">Site</th><th scope="col" data-numeric>Naive (tiles independent)</th><th scope="col" data-numeric>GP (spatially correlated)</th></tr></thead><tbody>{gp}</tbody></table></div></figure>'
         '<div class="a-prose"><p><strong>Why the GP error bar matters.</strong> Neighbouring 3.2 µm tiles are correlated (a big particle spans several), so treating them as independent '
         'understates uncertainty by about 1.7–2×. The GP learns the correlation length per image and widens the interval accordingly.</p>'
         '<p><strong>Known limits.</strong></p><ul><li>The bright phase is “likely Si”, not EDS-confirmed.</li><li>Binder and carbon black cannot be separated from graphite in BSE.</li>'
         '<li>2-D sections: no 3-D connectivity or tortuosity.</li><li>Porosity partly tracks the Batch_3 BSE black-level offset, so it is partly an imaging effect.</li></ul></div></div></section>')

# ---------- 2 batch differences
vt = {"consistent": ("Consistent", "done"), "investigate": ("Investigate", "at-risk"), "outlier": ("Outlier", "blocked")}
brow = []
for b in B:
    L = R["loo"][b]; top = L["rows"][0]
    brow.append([b, str(L["n_sites"]), f'<span class="a-status" data-state="{vt[L["verdict"]][1]}">{vt[L["verdict"]][0]}</span>',
                 ", ".join(L.get("unusual_sites", [])) or "none", f'{kn(top["kpi"])} (d = {top["d"]:+.2f}, adjusted p = {top["p_holm"]:.2f})'])
o.append('<section class="a-section"><h2 class="a-section__title">2 · What actually differs between the batches</h2>'
         '<p class="a-section__note">Each batch vs all other sites pooled; no batch is treated as approved. “Outlier” would mean different, not defective.</p>'
         + table("Batch verdicts", ["Batch", "Sites", "Verdict", "Unusual sites (≥2 KPIs)", "Strongest batch-mean difference"], brow, numeric_from=9) +
         f'<figure class="a-panel"><figcaption class="a-panel__title">Si fraction per site (95% GP interval) and Si fraction vs particle solidity</figcaption>{img("pitch/batches.jpg", "Si fraction per site and Si fraction versus solidity, with held-back sites as stars")}</figure>'
         '<p class="a-prose">No batch <em>average</em> differs after multiple-testing correction. The difference is a sub-population: 2 of 7 Batch_1 sites have ~2× the Si, '
         'irregular particles and a dimmer Si grey level. No Batch_2 or Batch_3 site looks like that; 3e122cbj does. Note that Si fraction alone overlaps (3e122cbj 8.2%, interval 6–10%); '
         'it is the <strong>combination</strong> with low solidity and circularity that is unique.</p>'
         f'<figure class="a-panel"><figcaption class="a-panel__title">Raw BSE: unusual Batch_1, typical Batch_2 / Batch_3, and the three held-back sites (22 × 15 µm)</figcaption>{img("pitch/gallery.jpg", "Raw BSE crops of known and held-back sites")}</figure></section>')

# ---------- 3 new cases
def stk(s):
    rows = "".join(f'<tr><th scope="row">{lbl}</th>' + "".join(f'<td data-numeric>{100 * P[k][s]["p"][b]:.0f}%</td>' for b in B) + "</tr>"
                   for k, lbl in [("material", "Material"), ("imaging", "Imaging"), ("combined", "Combined")])
    return (f'<figure class="a-panel" data-a-chart="stacked-bar" data-a-chart-unit="%" data-a-chart-max="100" data-a-chart-height="170">'
            f'<figcaption class="a-panel__title">{s}: batch probability</figcaption><div class="a-table-scroll"><table class="a-table">'
            f'<caption class="a-visually-hidden">{s} batch posterior by evidence type</caption><thead><tr><th scope="col">Evidence</th>'
            + "".join(f'<th scope="col" data-numeric>{b}</th>' for b in B) + f'</tr></thead><tbody>{rows}</tbody></table></div></figure>')


sb = "".join(f'<tr><th scope="row">{s} · {lbl}</th>' + "".join(f'<td data-numeric>{100 * P[k][s]["p"][b]:.0f}%</td>' for b in B) + "</tr>"
             for s in SITES for k, lbl in [("material", "material"), ("imaging", "imaging"), ("combined", "combined")])
tabs, panels = [], []
for i, s in enumerate(SITES):
    m = P["material"][s]; b, c, st, why = CALL[s]
    kr = [[E(q["label"]), f'{q["value"]:.3g}'] + [f'{q[f"mean_{x}"]:.3g}' for x in B] + [f'{q[f"z_{x}"]:+.1f}' for x in B] for q in m["kpis"][:6]]
    vs3 = "; ".join(f'{q["label"]} {"higher" if q["z_Batch_3"] > 0 else "lower"} (z {q["z_Batch_3"]:+.1f})' for q in m["kpis"][:3])
    tabs.append(f'<button role="tab" id="tab-{s}" aria-controls="panel-{s}" aria-selected="{"true" if i == 0 else "false"}">{s}</button>')
    panels.append(f'<div role="tabpanel" id="panel-{s}" aria-labelledby="tab-{s}"{"" if i == 0 else " hidden"}>'
                  f'<p class="a-prose"><strong>{s} → {b}</strong> <span class="a-status" data-state="{st}">{c} confidence</span> {E(why)}. '
                  f'Our teammates’ KPI classifier: {TEAM[s][0]} ({TEAM[s][1]:.2f}); fingerprint classifier: {TEAM_FP[s]}.</p>'
                  f'<p class="a-prose"><strong>How it differs from Batch_3</strong> (organisers’ requested comparison; Batch_3 is not treated as approved): {E(vs3)}.</p>'
                  + table(f"{s} KPI z-scores", ["KPI (most informative first)", "Value", "B1 mean", "B2 mean", "B3 mean", "z vs B1", "z vs B2", "z vs B3"], kr) + "</div>")
agree = [[s, CALL[s][0], TEAM[s][0], TEAM_FP[s], "agree" if CALL[s][0] == TEAM[s][0] == TEAM_FP[s] else "disagree"] for s in SITES]
o.append('<section class="a-section"><h2 class="a-section__title">3 · The three new cases</h2>'
         '<p class="a-section__note">Posterior probability per batch. Material = 10 analytical KPIs; imaging = acquisition fingerprint (dark level, gain, contrast); combined = product, assuming independence.</p>'
         + "".join(stk(x) for x in SITES) +
         f'<div class="a-panel" data-a-tabs><div role="tablist" aria-label="Held-back site">{"".join(tabs)}</div>{"".join(panels)}</div>'
         '<div class="a-panel"><div class="a-panel__head"><h3 class="a-panel__title">Do independent methods agree?</h3></div>'
         + table("Agreement with teammates", ["Site", "This analysis", "Team KPI classifier", "Team fingerprint classifier", "Agreement"], agree, numeric_from=9) +
         '<p class="a-prose">All three methods agree only on 3e122cbj. Disagreement on the other two is itself evidence that their batch origin is not recoverable from these images.</p></div></section>')

# ---------- 4 validation
def conf(ev):
    c = ev["confusion"]; rows = []
    for t in B:
        n = sum(c.get(p, {}).get(t, 0) for p in B)
        rows.append([t] + [str(c.get(p, {}).get(t, 0)) for p in B] + [f"{100 * c.get(t, {}).get(t, 0) / n:.0f}%"])
    return rows
o.append('<section class="a-section"><h2 class="a-section__title">4 · Validation: better than chance?</h2>'
         '<p class="a-section__note">Leave-one-site-out on the 31 labelled sites: hide one site, select KPIs and fit on the other 30, predict the hidden one. The null is the same procedure with shuffled batch labels (200×).</p>'
         '<figure class="a-panel" data-a-chart="bar" data-a-chart-unit="%" data-a-chart-max="100"><figcaption class="a-panel__title">Balanced accuracy, leave-one-site-out</figcaption>'
         '<div class="a-table-scroll"><table class="a-table"><caption class="a-visually-hidden">Balanced accuracy vs shuffle null and chance</caption>'
         '<thead><tr><th scope="col">Evidence</th><th scope="col" data-numeric>Observed</th><th scope="col" data-numeric>Shuffled labels, 95th pct</th><th scope="col" data-numeric>Chance</th></tr></thead><tbody>'
         f'<tr><th scope="row">Material KPIs (p = {mat["p_perm"]:.3f})</th><td data-numeric>{pct(mat["bal_acc"])}</td><td data-numeric>{pct(mat["null_bal_95"])}</td><td data-numeric>33%</td></tr>'
         f'<tr><th scope="row">Imaging fingerprint (p = {ima["p_perm"]:.3f})</th><td data-numeric>{pct(ima["bal_acc"])}</td><td data-numeric>{pct(ima["null_bal_95"])}</td><td data-numeric>33%</td></tr>'
         '</tbody></table></div></figure><div class="a-grid a-grid--halves">'
         '<div class="a-panel"><div class="a-panel__head"><h3 class="a-panel__title">Material KPIs: true (rows) vs predicted (columns)</h3></div>'
         + table("Material confusion", ["True", "→ B1", "→ B2", "→ B3", "Recall"], conf(mat)) + '</div>'
         '<div class="a-panel"><div class="a-panel__head"><h3 class="a-panel__title">Imaging fingerprint: true vs predicted</h3></div>'
         + table("Imaging confusion", ["True", "→ B1", "→ B2", "→ B3", "Recall"], conf(ima)) + '</div></div>'
         f'<p class="a-prose">Calibrated log-loss: material {mat["nll"]:.2f}, imaging {ima["nll"]:.2f}, chance {mat["chance_nll"]:.2f} (lower is better). '
         'The probabilities are temperature-scaled on these hold-out predictions, which is why the material posteriors are deliberately flat.</p>'
         '<div class="a-callout a-callout--risk"><p class="a-prose"><strong>Why imaging is not used for the QC verdict.</strong> The fingerprint identifies the microscope session '
         '(black level, grey-level quantisation, frame size), not the powder. A perfectly normal batch imaged on a new day would look “different”, and a changed batch imaged in a familiar '
         'session would look “the same”. It is shown only as an audit of where the images came from.</p></div></section>')

# ---------- 5 QC decision
rng = np.random.default_rng(0); n = np.arange(1, 11); p = rng.beta(2 + .5, 5 + .5, 20000)  # Jeffreys posterior for 2/7 unusual sites
det = 1 - (1 - p[:, None]) ** n; fa = 1 - 0.05 ** (1 / 24)  # one-sided 95% upper bound for 0/24 sites
lr = "".join(f'<tr><th scope="row">{k}</th><td data-numeric>{100 * (1 - (5 / 7) ** k):.0f}%</td><td data-numeric>{100 * np.percentile(det[:, k - 1], 10):.0f}%</td>'
             f'<td data-numeric>{100 * (1 - (1 - fa) ** k):.0f}%</td></tr>' for k in n)
dc = [[s, CALL[s][0], f'<span class="a-status" data-state="{QC[s][1]}">{QC[s][0]}</span>', E(QC[s][2]), E(QC[s][3])] for s in SITES]
o.append('<section class="a-section"><h2 class="a-section__title">5 · Using it for a QC decision</h2>'
         '<p class="a-section__note">Verdict vocabulary: consistent / investigate / outlier. Nothing is “rejected” without an external specification.</p>'
         '<div class="a-panel"><div class="a-panel__head"><h3 class="a-panel__title">Decision card</h3></div>'
         + table("QC decision per held-back site", ["Site", "Batch call", "Material QC verdict", "Evidence", "Recommended action"], dc, numeric_from=9) + '</div>'
         '<figure class="a-panel" data-a-chart="line" data-a-chart-unit="%" data-a-chart-max="100"><figcaption class="a-panel__title">How many images per incoming batch? Chance of catching a Batch_1-like batch</figcaption>'
         '<div class="a-table-scroll"><table class="a-table"><caption class="a-visually-hidden">Detection probability and false-alarm bound vs images per batch</caption>'
         '<thead><tr><th scope="col">Images per batch</th><th scope="col" data-numeric>Detect (estimate, 2/7 sites unusual)</th><th scope="col" data-numeric>Detect (pessimistic, 10th pct)</th>'
         f'<th scope="col" data-numeric>False alarm (worst case, 95% bound)</th></tr></thead><tbody>{lr}</tbody></table></div></figure>'
         '<p class="a-prose">Rule: flag a batch “investigate” if at least one image shows the unusual-site signature. 2 of 7 Batch_1 images had it and 0 of 24 others, so 3 images catch such a batch '
         'about 64% of the time and 5 images about 81%. The false-alarm line is a worst-case bound from 0/24 (it could be much lower); more labelled images would tighten both lines. '
         'The signature was defined after looking at the data, so these are planning numbers, not validated rates.</p>'
         '<dl class="a-meta-list"><div><dt>Run</dt><dd><code>./analytical_benchmarks/run_all.sh</code>, about 2 min for 31 sites on 8 cores; no retraining</dd></div>'
         '<div><dt>New unlabelled sites</dt><dd><code>heldout_kpis.py</code> → <code>classify_heldout.py</code> → <code>fig_heldout.py</code>, about 1 min</dd></div>'
         '<div><dt>Output</dt><dd>Per-site KPIs with GP error bars, batch verdicts, per-KPI explanations, images, and this report</dd></div></dl></section>')

# ---------- limits
o.append('<section class="a-section"><h2 class="a-section__title">What could be wrong, and what would change our mind</h2><div class="a-callout a-callout--risk"><ul class="a-prose">'
         '<li>Only 31 labelled sites (7 / 7 / 17); every rate here has wide intervals.</li>'
         '<li>KPIs are selected inside each validation fold, but the overall method was designed after seeing the labelled data.</li>'
         '<li>The combined probability assumes material and imaging evidence are independent.</li>'
         '<li>If the dim-grey particles are SiOx or Si–C rather than Si, the Batch_1 signature is a formulation change, not just a size/shape change; EDS would settle it.</li>'
         '<li>fn0mhxef could be Batch_1 or Batch_2 (0.54 vs 0.45); xrv9xvzb could be Batch_2 (0.42). We would not bet on either.</li></ul></div></section>')
open("pitch/body.html", "w").write("\n".join(o)); print("body", sum(map(len, o)) // 1024, "kB")
