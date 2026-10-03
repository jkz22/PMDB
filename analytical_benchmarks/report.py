"""Build the baseline-free QC report body from compare.json + site_kpis.csv.
Writes body.html; if qc_report.html (compiled once with the artifact kit) exists,
replaces the content between the REPORT markers so re-runs stay standalone."""
import base64, html, json, os, re
import numpy as np, pandas as pd
from compare import KPI_INFO

R = json.load(open("compare.json")); D = pd.read_csv("site_kpis.csv")
B = sorted(D.batch.unique()); E = html.escape
VLABEL = {"consistent": "Consistent", "investigate": "Investigate", "outlier": "Outlier"}
VSTATE = {"consistent": "done", "investigate": "at-risk", "outlier": "blocked"}

EXTRA = {"si_n_cracked": "Cracked/irregular Si (count)", "si_n_anomalous": "Anomalous Si (count)"}
def kname(k): return KPI_INFO.get(k, (EXTRA.get(k, k), ""))[0]
def fmt(x, k=""):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "–"
    a = abs(x)
    return f"{x:.3f}" if a < 1 else f"{x:.2f}" if a < 100 else f"{x:.0f}"
def pv(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
def img(path, alt):
    if not os.path.exists(path): return f'<p class="a-empty">{E(path)} not found; run figs.py.</p>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/{"jpeg" if path.endswith(".jpg") else "png"};base64,{b}" alt="{E(alt)}" style="width:100%;height:auto">'
def status(v): return f'<span class="a-status" data-state="{VSTATE[v]}">{VLABEL[v]}</span>'

flags = pd.DataFrame(R["site_flags"])
def explain(b, L):
    rows = sorted((r for r in L["rows"] if not r.get("insufficient")), key=lambda r: -abs(r["d"]))[:2]
    drv = "; ".join(f'{kname(r["kpi"])} {"higher" if r["diff"] > 0 else "lower"} by {fmt(abs(r["diff"]))} '
                    f'(effect size {r["d"]:+.1f}, adjusted p={pv(r["p_holm"])})' for r in rows)
    s = f"Strongest differences vs the other sites: {drv}."
    if L["flagged_sites"]:
        f = flags[flags.batch == b]
        per = "; ".join(f'{st} ({", ".join(kname(k) for k in g.kpi)})' for st, g in f.groupby("site"))
        s += f" Unusual individual sites: {per}."
    if L["verdict"] == "consistent" and not L["flagged_sites"]:
        s += " None of these is larger than normal site-to-site variation."
    elif L["verdict"] == "investigate" and all(r["p_holm"] >= 0.05 for r in L["rows"]):
        spread = [kname(r["kpi"]) for r in L["rows"] if r["p_spread"] < 0.01 and r["spread_diff"] > 0]
        why = ([f"a larger site-to-site spread in {', '.join(spread)}"] if spread else []) + \
              (["multiple unusual sites"] if len(L.get("unusual_sites", [])) >= 2 else [])
        if why: s += f" The batch average is not decisively different; the verdict comes from {' and '.join(why)}."
    return s

o = []
# 1. verdicts
o.append('<section class="a-section"><h2 class="a-section__title">Verdict per batch</h2>'
         '<p class="a-section__note">No batch is used as a reference. Each batch is compared with all other sites pooled. '
         'Without a material specification, “outlier” means <em>different</em>, not <em>defective</em>.</p><div class="a-grid">')
for b in B:
    L = R["loo"][b]
    o.append(f'<div class="a-metric"><div class="a-metric__label">{E(b)} · {L["n_sites"]} sites</div>'
             f'<div class="a-metric__value">{VLABEL[L["verdict"]]}</div>'
             f'<div class="a-metric__delta">{len(L.get("unusual_sites", []))} unusual site(s), {len(L["flagged_sites"])} with any flag</div></div>')
o.append('</div><div class="a-prose">')
for b in B:
    L = R["loo"][b]
    o.append(f'<p><strong>{E(b)}</strong> {status(L["verdict"])} {E(explain(b, L))}</p>')
o.append('</div><details class="a-disclosure"><summary>How the verdict is decided</summary><ul class="a-prose">'
         '<li><strong>Outlier</strong>: a KPI differs with Holm-adjusted permutation p&lt;0.01 <em>and</em> effect size |d|≥0.8.</li>'
         '<li><strong>Investigate</strong>: adjusted p&lt;0.05, or the batch is significantly more variable (spread p&lt;0.01), '
         'or at least 2 sites are unusual on 2 or more KPIs each (robust z&gt;3.5 vs all other sites). A single flag on one KPI is listed but expected by chance with 20 KPIs.</li>'
         '<li><strong>Consistent</strong>: none of the above. This means “no evidence of change”, which is weaker with few sites.</li>'
         '</ul></details></section>')

# 2. leave-one-batch-out tables
def comp_table(rows, cap, la, lb):
    t = [f'<div class="a-table-scroll"><table class="a-table" data-a-sticky-columns="1"><caption class="a-visually-hidden">{E(cap)}</caption>'
         f'<thead><tr><th scope="col">KPI</th><th scope="col" data-numeric>{E(la)}</th><th scope="col" data-numeric>{E(lb)}</th>'
         '<th scope="col" data-numeric>Difference</th><th scope="col" data-numeric>95% CI</th><th scope="col" data-numeric>Effect size d</th>'
         '<th scope="col" data-numeric>p</th><th scope="col" data-numeric>p (Holm)</th><th scope="col">Result</th></tr></thead><tbody>']
    for r in sorted(rows, key=lambda r: r["p_holm"]):
        t.append(f'<tr><th scope="row">{E(kname(r["kpi"]))}</th><td data-numeric>{fmt(r["mean_a"])}</td><td data-numeric>{fmt(r["mean_b"])}</td>'
                 f'<td data-numeric>{fmt(r["diff"])}</td><td data-numeric>[{fmt(r["ci95"][0])}, {fmt(r["ci95"][1])}]</td>'
                 f'<td data-numeric>{"–" if r.get("insufficient") else format(r["d"], "+.2f")}</td><td data-numeric>{pv(r["p"])}</td><td data-numeric>{pv(r["p_holm"])}</td>'
                 f'<td>{"Too few sites" if r.get("insufficient") else status(r["verdict"])}</td></tr>')
    t.append('</tbody></table></div>'); return "".join(t)

def tabs(items, label):
    t = [f'<div data-a-tabs><div role="tablist" aria-label="{E(label)}">']
    for i, (name, _) in enumerate(items):
        t.append(f'<button role="tab" type="button" aria-selected="{"true" if i == 0 else "false"}">{E(name)}</button>')
    t.append('</div>')
    for i, (_, body) in enumerate(items):
        t.append(f'<div role="tabpanel"{"" if i == 0 else " hidden"}>{body}</div>')
    t.append('</div>'); return "".join(t)

o.append('<section class="a-section"><h2 class="a-section__title">What drives the differences</h2>'
         '<p class="a-section__note">Each batch vs all other sites. Unit = one site; 95% CI by bootstrap over sites, p by permutation of batch labels.</p>')
o.append(tabs([(b, comp_table(R["loo"][b]["rows"], f"{b} vs other sites", b, "Other sites")) for b in B], "Batch"))
o.append('<details class="a-disclosure"><summary>Pairwise comparisons</summary>')
o.append(tabs([(k, comp_table(v, k, *k.split(" vs "))) for k, v in R["pairs"].items()], "Pair"))
o.append('</details></section>')

# 3. unusual sites
o.append('<section class="a-section"><h2 class="a-section__title">Unusual individual sites</h2>')
if len(flags):
    o.append('<div class="a-table-scroll"><table class="a-table"><caption class="a-visually-hidden">Sites with robust z above 3.5</caption><thead><tr>'
             '<th scope="col">Site</th><th scope="col">Batch</th><th scope="col">KPI</th><th scope="col" data-numeric>Value</th>'
             '<th scope="col" data-numeric>Median of other sites</th><th scope="col" data-numeric>Robust z</th></tr></thead><tbody>')
    for _, f in flags.sort_values("robust_z", key=abs, ascending=False).iterrows():
        med = D[D.site != f.site][f.kpi].median()
        o.append(f'<tr><th scope="row">{E(f.site)}</th><td>{E(f.batch)}</td><td>{E(kname(f.kpi))}</td><td data-numeric>{fmt(f.value)}</td>'
                 f'<td data-numeric>{fmt(med)}</td><td data-numeric>{f.robust_z:+.1f}</td></tr>')
    o.append('</tbody></table></div>')
else:
    o.append('<p class="a-empty">No site is beyond robust z 3.5 on any KPI.</p>')
o.append('</section>')

# 3a. physics estimates (not used in the verdict)
if os.path.exists("physics.json"):
    PH = json.load(open("physics.json")); PD = pd.read_csv("physics.csv")
    from physics import KPI_INFO as PK
    o.append('<section class="a-section"><h2 class="a-section__title">Physics estimates: what the differences mean for the battery</h2>'
             '<p class="a-section__note">Estimated from the phase fractions and Si sizes with textbook constants (Si 3579 mAh/g, +280% volume; '
             'graphite 372 mAh/g, +10%). Not measurements, and not used in the verdict: they are functions of KPIs already counted. '
             'SiOx column: same estimate if the bright particles are SiOx (1600 mAh/g, +160%).</p>'
             '<div class="a-table-scroll"><table class="a-table" data-a-sticky-columns="1"><caption class="a-visually-hidden">Physics estimates per site</caption><thead><tr>'
             '<th scope="col">Site</th><th scope="col">Batch</th>'
             + "".join(f'<th scope="col" data-numeric>{E(v[0])}</th>' for v in PK.values())
             + '<th scope="col" data-numeric>Capacity if SiOx (mAh/g)</th></tr></thead><tbody>')
    pfl = {(f["site"], f["kpi"]) for f in PH["site_flags"]}
    for _, r in PD.sort_values(["batch", "site"]).iterrows():
        cells = "".join(f'<td data-numeric>{"<strong>" if (r.site, k) in pfl else ""}{fmt(r[k])} ± {fmt(r[k + "__err"])}{"</strong>" if (r.site, k) in pfl else ""}</td>' for k in PK)
        o.append(f'<tr><th scope="row">{E(r.site)}</th><td>{E(r.batch)}</td>{cells}<td data-numeric>{fmt(r.spec_capacity__siox)}</td></tr>')
    allr = [(b, x) for b in B for x in PH["loo"][b]]
    best = min((x for _, x in allr), key=lambda x: x["p_holm"])
    sig = sorted(((b, x) for b, x in allr if x["p_holm"] < 0.05), key=lambda t: t[1]["p_holm"])
    concl = ("Batch-level differences after Holm correction (adjusted p&lt;0.05): "
             + "; ".join(f'{E(b)}: {E(PK[x["kpi"]][0])} (adjusted p={pv(x["p_holm"])})' for b, x in sig) + ". ") if sig else \
            f'Smallest batch-level adjusted p: {pv(best["p_holm"])} ({E(PK[best["kpi"]][0])}), so no batch differs as a whole. '
    n_over = int((PD.swell_to_pore > 1).sum())
    swell = ("Swelling exceeds the pore volume at every site (swelling ÷ porosity &gt; 1), so the electrode must thicken on charging; "
             if n_over == len(PD) else
             f"Swelling exceeds the pore volume at {n_over} of {len(PD)} sites (swelling ÷ porosity &gt; 1); where it does, the electrode must thicken on charging; ")
    o.append('</tbody></table></div><p class="a-section__note">± = 1 standard error (GP for tile-based estimates, bootstrap over particles for diffusion time). '
             'For swelling ÷ porosity the ± treats the swelling and porosity errors as independent; both come from the same tiles, so it is approximate. '
             'Bold = robust z &gt; 3.5 vs all other sites.</p>'
             f'<p class="a-prose">{concl}{swell}the higher the Si fraction, the more it thickens and the slower its coarse particles fill with lithium.</p>'
             f'<figure class="a-panel">{img("fig_physics.png", "Physics estimates per site and per tile maps of capacity and unabsorbed expansion")}</figure></section>')

# 3b. feature importance
import os
if os.path.exists("importance.json"):
    I = json.load(open("importance.json")); M = I["model"]
    o.append('<section class="a-section"><h2 class="a-section__title">Feature importance</h2>'
             '<p class="a-section__note">Model-free: η² = share of site-to-site variance explained by batch (on ranks); '
             'Kruskal-Wallis p, Holm-corrected over all KPIs; d = largest effect size of one batch vs all other sites.</p>'
             '<div class="a-table-scroll"><table class="a-table"><caption class="a-visually-hidden">KPI importance ranking</caption><thead><tr>'
             '<th scope="col">Rank</th><th scope="col">KPI</th><th scope="col" data-numeric>η²</th><th scope="col" data-numeric>p</th>'
             '<th scope="col" data-numeric>p (Holm)</th><th scope="col">Batch that differs most</th><th scope="col" data-numeric>d</th></tr></thead><tbody>')
    for i, u in enumerate(I["univariate"], 1):
        o.append(f'<tr><td data-numeric>{i}</td><th scope="row">{E(u["name"])}</th><td data-numeric>{u["eta2"]:.3f}</td>'
                 f'<td data-numeric>{u["kw_p"]:.3f}</td><td data-numeric>{u["kw_p_holm"]:.2f}</td><td>{E(u["batch"])}</td>'
                 f'<td data-numeric>{u["d_signed"]:+.2f}</td></tr>')
    o.append('</tbody></table></div>'
             + (f'<p class="a-prose">Random-forest cross-check skipped: {E(M["skipped"])}.</p>' if "skipped" in M else
             f'<p class="a-prose">Cross-check: a random forest predicting batch from all {len(I["univariate"])} KPIs reaches '
             f'{M["cv_balanced_accuracy"]:.2f} balanced accuracy in cross-validation (chance {M["chance"]:.2f}; shuffled labels '
             f'{M["null_mean"]:.2f}, 95th percentile {M["null_95"]:.2f}; p = {M["p"]:.2f}). That is barely above chance, so '
             'model-based importances are not reported: the batches are not separable as a whole.</p>') +
             f'<figure class="a-panel">{img("fig_importance.png", "Bar chart of KPI importance and heatmap of robust z per site and KPI")}</figure></section>')

# 3c. where the outliers are
if os.path.exists("spots.json"):
    S = json.load(open("spots.json"))
    o.append('<section class="a-section"><h2 class="a-section__title">Where the outliers are in the images</h2>'
             '<p class="a-section__note">Red boxes: 3.2 µm tiles whose KPI is above the 99th percentile of all tiles from the other 30 sites '
             '(about 1% of tiles would be expected for a normal site). Magenta: cracked/irregular Si particles; cyan: anomalous ones. '
             'Numbered crops show the four strongest hotspots.</p>'
             '<div class="a-table-scroll"><table class="a-table"><caption class="a-visually-hidden">Hotspot tiles per site</caption><thead><tr>'
             '<th scope="col">Site</th><th scope="col">KPI</th><th scope="col" data-numeric>Hotspot tiles</th><th scope="col" data-numeric>% of tiles</th>'
             '<th scope="col" data-numeric>Share of excess in hotspots</th><th scope="col" data-numeric>Cracked / anomalous particles</th></tr></thead><tbody>')
    for s, v in S.items():
        o.append(f'<tr><th scope="row">{E(v["batch"])} {E(s)}</th><td>{E(kname(v["kpi"]))}</td><td data-numeric>{v["n_hot"]} / {v["n_tiles"]}</td>'
                 f'<td data-numeric>{v["pct_hot"]:.1f}%</td><td data-numeric>{100 * v["hot_share_of_excess"]:.0f}%</td>'
                 f'<td data-numeric>{v["n_cracked"]} / {v["n_anomalous"]}</td></tr>')
    o.append('</tbody></table></div>')
    o.append(tabs([(f'{v["batch"]} {s}', f'<figure class="a-panel">{img(f"fig_spots_{s}.jpg", f"Hotspot tiles and flagged particles on {s}")}</figure>') for s, v in S.items()], "Site"))
    o.append('</section>')

# 4. figures
o.append('<section class="a-section"><h2 class="a-section__title">Site-level KPIs</h2>'
         '<p class="a-section__note">One dot per site. Error bars are the GP 95% interval, which accounts for neighbouring tiles being correlated.</p>'
         f'<figure class="a-panel">{img("fig_kpis.png", "Strip plots of nine KPIs, one dot per site, grouped by batch")}</figure>')
o.append('<h3>Spatial GP maps</h3><p class="a-prose">Left: segmentation. Middle: GP mean of the KPI over the image (bright = high). '
         'Right: GP standard deviation (bright = less certain). Blob size is the GP lengthscale.</p>'
         '<h3>Si particle size and shape distributions</h3><p class="a-prose">All Si particles of at least 1 µm pooled per batch. Shifts in these curves show changes in milling, supplier PSD or particle breakage.</p>'
         f'<figure class="a-panel">{img("fig_psd.png", "Histograms of Si particle diameter, aspect ratio and circularity per batch")}</figure>'
         '<h3>Cracked and anomalous particle examples</h3><p class="a-prose">The crack rule is simple (internal gaps of at least 2% of the particle area, or solidity below 0.8). Check these crops by eye before trusting the counts.</p>'
         f'<figure class="a-panel">{img("fig_particles.png", "BSE crops of particles flagged as cracked or anomalous")}</figure>'
         f'<figure class="a-panel">{img("fig_gpmaps.png", "Segmentation, GP mean and GP uncertainty maps for three sites")}</figure>')
o.append('<h3>Segmentation check</h3><p class="a-prose">Blue = pore, orange = bright phase (likely Si), grey = graphite/binder. '
         'Every KPI depends on this; it needs confirmation by a materials expert.</p>'
         f'<figure class="a-panel">{img("overlays_small.jpg", "Segmentation overlays for six sites across all batches")}</figure></section>')

# 5. full site table
cols = [("porosity", True), ("si_frac", True), ("interface_um_per_um2", True), ("inlens_texture", True), ("porosity__ls_um", False),
        ("si_d10_um", False), ("si_aspect_ratio", False), ("si_circularity", False), ("si_nn_um", False), ("si_clark_evans", False),
        ("si_cluster_size", False), ("si_n_cracked", False), ("si_cracked_frac", False), ("si_n_anomalous", False), ("inlens_edge_binder", False),
        ("si_frac__ls_um", False), ("si_d50_um", False), ("si_d90_um", False), ("si_count_per_1000um2", False), ("si_solidity", False)]
o.append('<section class="a-section"><h2 class="a-section__title">All sites</h2>'
         '<label>Batch <select data-a-filter="#site-table" data-a-filter-key="batch"><option value="">All</option>'
         + "".join(f'<option value="{E(b)}">{E(b)}</option>' for b in B) + '</select></label>'
         '<div class="a-table-scroll"><table class="a-table" id="site-table" data-a-sticky-columns="1"><caption class="a-visually-hidden">KPIs for every site</caption><thead><tr>'
         '<th scope="col">Site</th><th scope="col">Batch</th><th scope="col">3rd detector</th>'
         + "".join(f'<th scope="col" data-numeric>{E(kname(c))}{" ±GP err" if e else ""}</th>' for c, e in cols)
         + '<th scope="col" data-numeric>Uncertainty ratio GP/naive</th></tr></thead><tbody>')
fl = set(flags.site) if len(flags) else set()
for _, r in D.sort_values(["batch", "site"]).iterrows():
    cells = "".join("<td data-numeric>" + fmt(r[c]) + ((" ± " + fmt(r[c + "__err"])) if e else "") + "</td>" for c, e in cols)
    ratio = np.mean([r[f"{k}__err"] / r[f"{k}__naive_err"] for k in ("porosity", "si_frac", "interface_um_per_um2")])
    mark = " ⚑" if r.site in fl else ""
    o.append(f'<tr data-batch="{E(r.batch)}"><th scope="row">{E(r.site)}{mark}</th><td>{E(r.batch)}</td><td>{E(r.se_detector)}</td>{cells}<td data-numeric>{ratio:.1f}×</td></tr>')
o.append('</tbody></table></div><p class="a-section__note">⚑ = unusual site. Uncertainty ratio: how much wider the GP error bar is than the naive std/√n.</p></section>')

# 6. confounds
o.append('<section class="a-section"><h2 class="a-section__title">Imaging confounds and limits</h2>')
sig = [c for c in R["confounds"] if c.get("p", 1) < 0.05]
if sig:
    o.append('<div class="a-callout a-callout--risk"><p><strong>These KPIs also track imaging conditions</strong>, so part of their variation may come from the microscope rather than the material:</p><ul>')
    for c in sig:
        extra = (f'median {fmt(c["median_with"])} with vs {fmt(c["median_without"])} without' if "median_with" in c
                 else f'Spearman ρ={c["rho"]:+.2f}' if "rho" in c else "")
        o.append(f'<li>{E(kname(c["kpi"]))} vs {E(c["confound"])}: {extra} (p={pv(c["p"])})</li>')
    o.append('</ul></div>')
o.append('<ul class="a-prose"><li>Only 7, 7 and 17 sites per batch: “consistent” is weak evidence of no change, especially for Batch_1 and Batch_2.</li>'
         '<li>Phase identity (bright = Si) is inferred from BSE brightness and the dataset description, not confirmed.</li>'
         '<li>Without an approved reference or specification, the system can say which batch differs and how, not whether it is acceptable.</li></ul></section>')

# 7. method
o.append('<section class="a-section"><h2 class="a-section__title">Method</h2><ol class="a-prose">'
         '<li>Load each site from the PMDB half-resolution cache (50 nm/px, raw uint8 BSE).</li>'
         '<li>Normalise BSE per image: 0 = its own pore-black level, 1 = its own graphite peak. No batch-wide threshold, so the Batch_3 black-level offset is absorbed.</li>'
         '<li>Segment: pore below 0.5; bright phase above the histogram valley before the bright peak; graphite/binder in between.</li>'
         '<li>Cut into 64 px (3.2 µm) tiles; compute porosity, Si fraction, graphite fraction and phase-boundary density per tile.</li>'
         '<li>Fit one GP per KPI per site on tile position (RBF + white noise). Keep the GP mean, its correlated-error standard error (1/√(1ᵀK⁻¹1)) and the lengthscale in µm.</li>'
         '<li>Particle KPIs (d50, d90, count, solidity) from connected bright regions.</li>'
         '<li>Compare batches with site as the unit: permutation tests, bootstrap CIs, Holm correction over KPIs, spread test, robust-z site flags, and confound checks.</li></ol>'
         '<p class="a-prose">Surprise batch: copy it to <code>PMDB/data/Batch_N</code> and run <code>./run_all.sh</code>. Nothing is retrained; the new batch is compared with all other sites.</p></section>')

body = "\n".join(o)
open("body.html", "w").write("<!--REPORT_START-->\n" + body + "\n<!--REPORT_END-->\n")
if os.path.exists("qc_report.html"):
    s = open("qc_report.html").read()
    s2 = re.sub(r"(<!--REPORT_START-->).*?(<!--REPORT_END-->)", lambda m: m.group(1) + body + m.group(2), s, flags=re.S)
    open("qc_report.html", "w").write(s2)
    print("updated qc_report.html" if s2 != s else "qc_report.html already up to date" if "REPORT_START" in s else "markers missing in qc_report.html; recompile body.html")
