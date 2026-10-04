"""Interpret the released held-back answers -> pitch/slide_3.png. usage: PMDB_HELDOUT=<main checkout> python3 pitch_truth.py"""
import json, os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.environ.get("PMDB_HELDOUT", "..")
TRUTH = {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}  # released by organisers after scoring
COL = {"Batch_1": "#d62728", "Batch_2": "#1f77b4", "Batch_3": "#2ca02c"}; B = list(COL); SITES = list(TRUTH)
A = json.load(open("heldout_assign.json")); P = {k: {r["site"]: r for r in A[k]["sites"]} for k in ["material", "imaging", "combined"]}
lab = pd.read_csv("site_kpis.csv"); hel = pd.read_csv("heldout_kpis.csv")
img = pd.read_csv(f"{ROOT}/outputs/clean/summary.csv"); imh = pd.read_csv(f"{ROOT}/outputs/clean_heldout/summary.csv")
TEAM = {"Team KPI model": {"3e122cbj": "Batch_1", "fn0mhxef": "Batch_3", "xrv9xvzb": "Batch_3"},
        "Team fingerprint": {"3e122cbj": "Batch_1", "fn0mhxef": "Batch_3", "xrv9xvzb": "Batch_2"}}
plt.rcParams.update({"font.size": 12.5, "axes.spines.top": False, "axes.spines.right": False})
short = lambda b: b.replace("Batch_", "B")

fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.text(0.03, 0.955, "Checking our answers: what the true batches tell us", fontsize=28, weight="bold", va="center")
fig.text(0.03, 0.912, "Truth: 3e122cbj = Batch_2 · fn0mhxef = Batch_1 · xrv9xvzb = Batch_3.   Our combined call: 2 of 3 right, "
         "but the miss was our most confident call (0.98).", fontsize=15, color="#444", va="center")

# A. probability given to the true batch
a = fig.add_axes([0.05, 0.53, 0.27, 0.30]); w = 0.26
for i, (k, c) in enumerate([("material", "#8c6d31"), ("imaging", "#7f7f7f"), ("combined", "#222")]):
    v = [P[k][s]["p"][TRUTH[s]] for s in SITES]; x = np.arange(3) + (i - 1) * w
    a.bar(x, v, w, color=c, label=k)
    for xi, vi, s in zip(x, v, SITES):
        ok = P[k][s]["assigned"] == TRUTH[s]
        a.text(xi, vi + 0.02, "✓" if ok else "✗", ha="center", fontsize=15, color="#2a7" if ok else "#c00", weight="bold")
a.axhline(1 / 3, ls="--", c="#999"); a.text(2.45, 1 / 3 + .015, "chance", color="#777", fontsize=11, ha="right")
a.set_xticks(range(3), [f"{s}\n(true {short(TRUTH[s])})" for s in SITES]); a.set_ylim(0, 1.08); a.set_ylabel("probability we gave the TRUE batch")
a.legend(frameon=False, fontsize=11, loc="upper left", ncol=3); a.set_title("A · Which evidence pointed to the truth? (✓ = top pick right)", fontsize=14, loc="left")

# B. material space with true colours
a = fig.add_axes([0.385, 0.53, 0.27, 0.30])
for b, d in lab.groupby("batch"): a.scatter(d.si_frac * 100, d.si_solidity, c=COL[b], s=45, alpha=.75, label=b)
for _, r in hel.iterrows():
    a.scatter(r.si_frac * 100, r.si_solidity, marker="*", s=420, c=COL[TRUTH[r.site]], edgecolors="k", lw=1.2, zorder=5)
    a.annotate(f"{r.site} (true {short(TRUTH[r.site])})", (r.si_frac * 100, r.si_solidity), xytext=(9, -4), textcoords="offset points", fontsize=11, weight="bold")
a.axvspan(7.5, 15, ymax=(0.88 - 0.845) / (0.93 - 0.845), color="#999", alpha=.12); a.set_ylim(0.845, 0.93)
a.text(7.7, 0.848, "unusual-site corner: now B1 AND B2", fontsize=11, color="#333", weight="bold")
a.set_xlabel("Si area fraction (%)"); a.set_ylabel("Si solidity"); a.legend(fontsize=10, frameon=False, loc="upper right")
a.set_title("B · Material: the odd morphology is not Batch_1-only", fontsize=14, loc="left")

# C. imaging fingerprint with true colours
a = fig.add_axes([0.715, 0.53, 0.26, 0.30])
for b, d in img.groupby("batch"): a.scatter(d.BSE_D, d.BSE_noise_sigma_g, c=COL[b], s=30 + 35 * (d.BSE_grey_step - 1), alpha=.75)
for _, r in imh.iterrows():
    a.scatter(r.BSE_D, r.BSE_noise_sigma_g, marker="*", s=420, c=COL[TRUTH[r.site]], edgecolors="k", lw=1.2, zorder=5)
    a.annotate(f"{r.site}", (r.BSE_D, r.BSE_noise_sigma_g), xytext={"3e122cbj": (10, 10), "fn0mhxef": (10, -16)}.get(r.site, (9, 4)), textcoords="offset points", fontsize=11, weight="bold")
a.set_xlabel("BSE dark level (DN)"); a.set_ylabel("BSE noise σ"); a.set_title("C · Imaging: identifies the session, not the batch", fontsize=14, loc="left")

# D. why 3e122cbj went to Batch_1: wide-batch effect
ks = ["si_solidity", "si_circularity", "si_frac", "si_cracked_frac", "si_count_per_1000um2"]
nm = {"si_solidity": "solidity", "si_circularity": "circularity", "si_frac": "Si fraction", "si_cracked_frac": "cracked frac.", "si_count_per_1000um2": "Si count"}
h3 = hel.set_index("site").loc["3e122cbj"]; a = fig.add_axes([0.05, 0.08, 0.40, 0.33]); w = 0.26
for i, b in enumerate(B):
    d = lab[lab.batch == b]; z = [(h3[k] - d[k].mean()) / d[k].std() for k in ks]
    a.bar(np.arange(len(ks)) + (i - 1) * w, np.abs(z), w, color=COL[b], label=f"vs {b} (SD ×{np.mean([d[k].std() / lab[lab.batch != 'Batch_1'][k].std() for k in ks]):.1f})")
a.set_xticks(range(len(ks)), [nm[k] for k in ks]); a.set_ylabel("|z| of 3e122cbj vs batch")
a.legend(frameon=False, fontsize=10.5, title="spread relative to B2+B3", title_fontsize=10.5)
a.set_title("D · Why 3e122cbj went to Batch_1: it is unusual for EVERY batch,\n     but Batch_1's spread is 2–3× wider (its 2 odd sites), so it looks least unusual there", fontsize=13.5, loc="left")

# E. scorecard
rows = [["Ours · material"] + [short(P["material"][s]["assigned"]) for s in SITES],
        ["Ours · imaging"] + [short(P["imaging"][s]["assigned"]) for s in SITES],
        ["Ours · combined (sent)"] + [short(P["combined"][s]["assigned"]) for s in SITES]] + \
       [[t] + [short(v[s]) for s in SITES] for t, v in TEAM.items()]
for r in rows: r.append(f"{sum(r[i + 1] == short(TRUTH[s]) for i, s in enumerate(SITES))}/3")
a = fig.add_axes([0.48, 0.06, 0.27, 0.33]); a.axis("off")
tb = a.table(cellText=rows, colLabels=["method", *[f"{s}\ntrue {short(TRUTH[s])}" for s in SITES], "right"], loc="center", cellLoc="center",
             colWidths=[0.34, 0.18, 0.18, 0.18, 0.12])
tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 2.3)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#cfd6dd")
    if r == 0: cell.set_facecolor("#eef1f4"); cell.set_text_props(weight="bold")
    elif 1 <= c <= 3:
        ok = rows[r - 1][c] == short(TRUTH[SITES[c - 1]]); cell.set_facecolor("#e3f4e8" if ok else "#fbe3e3")
a.set_title("E · Scorecard", fontsize=14, loc="left")

# F. lessons
a = fig.add_axes([0.765, 0.05, 0.215, 0.37]); a.set_xticks([]); a.set_yticks([]); a.set_facecolor("#f4f6f8")
for sp in a.spines.values(): sp.set_color("#cfd6dd")
L = [("What it means", dict(weight="bold", fontsize=15)),
     ("• Material KPIs got 0/3 — as their", {}), ("  49% ≈ chance validation predicted.", {}),
     ("  The batches are the same material.", {}),
     ("• The odd Si morphology appears in", {}), ("  B1 and B2: a site-level feature,", {}),
     ("  not a batch identity → flag sites.", {}),
     ("• Imaging got 2/3 by recognising", {}), ("  sessions, and gave the 0.98 miss.", {}),
     ("• Fix: keep imaging separate; multiplying", dict(color="#a00")), ("  it in turned 0.60 into a 0.98 miss.", dict(color="#a00")),
     ("• 3 cases: 0/3 by chance has p ≈ 0.30.", dict(color="#555"))]
y = 0.95
for t, kw in L: a.text(0.04, y, t, transform=a.transAxes, va="top", **{"fontsize": 12, **kw}); y -= 0.076
fig.savefig("pitch/slide_3.png"); print("ok")
