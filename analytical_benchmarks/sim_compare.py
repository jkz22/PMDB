"""Compare the 50-cycle simulation of every site by batch, with the 3 held-back sites (released answers).

python3 sim_compare.py [sim_all]  ->  <dir>/sim_by_site.csv, pitch/slide_5.png
"""
import json, os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "sim_all"
TRUTH = {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}
COL = {"Batch_1": "#d62728", "Batch_2": "#1f77b4", "Batch_3": "#2ca02c"}
B = list(COL)
M = [("retention_pct", "Capacity left after 50 cycles (%)"), ("n_cracks", "Cracks after 50 cycles"),
     ("thickness_charged_pct", "Swelling when charged, cycle 50 (%)"), ("thickness_irrev_pct", "Permanent thickening (%)")]

S = json.load(open(f"{D}/sim.json")); N = S["cycles"]
lab = pd.read_csv("site_kpis.csv"); hel = pd.read_csv("heldout_kpis.csv")
kp = pd.concat([lab, hel.assign(batch="Batch_heldout")])[["site", "batch", "si_peak", "si_frac", "si_solidity"]]
rows = []
for name, v in S["summary"].items():
    site = name.replace("img_", "").replace("_BSE", "")
    rows.append(dict(site=site, **{k: v[k]["end"] for k, _ in M}, si_frac0=v["si_frac"]["cycle1"]))
t = pd.DataFrame(rows).merge(kp, on="site")
t["truth"] = [TRUTH.get(s, b) for s, b in zip(t.site, t.batch)]
t["heldout"] = t.site.isin(TRUTH)
t.to_csv(f"{D}/sim_by_site.csv", index=False)

L = t[~t.heldout]
z = {}
for s in TRUTH:
    r = t[t.site == s].iloc[0]
    z[s] = {b: float(np.sqrt(np.mean([((r[k] - L[L.batch == b][k].median()) / L[k].std()) ** 2 for k, _ in M]))) for b in B}
print(L.groupby("batch")[[k for k, _ in M]].median().round(2))
print(t[t.heldout][["site", "truth"] + [k for k, _ in M] + ["si_peak"]].round(3).to_string(index=False))
print({s: {b: round(v, 2) for b, v in d.items()} for s, d in z.items()})
print("corr cracks~si_peak (labelled):", round(L.n_cracks.corr(L.si_peak), 2), " cracks~si_frac:", round(L.n_cracks.corr(L.si_frac), 2))

fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.text(0.03, 0.955, "Simulating all 34 images: does predicted cracking explain the answers?", fontsize=26, weight="bold")
fig.text(0.03, 0.918, f"{N} cycles at 1C, 1 run per image, same 58 µm crop and parameters for all; stars = 3 held-back images coloured by their TRUE batch. "
         "Illustrative model, uncalibrated.", fontsize=13.5, color="#555")
rng = np.random.default_rng(0)
for i, (k, title) in enumerate(M):
    a = fig.add_axes([0.04 + i * 0.165, 0.50, 0.135, 0.36])
    for j, b in enumerate(B):
        y = L[L.batch == b][k]; a.scatter(j + rng.uniform(-0.15, 0.15, len(y)), y, s=40, color=COL[b], alpha=0.75, zorder=2)
        a.hlines(y.median(), j - 0.3, j + 0.3, color="k", lw=2, zorder=3)
    for s, b in TRUTH.items():
        r = t[t.site == s].iloc[0]; j = B.index(b)
        a.scatter(j + 0.38, r[k], marker="*", s=420, color=COL[b], edgecolor="k", lw=1.2, zorder=4)
        a.annotate(s[:2], (j + 0.38, r[k]), xytext=(9, -4), textcoords="offset points", fontsize=11, weight="bold")
    a.set_xticks(range(3)); a.set_xticklabels(["B1", "B2", "B3"], fontsize=13); a.set_xlim(-0.5, 2.8)
    a.set_title(title, fontsize=13); a.tick_params(labelsize=11); a.grid(axis="y", alpha=0.3)

a = fig.add_axes([0.72, 0.50, 0.25, 0.36])
for b in B:
    q = L[L.batch == b]; a.scatter(q.si_peak, q.n_cracks, s=45, color=COL[b], alpha=0.8, label=b)
for s, b in TRUTH.items():
    r = t[t.site == s].iloc[0]; a.scatter(r.si_peak, r.n_cracks, marker="*", s=420, color=COL[b], edgecolor="k", lw=1.2, zorder=4)
    a.annotate(s, (r.si_peak, r.n_cracks), xytext=(9, -4), textcoords="offset points", fontsize=11, weight="bold")
a.axvline(1.9, ls="--", color="#888"); a.text(1.905, a.get_ylim()[1] * 0.95, "dull Si  |  normal Si", fontsize=10.5, color="#555", va="top")
a.set_xlabel("Si brightness / graphite  (si_peak, imaging contrast)", fontsize=12.5); a.set_ylabel(f"Cracks after {N} cycles", fontsize=12.5)
a.set_title("Most cracks = the 3 dull-Si images", fontsize=14); a.legend(fontsize=10.5, loc="upper right"); a.tick_params(labelsize=11)

a = fig.add_axes([0.04, 0.06, 0.40, 0.33]); a.axis("off")
a.text(0, 1.0, "Which batch do the simulated outcomes point to?", fontsize=16, weight="bold", va="top")
hdr = ["image", "true batch", "sim. closest", "dist. B1 / B2 / B3"]
cells = []
for s, b in TRUTH.items():
    d = z[s]; best = min(d, key=d.get)
    cells.append([s, b.replace("Batch_", "B"), best.replace("Batch_", "B") + ("  ✓" if best == b else "  ✗"),
                  " / ".join(f"{d[x]:.2f}" for x in B)])
tb = a.table(cellText=cells, colLabels=hdr, loc="upper left", bbox=[0, 0.15, 1, 0.7], cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(13)
for (r_, c_), cell in tb.get_celld().items():
    if r_ == 0: cell.set_facecolor("#e8ecf0"); cell.set_text_props(weight="bold")
a.text(0, 0.06, "distance = RMS of the 4 outcomes above vs each batch median, in units of the spread across all sites", fontsize=11, color="#555", va="top")

a = fig.add_axes([0.48, 0.06, 0.49, 0.33]); a.set_xticks([]); a.set_yticks([]); a.set_facecolor("#f4f6f8")
for sp in a.spines.values(): sp.set_color("#cfd6dd")
TXT = ["# Simulation does not explain the answers (0 of 3)",
       "• The 3 batches behave the same: median 96–96.5% capacity, 23–29 cracks (p > 0.19)",
       "• Cracks follow Si amount (r = 0.84), and 3 images crack most: 4ih2ggld, 5n1q8atc, 3e122cbj",
       "• Those 3 are the dull-Si images (si_peak ≈ 1.65); excluding them, r = 0.01",
       "• So 3e122cbj's high cracking comes from the same imaging problem, not from Batch_2",
       "# Takeaway: the simulator inherits whatever the segmentation measures;",
       "      fix the image contrast first, then predictions become meaningful"]
y = 0.92
for line in TXT:
    a.text(0.025, y, line.lstrip("# "), fontsize=14, va="top", transform=a.transAxes, weight="bold" if line.startswith("#") else None); y -= 0.125
os.makedirs("pitch", exist_ok=True); fig.savefig("pitch/slide_5.png"); print("ok")
