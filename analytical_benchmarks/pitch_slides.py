"""Two 16:9 summary slides of the analytical branch -> pitch/slide_1.png, pitch/slide_2.png.
usage: PMDB_HELDOUT=<main checkout> python3 pitch_slides.py [evolve_video.mp4]"""
import os, subprocess, sys, tempfile, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pmdb.io import load_site
from seg import segment, overlay

COL = {"Batch_1": "#d62728", "Batch_2": "#1f77b4", "Batch_3": "#2ca02c"}
lab = pd.read_csv("site_kpis.csv"); hel = pd.read_csv("heldout_kpis.csv"); phy = pd.read_csv("physics.csv")
FLAG = ["4ih2ggld", "5n1q8atc"]
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def title(fig, t, s):
    fig.text(0.03, 0.945, t, fontsize=30, weight="bold", va="center")
    fig.text(0.03, 0.895, s, fontsize=16, color="#444", va="center")


def box(fig, rect, lines, fc="#f4f6f8", ec="#cfd6dd"):
    a = fig.add_axes(rect); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(True); sp.set_color(ec)
    a.set_facecolor(fc); y = 0.93
    for txt, kw in lines:
        kw = dict(kw); dy = kw.pop("dy", 0.115)
        a.text(0.03, y, txt, transform=a.transAxes, va="top", **{"fontsize": 13.5, **kw}); y -= dy
    return a


# ---------------- slide 1
bse = load_site("Batch_1", "4ih2ggld", resolution="half", normalise="none").image[..., 0]; L, _ = segment(bse)
H, W = bse.shape; c = (slice(H // 2 - 220, H // 2 + 220), slice(W // 2 - 330, W // 2 + 330))
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
title(fig, "From an SEM image to a batch verdict, with no baseline",
      "31 labelled sites (Batch_1 7 · Batch_2 7 · Batch_3 17) + 3 held-back sites · same code, one command, ~2 min")
steps = [("1 · Raw BSE (50 nm/px)", bse[c], "gray"), ("2 · Segmentation\npore · graphite+binder · likely Si", overlay(bse, L)[c], None)]
for i, (t, im, cm) in enumerate(steps):
    a = fig.add_axes([0.03 + i * 0.205, 0.50, 0.19, 0.33]); a.imshow(im, cmap=cm, vmin=0, vmax=255); a.axis("off"); a.set_title(t, fontsize=14, loc="left")
box(fig, [0.445, 0.53, 0.25, 0.28], [
    ("3 · 20 physical KPIs per site", dict(weight="bold", fontsize=15, dy=0.15)),
    ("porosity · Si fraction · phase-boundary density", {}), ("Si d10/d50/d90 · PSD span · aspect", {}),
    ("solidity · circularity · cracked fraction", {}), ("clustering (Clark–Evans, NN distance)", {}),
    ("Inlens texture · anomalous Si", {}), ("each ± spatial GP error bar", dict(color="#0b5394", weight="bold"))])
box(fig, [0.715, 0.53, 0.255, 0.28], [
    ("4 · Each batch vs all other sites", dict(weight="bold", fontsize=15, dy=0.15)),
    ("permutation test, Holm-corrected, effect size", {}), ("+ unusual-site check (robust z > 3.5)", dict(dy=0.16)),
    ("Batch_1  →  INVESTIGATE", dict(color=COL["Batch_1"], weight="bold", fontsize=16, dy=0.13)),
    ("Batch_2  →  consistent", dict(color=COL["Batch_2"], weight="bold", fontsize=16, dy=0.13)),
    ("Batch_3  →  consistent", dict(color=COL["Batch_3"], weight="bold", fontsize=16))])
for x in [0.428, 0.700]: fig.text(x, 0.67, "→", fontsize=34, ha="center", va="center", color="#888")

a = fig.add_axes([0.05, 0.08, 0.47, 0.34]); x0 = 0; ticks = []
for b, d in lab.groupby("batch"):
    d = d.sort_values("si_frac"); x = np.arange(len(d)) + x0
    a.errorbar(x, d.si_frac * 100, yerr=1.96 * d.si_frac__err * 100, fmt="o", c=COL[b], capsize=2.5, ms=6)
    for xi, (_, r) in zip(x, d.iterrows()):
        if r.site in FLAG: a.annotate(r.site, (xi, r.si_frac * 100), xytext=(-70, 0), textcoords="offset points", fontsize=11, color=COL[b])
    ticks.append((x.mean(), b)); x0 += len(d) + 1.5
x = np.arange(len(hel)) + x0
a.errorbar(x, hel.si_frac * 100, yerr=1.96 * hel.si_frac__err * 100, fmt="*", ms=15, c="k", capsize=2.5)
for xi, (_, r) in zip(x, hel.iterrows()): a.annotate(r.site, (xi, r.si_frac * 100), xytext=(7, 3), textcoords="offset points", fontsize=10.5)
ticks.append((x.mean(), "held-back")); a.set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
a.set_ylabel("Si area fraction (%)\n95% GP interval"); a.set_title("Batch_2 ≈ Batch_3; Batch_1 has a sub-population with ~2× Si", fontsize=15, loc="left")

box(fig, [0.56, 0.04, 0.41, 0.42], [
    ("What the KPIs found", dict(weight="bold", fontsize=16, dy=0.095)),
    ("• No batch average differs after correction.", {}),
    ("• 2 of 7 Batch_1 sites: ~2× Si, irregular (solidity 0.86 vs 0.905),", dict(dy=0.075)),
    ("   dimmer, more cracked Si particles. No Batch_2/3 site looks like this.", dict(dy=0.11)),
    ("Held-back sites", dict(weight="bold", fontsize=16, dy=0.095)),
    ("• 3e122cbj → Batch_1, high confidence (0.98): same signature.", {}),
    ("• fn0mhxef, xrv9xvzb: typical material → batch not recoverable", dict(dy=0.075)),
    ("   from material (Batch_1/2 at 0.54/0.45; Batch_3 at 0.48).", dict(dy=0.11)),
    ("Honest accuracy: material KPIs place 49% of known sites (shuffled", dict(color="#a00", dy=0.075)),
    ("labels: 49%) — because Batch_2 and Batch_3 are the same material.", dict(color="#a00"))])
fig.savefig("pitch/slide_1.png"); plt.close(fig)

# ---------------- slide 2
vid = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/evo_100/evolve_img_4ih2ggld_BSE.mp4")
tmp = tempfile.mkdtemp(); fr = {}
for t in [6, 34]:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", vid, "-frames:v", "1", f"{tmp}/f{t}.png"], check=True)
    fr[t] = np.asarray(Image.open(f"{tmp}/f{t}.png").convert("RGB"))
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
title(fig, "What the Batch_1 difference means for the battery",
      "Physics estimates from the KPIs + image-based cycling simulation (literature parameters, uncalibrated → compare sites, not lifetimes)")
a = fig.add_axes([0.05, 0.53, 0.27, 0.30])
for i, (b, d) in enumerate(phy.groupby("batch")):
    xs = i + np.random.default_rng(i).uniform(-.18, .18, len(d)); a.scatter(xs, d.swelling * 100, c=COL[b], s=55, alpha=.85)
    for xx, (_, r) in zip(xs, d.iterrows()):
        if r.site in FLAG: a.annotate(r.site, (xx, r.swelling * 100), xytext=(8, -4), textcoords="offset points", fontsize=11, color=COL[b])
a.set_xticks([0, 1, 2], ["Batch_1", "Batch_2", "Batch_3"]); a.set_ylabel("added volume at full charge\n(% of coating, before pores absorb it)")
a.set_title("Estimated swelling per site", fontsize=15, loc="left")
a = fig.add_axes([0.05, 0.10, 0.27, 0.32])
for s, b, lbl in [("5n1q8atc", "Batch_1", "Batch_1 5n1q8atc (flagged)"), ("4ih2ggld", "Batch_1", "Batch_1 4ih2ggld (flagged)"),
                  ("epqdaau9", "Batch_2", "Batch_2 epqdaau9 (typical)"), ("x77cy643", "Batch_3", "Batch_3 x77cy643 (typical)")]:
    t = pd.read_csv(f"sim/traj_{s}.csv").groupby("cycle").retention_pct.agg(["mean", "min", "max"]).dropna()
    ls = "-" if s in FLAG else "--"; a.plot(t.index, t["mean"], ls, c=COL[b], lw=2.2, label=lbl); a.fill_between(t.index, t["min"], t["max"], color=COL[b], alpha=.15)
a.set_xlabel("cycle"); a.set_ylabel("capacity retention (%)"); a.legend(fontsize=10.5, frameon=False)
a.set_title("Simulated capacity over 50 cycles (3 runs)", fontsize=15, loc="left")

crops = [(fr[6][195:735, 555:1035], "Cycle 1, charging: local swelling\n(red = swollen Si, blue = squeezed pores)"),
         (fr[34][205:560, 1163:1500], "Largest Si particle, cycle 0"), (fr[34][205:560, 1540:1877], "Same particle after 100 cycles:\ncracks + SEI")]
for i, (im, t) in enumerate(crops):
    a = fig.add_axes([0.36 + i * 0.205, 0.43, 0.195, 0.40]); a.imshow(im); a.axis("off"); a.set_title(t, fontsize=13, loc="left")
fig.text(0.36, 0.405, "Real BSE texture of 4ih2ggld deformed by the simulated expansion at true scale (from the 100-cycle video)", fontsize=12, color="#444")

S = {"5n1q8atc": ("92.3%", "12.9%", "91"), "4ih2ggld": ("92.6%", "13.7%", "77"), "epqdaau9": ("96.1%", "7.4%", "22"), "x77cy643": ("95.2%", "9.1%", "33")}
a = fig.add_axes([0.36, 0.10, 0.33, 0.25]); a.axis("off")
tb = a.table(cellText=[[f"{s}{' *' if s in FLAG else ''}", *S[s]] for s in S], colLabels=["site", "capacity left", "thickness swing", "cracks"],
             loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(13); tb.scale(1, 2.0)
for (r, cc), cell in tb.get_celld().items():
    cell.set_edgecolor("#cfd6dd")
    if r == 0: cell.set_facecolor("#eef1f4"); cell.set_text_props(weight="bold")
    elif r <= 2: cell.set_text_props(color=COL["Batch_1"], weight="bold")
a.set_title("After 50 simulated cycles. * flagged Batch_1 sites:\n~2× thickness swing, 3–4× cracks, ~2× capacity loss", fontsize=13.5, loc="left")
box(fig, [0.72, 0.08, 0.25, 0.29], [
    ("QC decision", dict(weight="bold", fontsize=16, dy=0.15)),
    ("Batch_1 / 3e122cbj → INVESTIGATE", dict(color=COL["Batch_1"], weight="bold", dy=0.13)),
    ("• EDS on the dim-grey Si particles", {}), ("  (Si vs SiOx / Si–C)", dict(dy=0.13)),
    ("• Image ≥5 fields per batch:", {}), ("  catches a Batch_1-like batch ~81%", dict(dy=0.13)),
    ("• Batch_2, Batch_3 → consistent", {})])
fig.savefig("pitch/slide_2.png"); plt.close(fig)
print("ok")
