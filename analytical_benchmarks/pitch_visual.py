"""Visual reading of the released held-back answers -> pitch/slide_4.png. usage: PMDB_HELDOUT=<main checkout> python3 pitch_visual.py"""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from pmdb.io import load_site
from seg import anchor, segment

ROOT = os.environ.get("PMDB_HELDOUT", "..")
COL = {"Batch_1": "#d62728", "Batch_2": "#1f77b4", "Batch_3": "#2ca02c"}
TRUTH = {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}
ROWS = [("3e122cbj", [("Batch_1", "4ih2ggld", "same session"), ("Batch_2", "i9jiqjwl", "nearest B2 by KPIs")],
         "Dim Si (Si/graphite 1.69) like the 2 'odd' Batch_1 sites (1.66, 1.68); every other site is 1.98–2.73"),
        ("fn0mhxef", [("Batch_1", "fzrt2k6r", "nearest by KPIs"), ("Batch_2", "3806gxp0", "nearest B2 by KPIs")],
         "Typical bright Si; Batch_1 and Batch_2 look the same"),
        ("xrv9xvzb", [("Batch_3", "hzumfsms", "same session"), ("Batch_2", "3806gxp0", "nearest by KPIs")],
         "Typical bright Si; matches Batch_2 material, Batch_3 session")]
W, H = 640, 400  # 32 x 20 um at 50 nm/px


def load(b, s):
    kw = dict(data_root=f"{ROOT}/data_heldout", cache_root=f"{ROOT}/cache_heldout") if b == "Batch_heldout" else {}
    g = load_site(b, s, resolution="half", normalise="none", **kw).image[..., 0].astype(np.uint8)
    a, _ = anchor(g); L, info = segment(g)
    si = ndi.uniform_filter((L == 2).astype(np.float32), (H, W))[H // 2::40, W // 2::40][: -(H // 80 + 1), : -(W // 80 + 1)]
    i, j = np.unravel_index(np.argmax(si), si.shape); y, x = i * 40, j * 40  # Si-rich window, for illustration
    return a, L, info, (slice(y, y + H), slice(x, x + W))


img = pd.concat([pd.read_csv(f"{ROOT}/outputs/clean/summary.csv"), pd.read_csv(f"{ROOT}/outputs/clean_heldout/summary.csv")])
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.text(0.02, 0.962, "Looking at the 3 images: why the answers are B2, B1, B3", fontsize=26, weight="bold", va="center")
fig.text(0.02, 0.923, "Brightness rescaled so pore = 0, graphite = 1 on every image (same grey scale in all panels). "
         "Orange outline = what the segmentation calls Si. Each crop 32 × 20 µm.", fontsize=13.5, color="#444", va="center")
x0, cw, ch, gap = 0.02, 0.165, 0.235, 0.008
for r, (h, refs, note) in enumerate(ROWS):
    y = 0.62 - r * 0.29; hist = fig.add_axes([x0 + 3 * (cw + gap) + 0.03, y + 0.02, 0.12, ch - 0.04])
    for c, (b, s, why) in enumerate([("Batch_heldout", h, f"true {TRUTH[h]}")] + refs):
        a, L, info, cr = load(b, s); ax = fig.add_axes([x0 + c * (cw + gap), y, cw, ch])
        ax.imshow(a[cr], cmap="gray", vmin=0, vmax=3, interpolation="lanczos"); ax.contour(L[cr] == 2, [0.5], colors="#ff8c00", linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        clr = COL[TRUTH[h]] if c == 0 else COL[b]
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(clr); sp.set_linewidth(4 if c == 0 else 2)
        ax.set_title(f"{'★ ' + h if c == 0 else b + ' ' + s}  ({why})", fontsize=11.5, color=clr, weight="bold" if c == 0 else None, loc="left")
        hh, e = np.histogram(a, bins=240, range=(0.3, 3.3), density=True)
        hist.semilogy(e[:-1], ndi.gaussian_filter1d(hh, 1.5) + 1e-4, c=clr, lw=2.4 if c == 0 else 1.5, ls="-" if c == 0 else "--")
        hist.axvline(info["si_peak"], c=clr, lw=0.8, ls=":")
    hist.set_ylim(1e-3, 10); hist.set_yticks([]); hist.set_xlabel("grey / graphite", fontsize=10.5)
    hist.text(1.12, 4, "graphite", fontsize=9.5); hist.text(2.3, 0.4, "Si", fontsize=9.5, ha="center")
    fig.text(x0, y - 0.022, note, fontsize=12.5, style="italic", color="#333")

a = fig.add_axes([0.755, 0.50, 0.225, 0.36])
for b, d in img[img.batch.isin(COL)].groupby("batch"): a.scatter(d.BSE_si_graphite, d.height / 2, c=COL[b], s=50, alpha=.8, label=b)
for _, r in img[img.site.isin(TRUTH)].iterrows():
    a.scatter(r.BSE_si_graphite, r.height / 2, marker="*", s=380, c=COL[TRUTH[r.site]], edgecolors="k", lw=1.2, zorder=5)
    a.annotate(r.site, (r.BSE_si_graphite, r.height / 2), xytext={"fn0mhxef": (8, -14), "3e122cbj": (10, -14)}.get(r.site, (8, 6)), textcoords="offset points", fontsize=10.5, weight="bold")
a.add_patch(plt.Rectangle((1.63, 1145), 0.09, 26, fill=False, ec="#555", ls="--")); a.text(1.64, 1176, "session A: 3e122cbj + 4ih2ggld + 5n1q8atc", fontsize=9.5)
a.add_patch(plt.Rectangle((1.96, 1032), 0.11, 24, fill=False, ec="#555", ls="--")); a.text(1.62, 1066, "session B: xrv9xvzb\n+ 3 B3 sites", fontsize=9.5)
a.set_xlabel("BSE Si / graphite contrast (imaging)"); a.set_ylabel("image height (px, half-res)"); a.legend(fontsize=9.5, frameon=False, loc="lower right")
a.set_ylim(780, 1200); a.set_title("Imaging sessions mix batches", fontsize=13.5, loc="left")

t = fig.add_axes([0.755, 0.04, 0.225, 0.40]); t.set_xticks([]); t.set_yticks([]); t.set_facecolor("#f4f6f8")
for sp in t.spines.values(): sp.set_color("#cfd6dd")
L_ = [("What the images show", dict(weight="bold", fontsize=14.5)),
      ("• 3e122cbj, 4ih2ggld, 5n1q8atc share a", {}), ("  session (frame, black level, low Si", {}),
      ("  contrast) but are B2, B1, B1.", {}),
      ("• So the dim, 'irregular, high-Si' look", {}), ("  follows the session, not the batch.", {}),
      ("  Low contrast also lets the Si mask", {}), ("  grab dim edges → Si% up, solidity down.", {}),
      ("• fn0mhxef, xrv9xvzb: Si looks the same", {}), ("  as typical B1, B2 and B3 sites.", {}),
      ("• Lesson: the 'Batch_1 sub-population'", dict(color="#a00")), ("  is likely partly an imaging artefact.", dict(color="#a00")),
      ("  Needs EDS / re-imaging to settle.", dict(color="#a00"))]
yy = 0.95
for s_, kw in L_: t.text(0.04, yy, s_, transform=t.transAxes, va="top", **{"fontsize": 11.8, **kw}); yy -= 0.071
fig.savefig("pitch/slide_4.png"); print("ok")
