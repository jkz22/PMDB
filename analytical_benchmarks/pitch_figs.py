"""Figures for the held-back presentation report (pitch/): segmentation check, batch differences, image gallery."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from pmdb.io import load_site
from seg import segment, overlay

ROOT = os.environ.get("PMDB_HELDOUT", "..")
HK = dict(data_root=f"{ROOT}/data_heldout", cache_root=f"{ROOT}/cache_heldout")
COL = {"Batch_1": "#d62728", "Batch_2": "#1f77b4", "Batch_3": "#2ca02c"}
lab = pd.read_csv("site_kpis.csv"); hel = pd.read_csv("heldout_kpis.csv")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def crop(b, s, kw={}, h=200, w=300):
    bse = load_site(b, s, resolution="half", normalise="none", **kw).image[..., 0]; L, info = segment(bse)
    H, W = bse.shape; c = (slice(H // 2 - h, H // 2 + h), slice(W // 2 - w, W // 2 + w))
    return bse[c], overlay(bse, L)[c], info


def save(fig, name, w=1500):
    fig.savefig(f"pitch/{name}.png", dpi=110); plt.close(fig)
    im = Image.open(f"pitch/{name}.png").convert("RGB")
    im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS).save(f"pitch/{name}.jpg", quality=80); os.remove(f"pitch/{name}.png")


# 1. segmentation check on the held-back sites
fig, ax = plt.subplots(3, 2, figsize=(13, 13.5))
for r, s in enumerate(hel.site):
    g, o, info = crop("Batch_heldout", s, HK)
    ax[r, 0].imshow(g, cmap="gray", vmin=0, vmax=255); ax[r, 0].set_title(f"{s}: raw BSE, 30 × 20 µm", fontsize=12)
    ax[r, 1].imshow(o); ax[r, 1].set_title(f"{s}: segmentation (blue = pore, orange = bright phase / likely Si)", fontsize=12)
    for a in ax[r]: a.axis("off")
fig.tight_layout(); save(fig, "seg_heldout")

# 2. gallery: unusual Batch_1 vs typical sites vs held-back
G = [("Batch_1", "4ih2ggld", "Batch_1 4ih2ggld (unusual)"), ("Batch_2", "epqdaau9", "Batch_2 epqdaau9 (typical)"),
     ("Batch_3", "x77cy643", "Batch_3 x77cy643 (typical)")] + [("Batch_heldout", s, f"held-back {s}") for s in hel.site]
fig, ax = plt.subplots(2, 3, figsize=(16, 7.6))
for a, (b, s, t) in zip(ax.ravel(), G):
    g, _, _ = crop(b, s, HK if b == "Batch_heldout" else {}, 150, 225)
    a.imshow(g, cmap="gray", vmin=0, vmax=255); a.set_title(t, fontsize=12); a.axis("off")
fig.tight_layout(); save(fig, "gallery")

# 3. batch differences: Si fraction per site with GP error bars, and Si fraction vs solidity
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
x0 = 0; ticks = []
for b, d in lab.groupby("batch"):
    d = d.sort_values("si_frac"); x = np.arange(len(d)) + x0
    ax[0].errorbar(x, d.si_frac * 100, yerr=1.96 * d.si_frac__err * 100, fmt="o", c=COL[b], capsize=3, label=b)
    ticks.append((x.mean(), b)); x0 += len(d) + 1.5
x = np.arange(len(hel)) + x0
ax[0].errorbar(x, hel.si_frac * 100, yerr=1.96 * hel.si_frac__err * 100, fmt="*", ms=14, c="k", capsize=3, label="held-back")
for xi, (_, r) in zip(x, hel.iterrows()): ax[0].annotate(r.site, (xi, r.si_frac * 100), xytext=(6, 4), textcoords="offset points", fontsize=9)
ticks.append((x.mean(), "held-back"))
ax[0].set_xticks([t[0] for t in ticks], [t[1] for t in ticks]); ax[0].set_ylabel("Si area fraction (%), 95% GP interval")
ax[0].set_title("Si fraction per site: two Batch_1 sites and 3e122cbj stand apart", fontsize=12); ax[0].legend(fontsize=9, loc="upper left")
for b, d in lab.groupby("batch"): ax[1].scatter(d.si_frac * 100, d.si_solidity, c=COL[b], s=45, label=b, alpha=.85)
ax[1].scatter(hel.si_frac * 100, hel.si_solidity, marker="*", s=300, c="k", label="held-back")
for _, r in hel.iterrows(): ax[1].annotate(r.site, (r.si_frac * 100, r.si_solidity), xytext=(6, 4), textcoords="offset points", fontsize=10)
ax[1].axvspan(7.5, 15, ymax=(0.88 - 0.845) / (0.93 - 0.845), color="#d62728", alpha=.08)
ax[1].text(7.7, 0.848, "unusual-site signature\n(Si > 7.5 %, solidity < 0.88)", fontsize=9, color="#a00")
ax[1].set_ylim(0.845, 0.93); ax[1].set_xlabel("Si area fraction (%)"); ax[1].set_ylabel("Si particle solidity (1 = convex)")
ax[1].set_title("More Si and more irregular Si go together", fontsize=12); ax[1].legend(fontsize=9)
fig.tight_layout(); save(fig, "batches")
print("ok")
