"""Held-back sites placed among the 31 labelled sites: material KPIs, imaging fingerprint, BSE crops + segmentation."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pmdb.io import load_site
from seg import segment, overlay

ROOT = os.environ.get("PMDB_HELDOUT", "..")
COL = {"Batch_1": "#d62728", "Batch_2": "#1f77b4", "Batch_3": "#2ca02c"}
lab = pd.read_csv("site_kpis.csv"); hel = pd.read_csv("heldout_kpis.csv")
img = pd.read_csv(f"{ROOT}/outputs/clean/summary.csv"); imh = pd.read_csv(f"{ROOT}/outputs/clean_heldout/summary.csv")
PAIRS = [("3e122cbj", "Batch_1", "4ih2ggld"), ("fn0mhxef", "Batch_1", "fzrt2k6r"), ("xrv9xvzb", "Batch_3", "hzumfsms")]

fig = plt.figure(figsize=(16, 15)); g = fig.add_gridspec(4, 3, height_ratios=[1.1, 1, 1, 1])
for n, (x, y, t) in enumerate([("si_frac", "si_solidity", "Material: Si fraction vs Si solidity"),
                               ("si_count_per_1000um2", "si_circularity", "Material: Si count vs circularity")]):
    a = fig.add_subplot(g[0, n])
    for b, d in lab.groupby("batch"): a.scatter(d[x], d[y], c=COL[b], s=40, label=b, alpha=.8)
    a.scatter(hel[x], hel[y], marker="*", s=320, c="k", label="held-back")
    for _, r in hel.iterrows(): a.annotate(r.site, (r[x], r[y]), xytext=(6, 4), textcoords="offset points", fontsize=10, weight="bold")
    a.set_xlabel(x); a.set_ylabel(y); a.set_title(t, fontsize=11); a.legend(fontsize=8)
a = fig.add_subplot(g[0, 2])
for b, d in img.groupby("batch"):
    a.scatter(d.BSE_D, d.BSE_noise_sigma_g, c=COL[b], s=40 + 40 * (d.BSE_grey_step - 1), label=b, alpha=.8)
a.scatter(imh.BSE_D, imh.BSE_noise_sigma_g, marker="*", s=320, c="k", label="held-back")
for _, r in imh.iterrows(): a.annotate(f"{r.site} (step {r.BSE_grey_step}, H {r.height})", (r.BSE_D, r.BSE_noise_sigma_g),
                                       xytext=(6, 4), textcoords="offset points", fontsize=9, weight="bold")
a.set_xlabel("BSE dark level D (DN)"); a.set_ylabel("BSE noise σ_g"); a.legend(fontsize=8)
a.set_title("Imaging fingerprint (acquisition, not material)\nmarker size = grey-level step", fontsize=11)
for r, (h, b, s) in enumerate(PAIRS, start=1):
    for c, (bb, ss, kw) in enumerate([("Batch_heldout", h, dict(data_root=f"{ROOT}/data_heldout", cache_root=f"{ROOT}/cache_heldout")),
                                      (b, s, {})]):
        bse = load_site(bb, ss, resolution="half", normalise="none", **kw).image[..., 0]; L, info = segment(bse)
        H, W = bse.shape; cr = (slice(H // 2 - 200, H // 2 + 200), slice(W // 2 - 300, W // 2 + 300))
        ax = fig.add_subplot(g[r, c]); ax.imshow(bse[cr], cmap="gray", vmin=0, vmax=255); ax.axis("off")
        ax.set_title(f"{'held-back ' if c == 0 else 'nearest: ' + bb + ' '}{ss}  BSE (30×20 µm)", fontsize=10)
        if c == 0:
            ax = fig.add_subplot(g[r, 2]); ax.imshow(overlay(bse, L)[cr]); ax.axis("off")
            ax.set_title(f"{ss} segmentation: blue pore, orange Si (thr {info['si_thr']:.2f})", fontsize=10)
fig.tight_layout(); fig.savefig("fig_heldout.png", dpi=90)
