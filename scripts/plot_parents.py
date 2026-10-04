"""Figure: rebuilt parent images with their child crops outlined and labelled by batch.

    python scripts/plot_parents.py

Reads outputs/stitching/chains.csv (left-to-right crop order per parent, from
scripts/stitch_report.py) and the half-res BSE cache. Crops inside a chain abut
(seam |dy| <= 5 full-res px, drawn as 0); separate chains of one parent have an
unknown gap and are drawn apart with a "gap unknown" marker.

Writes outputs/stitching/parents_annotated.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmdb.io import load_site  # noqa: E402

OUT = ROOT / "outputs/stitching/parents_annotated.png"
UM_PER_PX = 0.05 * 4          # half-res (50 nm/px) further downsampled 4x for display
CHAIN_GAP_UM = 25.0
# Categorical slots 1-4 of the dataviz reference palette (validated: CVD dE 9.1, normal 22.9)
COLORS = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a", "Batch_test": "#7a3fb8"}
LABELS = {"Batch_1": "Batch 1", "Batch_2": "Batch 2", "Batch_3": "Batch 3", "Batch_test": "test site"}
# Organiser-released ground truth for the held-out sites (2026-10-04); drawn dashed in the true colour.
HELDOUT_TRUTH = {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}
# Held-out parents first, then parents split across batches, then single-batch parents.
ORDER = ["G2316", "G2088", "G2048", "G2080", "G2068SE", "G2148", "G2156", "G2272",
         "G2060", "G1904", "G1612", "G1780", "G1880"]


def _bse(batch: str, site: str) -> np.ndarray:
    kw = {"Batch_heldout": dict(data_root=ROOT / "data_heldout", cache_root=ROOT / "cache_heldout"),
          "Batch_test": dict(data_root=ROOT / "data_test", cache_root=ROOT / "cache_test")}.get(batch, {})
    img = load_site(batch, site, resolution="half", **kw).image[..., 0]
    h, w = (img.shape[0] // 4) * 4, (img.shape[1] // 4) * 4
    return img[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))


def main() -> None:
    chains = pd.read_csv(ROOT / "outputs/stitching/chains.csv")
    parents = {p: g for p, g in chains.groupby("parent")}
    assert set(parents) == set(ORDER), sorted(set(parents) ^ set(ORDER))

    rows = []
    for p in ORDER:
        x, items, gaps, n_seams = 0.0, [], [], 0
        for k, (_, ch) in enumerate(parents[p].iterrows()):
            if k:
                gaps.append((x, x + CHAIN_GAP_UM))
                x += CHAIN_GAP_UM
            sites = [s.strip() for s in ch["chain"].split("|")]
            batches = [b.strip() for b in ch["batches"].split("|")]
            n_seams += len(sites) - 1
            for s, b in zip(sites, batches):
                img = _bse(b, s)
                w, h = img.shape[1] * UM_PER_PX, img.shape[0] * UM_PER_PX
                items.append((s, b, x, w, h, img))
                x += w
        rows.append((p, items, gaps, x, n_seams))

    max_w = max(r[3] for r in rows)
    heights = [max(it[4] for it in r[1]) for r in rows]
    fig_w = 22.0
    scale = fig_w / max_w  # inches per um
    pad_in = 0.55
    fig_h = sum(h * scale + pad_in for h in heights) + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=110)

    y_top = fig_h - 0.9
    for (p, items, gaps, width, n_seams), h in zip(rows, heights):
        ax_h = h * scale
        y_top -= pad_in
        ax = fig.add_axes([0.0, (y_top - ax_h) / fig_h, width * scale / fig_w, ax_h / fig_h])
        y_top -= ax_h
        ax.set_xlim(0, width)
        ax.set_ylim(h, 0)
        ax.axis("off")
        for s, b, x0, w, hh, img in items:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(x0, x0 + w, hh, 0),
                      interpolation="bilinear")
            # three site kinds: known (solid, batch colour), held-out with revealed truth (dotted, true
            # batch colour), test-day unknown (dashed purple + purple tint)
            kind = {"Batch_heldout": "held", "Batch_test": "test"}.get(b, "known")
            true_b = HELDOUT_TRUTH[s] if kind == "held" else b
            label = {"known": f"{s} · {LABELS.get(b)}",
                     "held": f"{s} · held-out → revealed {LABELS[true_b]}",
                     "test": f"{s} · UNKNOWN (test day)"}[kind]
            ls = {"known": "-", "held": ":", "test": "--"}[kind]
            if kind == "test":
                ax.add_patch(Rectangle((x0, 0), w, hh, fc=COLORS["Batch_test"], alpha=0.10, ec="none"))
            ax.add_patch(Rectangle((x0 + 0.6, 0.6), w - 1.2, hh - 1.2, fill=False,
                                   ec=COLORS[true_b], lw=3.5 if kind != "known" else 3.0, ls=ls))
            ax.text(x0 + 3, 3, label, va="top", ha="left", fontsize=10,
                    color="white" if kind == "test" else "#1a1a19",
                    fontweight="normal" if kind == "known" else "bold",
                    bbox=dict(fc=COLORS["Batch_test"] if kind == "test" else "white", ec=COLORS[true_b],
                              lw=2.0, pad=2.5, ls=ls))
        for g0, g1 in gaps:
            ax.text((g0 + g1) / 2, h / 2, "gap\nunknown", ha="center", va="center",
                    fontsize=8, color="#555555")
        n = len(items)
        seam_txt = f"{n_seams} seam{'s' if n_seams != 1 else ''} found" if n > 1 else "single crop"
        fig.text(0.002, (y_top + ax_h + 0.08) / fig_h, f"{p}  ({n} crop{'s' if n != 1 else ''}, {seam_txt})",
                 fontsize=11, fontweight="bold", color="#1a1a19", va="bottom")

    handles = [Patch(fc="white", ec=COLORS[b], lw=3, label=f"{LABELS[b]} (known, solid)")
               for b in COLORS if b != "Batch_test"]
    handles.append(Patch(fc="white", ec="#555555", lw=3, ls=":",
                         label="held-out, truth revealed (dotted, true batch colour)"))
    handles.append(Patch(fc="#d9c9ec", ec=COLORS["Batch_test"], lw=3, ls="--",
                         label="test-day unknown (dashed purple, tinted)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1 - 0.6 / fig_h), ncol=5,
               frameon=False, fontsize=12)
    fig.text(0.002, 1 - 0.35 / fig_h,
             "Parent images (BSE, left to right), all 13 groups incl. test-day sites, single crops and non-adjacent pieces; each outlined crop is one site",
             fontsize=14, fontweight="bold", color="#1a1a19", va="top")
    fig.savefig(OUT, dpi=110, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
