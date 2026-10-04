#!/usr/bin/env python3
"""Depth-resolved swelling: the SOC-1 swelling test per through-thickness band (rows = z).

The fingerprint says the batches differ in *where* the Si sits through the coating; this asks whether the
lithiation consequence localises the same way (constrained share, pore loss, swollen Si per band).

    python scripts/run_depth_swelling.py [--bands 5] [--jobs 7] -> outputs/functional/depth_swelling.csv
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmdb import functional, segment as segment_mod  # noqa: E402
from pmdb.io import list_sites, load_site  # noqa: E402

OUT = ROOT / "outputs" / "functional"
HELDOUT_DATA, HELDOUT_CACHE = ROOT / "data_heldout", ROOT / "cache_heldout"


def one(args: tuple) -> list[dict]:
    batch, site, heldout, n_bands = args
    kw = {"data_root": HELDOUT_DATA, "cache_root": HELDOUT_CACHE} if heldout else {}
    raw = load_site(batch, site, resolution="half", normalise="none", **kw)
    bse = np.asarray(raw.image[..., 0], dtype=np.float64)
    masks = segment_mod.segment_bse(bse, raw.nm_per_px, params=None)
    valid_rows = np.where(masks.fraction_space.mean(1) > 0.5)[0]
    edges = np.linspace(valid_rows[0], valid_rows[-1] + 1, n_bands + 1).astype(int)
    rows = []
    for b in range(n_bands):
        m = masks.crop(slice(edges[b], edges[b + 1]), slice(None))
        sw = functional.swelling_test(m, 1.0)
        dom = m.fraction_space
        rows.append({"batch": batch, "site": site, "heldout": heldout, "band": b, "n_bands": n_bands,
                     "depth_rel": (b + 0.5) / n_bands, "si0_frac": float((m.si & dom).sum() / max(dom.sum(), 1)),
                     "pore0_frac": float((m.pore & dom).sum() / max(dom.sum(), 1)),
                     **{f"F02_{k}": v for k, v in sw.items()}})
    return rows


def plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"Batch_1": "tab:blue", "Batch_2": "tab:orange", "Batch_3": "tab:green"}
    lab = df[~df.heldout]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (val, lbl) in zip(axes, (("si0_frac", "Si fraction (today)"), ("F02_pore_loss", "pore loss at SOC 1"),
                                     ("F02_into_graphite", "constrained share at SOC 1"))):
        for b, g in lab.groupby("batch"):
            med = g.groupby("band")[val].median()
            q1, q3 = g.groupby("band")[val].quantile(0.25), g.groupby("band")[val].quantile(0.75)
            ax.plot(med.values, med.index, "-o", c=col[b], label=b)
            ax.fill_betweenx(med.index, q1.values, q3.values, color=col[b], alpha=0.15)
        for s, g in df[df.heldout].groupby("site"):
            g = g.sort_values("band")
            ax.plot(g[val].values, g["band"].values, "--", c="k", lw=0.8)
            ax.annotate(s, (g[val].values[-1], g["band"].values[-1]), fontsize=6, xytext=(2, 0), textcoords="offset points")
        ax.set_xlabel(lbl, fontsize=9)
        ax.set_ylabel("depth band (0 = free surface, 4 = collector)", fontsize=8)
        ax.invert_yaxis()
    axes[0].legend(fontsize=7)
    fig.suptitle("Depth-resolved swelling test (median, IQR per batch; dashed = held-out)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "depth_swelling.png", dpi=130)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        plot(pd.read_csv(OUT / "depth_swelling.csv"))
        return 0
    jobs = [(r.batch, r.site, False, a.bands) for r in list_sites().itertuples()] + \
           [(r.batch, r.site, True, a.bands) for r in list_sites(HELDOUT_DATA).itertuples()]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [r for rs in ex.map(one, jobs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "depth_swelling.csv", index=False)
    plot(df)
    lab = df[~df.heldout]
    for col in ("si0_frac", "F02_into_graphite", "F02_pore_loss"):
        print(col)
        print(lab.pivot_table(index="band", columns="batch", values=col, aggfunc="median").round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
