#!/usr/bin/env python3
"""Segmenter agreement: do the functional conclusions survive a different, independently written segmenter?

`pmdb.segment.segment_bse` (this repo, v0r1) vs the analytical-benchmarks session's per-image anchored
segmenter (`scripts/_ab_seg_ref.py`, copied verbatim from branch `analytical-benchmarks`). Both are run on
the same half-res BSE; Si/pore IoU, K01 and the SOC-1 swelling test are compared per field, and the
batch tests of docs/functional.md are repeated under the alternative masks.

    python scripts/run_segmenter_agreement.py [--jobs 7] -> outputs/segagree/
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _ab_seg_ref as ab  # noqa: E402
from pmdb import functional, segment as segment_mod  # noqa: E402
from pmdb.io import list_sites, load_site  # noqa: E402
from pmdb.segment import Masks  # noqa: E402

OUT = ROOT / "outputs" / "segagree"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def one(args: tuple) -> dict:
    batch, site = args
    raw = load_site(batch, site, resolution="half", normalise="none")
    bse8 = np.asarray(raw.image[..., 0])
    m1 = segment_mod.segment_bse(bse8.astype(np.float64), raw.nm_per_px, params=None)
    lab, info = ab.segment(bse8.astype(np.uint8))
    si2, pore2 = lab == 2, lab == 0
    gr2 = ~(si2 | pore2) & ~m1.artefact
    m2 = Masks(si=si2 & ~m1.artefact, graphite=gr2, pore=pore2 & ~m1.artefact, artefact=m1.artefact,
               admissible=~gr2 & ~m1.artefact, version="ab-anchored", params=info)
    dom = m1.fraction_space
    r = {"batch": batch, "site": site, "si_iou": iou(m1.si, m2.si), "pore_iou": iou(m1.pore, m2.pore),
         "K01_v0r1": float(m1.si[dom].mean()), "K01_ab": float(m2.si[dom].mean()),
         "pore_v0r1": float(m1.pore[dom].mean()), "pore_ab": float(m2.pore[dom].mean()), "ab_si_thr": info["si_thr"]}
    for tag, m in (("v0r1", m1), ("ab", m2)):
        sw = functional.swelling_test(m, 1.0)
        r.update({f"{k}_{tag}": v for k, v in sw.items()})
        r[f"not_into_pore_{tag}"] = 1.0 - sw["into_pore"]
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    a = ap.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    jobs = [(r.batch, r.site) for r in list_sites().itertuples()]
    with ProcessPoolExecutor(a.jobs) as ex:
        D = pd.DataFrame(list(ex.map(one, jobs)))
    D.to_csv(OUT / "per_site.csv", index=False)
    summ = {"n_sites": len(D), "si_iou_median": float(D.si_iou.median()), "si_iou_min": float(D.si_iou.min()),
            "pore_iou_median": float(D.pore_iou.median())}
    for q in ("K01", "pore", "into_graphite", "not_into_pore", "pore_loss"):
        x, y = D[f"{q}_v0r1"], D[f"{q}_ab"]
        rho, p = spearmanr(x, y)
        summ[q] = {"spearman": float(rho), "p": float(p), "median_v0r1": float(x.median()), "median_ab": float(y.median()),
                   "kw_batch_v0r1": float(kruskal(*[g for _, g in x.groupby(D.batch)]).pvalue),
                   "kw_batch_ab": float(kruskal(*[g for _, g in y.groupby(D.batch)]).pvalue),
                   "batch_medians_ab": {b: float(v) for b, v in y.groupby(D.batch).median().items()}}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"Batch_1": "tab:blue", "Batch_2": "tab:orange", "Batch_3": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, q, lbl in zip(axes, ("K01", "into_graphite", "pore_loss"), ("Si fraction", "constrained share (SOC 1)", "pore loss (SOC 1)")):
        for b, g in D.groupby("batch"):
            ax.scatter(g[f"{q}_v0r1"], g[f"{q}_ab"], c=col[b], s=22, label=b)
        lo, hi = D[[f"{q}_v0r1", f"{q}_ab"]].min().min(), D[[f"{q}_v0r1", f"{q}_ab"]].max().max()
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xlabel(f"{lbl}, pmdb.segment v0r1", fontsize=9)
        ax.set_ylabel(f"{lbl}, anchored segmenter (analytical-benchmarks)", fontsize=9)
        ax.set_title(f"Spearman {summ[q]['spearman']:+.2f}; KW p {summ[q]['kw_batch_v0r1']:.3f} -> {summ[q]['kw_batch_ab']:.3f}", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Two independently written segmenters, same 31 BSE fields", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "segmenter_agreement.png", dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
