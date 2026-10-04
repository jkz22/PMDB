#!/usr/bin/env python3
"""Does the v0r1 'binder / carbon' residue class have an intensity identity in any channel?

For every labelled site: segment BSE with pmdb.segment v0r1, take the residue (admissible, not Si / graphite /
pore), and compare its BSE / Inlens / SE intensities with graphite (median per class, pixel-level ROC AUC on
20k samples). Writes outputs/segagree/binder_probe.csv and prints per-batch medians.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmdb import segment as segment_mod  # noqa: E402
from pmdb.io import list_sites, load_site  # noqa: E402

OUT = ROOT / "outputs" / "segagree"
N_SAMPLE = 20000


def one(a: tuple[str, str]) -> dict:
    b, s = a
    raw = load_site(b, s, resolution="half", normalise="none")
    img = np.asarray(raw.image, float)
    m = segment_mod.segment_bse(img[..., 0], raw.nm_per_px, params=None)
    dom = m.fraction_space
    residue = dom & ~m.si & ~m.graphite & ~m.pore
    graphite = m.graphite & dom
    r = dict(batch=b, site=s, residue_frac=float(residue.sum() / dom.sum()))
    rng = np.random.default_rng(0)
    ig = rng.choice(np.flatnonzero(graphite), N_SAMPLE)
    ir = rng.choice(np.flatnonzero(residue), min(N_SAMPLE, int(residue.sum())))
    y = np.r_[np.zeros(len(ig)), np.ones(len(ir))]
    for ch, name in ((0, "bse"), (1, "inlens"), (2, "se")):
        x = img[..., ch]
        r[f"{name}_graphite_median"] = float(np.median(x[graphite]))
        r[f"{name}_residue_median"] = float(np.median(x[residue]))
        r[f"{name}_auc_residue_vs_graphite"] = float(roc_auc_score(y, np.r_[x.ravel()[ig], x.ravel()[ir]]))
    return r


def main() -> int:
    with ProcessPoolExecutor(6) as ex:
        D = pd.DataFrame(list(ex.map(one, [(r.batch, r.site) for r in list_sites().itertuples()])))
    D.to_csv(OUT / "binder_probe.csv", index=False)
    cols = ["residue_frac"] + [c for c in D.columns if c.endswith("auc_residue_vs_graphite")]
    print(D.groupby("batch")[cols].median().round(3).to_string())
    print(D[cols].describe().loc[["min", "50%", "max"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
