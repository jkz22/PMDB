#!/usr/bin/env python3
"""Is the constrained-share batch effect the Batch 3 grey-level artefact? Re-run segmentation + SOC-1
swelling on hybrid-harmonised BSE and compare with raw (`outputs/segagree/harmonised_swelling.csv`)."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmdb import functional, segment as segment_mod  # noqa: E402
from pmdb.io import list_sites, load_site  # noqa: E402

OUT = ROOT / "outputs" / "segagree"


def one(a: tuple[str, str]) -> dict:
    b, s = a
    raw = load_site(b, s, resolution="half", normalise="none")
    m = segment_mod.segment_bse(np.asarray(raw.image[..., 0], float), raw.nm_per_px, params=None)
    h = load_site(b, s, resolution="half", normalise="fixed", harmonise="hybrid")
    mh = segment_mod.segment_bse(np.asarray(h.image[..., 0], float) * 255.0, raw.nm_per_px, params=None)
    r = {"batch": b, "site": s, "graphite_iou": float((m.graphite & mh.graphite).sum() / (m.graphite | mh.graphite).sum())}
    for tag, mm in (("raw", m), ("harm", mh)):
        sw = functional.swelling_test(mm, 1.0)
        r.update({f"{k}_{tag}": v for k, v in sw.items()})
    return r


def main() -> int:
    with ProcessPoolExecutor(6) as ex:
        D = pd.DataFrame(list(ex.map(one, [(r.batch, r.site) for r in list_sites().itertuples()])))
    D.to_csv(OUT / "harmonised_swelling.csv", index=False)
    C = pd.read_csv(ROOT / "outputs" / "clean" / "summary.csv")[["batch", "site", "BSE_D", "BSE_G"]].drop_duplicates(["batch", "site"])
    D = D.merge(C, on=["batch", "site"])
    b3 = D[D.batch == "Batch_3"]
    summ = {"graphite_iou_min": float(D.graphite_iou.min()),
            "into_graphite_rho_raw_harm": float(spearmanr(D.into_graphite_raw, D.into_graphite_harm)[0]),
            "kw_into_graphite_raw": float(kruskal(*[g for _, g in D.into_graphite_raw.groupby(D.batch)]).pvalue),
            "kw_into_graphite_harm": float(kruskal(*[g for _, g in D.into_graphite_harm.groupby(D.batch)]).pvalue),
            "rho_into_graphite_vs_BSE_D_all": float(spearmanr(D.into_graphite_raw, D.BSE_D)[0]),
            "rho_into_graphite_vs_BSE_D_within_B3": float(spearmanr(b3.into_graphite_raw, b3.BSE_D)[0])}
    (OUT / "harmonised_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
