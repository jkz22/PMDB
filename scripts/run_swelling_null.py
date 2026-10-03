#!/usr/bin/env python3
"""Arrangement null for the swelling budget: relocate every Si object at random inside the admissible
space (same objects, same graphite skeleton) and compare the observed constrained share, pore loss
and K15 contact with the null distribution.

    python scripts/run_swelling_null.py --jobs 7 --n-sims 20   -> outputs/functional/null_relocation.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import pmdb.segment as segment_mod  # noqa: E402
from pmdb.functional import swelling_null  # noqa: E402
from pmdb.io import list_sites, load_site  # noqa: E402
from pmdb.kpis.common import stable_seed  # noqa: E402

OUT = REPO_ROOT / "outputs" / "functional"


def process_site(args: tuple) -> dict:
    batch, site, soc, n_sims = args
    t0 = time.time()
    raw = load_site(batch, site, resolution="half", normalise="none")
    masks = segment_mod.segment_bse(np.asarray(raw.image[..., 0], dtype=np.float64), raw.nm_per_px)
    seed = stable_seed(f"{batch}/{site}/relocate") % (2 ** 32)
    out = swelling_null(masks, soc, n_sims, seed)
    return {"batch": batch, "site": site, "soc": soc, **out, "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--n-sims", type=int, default=20)
    ap.add_argument("--soc", type=float, default=1.0)
    a = ap.parse_args()
    manifest = list_sites()
    jobs = [(r.batch, r.site, a.soc, a.n_sims) for r in manifest.itertuples()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(process_site, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "null_relocation.csv", index=False)

    summary = {}
    for k in ("into_graphite", "pore_loss", "contact"):
        df[f"excess_{k}"] = df[f"obs_{k}"] - df[f"null_mean_{k}"]
        g = [df.loc[df.batch == b, f"excess_{k}"] for b in ("Batch_1", "Batch_2", "Batch_3")]
        gz = [df.loc[df.batch == b, f"z_{k}"] for b in ("Batch_1", "Batch_2", "Batch_3")]
        summary[k] = {
            "median_excess_by_batch": [float(x.median()) for x in g],
            "median_z_by_batch": [float(x.median()) for x in gz],
            "sites_z_above_2": int((df[f"z_{k}"] > 2).sum()),
            "sites_z_below_-2": int((df[f"z_{k}"] < -2).sum()),
            "kruskal_p_excess": float(stats.kruskal(*g).pvalue),
            "kruskal_p_null_mean": float(stats.kruskal(*[df.loc[df.batch == b, f"null_mean_{k}"]
                                                        for b in ("Batch_1", "Batch_2", "Batch_3")]).pvalue),
            "mannwhitney_p_B12_vs_B3_excess": float(stats.mannwhitneyu(pd.concat(g[:2]), g[2]).pvalue),
        }
    summary["null_kept_frac_max"] = float(df["null_kept_frac"].max())
    summary["n_sims"] = a.n_sims
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    (OUT / "null_relocation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    cols = ["batch", "site", "obs_into_graphite", "null_mean_into_graphite", "z_into_graphite",
            "obs_contact", "null_mean_contact", "z_contact", "obs_pore_loss", "null_mean_pore_loss", "z_pore_loss"]
    print(df[cols].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
