"""Aggregate the per-run occlusion / Grad-CAM summaries (attribution_summary.json) over folds, per
architecture x input route, so the clean routes can be compared with raw and the PR #16 LUTs.
Writes outputs/v2/attribution_by_route.csv."""
from __future__ import annotations

import json

import pandas as pd

from src.v2.classify import CLS_RUNS
from src.v2.common import OUT

KEYS = ("occ_ratio_si", "occ_ratio_graphite", "occ_ratio_pore", "cam_ratio_si", "cam_ratio_graphite", "cam_ratio_pore",
        "occ_corr_brightness", "occ_corr_contrast", "occ_corr_edges", "occ_corr_noise",
        "cam_corr_brightness", "cam_corr_contrast", "cam_corr_edges", "cam_corr_noise")


def main() -> pd.DataFrame:
    rows = []
    for f in CLS_RUNS.glob("*/attribution_summary.json"):
        c = json.loads((f.parent / "config.json").read_text())
        s = json.loads(f.read_text())
        for grp in ("all", "Batch_1", "Batch_2", "Batch_3"):
            if grp in s and isinstance(s[grp], dict) and "occ_ratio_si" in s[grp]:
                rows.append(dict(arch=c["arch"], view=c["view"], harmonise=str(c["harmonise"]), aug=c["aug"], dequant=c.get("dequant") or 0,
                                 fold=c["fold"], group=grp, n=s[grp].get("n", s.get("n_crops")), **{k: s[grp].get(k) for k in KEYS}))
    df = pd.DataFrame(rows)
    df["harmonise"] = df.harmonise.replace({"False": "raw", "True": "gmm"})
    agg = df.groupby(["arch", "view", "harmonise", "aug", "dequant", "group"]).agg(
        n_folds=("fold", "nunique"), n_crops=("n", "sum"), **{k: (k, "mean") for k in KEYS}).reset_index()
    agg.to_csv(OUT / "attribution_by_route.csv", index=False)
    pd.set_option("display.width", 250)
    show = agg[(agg.group == "all") & (agg.dequant == 0)]
    print(show[["arch", "view", "harmonise", "aug", "n_folds", "n_crops", "occ_ratio_si", "occ_ratio_graphite", "occ_ratio_pore",
                "occ_corr_brightness", "occ_corr_edges", "occ_corr_noise", "cam_corr_edges", "cam_corr_noise"]].round(3).to_string())
    return agg


if __name__ == "__main__":
    main()
