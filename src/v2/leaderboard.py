"""Collect <runs>/*/metrics.json into outputs/v2/leaderboard.csv with the rank-average selection score.

    python -m src.v2.leaderboard [--runs DIR] [--stage A,B,OTS]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.v2.common import OUT
from src.v2.evaluate import selection_score

KEEP = ("hash", "stage", "factor", "family", "view", "input", "train_set", "aug", "vae_mask", "mae_mask", "harmonise",
        "seed", "fold", "lobo", "steps", "kpi_r2", "img_r2", "image_id_ratio", "lift_shift", "recon_kpi_err",
        "knn_batch_acc", "n_crops", "emb_dim", "kpi_commit_full")


def collect(runs: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(runs.glob("*/metrics.json")):
        m = json.load(open(p))
        cfg = json.load(open(p.parent / "config.json"))
        rows.append({**cfg, **m})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(OUT / "runs"))
    ap.add_argument("--stage", default="OTS,A,B,R2")
    a = ap.parse_args(argv)
    df = collect(Path(a.runs))
    df = df[df.stage.isin(a.stage.split(","))].copy()
    df["selection_score"] = selection_score(df)
    df["selection_rank"] = df.selection_score.rank(method="min").astype(int)
    cols = [c for c in KEEP if c in df] + ["selection_score", "selection_rank"]
    rest = sorted(c for c in df if "__" in c)
    df = df.sort_values("selection_rank")[cols + rest]
    df.to_csv(OUT / "leaderboard.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[cols].head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
