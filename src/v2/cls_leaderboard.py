"""Aggregate the supervised batch-classification runs over folds -> outputs/v2/cls_leaderboard.csv.

One row per (arch, view, input, harmonise, aug): mean +- sd over the grouped folds of crop- and
field-level accuracy / macro-F1, pooled field confusion matrix, and the raw-minus-harmonised
accuracy gap for the same arch/view (how much of the separability is brightness/gain).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.common import OUT

KEYS = ["arch", "view", "input", "harmonise", "aug"]


def collect(runs: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(runs.glob("*/metrics.json"))]
    rows = [r for r in rows if r.get("split", "none") == "strat"]  # drop the first (non-stratified) launch
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for k, g in df.groupby(KEYS, dropna=False):
        cm = np.sum([np.array(c) for c in g.field_confusion], axis=0)
        out.append({**dict(zip(KEYS, k)), "n_folds_done": len(g), "folds": sorted(g.fold.tolist()),
                    "crop_acc": g.crop_acc.mean(), "crop_acc_sd": g.crop_acc.std(ddof=0),
                    "crop_f1": g.crop_f1_macro.mean(), "crop_logloss": g.crop_logloss.mean(),
                    "field_acc": g.field_acc.mean(), "field_acc_sd": g.field_acc.std(ddof=0),
                    "field_f1": g.field_f1_macro.mean(),
                    "field_acc_Batch_1": cm[0, 0] / max(1, cm[0].sum()), "field_acc_Batch_2": cm[1, 1] / max(1, cm[1].sum()),
                    "field_acc_Batch_3": cm[2, 2] / max(1, cm[2].sum()), "field_confusion": cm.tolist(),
                    "n_test_fields": int(g.n_test_fields.sum()), "train_sec_mean": g.train_sec.mean()})
    lb = pd.DataFrame(out)
    raw = lb[(~lb.harmonise.astype(bool)) & (lb.aug == "aug1") & (lb.input == "raw")].set_index(["arch", "view"])
    harm = lb[lb.harmonise.astype(bool) & (lb.aug == "aug1") & (lb.input == "raw")].set_index(["arch", "view"])
    gap = (raw.field_acc - harm.field_acc).rename("raw_minus_harm_field_acc")
    lb = lb.merge(gap.reset_index(), on=["arch", "view"], how="left")
    return lb.sort_values(["field_acc", "crop_acc"], ascending=False).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(OUT / "cls_runs"))
    ap.add_argument("--out", default=str(OUT / "cls_leaderboard.csv"))
    a = ap.parse_args(argv)
    df = collect(Path(a.runs))
    if df.empty:
        print("no classification runs"); return
    lb = aggregate(df)
    lb.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(lb[KEYS + ["n_folds_done", "crop_acc", "field_acc", "field_acc_sd", "field_f1", "raw_minus_harm_field_acc"]].round(3).to_string())
    return lb


if __name__ == "__main__":
    main()
