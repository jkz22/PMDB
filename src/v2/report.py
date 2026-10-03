"""Tables for RESULTS.md from leaderboard.csv, Stage C, audits and round-2 reruns.

    python -m src.v2.report  -> outputs/v2/results_tables.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.v2.common import OUT
from src.v2.leaderboard import collect

M = ["kpi_r2", "img_r2", "image_id_ratio", "lift_shift", "recon_kpi_err", "psnr", "ssim", "knn_batch_acc"]
CFG = ["family", "view", "input", "train_set", "aug", "vae_mask", "mae_mask", "harmonise"]


def md(df: pd.DataFrame, nd=3, index=False) -> str:
    """GitHub markdown table without the optional ``tabulate`` dependency."""
    d = df.round(nd).reset_index() if index else df.round(nd)
    cells = [[("" if pd.isna(v) else str(v)) for v in row] for row in d.itertuples(index=False)]
    head = [str(c) for c in d.columns]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
                       *["| " + " | ".join(r) + " |" for r in cells]])


def cost_lines() -> list[str]:
    out = []
    for p in sorted((OUT / "status").glob("*_summary.json")):
        s = json.loads(p.read_text())
        out.append(f"- `{p.stem.replace('_summary', '')}`: {s.get('n_done', '?')} done, {s.get('n_failed', '?')} failed, "
                   f"{s.get('wall_h', float('nan')):.2f} h wall, ${s.get('usd', float('nan')):.2f}, "
                   f"GPU util mean {s.get('util_mean', float('nan')):.0f}% (busy {s.get('util_busy_mean', float('nan')):.0f}%)")
    return out


def factor_effects(lb: pd.DataFrame) -> pd.DataFrame:
    """Stage B one-factor change minus the Stage A default of the same family."""
    a = lb[lb.stage == "A"].set_index("family")
    rows = []
    for _, r in lb[lb.stage == "B"].iterrows():
        if r.family not in a.index:
            continue
        base = a.loc[r.family]
        lvl = {"view": r["view"], "train_set": r["train_set"], "aug": r["aug"], "input": r["input"],
               "mask": r["vae_mask"] if r.family.startswith("vae") else r["mae_mask"]}.get(r.factor, "")
        rows.append({"family": r.family, "factor": r.factor, "level": str(lvl),
                     **{f"d_{m}": r[m] - base[m] for m in ["kpi_r2", "img_r2", "image_id_ratio", "knn_batch_acc", "lift_shift"]},
                     "rank": r.selection_rank, "rank_default": base.selection_rank})
    return pd.DataFrame(rows).sort_values(["factor", "level", "family"])


def main():
    lb = pd.read_csv(OUT / "leaderboard.csv")
    S = ["# Auto-generated tables (`python -m src.v2.report`)\n", "## Compute\n", *cost_lines(), ""]
    show = ["selection_rank", "stage", "factor", *CFG, *[m for m in M if m in lb]]
    S += ["## Leaderboard (Stage OTS/A/B/R2, top 20)\n", md(lb[show].head(20)), ""]
    best = lb.sort_values("selection_rank").groupby("family").head(1)
    S += ["## Best configuration per family\n", md(best[show]), ""]
    fe = factor_effects(lb)
    if len(fe):
        S += ["## Stage B: change vs the family's Stage A default (B minus A)\n", md(fe), ""]
        for f in ["view", "train_set", "aug", "input", "mask"]:
            g = fe[fe.factor == f].groupby("level")[[c for c in fe if c.startswith("d_")]].median()
            if len(g):
                S += [f"### Median effect over families: `{f}`\n", md(g, index=True), ""]
    runs = collect(OUT / "runs")
    c = runs[runs.stage == "C"] if "stage" in runs else runs.iloc[:0]
    if len(c):
        key = c[CFG].astype(str).agg("|".join, axis=1)
        cv = c[c.fold.notna() & c.lobo.isna()] if "lobo" in c else c[c.fold.notna()]
        if len(cv):
            g = cv.groupby(key[cv.index])[M].agg(["mean", "std"])
            g.columns = [f"{a}_{b}" for a, b in g.columns]
            S += ["## Stage C: grouped 5-fold cross-fitting (mean, sd over folds)\n", md(g, index=True), ""]
        sd = c[c.factor == "seed"] if "factor" in c else c.iloc[:0]
        if len(sd):
            S += ["## Stage C: extra seeds of the best configuration\n", md(sd[["seed", *M]]), ""]
        lo = c[c.lobo.notna()] if "lobo" in c else c.iloc[:0]
        if len(lo):
            S += ["## Stage C: leave-one-batch-out retraining of the best configuration\n", md(lo[["lobo", *M]]), ""]
    aud = sorted((OUT / "audit").glob("*/audit.json"))
    if aud:
        rows = [json.loads(p.read_text()) for p in aud]
        flat = pd.json_normalize(rows)
        keep = [k for k in flat if not k.startswith("field_inference.per_field")]
        S += ["## Phase 6 latent audit (best configuration per family, all 31 fields)\n", md(flat[keep]), ""]
    fl = OUT / "round2_flags.csv"
    if fl.exists():
        S += ["## Round 2 trigger (imaging-probe R^2 > 0.5 or image-ID ratio > 2x chance)\n", md(pd.read_csv(fl)), ""]
        r2 = lb[lb.stage == "R2"]
        if len(r2):
            par = lb.set_index("hash")
            r2 = r2.assign(parent_rank=[par.selection_rank.get(p, np.nan) for p in runs.set_index("hash").loc[r2.hash, "parent"]])
            S += ["### Round 2 reruns\n", md(r2[["family", "factor", "parent_rank", "selection_rank", *M]]), ""]
    (OUT / "results_tables.md").write_text("\n".join(S))
    print("\n".join(S)[:3000])


if __name__ == "__main__":
    main()
