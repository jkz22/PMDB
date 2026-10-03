"""Tables for RESULTS.md from leaderboard.csv, Stage C, audits and round-2 reruns.

    python -m src.v2.report  -> outputs/v2/results_tables.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.v2.common import OUT, harm_method
from src.v2.leaderboard import collect

M = ["kpi_r2", "img_r2", "image_id_ratio", "lift_shift", "recon_kpi_err", "psnr", "ssim", "knn_batch_acc"]
CFG = ["family", "view", "input", "train_set", "aug", "vae_mask", "mae_mask", "harmonise"]


def md(df: pd.DataFrame, nd=3, index=False) -> str:
    """GitHub markdown table without the optional ``tabulate`` dependency."""
    d = df.round(nd).reset_index() if index else df.round(nd)
    cells = [[("" if np.isscalar(v) and pd.isna(v) else str(v)) for v in row] for row in d.itertuples(index=False)]
    head = [str(c) for c in d.columns]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
                       *["| " + " | ".join(r) + " |" for r in cells]])


def cost_lines() -> list[str]:
    """One line per Modal job from status/<tag>.csv (run states, run-seconds) and the job log's last cost line."""
    import re
    out = []
    for p in sorted((OUT / "status").glob("*.csv")):
        tag = p.stem
        if tag.startswith("calibrate"):
            continue
        d = pd.read_csv(p)
        cnt = d.state.value_counts().to_dict()
        usd, mins = float("nan"), float("nan")
        log = OUT / f"{tag}.log"
        if log.exists():
            m = re.findall(r"\[\s*([\d.]+) min \$\s*([\d.]+)\]", log.read_text(errors="ignore"))
            if m:
                mins, usd = float(m[-1][0]), float(m[-1][1])
        out.append(f"- `{tag}`: {cnt}, run-time sum {d.sec.sum() / 3600:.2f} GPU-h (8 concurrent), "
                   f"wall {mins / 60:.2f} h, ${usd:.2f} (H100 + 24 CPU + 96 GiB at $5.85/h)")
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


def extra_sections(lb: pd.DataFrame) -> list[str]:
    S = []
    ak = OUT / "leaderboard_allkpi.csv"
    if ak.exists():
        a = pd.read_csv(ak)
        cols = ["selection_rank", *CFG, "kpi_set", "kpi_r2", "kpi_r2_all", "kpi_r2_ungated", "img_r2", "image_id_ratio", "knn_batch_acc"]
        S += ["## All-KPI track (Kevin's 10 crop KPIs incl. ungated K02/K03/K04-density; top 15)\n",
              "Ungated KPIs fail gates G2/G3 (size/count not scale-stable); shown for comparison only.\n",
              md(a[[c for c in cols if c in a]].head(15)), ""]
    if "baseline" in lb:
        b = lb[lb.train_set == "baseline"].copy()
        b["baseline"] = b.baseline.fillna("Batch_1")
        if len(b):
            S += ["## Baseline-only training: Batch_1 (6 fields, original assumption) vs Batch_3 (14 fields, corrected baseline)\n",
                  md(b.sort_values(["family", "baseline"])[["family", "kpi_set", "baseline", *[m for m in M if m in b]]]), ""]
    cl = OUT / "cls_leaderboard.csv"
    if cl.exists():
        c = pd.read_csv(cl)
        S += ["## Supervised 3-class batch classification (stratified grouped 5-fold by field; mean over folds)\n",
              md(c[["arch", "view", "input", "harmonise", "aug", "n_folds_done", "crop_acc", "field_acc", "field_acc_sd",
                    "field_f1", "field_acc_Batch_1", "field_acc_Batch_2", "field_acc_Batch_3", "raw_minus_harm_field_acc"]]), ""]
    r3 = lb[lb.stage == "R3"] if "stage" in lb else lb.iloc[:0]
    if len(r3):
        runs = collect(OUT / "runs").set_index("hash")
        parents = runs.loc[runs.index.intersection(r3.hash), "parent"].dropna().unique()
        fam = pd.concat([lb[lb.hash.isin(parents)], r3]).copy()
        fam["harmonise"] = fam.harmonise.map(harm_method)
        cols = ["family", "harmonise", "aug", "input", "view", "train_set", "kpi_r2", "img_r2", "img_r2__p1_c0", "image_id_ratio",
                "knn_batch_acc", "lift_shift"]
        S += ["## Round 3: best config per family retrained on PR #16 per-site LUT harmonisation "
              "(`none` = raw grey levels; `hybrid` = affine BSE/SE + histmatch Inlens; `affine2`; `histmatch`)\n",
              "`gmm` is the earlier 3-peak per-image linear map (Round 2). Lower `img_r2`/`knn_batch_acc` = less grey-level shortcut.\n",
              md(fam.sort_values(["family", "harmonise"])[[c for c in cols if c in fam]]), ""]
    if cl.exists():
        c = pd.read_csv(cl)
        base = c[(c.aug == "aug1") & (c.input == "raw")]
        if base.harmonise.nunique() > 1:
            pv = base.pivot_table(index=["arch", "view"], columns="harmonise", values="field_acc")
            S += ["## Classification: field accuracy by harmonisation method (raw input, aug1; chance = 0.33)\n",
                  "If accuracy survives `hybrid`, the batch signal is not the Batch_3 black-level/gain artefact.\n",
                  md(pv.round(3), index=True), ""]
            pb = base.pivot_table(index=["arch", "view"], columns="harmonise", values="field_acc_Batch_3")
            S += ["### Batch_3 field recall by harmonisation method\n", md(pb.round(3), index=True), ""]
    cp = OUT / "cls_probe.csv"
    if cp.exists():
        p = pd.read_csv(cp)
        cols = ["name", "clf", "n_crops", "crop_acc", "field_acc", "field_f1", "field_acc_Batch_1", "field_acc_Batch_2", "field_acc_Batch_3"]
        S += ["## Batch classification from features only (leave-one-field-out)\n", md(p[p.source == "features"][cols]), "",
              "## Batch classification from saved embeddings (leave-one-field-out; top 15 by field accuracy)\n",
              md(p[p.source != "features"].sort_values("field_acc", ascending=False)[cols].head(15)), ""]
    return S


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
    S += extra_sections(lb)
    (OUT / "results_tables.md").write_text("\n".join(S))
    print("\n".join(S)[:3000])


if __name__ == "__main__":
    main()
