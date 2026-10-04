"""Explanation of a held-out call versus Batch_3: stage-1 importance x z-score (classifier plan C14)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def site_means(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    g = table.groupby(["batch", "site"], sort=True)[features]
    return g.agg(lambda s: np.nan if s.isna().all() else float(np.nanmean(s.to_numpy(float)))).reset_index()


def zscores_vs_b3(site_vals: pd.DataFrame, features: list[str], importance: dict[str, float]) -> pd.DataFrame:
    ref = site_vals[site_vals["batch"] == "Batch_3"]
    tgt = site_vals[site_vals["batch"] == "Batch_heldout"]
    rows = []
    for _, r in tgt.iterrows():
        for f in features:
            mu = float(np.nanmean(ref[f].to_numpy(float)))
            sd = float(np.nanstd(ref[f].to_numpy(float), ddof=1))
            x = float(r[f])
            z = (x - mu) / sd if np.isfinite(x) and np.isfinite(sd) and sd > 0 else np.nan
            imp = float(importance[f])
            rows.append({"site": r["site"], "feature": f, "site_value": x, "b3_mean": mu, "b3_std": sd,
                         "z": z, "importance_stage1": imp, "score": imp * abs(z) if np.isfinite(z) else np.nan})
    out = pd.DataFrame(rows)
    out["rank"] = np.nan
    for _, g in out.groupby("site"):
        g = g[g["score"].notna()].sort_values(["score", "feature"], ascending=[False, True])
        out.loc[g.index, "rank"] = np.arange(1, len(g) + 1)
    return out


def explanation_text(site: str, pred: dict, z: pd.DataFrame, meanings: dict[str, str], top_k: int = 3) -> str:
    top = z[(z["site"] == site) & z["rank"].notna()].sort_values("rank").head(top_k)
    parts = []
    for r in top.itertuples():
        d = "higher" if r.z > 0 else "lower"
        parts.append(f"{meanings.get(r.feature, r.feature)} is {d} ({r.feature} {r.site_value:.3g} vs Batch_3 "
                     f"{r.b3_mean:.3g} ± {r.b3_std:.3g}, z = {r.z:+.1f})")
    text = (f"{site}: assigned {pred['predicted']} (confidence {pred['confidence']:.2f}; "
            f"P(Batch_3) = {pred['p_b3']:.2f}, P(Batch_1 | not Batch_3) = {pred['q_b1']:.2f}; "
            f"{pred['stage1_votes_not_b3']}/{pred['n_tiles']} tiles vote not-Batch_3). Versus Batch_3: "
            + "; ".join(parts) + ".")
    if len(top) == 0 or (top["z"].abs() < 1).all():
        text += " No top feature deviates from Batch_3 by more than 1 SD."
    return text
