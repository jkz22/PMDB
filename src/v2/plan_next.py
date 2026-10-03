"""Turn the leaderboard into the next run specs.

    python -m src.v2.plan_next stage_c     -> outputs/v2/stage_c_specs.json (top-3 cross-fit, best +2 seeds, best LOBO)
    python -m src.v2.plan_next round2      -> outputs/v2/round2_specs.json (confounder reruns for flagged families)
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src.v2.common import OUT
from src.v2.rungrid import TRAIN_KEYS, stage_c

TRAINED = ("A", "B")


def _spec(row) -> dict:
    d = {k: row[k] for k in TRAIN_KEYS if k in row and pd.notna(row[k])}
    for k in ("seed", "steps", "batch_size", "n_folds"):
        if k in d:
            d[k] = int(d[k])
    for k in ("fold", "lobo"):
        d.pop(k, None)
    h = d.get("harmonise", False)
    d["harmonise"] = False if h in (False, "False", "none") else True if h in (True, "True") else h
    return d


def plan_stage_c():
    lb = pd.read_csv(OUT / "leaderboard.csv")
    lb = lb[lb.stage.isin(TRAINED)].sort_values("selection_rank")
    top3 = [_spec(r) for _, r in lb.head(3).iterrows()]
    specs = stage_c(top3, top3[0])
    (OUT / "stage_c_specs.json").write_text(json.dumps(specs, indent=1))
    print(json.dumps(top3, indent=1), len(specs))


def plan_round2():
    """Best config per trainable family; rerun if imaging probe R^2 > 0.5 or image-ID > 2x chance."""
    lb = pd.read_csv(OUT / "leaderboard.csv")
    lb = lb[lb.stage.isin(TRAINED)].sort_values("selection_rank")
    best = lb.groupby("family").head(1)
    img_cols = [c for c in lb if c.startswith("img_r2__")]
    specs, flags = [], []
    for _, r in best.iterrows():
        img_max = r[img_cols].max()
        flag = bool(img_max > 0.5 or r.image_id_ratio > 2.0)
        flags.append(dict(family=r.family, hash=r.hash, img_r2_max=float(img_max), image_id_ratio=float(r.image_id_ratio), rerun=flag))
        if not flag:
            continue
        base = _spec(r)
        variants = [dict(harmonise=True, factor="r2_harmonise"), dict(aug="aug2", factor="r2_aug2"),
                    dict(harmonise=True, aug="aug2", factor="r2_harm_aug2"), dict(input="norm", factor="r2_norm")]
        for v in variants:
            s = {**base, **v, "stage": "R2", "parent": r.hash}
            if s["family"] == "dino_ft" and s.get("aug") == "aug0":
                s["aug"] = "aug1"
            specs.append(s)
    pd.DataFrame(flags).to_csv(OUT / "round2_flags.csv", index=False)
    (OUT / "round2_specs.json").write_text(json.dumps(specs, indent=1))
    print(pd.DataFrame(flags).to_string(index=False), "\n", len(specs), "round-2 runs")


def plan_round3(methods=("hybrid", "affine2", "histmatch")):
    """Round 3: best Stage A/B config per trainable family retrained on the PR #16 per-site LUT
    harmonised caches (cache/harmonised/<method>/half), plus the top config with hybrid + aug2."""
    lb = pd.read_csv(OUT / "leaderboard.csv")
    lb = lb[lb.stage.isin(TRAINED)].sort_values("selection_rank")
    best = lb.groupby("family").head(1)
    specs = []
    for _, r in best.iterrows():
        base = _spec(r)
        for m in methods:
            specs.append({**base, "harmonise": m, "factor": f"r3_{m}", "stage": "R3", "parent": r.hash})
    top = _spec(best.iloc[0])
    specs.append({**top, "harmonise": "hybrid", "aug": "aug2", "factor": "r3_hybrid_aug2", "stage": "R3", "parent": best.iloc[0].hash})
    (OUT / "round3_specs.json").write_text(json.dumps(specs, indent=1))
    print(len(specs), "round-3 runs:", sorted({(s["family"], s["factor"]) for s in specs}))


if __name__ == "__main__":
    {"stage_c": plan_stage_c, "round2": plan_round2, "round3": plan_round3}[sys.argv[1]]()
