"""Fixed Phase 3 run design. Stage A defaults, Stage B one-factor-at-a-time; Stage C is
generated from the Stage A/B leaderboard by ``stage_c``."""
from __future__ import annotations

from src.v2.models import TRAINABLE

STEPS = {"vae_a": 4000, "vae_b": 4000, "vae_c": 4000, "mae_adapted": 6000, "mae_scratch": 8000, "dino_ft": 4000}
ALT_MASK = {"vae": ("vae_mask", 0.5), "mae": ("mae_mask", 0.5)}
CUT_ORDER = ("mae_scratch", "stageB_detector_vae_bc", "extra_seeds")  # cut in this exact order if short


def stage_a():
    return [dict(family=f, steps=STEPS[f], stage="A") for f in TRAINABLE]


def stage_b():
    runs = []
    for f in TRAINABLE:
        base = dict(family=f, steps=STEPS[f], stage="B")
        for v in ("BSE", "Inlens", "SE_type"):
            runs.append({**base, "view": v, "factor": "view", "tag": "stageB_detector_vae_bc" if f in ("vae_b", "vae_c") else ""})
        runs.append({**base, "train_set": "baseline", "factor": "train_set"})
        for a in ("aug0", "aug2"):
            if f == "dino_ft" and a == "aug0":
                continue
            runs.append({**base, "aug": a, "factor": "aug"})
        fam = f.split("_")[0]
        if fam in ALT_MASK:
            k, val = ALT_MASK[fam]
            runs.append({**base, k: val, "factor": "mask"})
        runs.append({**base, "input": "norm", "factor": "input"})
    return runs


def stage_c(top3: list[dict], best: dict, batches=("Batch_1", "Batch_2", "Batch_3")):
    runs = [{**c, "fold": k, "stage": "C", "factor": "crossfit"} for c in top3 for k in range(5)]
    runs += [{**best, "seed": s, "stage": "C", "factor": "seed", "tag": "extra_seeds"} for s in (1, 2)]
    runs += [{**best, "lobo": b, "stage": "C", "factor": "lobo"} for b in batches]
    return runs


TRAIN_KEYS = ("family", "view", "input", "train_set", "aug", "vae_mask", "mae_mask", "harmonise", "seed",
              "steps", "batch_size", "lr", "fold", "n_folds", "lobo", "kpi_commit", "kpi_set", "baseline", "phase_mask")


def train_cfg(run: dict) -> dict:
    return {k: v for k, v in run.items() if k in TRAIN_KEYS}
