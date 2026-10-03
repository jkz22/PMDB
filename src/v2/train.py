"""Train one config (Phase 3/4). Config-hash keyed, checkpointed, resumable; bf16 on CUDA.

CUDA is required: there is no CPU fallback. ``PMDB_SMOKE=1`` allows a tiny CPU run for tests only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.v2.augment import ForwardRanges, augment
from src.v2.common import OUT, SEED
from src.v2.common import BASELINE
from src.v2.data import CropDataset, FieldStore, GPUCropLoader, grouped_folds, heldout_split
from src.v2.kpi_adapter import crop_kpi_frame, kpi_cols
from src.v2.models import build

DEFAULTS = dict(family="vae_a", view="stack", input="raw", train_set="all", aug="aug1", vae_mask=0.0,
                mae_mask=0.75, harmonise=False, seed=SEED, steps=3000, batch_size=64, lr=None,
                fold=None, n_folds=5, lobo=None, kpi_commit="d23a116")
LR = {"vae": 1e-3, "mae": 1.5e-4, "dino": 5e-5}
RUNS = Path(os.environ.get("PMDB_RUNS", OUT / "runs"))
LOCK_STALE_S = 900


class RunLocked(RuntimeError):
    """Another scheduler is training this config hash right now."""


def full_cfg(cfg: dict) -> dict:
    c = {**DEFAULTS, **cfg}
    if c["lr"] is None:
        c["lr"] = LR.get(c["family"].split("_")[0], 0.0)
    if c["train_set"] == "baseline":  # which batch is the baseline is part of the config identity
        c.setdefault("baseline", BASELINE)
    return c


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(full_cfg(cfg), sort_keys=True).encode()).hexdigest()[:12]


def train_fields(c: dict) -> pd.DataFrame:
    m = heldout_split()
    if c["fold"] is not None:  # Stage C cross-fitting over all fields, grouped by field
        m["fold"] = grouped_folds(m.group_id.to_numpy(), c["n_folds"], c["seed"])
        f = m[m.fold != c["fold"]]
    else:
        f = m[~m.heldout]
    if c["lobo"] is not None:
        f = f[f.batch != c["lobo"]]
    if c["train_set"] == "baseline":
        f = f[f.batch == c.get("baseline", BASELINE)]
    return f


def kpi_table(fields: pd.DataFrame, cols=kpi_cols(None)):
    cols = list(cols)
    k = crop_kpi_frame("train", cols)
    tr = k[k.group_id.isin(fields.group_id)]
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
    k[cols] = (k[cols] - mu) / sd
    return k, {"mean": mu.to_dict(), "std": sd.to_dict()}


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("PMDB_SMOKE") == "1":
        return torch.device("cpu")
    raise RuntimeError("CUDA not available: refusing to train on CPU")


def run(cfg: dict, status_cb=None) -> Path:
    c = full_cfg(cfg)
    if c["family"] == "dino_ft" and c["aug"] == "aug0":
        raise ValueError("DINOv2-FT needs two augmented views (aug0 excluded by design)")
    h = cfg_hash(c)
    d = RUNS / h
    d.mkdir(parents=True, exist_ok=True)
    if (d / "done.json").exists():
        return d
    lock = d / "lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < LOCK_STALE_S:
        raise RunLocked(f"{h} locked by pid {lock.read_text()}")
    lock.write_text(str(os.getpid()))
    (d / "config.json").write_text(json.dumps({**c, "hash": h}, indent=2))
    dev = device()
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])

    fields = train_fields(c)
    store = FieldStore(fields, c["input"], harmonise=c["harmonise"])
    kcols = kpi_cols(c.get("kpi_set"))
    kpis, kpi_norm = kpi_table(fields, kcols)
    cond = c["family"] in ("vae_b", "vae_c")
    ds = CropDataset(store, c["view"], random_offset=c["aug"] != "aug0", kpis=kpis if cond else None,
                     kpi_cols=kcols if cond else ())
    g = torch.Generator().manual_seed(c["seed"])
    dl = GPUCropLoader(ds, c["batch_size"], dev, g)

    model = build(c["family"], n_kpi=len(kcols), vae_mask=c["vae_mask"], mae_mask=c["mae_mask"]).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=0.05)
    warm = max(1, c["steps"] // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / c["steps"]))))
    fr_path = OUT / "imaging_stats" / "forward_ranges.json"
    fr = ForwardRanges.from_json(fr_path) if fr_path.exists() else None
    if c["aug"] == "aug2" and fr is None:
        raise FileNotFoundError("aug2 needs Phase 0 forward_ranges.json")

    step, ck = 0, d / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        step = s["step"]; g.set_state(s["gen"].cpu())
    gen = torch.Generator(device=dev).manual_seed(c["seed"] + step)
    log = open(d / "train_log.jsonl", "a")
    t0 = time.time()
    model.train()
    while step < c["steps"]:
        for b in dl:
            if step >= c["steps"]:
                break
            x = b["x"]
            b["x"] = augment(x, c["aug"], fr, gen)
            if c["family"] == "dino_ft":
                b["x2"] = augment(x, c["aug"], fr, gen)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss, logs = model.loss(b)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if hasattr(model, "after_step"):
                model.after_step()
            step += 1
            if step % 50 == 0 or step == c["steps"]:
                rec = {"step": step, "loss": loss.item(), **logs, "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
                log.write(json.dumps(rec) + "\n"); log.flush(); lock.touch()
                if status_cb:
                    status_cb(h, rec)
            if step % 500 == 0 or step == c["steps"]:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "gen": g.get_state()}, ck.with_suffix(".tmp"))
                os.replace(ck.with_suffix(".tmp"), ck)
    torch.save({"model": model.state_dict(), "cfg": c, "kpi_norm": kpi_norm,
                "train_groups": fields.group_id.tolist()}, d / "final.pt")
    (d / "done.json").write_text(json.dumps({"hash": h, "steps": step, "sec": time.time() - t0}))
    lock.unlink(missing_ok=True)
    return d
