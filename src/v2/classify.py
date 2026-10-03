"""Supervised 3-class batch classification (Batch_1 / Batch_2 / Batch_3), split by field code.

Grouped k-fold over all 31 fields (every crop of a field in one fold). Train on the fold's
training fields (overlapping crops, aug1/aug2), test on the held-out fields' non-overlapping
crops: crop-level and field-level (mean logit) accuracy, macro-F1, confusion matrix, log-loss.

Spec keys: arch (resnet18_scratch | resnet18_imnet | effb4_imnet | dinov2_ft | dinov2_linear),
view, input (raw|norm), harmonise, aug, fold, n_folds, steps, batch_size, lr, seed.
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
import torch.nn as nn
import torch.nn.functional as F

from src.v2.augment import ForwardRanges, augment
from src.v2.common import CROP, OUT, SEED, manifest
from src.v2.data import CropDataset, FieldStore, GPUCropLoader, grouped_folds, stratified_group_folds
from src.v2.models import dino_crop, imnet
from src.v2.train import LOCK_STALE_S, RUNS, RunLocked, device

BATCHES = ("Batch_1", "Batch_2", "Batch_3")
CLS_RUNS = Path(os.environ.get("PMDB_CLS_RUNS", RUNS.parent / "cls_runs"))  # beside PMDB_RUNS (/vol/runs on Modal)
DEFAULTS = dict(task="cls", arch="resnet18_imnet", view="stack", input="raw", harmonise=False, aug="aug1",
                fold=0, n_folds=5, split="strat", steps=1500, batch_size=64, lr=None, seed=SEED, label_smoothing=0.1)
LR = {"resnet18_scratch": 1e-3, "resnet18_imnet": 3e-4, "effb4_imnet": 3e-4, "dinov2_ft": 5e-5, "dinov2_linear": 1e-3}


def full_cfg(cfg: dict) -> dict:
    c = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
    if c["lr"] is None:
        c["lr"] = LR[c["arch"]]
    return c


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(full_cfg(cfg), sort_keys=True).encode()).hexdigest()[:12]


class Classifier(nn.Module):
    def __init__(self, arch: str, n_cls: int = 3):
        super().__init__()
        self.arch = arch
        if arch.startswith("resnet18"):
            import torchvision
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if arch.endswith("imnet") else None
            self.m = torchvision.models.resnet18(weights=w)
            self.m.fc = nn.Linear(self.m.fc.in_features, n_cls)
        elif arch == "effb4_imnet":
            import timm
            self.m = timm.create_model("efficientnet_b4", pretrained=True, num_classes=n_cls)
        elif arch in ("dinov2_ft", "dinov2_linear"):
            from transformers import Dinov2Model
            self.bb = Dinov2Model.from_pretrained("facebook/dinov2-small")
            self.bb.requires_grad_(False)
            if arch == "dinov2_ft":
                for blk in self.bb.encoder.layer[-2:]:
                    blk.requires_grad_(True)
                self.bb.layernorm.requires_grad_(True)
            self.head = nn.Linear(2 * self.bb.config.hidden_size, n_cls)
        else:
            raise ValueError(arch)

    def forward(self, x):
        x = imnet(x)
        if self.arch.startswith("dinov2"):
            h = self.bb(pixel_values=dino_crop(x)).last_hidden_state
            return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))
        return self.m(x)


def _labels(fields: pd.DataFrame) -> torch.Tensor:
    return torch.tensor([BATCHES.index(b) for b in fields.batch])


def run_cls(cfg: dict, status_cb=None) -> Path:
    c = full_cfg(cfg)
    h = cfg_hash(c)
    d = CLS_RUNS / h
    d.mkdir(parents=True, exist_ok=True)
    if (d / "metrics.json").exists():
        return d
    lock = d / "lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < LOCK_STALE_S:
        raise RunLocked(f"cls {h} locked by pid {lock.read_text()}")
    lock.write_text(str(os.getpid()))
    (d / "config.json").write_text(json.dumps({**c, "hash": h}, indent=2))
    dev = device()
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])

    m = manifest()
    if c["split"] == "strat":
        m["fold"] = stratified_group_folds(m.group_id.to_numpy(), m.batch.to_numpy(), c["n_folds"], c["seed"])
    else:
        m["fold"] = grouped_folds(m.group_id.to_numpy(), c["n_folds"], c["seed"])
    tr_f, te_f = m[m.fold != c["fold"]], m[m.fold == c["fold"]]
    tr = CropDataset(FieldStore(tr_f, c["input"], harmonise=c["harmonise"]), c["view"], random_offset=c["aug"] != "aug0")
    y_crop = _labels(tr_f)[torch.tensor([i for i, _, _ in tr.index])].to(dev)
    g = torch.Generator().manual_seed(c["seed"])
    dl = GPUCropLoader(tr, c["batch_size"], dev, g)
    fr_path = OUT / "imaging_stats" / "forward_ranges.json"
    fr = ForwardRanges.from_json(fr_path) if fr_path.exists() else None
    gen = torch.Generator(device=dev).manual_seed(c["seed"])

    model = Classifier(c["arch"]).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=0.05)
    warm = max(1, c["steps"] // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / c["steps"]))))
    # class weights: fields per batch are 7/7/17, so weight inversely by training-crop frequency
    cnt = torch.bincount(y_crop, minlength=3).float()
    w = (cnt.sum() / (3 * cnt.clamp_min(1))).to(dev)
    log, t0, step = open(d / "train_log.jsonl", "w"), time.time(), 0
    model.train()
    while step < c["steps"]:
        for b in dl:
            if step >= c["steps"]:
                break
            x, y = augment(b["x"], c["aug"], fr, gen), y_crop[b["idx"]]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, weight=w, label_smoothing=c["label_smoothing"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0 or step == c["steps"]:
                rec = {"step": step, "loss": loss.item(), "acc": (logits.argmax(1) == y).float().mean().item(),
                       "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
                log.write(json.dumps(rec) + "\n"); log.flush(); lock.touch()
                if status_cb:
                    status_cb(h, rec)
    train_sec = time.time() - t0

    # ---- evaluate on the held-out fold: non-overlapping eval grid
    model.eval()
    te = CropDataset(FieldStore(te_f, c["input"], harmonise=c["harmonise"]), c["view"], stride=CROP)
    y_te = _labels(te_f)[torch.tensor([i for i, _, _ in te.index])].numpy()
    logits = []
    with torch.no_grad():
        for i in range(0, len(te), 128):
            x = torch.stack([te[j]["x"] for j in range(i, min(i + 128, len(te)))]).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits.append(model(x).float().cpu())
    L = torch.cat(logits).numpy()
    P = torch.softmax(torch.from_numpy(L), 1).numpy()
    pred = L.argmax(1)
    meta = pd.DataFrame({"group_id": te.group, "batch": te.batch, "y_true": y_te, "y_pred": pred,
                         **{f"p_{b}": P[:, k] for k, b in enumerate(BATCHES)}})
    meta.to_csv(d / "crop_predictions.csv", index=False)
    fld = meta.groupby("group_id").agg(y_true=("y_true", "first"), **{f"p_{b}": (f"p_{b}", "mean") for b in BATCHES})
    fld["y_pred"] = fld[[f"p_{b}" for b in BATCHES]].to_numpy().argmax(1)
    from sklearn.metrics import confusion_matrix, f1_score, log_loss
    out = {**c, "hash": h, "train_sec": train_sec, "n_train_fields": len(tr_f), "n_test_fields": len(te_f),
           "n_test_crops": len(te), "test_fields": te_f.group_id.tolist(),
           "crop_acc": float((pred == y_te).mean()), "crop_f1_macro": float(f1_score(y_te, pred, average="macro")),
           "crop_logloss": float(log_loss(y_te, P, labels=[0, 1, 2])),
           "field_acc": float((fld.y_pred == fld.y_true).mean()),
           "field_f1_macro": float(f1_score(fld.y_true, fld.y_pred, average="macro")),
           "field_confusion": confusion_matrix(fld.y_true, fld.y_pred, labels=[0, 1, 2]).tolist(),
           "crop_confusion": confusion_matrix(y_te, pred, labels=[0, 1, 2]).tolist(),
           "n_params_trained": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (d / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    lock.unlink(missing_ok=True)
    return d


def cls_grid(n_folds: int = 5, harms=(False, True), aug2: bool = True) -> list[dict]:
    """Classification baselines: arch x view (stack + single detectors) x input treatment, all folds.
    Harmonised input removes the per-image brightness/gain differences (the Batch_3 black level),
    so raw-vs-harmonised is the test of 'microstructure or imaging?'."""
    runs = []
    archs = ("resnet18_scratch", "resnet18_imnet", "effb4_imnet", "dinov2_ft", "dinov2_linear")
    for a in archs:
        for v in ("stack", "BSE", "Inlens", "SE_type"):
            for harm in harms:
                if v != "stack" and (a not in ("resnet18_imnet", "dinov2_ft") or harm not in (False, True, "hybrid")):
                    continue  # single-detector sweep on two representative archs (and the recommended LUT) only
                for k in range(n_folds):
                    runs.append(dict(task="cls", arch=a, view=v, harmonise=harm, fold=k, n_folds=n_folds, stage="CLS"))
    for k in range(n_folds if aug2 else 0):  # aug2 nuisance-robust variant of the main arch
        runs.append(dict(task="cls", arch="resnet18_imnet", view="stack", aug="aug2", fold=k, n_folds=n_folds, stage="CLS"))
        runs.append(dict(task="cls", arch="resnet18_imnet", view="stack", input="norm", fold=k, n_folds=n_folds, stage="CLS"))
    return runs


if __name__ == "__main__":
    import sys
    run_cls(json.loads(sys.argv[1]))
