"""Supervised 3-class batch classification (Batch_1 / Batch_2 / Batch_3), split by field code.

Grouped k-fold over all 31 fields (every crop of a field in one fold). Train on the fold's
training fields (overlapping crops, aug1/aug2), test on the held-out fields' non-overlapping
crops: crop-level and field-level (mean logit) accuracy, macro-F1, confusion matrix, log-loss.

Spec keys: arch (resnet18_scratch | resnet18_imnet | effb4_imnet | effb4_micronet | dinov2_ft | dinov2_linear),
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
                fold=0, n_folds=5, split="strat", steps=1500, batch_size=64, lr=None, seed=SEED, label_smoothing=0.1,
                save=False, dequant=0.0, labels="batch", exclude="")
NOHASH = ("save",)  # bookkeeping flags that do not change the trained model
NOHASH_IF_DEFAULT = ("dequant", "labels", "exclude")  # later additions: keep earlier hashes stable when left at default
# labels='batch': 3-class Batch_1/2/3; labels='off': Batch_3 (supplier baseline) vs Batch_1+2 ('off')
# exclude: comma-separated site ids dropped from training AND evaluation (e.g. imaging-outlier fields)
OFF_CLASSES = ("Batch_1+2", "Batch_3")
LR = {"resnet18_scratch": 1e-3, "resnet18_imnet": 3e-4, "effb4_imnet": 3e-4, "effb4_micronet": 3e-4,
      "dinov2_ft": 5e-5, "dinov2_linear": 1e-3}


def full_cfg(cfg: dict) -> dict:
    c = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
    if c["lr"] is None:
        c["lr"] = LR[c["arch"]]
    return c


def cfg_hash(cfg: dict) -> str:
    c = {k: v for k, v in full_cfg(cfg).items() if k not in NOHASH}
    c = {k: v for k, v in c.items() if not (k in NOHASH_IF_DEFAULT and v == DEFAULTS[k])}
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


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
        elif arch == "effb4_micronet":  # NASA MicroNet weights (ImageNet -> MicroNet), fully fine-tuned
            from src.v2.models import OffTheShelf
            self.m = OffTheShelf("micronet").m.requires_grad_(True).train()
            self.head = nn.Linear(self.m.out_channels[-1], n_cls)
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
        if self.arch == "effb4_micronet":
            return self.head(self.m(x)[-1].mean((2, 3)))
        return self.m(x)


B12_CLASSES = ("Batch_1", "Batch_2")  # labels='b12': stage-2 classifier trained on the 14 off fields only


def classes_of(c: dict) -> tuple[str, ...]:
    lab = c.get("labels", "batch")
    return BATCHES if lab == "batch" else B12_CLASSES if lab == "b12" else OFF_CLASSES


def _labels(fields: pd.DataFrame, c: dict | None = None) -> torch.Tensor:
    if c is not None and c.get("labels", "batch") == "off":
        return torch.tensor([int(b == "Batch_3") for b in fields.batch])
    if c is not None and c.get("labels", "batch") == "b12":
        return torch.tensor([B12_CLASSES.index(b) for b in fields.batch])
    return torch.tensor([BATCHES.index(b) for b in fields.batch])


def run_cls(cfg: dict, status_cb=None) -> Path:
    c = full_cfg(cfg)
    h = cfg_hash(c)
    d = CLS_RUNS / h
    d.mkdir(parents=True, exist_ok=True)
    if (d / "metrics.json").exists() and (not c["save"] or (d / "final.pt").exists()):
        return d
    lock = d / "lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < LOCK_STALE_S:
        raise RunLocked(f"cls {h} locked by pid {lock.read_text()}")
    lock.write_text(str(os.getpid()))
    (d / "config.json").write_text(json.dumps({**c, "hash": h}, indent=2))
    dev = device()
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])

    m = manifest()
    if c.get("exclude"):
        m = m[~m.group_id.str.split("/").str[1].isin(c["exclude"].split(","))].reset_index(drop=True)
    if c.get("labels", "batch") == "b12":
        m = m[m.batch.isin(B12_CLASSES)].reset_index(drop=True)
    if c["split"] == "parent":  # leave-parents-out: crops of one parent image never straddle train/test
        m["fold"] = parent_folds(m, c["n_folds"], c["seed"])
    elif c["split"] == "strat":
        m["fold"] = stratified_group_folds(m.group_id.to_numpy(), m.batch.to_numpy(), c["n_folds"], c["seed"])
    else:
        m["fold"] = grouped_folds(m.group_id.to_numpy(), c["n_folds"], c["seed"])
    tr_f, te_f = m[m.fold != c["fold"]], m[m.fold == c["fold"]]  # fold=-1: train on every labelled field, no test fold
    tr = CropDataset(FieldStore(tr_f, c["input"], harmonise=c["harmonise"]), c["view"], random_offset=c["aug"] != "aug0")
    classes = classes_of(c); n_cls = len(classes)
    y_crop = _labels(tr_f, c)[torch.tensor([i for i, _, _ in tr.index])].to(dev)
    g = torch.Generator().manual_seed(c["seed"])
    dl = GPUCropLoader(tr, c["batch_size"], dev, g)
    fr_path = OUT / "imaging_stats" / "forward_ranges.json"
    fr = ForwardRanges.from_json(fr_path) if fr_path.exists() else None
    gen = torch.Generator(device=dev).manual_seed(c["seed"])

    model = Classifier(c["arch"], n_cls=n_cls).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=0.05)
    warm = max(1, c["steps"] // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / c["steps"]))))
    # class weights: fields per batch are 7/7/17, so weight inversely by training-crop frequency
    cnt = torch.bincount(y_crop, minlength=n_cls).float()
    w = (cnt.sum() / (n_cls * cnt.clamp_min(1))).to(dev)
    log, t0, step = open(d / "train_log.jsonl", "w"), time.time(), 0
    model.train()
    while step < c["steps"]:
        for b in dl:
            if step >= c["steps"]:
                break
            x, y = augment(b["x"], c["aug"], fr, gen), y_crop[b["idx"]]
            if c["dequant"]:  # uniform +-dequant grey levels: fills the LUT comb (missing grey levels) left by harmonisation
                x = x + (torch.rand(x.shape, generator=gen, device=x.device) - 0.5) * (2 * c["dequant"] / 255.0)
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
    if len(te_f) == 0:  # final model: score the 3 organiser held-out sites instead (labels known post hoc)
        if c["save"]:
            torch.save(model.state_dict(), d / "final.pt")
        out = {**c, "hash": h, "train_sec": train_sec, "n_train_fields": len(tr_f), "n_test_fields": 0, "classes": list(classes),
               **heldout_eval(model, c, dev)}
        (d / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
        lock.unlink(missing_ok=True)
        return d
    te = CropDataset(FieldStore(te_f, c["input"], harmonise=c["harmonise"]), c["view"], stride=CROP)
    y_te = _labels(te_f, c)[torch.tensor([i for i, _, _ in te.index])].numpy()
    logits = []
    with torch.no_grad():
        for i in range(0, len(te), 128):
            x = torch.stack([te[j]["x"] for j in range(i, min(i + 128, len(te)))]).to(dev)
            if c["dequant"]:
                x = x + (torch.rand(x.shape, generator=gen, device=x.device) - 0.5) * (2 * c["dequant"] / 255.0)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits.append(model(x).float().cpu())
    L = torch.cat(logits).numpy()
    P = torch.softmax(torch.from_numpy(L), 1).numpy()
    pred = L.argmax(1)
    meta = pd.DataFrame({"group_id": te.group, "batch": te.batch, "y_true": y_te, "y_pred": pred,
                         **{f"p_{b}": P[:, k] for k, b in enumerate(classes)}})
    meta.to_csv(d / "crop_predictions.csv", index=False)
    fld = meta.groupby("group_id").agg(y_true=("y_true", "first"), **{f"p_{b}": (f"p_{b}", "mean") for b in classes})
    fld["y_pred"] = fld[[f"p_{b}" for b in classes]].to_numpy().argmax(1)
    labs = list(range(n_cls))
    from sklearn.metrics import confusion_matrix, f1_score, log_loss
    out = {**c, "hash": h, "train_sec": train_sec, "n_train_fields": len(tr_f), "n_test_fields": len(te_f),
           "n_test_crops": len(te), "test_fields": te_f.group_id.tolist(),
           "crop_acc": float((pred == y_te).mean()), "crop_f1_macro": float(f1_score(y_te, pred, average="macro")),
           "crop_logloss": float(log_loss(y_te, P, labels=labs)),
           "field_acc": float((fld.y_pred == fld.y_true).mean()),
           "field_f1_macro": float(f1_score(fld.y_true, fld.y_pred, average="macro")),
           "field_confusion": confusion_matrix(fld.y_true, fld.y_pred, labels=labs).tolist(),
           "crop_confusion": confusion_matrix(y_te, pred, labels=labs).tolist(), "classes": list(classes),
           "n_params_trained": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (d / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    if c["save"]:
        torch.save(model.state_dict(), d / "final.pt")
    lock.unlink(missing_ok=True)
    return d


PARENTS = Path(__file__).with_name("parent_groups.csv")  # outputs/parent_groups.csv from main: 31 sites = crops of 13 parent images


def parent_folds(m: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """Assign whole parent images to folds, largest parents first, balancing the Batch_3 share per fold."""
    par = pd.read_csv(PARENTS).set_index("site").parent_id
    site = m.group_id.str.split("/").str[1]
    pid = site.map(par).to_numpy()
    rng = np.random.default_rng(seed)
    ids, n = np.unique(pid, return_counts=True)
    order = np.argsort(-n + rng.uniform(0, 0.5, len(n)))
    fold_n, fold_b3, fold_of = np.zeros(n_folds), np.zeros(n_folds), {}
    for k in order:
        b3 = (m.batch.to_numpy()[pid == ids[k]] == "Batch_3").sum()
        score = fold_n + 0.5 * np.abs(fold_b3 + b3 - (fold_n + n[k]) * 17 / 31)
        f = int(np.argmin(score)); fold_of[ids[k]] = f; fold_n[f] += n[k]; fold_b3[f] += b3
    return np.array([fold_of[p] for p in pid])


HELDOUT_TRUTH = {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}  # organiser labels, given after the fact


def heldout_eval(model, c: dict, dev) -> dict:
    """Fold-free final model: P(class) per held-out site on the eval grid, compared with HELDOUT_TRUTH."""
    from src.v2.sae_ablate import heldout_crops
    classes = classes_of(c)
    X, meta = heldout_crops(c)
    with torch.no_grad():
        P = torch.cat([torch.softmax(model(torch.from_numpy(X[i:i + 64]).permute(0, 3, 1, 2).float().to(dev)).float(), 1).cpu()
                       for i in range(0, len(X), 64)]).numpy()
    rows, n_ok, nll = [], 0, 0.0
    for site, truth in HELDOUT_TRUTH.items():
        p = P[(meta.site == site).to_numpy()].mean(0)
        t = truth if c.get("labels", "batch") in ("batch", "b12") else ("Batch_3" if truth == "Batch_3" else "Batch_1+2")
        if t not in classes:  # stage-2 B1/B2 model: the Batch_3 site is out of scope, report its probabilities only
            rows.append({"site": site, "truth": t, "pred": classes[int(p.argmax())], "correct": None, **{f"p_{b}": float(v) for b, v in zip(classes, p)}})
            continue
        k = classes.index(t); pred = classes[int(p.argmax())]
        n_ok += pred == t; nll -= math.log(max(p[k], 1e-6))
        rows.append({"site": site, "truth": t, "pred": pred, "correct": pred == t, **{f"p_{b}": float(v) for b, v in zip(classes, p)}})
    scored = [r for r in rows if r["correct"] is not None]
    return {"heldout": rows, "heldout_correct": int(n_ok), "heldout_nll": nll / len(scored),
            "heldout_min_margin": float(min(r[f"p_{r['truth']}"] - max(v for kk, v in r.items() if kk.startswith("p_") and kk != f"p_{r['truth']}") for r in scored))}


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
