"""Throughput benchmark for one model alone on the GPU: real DataLoader step vs synthetic
on-GPU batches (isolates data loading), at batch 64 and 256. Prints steps/s and crops/s."""
from __future__ import annotations

import json
import time

import torch

from src.v2.augment import augment
from src.v2.data import CropDataset, FieldStore, GPUCropLoader
from src.v2.models import build
from src.v2.train import full_cfg, kpi_table, train_fields
from src.v2.kpi_adapter import kpi_cols


def _time(model, opt, batches, n, fam):
    def step(b):
        if fam == "dino_ft":
            b["x2"] = augment(b["x"], "aug1")
        b["x"] = augment(b["x"], "aug1")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.loss(b)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if hasattr(model, "after_step"):
            model.after_step()
    it = iter(batches)
    for _ in range(10):
        step(next(it))
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        step(next(it))
    torch.cuda.synchronize()
    return n / (time.time() - t)


def bench(family: str, n: int = 100) -> dict:
    dev = torch.device("cuda")
    c = full_cfg({"family": family})
    model = build(family, n_kpi=5).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    out = {"family": family}
    fields = train_fields(c)
    cond = family in ("vae_b", "vae_c")
    kp, _ = kpi_table(fields, kpi_cols(None))
    ds = CropDataset(FieldStore(fields), "stack", random_offset=True, kpis=kp if cond else None,
                     kpi_cols=kpi_cols(None) if cond else ())
    for bs in (64, 256):
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4,
                                         persistent_workers=True, pin_memory=True)

        def real():
            while True:
                for b in dl:
                    yield {k: v.to(dev, non_blocking=True) for k, v in b.items()}

        gl = GPUCropLoader(ds, bs, dev, torch.Generator().manual_seed(0))

        def gpu():
            while True:
                yield from gl

        def synth():
            b = {"x": torch.rand(bs, 3, 256, 256, device=dev)}
            if cond:
                b["kpi"] = torch.randn(bs, 5, device=dev)
            while True:
                yield dict(b)
        for name, src in (("loader", real), ("gpuloader", gpu), ("synthetic", synth)):
            try:
                s = _time(model, opt, src(), n if bs == 64 else n // 2, family)
                out[f"bs{bs}_{name}_steps_s"] = round(s, 2)
                out[f"bs{bs}_{name}_crops_s"] = round(s * bs)
            except torch.cuda.OutOfMemoryError:
                out[f"bs{bs}_{name}"] = "OOM"
                torch.cuda.empty_cache()
    out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    print(json.dumps(out), flush=True)
    return out
