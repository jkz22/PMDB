"""One run on the GPU: train (resumable) -> evaluate on held-out fields -> embed every field.

    python -m src.v2.worker '<json run dict>'
Writes <run_dir>/metrics.json and embeddings_eval.npz (all 31 fields, eval grid).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from src.v2.common import CROP
from src.v2.data import CropDataset, FieldStore, heldout_split
from src.v2.evaluate import crop_meta, embed_all, evaluate, recon_metrics
from src.v2.kpi_adapter import GATED_COLS, kpi_commit_hash
from src.v2.models import build
from src.v2.rungrid import train_cfg
from src.v2.train import RUNS, cfg_hash, device, full_cfg, run


def _std_kpis(kpi_norm):
    import pandas as pd
    from src.v2.common import OUT
    k = pd.read_csv(OUT / "kpis" / "crop_kpis_eval.csv")
    for col in GATED_COLS:
        k[col] = (k[col] - kpi_norm["mean"][col]) / kpi_norm["std"][col]
    return k


def main(run_spec: dict):
    dev = device()
    fam = run_spec["family"]
    if fam.startswith("ots_"):
        c = full_cfg({**train_cfg(run_spec), "steps": 0})
        d = RUNS / cfg_hash(c); d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({**c, "hash": d.name}, indent=2))
        model, kpi_norm = build(fam).to(dev), None
    else:
        d = run(train_cfg(run_spec))
        s = torch.load(d / "final.pt", map_location=dev, weights_only=False)
        c, kpi_norm = s["cfg"], s["kpi_norm"]
        model = build(fam, n_kpi=len(GATED_COLS), vae_mask=c["vae_mask"], mae_mask=c["mae_mask"]).to(dev)
        model.load_state_dict(s["model"])
    model.eval()
    m = heldout_split()
    if (d / "metrics.json").exists():
        out = json.loads((d / "metrics.json").read_text())
        if fam.startswith("vae") and "psnr" not in out:  # backfill runs evaluated before PSNR/SSIM existed
            held = FieldStore(m[m.heldout], c["input"], harmonise=c["harmonise"])
            ds = CropDataset(held, c["view"], stride=CROP, kpis=_std_kpis(kpi_norm) if fam in ("vae_b", "vae_c") else None,
                             kpi_cols=GATED_COLS if fam in ("vae_b", "vae_c") else ())
            out.update(recon_metrics(model, ds, dev, fam in ("vae_b", "vae_c")))
            (d / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
        return
    held = FieldStore(m[m.heldout], c["input"], harmonise=c["harmonise"])
    res, _, _ = evaluate(model, held, c["view"], dev, fam, kpi_norm)
    allf = FieldStore(m, c["input"], harmonise=c["harmonise"])
    kpi_needed = fam in ("vae_b", "vae_c")
    if kpi_needed:
        ds = CropDataset(allf, c["view"], stride=CROP, kpis=_std_kpis(kpi_norm), kpi_cols=GATED_COLS)
    else:
        ds = CropDataset(allf, c["view"], stride=CROP)
    E = embed_all(model, ds, dev, kpi_needed=kpi_needed)
    meta = crop_meta(ds)
    np.savez_compressed(d / "embeddings_eval.npz", E=E.astype(np.float32), **{k: meta[k].to_numpy() for k in meta})
    out = {**{k: v for k, v in run_spec.items()}, **res, "hash": d.name, "kpi_commit_full": kpi_commit_hash()}
    (d / "metrics.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
