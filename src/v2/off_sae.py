"""SAE analysis of the off-detector classifiers' penultimate features.

For one binary config (route x architecture) the out-of-fold penultimate features of the five fold
models are pooled on the non-overlapping eval grid of all 31 labelled fields and analysed with the
same TopK SAE / material-vs-imaging characterisation as the representation runs
(``src/v2/sae.analyse_embedding``). Output: ``outputs/v2/off/sae/<route>_<arch>/`` and a summary row
in ``outputs/v2/off/sae_summary.csv``.  Usage: python src/v2/off_sae.py [--arch resnet18_imnet] [--routes extreme,hybrid]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.v2.attribution import load_run, test_fold
from src.v2.common import CROP, OUT, harm_method
from src.v2.data import CropDataset, FieldStore
from src.v2.off_explain import ROUTE
from src.v2.sae import analyse_embedding


def binary_runs() -> pd.DataFrame:
    rows = []
    for d in (OUT / "cls_runs").iterdir():
        p = d / "config.json"
        if not p.exists() or not (d / "final.pt").exists():
            continue
        c = json.loads(p.read_text())
        if c.get("labels") == "off":
            rows.append({"hash": d.name, "arch": c["arch"], "input": c["input"], "harmonise": c["harmonise"], "fold": c["fold"], "view": c["view"]})
    return pd.DataFrame(rows)


@torch.no_grad()
def penultimate(model, x: torch.Tensor) -> torch.Tensor:
    feats = {}
    layer = model.m.fc if hasattr(model, "m") and hasattr(model.m, "fc") else model.head
    h = layer.register_forward_hook(lambda _, inp, __: feats.__setitem__("z", inp[0].detach()))
    model(x)
    h.remove()
    return feats["z"].float().cpu()


def pooled_features(runs: pd.DataFrame, dev) -> tuple[np.ndarray, pd.DataFrame]:
    E, meta = [], []
    for h in runs.sort_values("fold").hash:
        d = OUT / "cls_runs" / h
        model, c = load_run(d)
        model.to(dev)
        te_f = test_fold(c)
        store = FieldStore(te_f, c["input"], harmonise=c["harmonise"])
        ds = CropDataset(store, c["view"], stride=CROP)
        for i in range(len(ds)):
            fi, y0, x0 = ds.index[i]
            E.append(penultimate(model, ds[i]["x"].to(dev)[None])[0].numpy())
            meta.append({"group_id": ds.group[i], "batch": te_f.batch.iloc[fi], "y": y0, "x": x0})
    return np.stack(E), pd.DataFrame(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18_imnet")
    ap.add_argument("--routes", default="extreme,hybrid")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = binary_runs()
    rows = []
    for _, g in runs[(runs.arch == a.arch) & (runs.view == "stack")].groupby(["input", "harmonise"]):
        route = ROUTE[(g.input.iloc[0], g.harmonise.iloc[0])].split()[0]
        if route not in a.routes.split(","):
            continue
        out = OUT / "off" / "sae" / f"{route}_{a.arch}"
        out.mkdir(parents=True, exist_ok=True)
        E, meta = pooled_features(g, dev)
        np.savez(out / "features_eval.npz", E=E, **{k: meta[k].to_numpy() for k in meta})
        summ, _ = analyse_embedding(E, meta, out, harm_method(g.harmonise.iloc[0]), tag=f"{route} {a.arch}")
        rows.append({"route": route, "arch": a.arch, "n_crops": len(E), **{k: v for k, v in summ.items() if not isinstance(v, list)}})
        print(route, a.arch, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
    p = OUT / "off" / "sae_summary.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["route", "arch"], keep="last")
    new.to_csv(p, index=False)


if __name__ == "__main__":
    main()
