"""Model-level injection ablation: does a classifier trained on route R still change its prediction
when a known SEM artefact is injected into the *raw* image and the image is re-processed by R?

Uses the saved conditions of src.v2.injection_ablation (raw -> inject -> LUT / clean pipeline) and the
saved ResNet-18 stack classifiers (one per route and fold). For each ablation site the fold in which
the site is a *test* field is used, so every prediction is out-of-fold. Reports, per site x injection
x route, the mean P(batch) over the site's eval crops and the field-level decision, relative to the
un-injected condition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.v2.classify import BATCHES, CLS_RUNS, Classifier
from src.v2.common import CROP, OUT, SEED, grid, manifest
from src.v2.data import MIN_VALID, VIEWS, stratified_group_folds
from src.v2.injection_ablation import DETS, INJECTIONS, OUT_DIR as ABL_DIR, Z_SCALE

ROUTES = {"raw": False, "hybrid": "hybrid", "clean_norm": "clean_norm", "clean_harm": "clean_harm"}
OUT_DIR = OUT / "injection_ablation"


def saved_models(arch: str = "resnet18_imnet") -> pd.DataFrame:
    rows = []
    for d in CLS_RUNS.glob("*/final.pt"):
        c = json.loads((d.parent / "config.json").read_text())
        if c["arch"] == arch and c["view"] == "stack" and c["aug"] == "aug1" and not c.get("dequant") and c["n_folds"] == 5:
            rows.append(dict(hash=d.parent.name, harmonise=str(c["harmonise"]), fold=c["fold"]))
    return pd.DataFrame(rows)


def route_input(z: np.lib.npyio.NpzFile, route: str) -> tuple[np.ndarray, np.ndarray | None]:
    if route in ("raw", "hybrid", "affine2"):
        return z[route].astype(np.float32) / 255.0, None
    zh, va = z[f"{route}_z"], z[f"{route}_valid"]
    im = np.stack([np.round(np.clip(zh[..., i], 0, 255.0 / Z_SCALE[d]) * Z_SCALE[d]) for i, d in enumerate(DETS)], -1)
    return im.astype(np.float32) / 255.0, va.all(-1)


def crops(im: np.ndarray, valid: np.ndarray | None) -> torch.Tensor:
    ch = list(VIEWS["stack"])
    xs = []
    for y, x in grid(*im.shape[:2], CROP, CROP):
        if valid is None or valid[y:y + CROP, x:x + CROP].mean() >= MIN_VALID:
            xs.append(torch.from_numpy(np.ascontiguousarray(im[y:y + CROP, x:x + CROP][..., ch])).permute(2, 0, 1))
    return torch.stack(xs)


@torch.no_grad()
def predict(model, x: torch.Tensor) -> np.ndarray:
    return torch.cat([torch.softmax(model(x[i:i + 64]).float(), 1) for i in range(0, len(x), 64)]).numpy()


def main(sites=("4ih2ggld", "3806gxp0", "iv6g2oq0"), injections=INJECTIONS):
    m = manifest()
    m["fold"] = stratified_group_folds(m.group_id.to_numpy(), m.batch.to_numpy(), 5, SEED)
    models = saved_models()
    rows = []
    for s in sites:
        r = m[m.site == s].iloc[0]
        for route, harm in ROUTES.items():
            hit = models[(models.harmonise == str(harm)) & (models.fold == r.fold)]
            if hit.empty:
                print(f"no saved {route} model for fold {r.fold} (site {s})")
                continue
            model = Classifier("resnet18_imnet")
            model.load_state_dict(torch.load(CLS_RUNS / hit.hash.iloc[0] / "final.pt", map_location="cpu"))
            model.eval()
            for inj in injections:
                f = ABL_DIR / f"{s}__{inj}.npz"
                if not f.exists():
                    continue
                im, va = route_input(np.load(f), route)
                P = predict(model, crops(im, va))
                rows.append(dict(site=s, batch=r.batch, fold=int(r.fold), route=route, injection=inj, n_crops=len(P),
                                 **{f"p_{b}": float(P[:, k].mean()) for k, b in enumerate(BATCHES)},
                                 **{f"share_{b}": float((P.argmax(1) == k).mean()) for k, b in enumerate(BATCHES)},
                                 field_pred=BATCHES[int(P.mean(0).argmax())]))
    df = pd.DataFrame(rows)
    base = df[df.injection == "none"].set_index(["site", "route"])
    df["p_true"] = [row[f"p_{row.batch}"] for _, row in df.iterrows()]
    df["dp_true"] = df.p_true - [base.loc[(r.site, r.route), f"p_{r.batch}"] for r in df.itertuples()]
    df["dp_B3"] = df.p_Batch_3 - [base.loc[(r.site, r.route), "p_Batch_3"] for r in df.itertuples()]
    df["flip"] = df.field_pred != [base.loc[(r.site, r.route), "field_pred"] for r in df.itertuples()]
    df.to_csv(OUT_DIR / "model_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print("mean P(Batch_3) per route and injection (rows: site):")
    print(df.pivot_table(index=["site", "injection"], columns="route", values="p_Batch_3").round(2).to_string())
    print("change in P(true batch) vs un-injected, median over sites:")
    print(df[df.injection != "none"].pivot_table(index="injection", columns="route", values="dp_true", aggfunc="median").round(2).to_string())
    print("field-level decision flips (count over sites):")
    print(df[df.injection != "none"].pivot_table(index="injection", columns="route", values="flip", aggfunc="sum").to_string())
    return df


if __name__ == "__main__":
    main()
