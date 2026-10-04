"""Injection ablation for the imported harmonisation routes (PR #37: N4 + Nyul-Udupa `nyul`, BaSiC `basic`).

Re-uses the injected raw half-res conditions saved by src.v2.injection_ablation
(outputs/v2/injection_ablation/<site>__<injection>.npz, key `raw`) and re-processes each with the
*fitted* nyul / basic models (cache/harmonised_ext/<m>/model.*), exactly as the cache arrays were
built (N4 bias correction + landmark mapping; (I-D)/S - baseline). Reports
  * imaging-fingerprint residuals (injected - un-injected) in units of the across-site SD of that
    statistic on the same route (so 1.0 = one site-to-site SD), and
  * the out-of-fold ResNet-18 decisions of the nyul / basic classifiers (3-class and binary off-detector)
    on the injected images: mean P and field-level flips.
Outputs: outputs/v2/injection_ablation/ext_{residuals,model}.csv and a printed summary.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch

from pmdb import harmonise_ext as X
from src.v2 import kpi_adapter as K
from src.v2.classify import BATCHES, CLS_RUNS, Classifier, classes_of
from src.v2.common import EXT_METHODS, NM_HALF, OUT, REPO, SEED, load_half_raw, manifest
from src.v2.data import stratified_group_folds
from src.v2.injection_ablation import DETS, INJECTIONS, OUT_DIR, fingerprint
from src.v2.injection_model import crops, predict

SITES = ("4ih2ggld", "3806gxp0", "iv6g2oq0", "b3esycq1", "0grcilhi", "71vgq3fw")
EXT_ROOT = REPO / "cache" / "harmonised_ext"


def apply_route(raw_half: np.ndarray, method: str, models: dict) -> np.ndarray:
    """uint8 (H, W, 3) as stored in cache/harmonised_ext (fixed grey/255 scale for the models)."""
    out = []
    valid = np.ones(raw_half.shape[:2], bool)
    for i, d in enumerate(DETS):
        img = raw_half[..., i].astype(np.float32)
        if method == "nyul":
            n4, _ = X.n4_correct(img, valid)
            im, _ = X.nyul_apply(n4, valid, models[d])
        else:
            im, _ = X.basic_apply(img, valid, models[d])
        out.append(X.to_uint8(im, method))
    return np.stack(out, -1)


def stats(im: np.ndarray) -> dict:
    st = {f"{k}_{d}": v for i, d in enumerate(DETS) for k, v in fingerprint(im[..., i]).items()}
    m = K.segment(im[..., 0], NM_HALF)
    st.update({f"{k}_BSE": v for k, v in K.mask_fractions(m).items() if k != "frac_artefact"})
    return st


def site_spread(method: str) -> pd.Series:
    rows = [stats(load_half_raw(b, s, harm=method)) for b, s in zip(manifest().batch, manifest().site)]
    return pd.DataFrame(rows).std(ddof=1)


def saved_models(method: str, labels: str) -> dict[int, str]:
    out = {}
    for d in CLS_RUNS.glob("*/final.pt"):
        c = json.loads((d.parent / "config.json").read_text())
        if (c["arch"] == "resnet18_imnet" and c["view"] == "stack" and c["harmonise"] == method
                and c.get("labels", "batch") == labels and c["n_folds"] == 5):
            out[c["fold"]] = d.parent.name
    return out


def main(sites=SITES, injections=INJECTIONS):
    m = manifest()
    m["fold"] = stratified_group_folds(m.group_id.to_numpy(), m.batch.to_numpy(), 5, SEED)
    res_rows, mod_rows = [], []
    for method in EXT_METHODS:
        models = X.load_models(EXT_ROOT, method)
        spread = site_spread(method)
        cls = {lab: saved_models(method, lab) for lab in ("batch", "off")}
        for s in sites:
            r = m[m.site == s].iloc[0]
            cond = {}
            for inj in injections:
                f = OUT_DIR / f"{s}__{inj}.npz"
                if not f.exists():
                    continue
                im = apply_route(np.load(f)["raw"], method, models)
                np.savez_compressed(OUT_DIR / f"{s}__{inj}__{method}.npz", image=im)
                cond[inj] = stats(im)
                x = crops(im.astype(np.float32) / 255.0, None)
                for lab, folds in cls.items():
                    if r.fold not in folds:
                        continue
                    c = json.loads((CLS_RUNS / folds[r.fold] / "config.json").read_text())
                    net = Classifier("resnet18_imnet", n_cls=len(classes_of(c)))
                    net.load_state_dict(torch.load(CLS_RUNS / folds[r.fold] / "final.pt", map_location="cpu"))
                    P = predict(net.eval(), x)
                    names = classes_of(c)
                    mod_rows.append(dict(route=method, labels=lab, site=s, batch=r.batch, injection=inj, n_crops=len(P),
                                         **{f"p_{n}": float(P[:, k].mean()) for k, n in enumerate(names)},
                                         field_pred=names[int(P.mean(0).argmax())]))
                print(method, s, inj, "done", flush=True)
            base = cond.get("none")
            for inj, st in cond.items():
                for k, v in st.items():
                    res_rows.append(dict(route=method, site=s, batch=r.batch, injection=inj, stat=k, value=v,
                                         residual_sd=(v - base[k]) / spread[k] if base is not None and spread[k] > 0 else np.nan))
    R = pd.DataFrame(res_rows)
    R.to_csv(OUT_DIR / "ext_residuals.csv", index=False)
    M = pd.DataFrame(mod_rows)
    if len(M):
        key = ["route", "labels", "site"]
        base = M[M.injection == "none"].set_index(key)
        M["p_off"] = M.get("p_Batch_1+2", np.nan)
        M["flip"] = M.field_pred.to_numpy() != base.loc[[tuple(r) for r in M[key].to_numpy()], "field_pred"].to_numpy()
        M["dp_B3"] = M.p_Batch_3.to_numpy() - base.loc[[tuple(r) for r in M[key].to_numpy()], "p_Batch_3"].to_numpy()
        M.to_csv(OUT_DIR / "ext_model.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n|residual| in site-SD units, median over sites (BSE):")
    sub = R[(R.injection != "none") & R.stat.str.endswith("_BSE")]
    print(sub.assign(a=sub.residual_sd.abs()).pivot_table(index=["stat"], columns=["route", "injection"], values="a", aggfunc="median").round(2).to_string())
    if len(M):
        print("\nfield-level decision flips (count over %d sites):" % len(sites))
        print(M[M.injection != "none"].pivot_table(index="injection", columns=["route", "labels"], values="flip", aggfunc="sum").to_string())
        print("\nchange in P(Batch_3) vs un-injected, median over sites:")
        print(M[M.injection != "none"].pivot_table(index="injection", columns=["route", "labels"], values="dp_B3", aggfunc="median").round(2).to_string())


if __name__ == "__main__":
    main(injections=tuple(sys.argv[1:]) or INJECTIONS)
