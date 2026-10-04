"""Test-time perturbation of the saved binary off-detectors (Batch_3 vs Batch_1+2): inject a known SEM artefact
into the model's *input* crops (after the route's harmonisation) and record how P(off) and the field decision
move. Out-of-fold: every field is scored by the fold model that did not train on it."""
import json, sys
import numpy as np, pandas as pd, torch
from src.v2 import attribution as A
from src.v2.classify import CLS_RUNS
import torch.nn.functional as Fn


def gauss_blur(x, sigma):
    r = int(3 * sigma); k = torch.exp(-(torch.arange(-r, r + 1).float() ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    k2 = (k[:, None] * k[None, :])[None, None].repeat(x.shape[1], 1, 1, 1)
    return Fn.conv2d(Fn.pad(x, (r, r, r, r), mode="reflect"), k2, groups=x.shape[1])
A.device = lambda: torch.device("cpu")
torch.manual_seed(0); torch.set_num_threads(3)
P = {"none": lambda x: x, "noise_6grey": lambda x: x + torch.randn_like(x) * 6 / 255, "blur_0.7px": lambda x: gauss_blur(x, 0.7),
     "blur_1.0px": lambda x: gauss_blur(x, 1.0), "offset_+8grey": lambda x: x + 8 / 255, "gain_0.85": lambda x: x * 0.85}
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 512
rows = []
for d in sorted(CLS_RUNS.glob("*/final.pt")):
    c = json.loads((d.parent / "config.json").read_text())
    if c.get("labels") != "off" or c["arch"] != "resnet18_imnet" or c["view"] != "stack":
        continue
    model, c = A.load_run(d.parent); te_f = A.test_fold(c)
    ds = A.CropDataset(A.FieldStore(te_f, c["input"], harmonise=c["harmonise"]), c["view"], stride=STRIDE)
    xs = torch.stack([ds[j]["x"] for j in range(len(ds))]); fi = np.array([ds.index[j][0] for j in range(len(ds))])
    for name, fn in P.items():
        with torch.no_grad():
            p = torch.cat([torch.softmax(model(fn(xs[a:a + 64]).clamp(0, 1)).float(), 1) for a in range(0, len(xs), 64)]).numpy()
        for k in np.unique(fi):
            rows.append(dict(route=f'{c["input"]}/{c["harmonise"]}', fold=c["fold"], perturb=name, group_id=te_f.group_id.iloc[k],
                             batch=te_f.batch.iloc[k], n=int((fi == k).sum()), p_off=float(p[fi == k, 0].mean())))
    print(d.parent.name, c["input"], c["harmonise"], c["fold"], len(xs), flush=True)
df = pd.DataFrame(rows); df.to_csv("outputs/v2/off/testtime_perturbation.csv", index=False)
base = df[df.perturb == "none"].set_index(["route", "group_id"]).p_off
df["p_off_base"] = base.loc[list(zip(df.route, df.group_id))].to_numpy()
df["flip"] = (df.p_off > .5) != (df.p_off_base > .5)
df["off_true"] = df.batch != "Batch_3"
pd.set_option("display.width", 250)
print("\nfield flips out of 31 and mean dP(off), per route x perturbation:")
g = df[df.perturb != "none"].groupby(["route", "perturb"]).agg(flips=("flip", "sum"), dP_off=("p_off", lambda s: 0), n=("flip", "size"))
g["dP_off"] = df[df.perturb != "none"].assign(d=lambda t: t.p_off - t.p_off_base).groupby(["route", "perturb"]).d.mean()
g["bal_acc"] = df[df.perturb != "none"].groupby(["route", "perturb"]).apply(lambda t: 0.5 * (((t.p_off > .5) & t.off_true).sum() / t.off_true.sum() + ((t.p_off <= .5) & ~t.off_true).sum() / (~t.off_true).sum()))
print(g.round(2).to_string())
