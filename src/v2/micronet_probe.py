"""MicroNet deep-dive (CPU): what the frozen embeddings encode, and pooling/input variants."""
import numpy as np, pandas as pd, torch, json, time, sys
import torch.nn.functional as F
from src.v2 import kpi_adapter as K
from src.v2.common import CROP, OUT, manifest
from src.v2.data import CropDataset, FieldStore, heldout_split
from src.v2.evaluate import probe_r2, knn_batch_acc, crop_meta, IMG_STATS
from src.v2.models import OffTheShelf, imnet

m = heldout_split()
ev = m[m.heldout] if "heldout" in m else m[m.split == "heldout"]
print("eval fields", ev.group_id.tolist(), flush=True)
kpis = K.crop_kpi_frame("eval", K.GATED_COLS)
ist = pd.read_csv(OUT / "imaging_stats" / "per_crop.csv")

def targets(ds):
    meta = crop_meta(ds)
    Yk = meta.merge(kpis, on=["group_id", "y", "x"], how="left")[list(K.GATED_COLS)]
    Yi = meta.copy()
    for c in sorted(set(ds.ch)):
        s = ist[ist.channel == c][["group_id", "y_half", "x_half", *IMG_STATS]]
        s.columns = ["group_id", "y", "x", *[f"{k}_c{c}" for k in IMG_STATS]]
        Yi = Yi.merge(s, on=["group_id", "y", "x"], how="left")
    Yi = Yi.drop(columns=["group_id", "batch", "y", "x"])
    return meta, Yk, Yi

def score(E, meta, Yk, Yi):
    g = meta.group_id.to_numpy()
    kr, ir = probe_r2(E, Yk, g), probe_r2(E, Yi, g)
    return dict(kpi_r2=np.mean(list(kr.values())), img_r2=np.mean(list(ir.values())), knn_batch=knn_batch_acc(E, meta),
                **{f"kpi__{k}": v for k, v in kr.items()}, img_r2_max=max(ir.values()), img_r2_argmax=max(ir, key=ir.get), dim=E.shape[1])

rows = []
# 1) saved embeddings of the three OTS models (raw + hybrid) -> per-KPI / per-imaging-stat profile
lb = pd.read_csv(OUT / "leaderboard.csv")
for _, r in lb[lb.family.str.startswith("ots")].iterrows():
    if r.harmonise not in ("False", "hybrid"): continue
    z = np.load(OUT / "runs" / r.hash / "embeddings_eval.npz", allow_pickle=True)
    harm = "hybrid" if r.harmonise == "hybrid" else "none"
    ds = CropDataset(FieldStore(ev, "raw", harmonise=harm if harm != "none" else False), "stack", stride=CROP)
    meta, Yk, Yi = targets(ds)
    key = pd.DataFrame({"group_id": z["group_id"], "y": z["y"], "x": z["x"], "i": np.arange(len(z["E"]))})
    sel = meta.merge(key, on=["group_id", "y", "x"], how="left")
    assert sel.i.notna().all(), "eval crops missing from saved embeddings"
    E = z["E"][sel.i.to_numpy(int)]
    rows.append(dict(model=r.family, variant="saved", harm=harm, **score(E, meta, Yk, Yi)))
    print(rows[-1], flush=True)

# 2) MicroNet variants, computed here
net = OffTheShelf("micronet")
enc = net.m
def embed(ds, fn, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            x = torch.stack([ds[j]["x"] for j in range(i, min(i + bs, len(ds)))])
            out.append(fn(x).float())
    return torch.cat(out).numpy()
def feats(x, norm=True, scale=1):
    if scale != 1: x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    return enc(imnet(x) if norm else x)
V = {
 "last_mean": lambda x: feats(x)[-1].mean((2, 3)),
 "last_gem3": lambda x: feats(x)[-1].clamp_min(1e-6).pow(3).mean((2, 3)).pow(1 / 3),
 "last_max": lambda x: feats(x)[-1].amax((2, 3)),
 "stages345_mean": lambda x: torch.cat([f.mean((2, 3)) for f in feats(x)[3:]], 1),
 "stages2345_meanstd": lambda x: torch.cat([torch.cat([f.mean((2, 3)), f.std((2, 3))], 1) for f in feats(x)[2:]], 1),
 "last_mean_x2input": lambda x: feats(x, scale=2)[-1].mean((2, 3)),
 "last_mean_no_imnet_norm": lambda x: feats(x, norm=False)[-1].mean((2, 3)),
}
for harm in ("none", "hybrid"):
    for view in ("stack", "BSE"):
        ds = CropDataset(FieldStore(ev, "raw", harmonise=harm if harm != "none" else False), view, stride=CROP)
        meta, Yk, Yi = targets(ds)
        for name, fn in V.items():
            if view == "BSE" and name not in ("last_mean", "stages2345_meanstd"): continue
            t = time.time(); E = embed(ds, fn)
            rows.append(dict(model="ots_micronet", variant=f"{view}/{name}", harm=harm, **score(E, meta, Yk, Yi), sec=round(time.time() - t)))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT / "micronet" / "probe_variants.csv", index=False)
print("done")
