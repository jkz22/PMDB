"""Does the noise/sharpness cue survive each route? Stats on the models' own input crops (BSE channel), field means, LOFO logistic."""
import numpy as np, pandas as pd, torch
from scipy.ndimage import laplace, convolve, gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from src.v2.data import FieldStore, CropDataset
from src.v2.common import CROP, manifest
from src.v2.injection_ablation import fingerprint
torch.set_num_threads(2)
m = manifest()
ROUTES = {"raw": ("raw", False), "hybrid": ("raw", "hybrid"), "nyul": ("raw", "nyul"), "basic": ("raw", "basic"), "clean_harm": ("raw", "clean_harm"), "naive": ("naive", False), "extreme": ("extreme", False)}
rows = []
for route, (inp, harm) in ROUTES.items():
    try:
        ds = CropDataset(FieldStore(m, inp, harmonise=harm), "stack", stride=CROP)
    except Exception as e:
        print(route, "skipped", e); continue
    for j in range(len(ds)):
        fi = ds.index[j][0]; x = ds[j]["x"][0].numpy() * 255.0
        rows.append(dict(route=route, group_id=m.group_id.iloc[fi], batch=m.batch.iloc[fi], **fingerprint(x)))
    print(route, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv("outputs/v2/off/route_cue_crops.csv", index=False)
F = D.groupby(["route", "group_id"]).mean(numeric_only=True).reset_index(); F["batch"] = F.group_id.str.split("/").str[0]
out = []
for route, g in F.groupby("route"):
    y = (g.batch != "Batch_3").astype(int).to_numpy()
    for name, cols in {"noise+sharp": ["noise_sigma", "sharpness"], "p1+p99": ["p1", "p99"], "all4": ["p1", "p99", "noise_sigma", "sharpness"]}.items():
        X = g[cols].to_numpy(); p = np.zeros(len(y))
        for i in range(len(y)):
            mk = np.arange(len(y)) != i
            p[i] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[mk], y[mk]).predict_proba(X[i:i + 1])[0, 1]
        pred = p > .5
        out.append(dict(route=route, cue=name, field_acc=(pred == y).mean(), auroc=roc_auc_score(y, p),
                        rec_B3=float((~pred)[y == 0].mean()), rec_off=float(pred[y == 1].mean()),
                        d_noise_B3_vs_12=(g.noise_sigma[g.batch == "Batch_3"].mean() - g.noise_sigma[g.batch != "Batch_3"].mean()) / g.noise_sigma.std(),
                        d_sharp_B3_vs_12=(g.sharpness[g.batch == "Batch_3"].mean() - g.sharpness[g.batch != "Batch_3"].mean()) / g.sharpness.std()))
R = pd.DataFrame(out); R.to_csv("outputs/v2/off/route_cue_lofo.csv", index=False)
pd.set_option("display.width", 250); print(R.round(2).to_string())
