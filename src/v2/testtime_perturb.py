"""Test-time perturbation of saved classifiers: does Batch_3 recognition survive removing the LUT comb / adding noise / blur?"""
import sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
from src.v2 import attribution as A
A.device = lambda: torch.device("cpu")
torch.manual_seed(0)
def gauss_blur(x, sigma):
    r = int(3 * sigma); k = torch.exp(-(torch.arange(-r, r + 1).float() ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    k2 = (k[:, None] * k[None, :])[None, None].repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), k2, groups=x.shape[1])
P = {"none": lambda x: x,
     "dequant_1.5grey": lambda x: x + (torch.rand_like(x) - .5) * 3 / 255,
     "gauss_noise_3grey": lambda x: x + torch.randn_like(x) * 3 / 255,
     "gauss_noise_6grey": lambda x: x + torch.randn_like(x) * 6 / 255,
     "blur_0.7px": lambda x: gauss_blur(x, 0.7),
     "blur_1.2px": lambda x: gauss_blur(x, 1.2),
     "offset_+8grey": lambda x: (x + 8 / 255).clamp(0, 1),
     "gain_0.85": lambda x: x * 0.85}
rows = []
for h in sys.argv[1:]:
    d = Path("outputs/v2/cls_runs") / h
    model, c = A.load_run(d); te_f = A.test_fold(c)
    ds = A.CropDataset(A.FieldStore(te_f, c["input"], harmonise=c["harmonise"]), c["view"], stride=A.CROP)
    y = A._labels(te_f)[torch.tensor([i for i, _, _ in ds.index])].numpy()
    xs = torch.stack([ds[j]["x"] for j in range(len(ds))]); gid = np.array([ds.index[j][0] for j in range(len(ds))])
    fields = te_f.group_id.to_numpy()
    for name, fn in P.items():
        with torch.no_grad():
            p = torch.cat([A.predict(model, fn(xs[a:a + 64]).clamp(0, 1)) for a in range(0, len(xs), 64)]).numpy()
        fid = fields[gid] if gid.dtype.kind in "iu" else gid
        df = pd.DataFrame(p, columns=["p1", "p2", "p3"]); df["y"] = y; df["f"] = fid
        fm = df.groupby("f").mean(); pred = fm[["p1", "p2", "p3"]].to_numpy().argmax(1); yt = fm.y.round().astype(int).to_numpy()
        rec = {f"rec_B{b+1}": float((pred[yt == b] == b).mean()) if (yt == b).any() else np.nan for b in range(3)}
        rows.append(dict(run=h, arch=c["arch"], harm=str(c["harmonise"]), fold=c["fold"], perturb=name, field_acc=float((pred == yt).mean()), mean_p_true_B3=float(df[df.y == 2].p3.mean()), **rec))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("outputs/v2/cls_testtime_perturbation.csv", index=False)
