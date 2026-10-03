"""Figures for the report: per-site KPI strip plots + GP maps for selected sites."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from compare import KPI_INFO
d = pd.read_csv("site_kpis.csv"); B = sorted(d.batch.unique()); col = dict(zip(B, ["#2f6fdf", "#e08a1e", "#c23b3b", "#3a9a5b"]))
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for a, (k, (name, _)) in zip(ax.ravel(), KPI_INFO.items()):
    for i, b in enumerate(B):
        s = d[d.batch == b]; x = i + np.random.default_rng(i).uniform(-.12, .12, len(s))
        err = s.get(f"{k}__err")
        a.errorbar(x, s[k], yerr=None if err is None else 1.96 * err, fmt="o", color=col[b], alpha=.8, ms=5, capsize=2)
        a.hlines(s[k].mean(), i - .3, i + .3, color="k", lw=2)
        for _, r in s.iterrows():
            if r.site in ("5n1q8atc", "4ih2ggld"): a.annotate(r.site, (i + .14, r[k]), fontsize=7)
    a.set_xticks(range(len(B))); a.set_xticklabels(B); a.set_title(name, fontsize=10); a.grid(alpha=.3)
fig.suptitle("Each dot = one site (error bar = GP 95% for tile-KPIs). Black line = batch mean.", fontsize=11)
fig.tight_layout(); fig.savefig("fig_kpis.png", dpi=80)

def gpmap(b, s, k_idx, kname, ax3, title):
    z = np.load(f"tiles/{b}__{s}.npz"); X, y = z["X"], z["Y"][:, k_idx]
    mu0, sd0 = y.mean(), y.std(); k = C(1.0) * RBF(2.0, (0.5, 200)) + WhiteKernel(0.5, (1e-4, 2))
    gp = GaussianProcessRegressor(k, random_state=0).fit(X, (y - mu0) / sd0)
    yy, xx = np.mgrid[X[:, 0].min():X[:, 0].max():0.8, X[:, 1].min():X[:, 1].max():0.8]
    m, sd = gp.predict(np.c_[yy.ravel(), xx.ravel()], return_std=True)
    sd = np.sqrt(np.maximum(sd ** 2 - gp.kernel_.k2.noise_level, 0))
    ext = [0, z["lab"].shape[1] * 0.1, z["lab"].shape[0] * 0.1, 0]
    ax3[0].imshow(z["lab"], cmap=matplotlib.colors.ListedColormap(["#1f4fff", "#333333", "#ff9a00"]), extent=ext, interpolation="nearest")
    ax3[0].set_title(f"{title}: phases (blue pore, orange Si)", fontsize=9)
    im = ax3[1].imshow((mu0 + sd0 * m).reshape(yy.shape), cmap="magma", extent=ext, vmin=0); ax3[1].set_title(f"GP mean {kname}", fontsize=9); plt.colorbar(im, ax=ax3[1], fraction=.02)
    im = ax3[2].imshow((sd0 * sd).reshape(yy.shape), cmap="viridis", extent=ext); ax3[2].set_title(f"GP std {kname} (uncertainty)", fontsize=9); plt.colorbar(im, ax=ax3[2], fraction=.02)
    for a in ax3: a.set_xlabel("µm")

fig, ax = plt.subplots(3, 3, figsize=(20, 7.5))
gpmap("Batch_1", "5n1q8atc", 1, "Si fraction", ax[0], "Batch_1 5n1q8atc (flagged)")
gpmap("Batch_2", "epqdaau9", 1, "Si fraction", ax[1], "Batch_2 epqdaau9 (typical)")
gpmap("Batch_3", "hzumfsms", 0, "porosity", ax[2], "Batch_3 hzumfsms (largest pore patches)")
fig.tight_layout(); fig.savefig("fig_gpmaps.png", dpi=70)
