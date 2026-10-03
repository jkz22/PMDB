"""Figures for the report: per-site KPI strip plots + GP maps for selected sites."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from compare import KPI_INFO
d = pd.read_csv("site_kpis.csv"); B = sorted(d.batch.unique()); col = dict(zip(B, ["#2f6fdf", "#e08a1e", "#c23b3b", "#3a9a5b"]))
NC = 4; NR = int(np.ceil(len(KPI_INFO) / NC))
fig, ax = plt.subplots(NR, NC, figsize=(4.6 * NC, 3.2 * NR))
for a, (k, (name, _)) in zip(ax.ravel(), KPI_INFO.items()):
    for i, b in enumerate(B):
        s = d[d.batch == b]; x = i + np.random.default_rng(i).uniform(-.12, .12, len(s))
        err = s.get(f"{k}__err")
        a.errorbar(x, s[k], yerr=None if err is None else 1.96 * err, fmt="o", color=col[b], alpha=.8, ms=5, capsize=2)
        a.hlines(s[k].mean(), i - .3, i + .3, color="k", lw=2)
        for _, r in s.iterrows():
            if r.site in ("5n1q8atc", "4ih2ggld"): a.annotate(r.site, (i + .14, r[k]), fontsize=7)
    a.set_xticks(range(len(B))); a.set_xticklabels(B); a.set_title(name, fontsize=9); a.grid(alpha=.3)
for a in ax.ravel()[len(KPI_INFO):]: a.axis("off")
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


# --- particle distributions pooled per batch (size, aspect ratio, circularity)
P = {b: {k: [] for k in ("p_d", "p_ar", "p_circ")} for b in B}
for _, r in d.iterrows():
    z = np.load(f"tiles/{r.batch}__{r.site}.npz")
    for k in P[r.batch]:
        if k in z: P[r.batch][k].append(z[k])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for a, (k, lab, bins) in zip(ax, [("p_d", "Si equivalent diameter (µm, log scale)", np.logspace(0, np.log10(25), 30)),
                                   ("p_ar", "Si aspect ratio", np.linspace(1, 5, 33)),
                                   ("p_circ", "Si circularity", np.linspace(0, 1, 33))]):
    for b in B:
        v = np.concatenate(P[b][k]); a.hist(v, bins=bins, density=True, histtype="step", lw=2, color=col[b], label=f"{b} (n={len(v)})")
    a.set_xlabel(lab); a.set_ylabel("density"); a.grid(alpha=.3); a.legend(fontsize=8)
ax[0].set_xscale("log")
fig.suptitle("Si particle distributions, all particles ≥1 µm pooled per batch", fontsize=11)
fig.tight_layout(); fig.savefig("fig_psd.png", dpi=80)

# --- examples of particles flagged cracked/irregular and anomalous, for checking by eye
from pmdb.io import load_site
ex = []
for _, r in d.sort_values("si_cracked_frac", ascending=False).iterrows():
    z = np.load(f"tiles/{r.batch}__{r.site}.npz")
    if "p_cracked" not in z: continue
    for kind, m in (("cracked", z["p_cracked"]), ("anomalous", z["p_anom"])):
        for i in np.where(m)[0][:1]: ex.append((r.batch, r.site, kind, z["p_bbox"][i]))
cr = [e for e in ex if e[2] == "cracked"][:6]; an = [e for e in ex if e[2] == "anomalous"][:6]
sel = cr + an
if sel:
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.4))
    for a in ax.ravel(): a.axis("off")
    for a, (b, s_, kind, bb) in zip(list(ax[0][:len(cr)]) + list(ax[1][:len(an)]), sel):
        img = load_site(b, s_, resolution="half", normalise="none").image[..., 0]
        r0, c0, r1, c1 = bb; pad = 20
        a.imshow(img[max(r0 - pad, 0):r1 + pad, max(c0 - pad, 0):c1 + pad], cmap="gray", vmin=0, vmax=255)
        a.set_title(f"{kind}: {b} {s_}", fontsize=8)
    fig.suptitle("Top row: particles flagged cracked/irregular (internal gaps ≥2% or solidity <0.8). "
                 "Bottom row: anomalous (>8 µm or aspect ratio >3). BSE crops.", fontsize=10)
    fig.tight_layout(); fig.savefig("fig_particles.png", dpi=75)
