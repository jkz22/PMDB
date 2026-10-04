"""2-D map of the 31 labelled fields + 3 held-out sites in the extreme VAE-C latent with the session-imaging
directions projected out (see latent_cluster.inlp). Writes outputs/v2/cluster/latent_map.png and
latent_coords.csv (PC1-5, 2-D embedding, batch, Si / K04 field means, nearest neighbours)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE

from src.v2.common import OUT, manifest
from src.v2.latent_cluster import CL, IMG, KP, RUNS, field_means, imaging_table, inlp
from src.v2.sae_ablate import HELDOUT, embed, heldout_crops, load_model

COL = {"Batch_1": "#1f77b4", "Batch_2": "#2ca02c", "Batch_3": "#7f7f7f", "heldout": "#d62728"}


def main(name="extreme VAE-C", stage="session"):
    h = RUNS[name]; d = OUT / "runs" / h
    cfg = json.loads((d / "config.json").read_text())
    z = np.load(d / "embeddings_eval.npz", allow_pickle=True)
    E, g = z["E"].astype(np.float64), z["group_id"].astype(str)
    img = imaging_table()
    meta = pd.DataFrame({"group_id": g, "y": z["y"], "x": z["x"]}).merge(img, on=["group_id", "y", "x"], how="left")
    Z = meta[[c for c in meta.columns if c.rsplit("_", 1)[0] in IMG]].to_numpy(); Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    kp = pd.read_csv(OUT / "kpis" / "crop_kpis_eval.csv").merge(meta[["group_id", "y", "x"]], on=["group_id", "y", "x"], how="right")
    Y = np.nan_to_num(kp[KP].to_numpy()); Ys = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    mu, sd = E.mean(0), E.std(0) + 1e-6; Es = (E - mu) / sd
    Zres = Zs - Ridge(alpha=1.0).fit(Ys, Zs).predict(Ys)
    Ec, P, trace = inlp(Es, Zres if stage == "session" else Zs, g)
    model = load_model(d, cfg); Xh, mh = heldout_crops(cfg)
    Eh = ((embed(model, Xh).astype(np.float64) - mu) / sd) @ P
    gh = ("Batch_heldout/" + mh.site).to_numpy()
    F = pd.concat([field_means(Ec, g), field_means(Eh, gh)])
    m = manifest(); batch_of = dict(zip(m.group_id, m.batch))
    b = np.array([batch_of.get(x, "heldout") for x in F.index])
    pca = PCA(n_components=5, random_state=0).fit(F.to_numpy()[b != "heldout"])
    PC = pca.transform(F.to_numpy())
    ts = TSNE(n_components=2, perplexity=8, init="pca", random_state=0, metric="cosine").fit_transform(F.to_numpy())
    # KPI field means for colouring / table
    kf = kp.groupby("group_id")[KP].mean()
    kh = pd.read_csv(OUT / "off" / "heldout_crop_kpis.csv").groupby("group_id")[KP].mean()
    kall = pd.concat([kf, kh]).reindex(F.index)
    Fn = F.to_numpy() / (np.linalg.norm(F.to_numpy(), axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T; np.fill_diagonal(S, -1)
    lab = b != "heldout"
    nn = []
    for i in range(len(F)):
        s = S[i].copy(); s[~lab] = -1
        j = np.argsort(-s)[:3]
        nn.append("; ".join(f"{F.index[k].split('/')[1]} ({batch_of[F.index[k]][-1]}, {s[k]:.2f})" for k in j))
    out = pd.DataFrame({"field": [x.split("/")[1] for x in F.index], "batch": b, "tsne1": ts[:, 0], "tsne2": ts[:, 1],
                        **{f"PC{k + 1}": PC[:, k] for k in range(5)}, "frac_si": kall.frac_si.to_numpy(), "K04_agglom_frac": kall.K04_agglom_frac.to_numpy(),
                        "nn3": nn})
    out.round(4).to_csv(CL / "latent_coords.csv", index=False)
    print("explained var PC1-5:", pca.explained_variance_ratio_.round(3), " inlp trace:", [round(t["img_r2"], 3) for t in trace])
    r = pd.DataFrame({f"PC{k + 1}": [np.corrcoef(PC[lab, k], kall[c].to_numpy()[lab])[0, 1] for c in KP] for k in range(5)}, index=KP)
    print("corr(PC, KPI):\n", r.round(2).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    for ax, (xx, yy, xl, yl) in zip(axes, ((PC[:, 0], PC[:, 1], f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"),
                                          (ts[:, 0], ts[:, 1], "t-SNE 1 (cosine)", "t-SNE 2"))):
        for bb in ("Batch_1", "Batch_2", "Batch_3"):
            mm = b == bb
            ax.scatter(xx[mm], yy[mm], s=90 + 1500 * kall.frac_si.to_numpy()[mm], c=COL[bb], alpha=0.75, edgecolor="k", label=f"{bb} (size = Si fraction)")
        for i in np.where(lab)[0]:
            ax.annotate(F.index[i].split("/")[1], (xx[i], yy[i]), fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
        for i in np.where(~lab)[0]:
            ax.scatter(xx[i], yy[i], s=260, marker="*", c=COL["heldout"], edgecolor="k", zorder=5)
            ax.annotate(F.index[i].split("/")[1], (xx[i], yy[i]), fontsize=10, fontweight="bold", color=COL["heldout"], xytext=(6, -12), textcoords="offset points")
            s = S[i].copy(); s[~lab] = -1
            for k in np.argsort(-s)[:3]:
                ax.plot([xx[i], xx[k]], [yy[i], yy[k]], c=COL["heldout"], lw=1.2, alpha=0.6)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{name} latent, {'session-' if stage == 'session' else 'all '}imaging directions projected out ({int(round(Es.shape[1] - np.trace(P)))} dims removed; "
                 f"imaging probe R2 {trace[0]['img_r2']:.2f} -> {trace[-1]['img_r2']:.2f}). Field means; stars = held-out, lines = 3 nearest labelled fields")
    fig.tight_layout(); fig.savefig(CL / "latent_map.png", dpi=130)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(out[["field", "batch", "PC1", "PC2", "tsne1", "tsne2", "frac_si", "nn3"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
