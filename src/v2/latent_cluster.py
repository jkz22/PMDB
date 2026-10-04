"""Unsupervised view: VAE latents of all 31 labelled fields, imaging directions projected out, clustering, and
nearest-neighbour placement of the three held-out sites.

The batches are known to be crops of larger original images grouped by some calculated/observed feature, so
instead of a supervised batch classifier we ask: which fields does each held-out site resemble, in a latent
that (a) was trained without labels and (b) has had every linearly-decodable imaging direction (p1, p99,
noise sigma, sharpness on all three detectors, raw-image statistics) removed by iterative null-space
projection (INLP). The KPI space (5 gated KPIs, crop means) is run through the same pipeline as the
pure-material reference.
Outputs under outputs/v2/cluster/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_mutual_info_score, r2_score
from sklearn.mixture import GaussianMixture

from src.v2.common import OUT, REPO, manifest
from src.v2.sae_ablate import HELDOUT, embed, heldout_crops, load_model

torch.set_num_threads(2)
CL = OUT / "cluster"
RUNS = {"extreme VAE-C": "392f6eecf87f", "extreme VAE-C phase-inpaint": "2474181311ea", "hybrid VAE-C": "863083fb14d9", "raw VAE-C": "489232073a90"}
KP = ["frac_si", "frac_graphite", "frac_pore", "K01_si_frac_adm", "K04_agglom_frac"]
IMG = ["p1", "p99", "noise_sigma", "sharpness"]


def imaging_table() -> pd.DataFrame:
    c = pd.read_csv(OUT / "imaging_stats" / "per_crop.csv")
    w = c.pivot_table(index=["group_id", "y_half", "x_half"], columns="channel", values=IMG)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.reset_index().rename(columns={"y_half": "y", "x_half": "x"})


def heldout_imaging() -> pd.DataFrame:
    rows = []
    for s in HELDOUT:
        im = np.load(REPO / "cache_heldout" / "half" / f"Batch_heldout__{s}.npz")["image"]
        from src.v2.common import CROP, grid
        from src.v2.sae_ablate import crop_imaging_stats
        for y, x in grid(*im.shape[:2], CROP, CROP):
            r = dict(group_id=f"Batch_heldout/{s}", y=y, x=x)
            for ch in range(3):
                st = crop_imaging_stats(im[y:y + CROP, x:x + CROP, ch])
                r.update({f"{k}_{ch}": st[k] for k in IMG})
            rows.append(r)
    return pd.DataFrame(rows)


def lofo_r2(E: np.ndarray, Y: np.ndarray, groups: np.ndarray, alpha=10.0) -> float:
    """Leave-one-field-out ridge probe, pooled R^2 over targets (standardised)."""
    pred = np.zeros_like(Y)
    for g in np.unique(groups):
        m = groups == g
        pred[m] = Ridge(alpha=alpha).fit(E[~m], Y[~m]).predict(E[m])
    return float(np.mean([r2_score(Y[:, j], pred[:, j]) for j in range(Y.shape[1])]))


def lofo_knn_batch(E: np.ndarray, batch: np.ndarray, groups: np.ndarray, k=20) -> float:
    """Field-level batch accuracy from crop-level kNN votes, leave-one-field-out."""
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    correct = []
    for g in np.unique(groups):
        m = groups == g
        S = En[m] @ En[~m].T
        nb = batch[~m][np.argsort(-S, axis=1)[:, :k]]
        votes = pd.Series(nb.ravel()).value_counts()
        correct.append(votes.idxmax() == batch[m][0])
    return float(np.mean(correct))


def inlp(E: np.ndarray, Z: np.ndarray, groups: np.ndarray, max_iter=12, tol=0.05) -> tuple[np.ndarray, np.ndarray, list]:
    """Iteratively remove the ridge-regression directions that predict the imaging stats Z until the LOFO
    probe R^2 for Z is <= tol. Returns (projected E, projection matrix P, trace)."""
    P = np.eye(E.shape[1])
    trace = []
    for it in range(max_iter):
        Ep = E @ P
        r2 = lofo_r2(Ep, Z, groups)
        trace.append(dict(iter=it, img_r2=r2))
        if r2 <= tol:
            break
        W = Ridge(alpha=10.0).fit(Ep, Z).coef_.T  # d x nz
        Q, _ = np.linalg.qr(W)
        P = P @ (np.eye(E.shape[1]) - Q @ Q.T)
    return E @ P, P, trace


def field_means(E: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(E).groupby(groups).mean()


def cluster_report(F: pd.DataFrame, batch_of: dict, tag: str) -> tuple[dict, pd.DataFrame]:
    X = PCA(n_components=min(10, F.shape[1], len(F) - 1), random_state=0).fit_transform(F.to_numpy())
    b = np.array([batch_of.get(g, "heldout") for g in F.index])
    lab = b != "heldout"
    bic = {k: GaussianMixture(k, covariance_type="full", reg_covar=1e-3, random_state=0, n_init=5).fit(X[lab]).bic(X[lab]) for k in range(1, 6)}
    kbest = min(bic, key=bic.get)
    gm = GaussianMixture(max(kbest, 2), covariance_type="full", reg_covar=1e-3, random_state=0, n_init=5).fit(X[lab])
    km3 = KMeans(3, n_init=20, random_state=0).fit(X[lab])
    km2 = KMeans(2, n_init=20, random_state=0).fit(X[lab])
    out = pd.DataFrame({"field": F.index, "batch": b, "gmm": gm.predict(X), "kmeans3": km3.predict(X), "kmeans2": km2.predict(X)})
    summ = dict(tag=tag, gmm_bic_best_k=kbest, ami_gmm=adjusted_mutual_info_score(b[lab], out.gmm[lab]),
                ami_kmeans3=adjusted_mutual_info_score(b[lab], out.kmeans3[lab]), ami_kmeans2=adjusted_mutual_info_score(b[lab], out.kmeans2[lab]),
                ami_kmeans2_vs_off=adjusted_mutual_info_score(b[lab] == "Batch_3", out.kmeans2[lab]))
    return summ, out


def neighbours(F: pd.DataFrame, Ec: np.ndarray, gc: np.ndarray, Eh: np.ndarray, gh: np.ndarray, batch_of: dict, k_field=5, k_crop=20) -> list[dict]:
    Fn = F.to_numpy(); Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    idx_lab = np.array([g in batch_of for g in F.index])
    Ecn = Ec / (np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-8)
    rows = []
    for s in HELDOUT:
        g = f"Batch_heldout/{s}"
        i = list(F.index).index(g)
        sims = Fn[idx_lab] @ Fn[i]
        order = np.argsort(-sims)[:k_field]
        labs = F.index[idx_lab]
        nn = [(labs[j].split("/")[1], labs[j].split("/")[0], float(sims[j])) for j in order]
        # crop-level kNN vote
        m = gh == g
        Ehn = Eh[m] / (np.linalg.norm(Eh[m], axis=1, keepdims=True) + 1e-8)
        S = Ehn @ Ecn.T
        nb = np.array([batch_of[x] for x in gc])[np.argsort(-S, axis=1)[:, :k_crop]]
        share = pd.Series(nb.ravel()).value_counts(normalize=True)
        # distance to each batch centroid (field means, standardised) in units of that batch's within spread
        cent = {b: Fn[[j for j, x in enumerate(F.index) if batch_of.get(x) == b]].mean(0) for b in ("Batch_1", "Batch_2", "Batch_3")}
        rows.append(dict(site=s, nn_fields="; ".join(f"{a} ({b.replace('Batch_', 'B')}, cos {c:.2f})" for a, b, c in nn),
                         nn_field_share_B3=float(np.mean([b == "Batch_3" for _, b, _ in nn])),
                         crop_knn_share_B1=float(share.get("Batch_1", 0)), crop_knn_share_B2=float(share.get("Batch_2", 0)), crop_knn_share_B3=float(share.get("Batch_3", 0)),
                         **{f"cos_centroid_{b[-1]}": float(Fn[i] @ cent[b] / (np.linalg.norm(cent[b]) + 1e-8)) for b in cent}))
    return rows


def run_latent(name: str, h: str, img: pd.DataFrame, imgh: pd.DataFrame, batch_of: dict) -> tuple[list, list, pd.DataFrame]:
    d = OUT / "runs" / h
    cfg = json.loads((d / "config.json").read_text())
    z = np.load(d / "embeddings_eval.npz", allow_pickle=True)
    E, g = z["E"].astype(np.float64), z["group_id"].astype(str)
    meta = pd.DataFrame({"group_id": g, "y": z["y"], "x": z["x"]}).merge(img, on=["group_id", "y", "x"], how="left")
    Z = meta[[c for c in meta.columns if c.split("_")[0] in IMG or c.rsplit("_", 1)[0] in IMG]].to_numpy()
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    kp = pd.read_csv(OUT / "kpis" / "crop_kpis_eval.csv").merge(meta[["group_id", "y", "x"]], on=["group_id", "y", "x"], how="right")
    Y = kp[KP].to_numpy(); Ys = (Y - np.nanmean(Y, 0)) / (np.nanstd(Y, 0) + 1e-8); Ys = np.nan_to_num(Ys)
    batch = np.array([batch_of[x] for x in g])
    mu, sd = E.mean(0), E.std(0) + 1e-6
    Es = (E - mu) / sd
    # held-out
    model = load_model(d, cfg)
    Xh, mh = heldout_crops(cfg)
    Eh = (embed(model, Xh).astype(np.float64) - mu) / sd
    gh = ("Batch_heldout/" + mh.site).to_numpy()
    summ, clus = [], []
    # session-imaging = the part of the imaging stats that the KPIs cannot explain (Si-rich crops are legitimately
    # brighter/sharper; that part is material and is kept)
    Zres = Zs - Ridge(alpha=1.0).fit(Ys, Zs).predict(Ys)
    for stage in ("as trained", "session-imaging projected out", "all imaging projected out"):
        Ec, Ehc, P, trace = Es, Eh, None, None
        if stage != "as trained":
            Ec, P, trace = inlp(Es, Zres if stage.startswith("session") else Zs, g)
            Ehc = Eh @ P
        s = dict(latent=name, stage=stage, dim_removed=int(round(Es.shape[1] - np.trace(P))) if P is not None else 0,
                 kpi_r2=lofo_r2(Ec, Ys, g), img_r2=lofo_r2(Ec, Zs, g), session_img_r2=lofo_r2(Ec, Zres, g), knn_batch_field_acc=lofo_knn_batch(Ec, batch, g),
                 knn_off_field_acc=lofo_knn_batch(Ec, np.where(batch == "Batch_3", "Batch_3", "Batch_1+2"), g))
        F = pd.concat([field_means(Ec, g), field_means(Ehc, gh)])
        cs, out = cluster_report(F, batch_of, f"{name} | {stage}")
        s.update({k: v for k, v in cs.items() if k != "tag"})
        for r in neighbours(F, Ec, g, Ehc, gh, batch_of):
            summ.append({**s, **r})
        out["latent"], out["stage"] = name, stage
        clus.append(out)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
        if trace:
            print("  inlp trace:", [round(t["img_r2"], 3) for t in trace], flush=True)
    return summ, clus, meta


def run_kpi_space(batch_of: dict) -> tuple[list, list]:
    kp = pd.read_csv(OUT / "kpis" / "crop_kpis_eval.csv")
    kh = pd.read_csv(OUT / "off" / "heldout_crop_kpis.csv")
    allk = pd.concat([kp[["group_id"] + KP], kh[["group_id"] + KP]])
    mu, sd = kp[KP].mean(), kp[KP].std() + 1e-8
    Z = ((allk[KP] - mu) / sd).to_numpy(); g = allk.group_id.to_numpy()
    lab = np.array([x in batch_of for x in g])
    F = pd.concat([field_means(Z[lab], g[lab]), field_means(Z[~lab], g[~lab])])
    cs, out = cluster_report(F, batch_of, "KPI space")
    batch = np.array([batch_of[x] for x in g[lab]])
    s = dict(latent="KPI space (5 gated KPIs)", stage="as is", dim_removed=0, kpi_r2=1.0, img_r2=np.nan,
             knn_batch_field_acc=lofo_knn_batch(Z[lab], batch, g[lab]),
             knn_off_field_acc=lofo_knn_batch(Z[lab], np.where(batch == "Batch_3", "Batch_3", "Batch_1+2"), g[lab]))
    s.update({k: v for k, v in cs.items() if k != "tag"})
    rows = [{**s, **r} for r in neighbours(F, Z[lab], g[lab], Z[~lab], g[~lab], batch_of)]
    out["latent"], out["stage"] = "KPI space", "as is"
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    return rows, [out]


def main(names=None):
    CL.mkdir(parents=True, exist_ok=True)
    m = manifest(); batch_of = dict(zip(m.group_id, m.batch))
    img = imaging_table(); imgh = heldout_imaging()
    summ, clus = run_kpi_space(batch_of)
    for name, h in RUNS.items():
        if names and name not in names:
            continue
        if not (OUT / "runs" / h / "embeddings_eval.npz").exists():
            print("missing", name, h); continue
        s, c, _ = run_latent(name, h, img, imgh, batch_of)
        summ += s; clus += c
        pd.DataFrame(summ).to_csv(CL / "summary.csv", index=False)
        pd.concat(clus).to_csv(CL / "clusters.csv", index=False)
    S = pd.DataFrame(summ)
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 120)
    print(S[["latent", "stage", "dim_removed", "kpi_r2", "img_r2", "session_img_r2", "knn_batch_field_acc", "knn_off_field_acc", "gmm_bic_best_k", "ami_gmm", "ami_kmeans3", "ami_kmeans2_vs_off"]].drop_duplicates().round(3).to_string())
    print(S[["latent", "stage", "site", "crop_knn_share_B1", "crop_knn_share_B2", "crop_knn_share_B3", "nn_fields"]].round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or None)
