"""Phase 3/5 selection metrics on held-out fields (non-overlapping 256-px eval grid).

1 KPI probe R^2 (higher better)          2 imaging-stat probe R^2 (lower better)
3 within-batch image-ID acc / chance      4 black-level-lift shift / between-field distance
5 VAE reconstruction KPI error            + kNN batch accuracy (reported, not selected on)
Probes: RidgeCV on standardised embeddings, leave-one-field-out over held-out fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.v2 import kpi_adapter as K
from src.v2.augment import black_level_lift
from src.v2.common import CROP, NM_HALF, OUT
from src.v2.data import CropDataset, FieldStore

IMG_STATS = ("p1", "p99", "dyn_range", "noise_sigma", "sharpness", "gmm_mu0", "gmm_mu1", "gmm_mu2")
LIFT = 7.0 / 255.0  # Batch_3 BSE black-level offset measured in Phase 0
ALPHAS = np.logspace(-2, 4, 13)


@torch.no_grad()
def embed_all(model, ds: CropDataset, dev, bs=128, transform=None, kpi_needed=False):
    out = []
    for i in range(0, len(ds), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        x = torch.stack([it["x"] for it in items]).to(dev)
        if transform is not None:
            x = transform(x)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            e = model.embed(x, torch.stack([it["kpi"] for it in items]).to(dev)) if kpi_needed else model.embed(x)
        out.append(e.float().cpu())
    return torch.cat(out).numpy()


def crop_meta(ds: CropDataset) -> pd.DataFrame:
    return pd.DataFrame({"group_id": ds.group, "batch": ds.batch, "y": [y for _, y, _ in ds.index],
                         "x": [x for _, _, x in ds.index]})


def probe_r2(E, Y: pd.DataFrame, groups) -> dict[str, float]:
    out = {}
    cv = LeaveOneGroupOut()
    for c in Y.columns:
        ok = Y[c].notna().to_numpy()
        if ok.sum() < 20 or len(np.unique(groups[ok])) < 2:
            continue
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        pred = cross_val_predict(pipe, E[ok], Y[c].to_numpy()[ok], groups=groups[ok], cv=cv)
        out[c] = float(r2_score(Y[c].to_numpy()[ok], pred))
    return out


def image_id_ratio(E, meta: pd.DataFrame, widths: dict[str, int]) -> float:
    """Train on left-half crops, test on right-half (crops crossing the midline dropped);
    accuracy / chance averaged over batches with >= 2 fields."""
    ratios = []
    for _, g in meta.groupby("batch"):
        if g.group_id.nunique() < 2:
            continue
        mid = g.group_id.map(widths) / 2
        left, right = (g.x + CROP <= mid).to_numpy(), (g.x >= mid).to_numpy()
        idx = g.index.to_numpy()
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(E[idx[left]], g.group_id.to_numpy()[left])
        acc = (clf.predict(E[idx[right]]) == g.group_id.to_numpy()[right]).mean()
        ratios.append(acc * g.group_id.nunique())
    return float(np.mean(ratios)) if ratios else np.nan


def knn_batch_acc(E, meta) -> float:
    pred = cross_val_predict(make_pipeline(StandardScaler(), KNeighborsClassifier(10)), E, meta.batch.to_numpy(),
                             groups=meta.group_id.to_numpy(), cv=LeaveOneGroupOut())
    return float((pred == meta.batch.to_numpy()).mean())


def lift_shift(E, E_lift, meta) -> float:
    shift = np.linalg.norm(E - E_lift, axis=1).mean()
    cent = pd.DataFrame(E).groupby(meta.group_id.to_numpy()).mean().to_numpy()
    d = np.linalg.norm(cent[:, None] - cent[None], axis=2)
    between = np.median(d[np.triu_indices(len(cent), 1)])
    return float(shift / between)


@torch.no_grad()
def ssim(a: np.ndarray, b: np.ndarray, sigma=1.5, L=1.0) -> float:
    """Gaussian-window SSIM (Wang et al. 2004) on one 2-D image pair in [0, L]."""
    from scipy.ndimage import gaussian_filter as g
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    ma, mb = g(a, sigma), g(b, sigma)
    va, vb, cab = g(a * a, sigma) - ma ** 2, g(b * b, sigma) - mb ** 2, g(a * b, sigma) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def psnr(a: np.ndarray, b: np.ndarray, L=1.0) -> float:
    return float(10 * np.log10(L ** 2 / max(float(np.mean((a - b) ** 2)), 1e-12)))


@torch.no_grad()
def recon_metrics(model, ds: CropDataset, dev, kpi_needed: bool, n_max=256) -> dict:
    """VAE reconstruction quality on held-out crops: PSNR/SSIM per input channel (mean over
    channels) and mean |standardised gated-KPI difference| between original and reconstructed
    BSE crops, both segmented with the teammate segmenter."""
    sel = np.linspace(0, len(ds) - 1, min(n_max, len(ds))).astype(int)
    a, b, ps, ss = [], [], [], []
    c0 = ds.ch.index(0) if 0 in ds.ch else None
    for j in sel:
        it = ds[int(j)]
        x = it["x"][None].to(dev)
        r = model.reconstruct(x, it["kpi"][None].to(dev)) if kpi_needed else model.reconstruct(x)
        xn, rn = x[0].float().cpu().numpy(), r[0].float().clamp(0, 1).cpu().numpy()
        ps.append(np.mean([psnr(xn[k], rn[k]) for k in range(xn.shape[0])]))
        ss.append(np.mean([ssim(xn[k], rn[k]) for k in range(xn.shape[0])]))
        if c0 is not None:
            for arr, store in ((xn, a), (rn, b)):
                store.append(K.kpis_from_masks(K.segment(arr[c0] * 255.0, NM_HALF), NM_HALF, "recon"))
    err = np.nan
    if c0 is not None:
        A, B = pd.DataFrame(a)[list(K.GATED_COLS)], pd.DataFrame(b)[list(K.GATED_COLS)]
        err = float(((A - B).abs() / A.std().replace(0, 1)).mean().mean())
    return dict(recon_kpi_err=err, psnr=float(np.mean(ps)), ssim=float(np.mean(ss)))


def recon_kpi_error(model, ds: CropDataset, dev, kpi_needed: bool, n_max=256) -> float:
    return recon_metrics(model, ds, dev, kpi_needed, n_max)["recon_kpi_err"]


def evaluate(model, store: FieldStore, view: str, dev, family: str, kpi_norm: dict | None = None,
             cond_cols=K.GATED_COLS) -> dict:
    kpis = K.crop_kpi_frame("eval", cond_cols)
    kpis_std = kpis.copy()
    if kpi_norm is not None:  # same standardisation as training (VAE-B/C inputs)
        for c in cond_cols:
            kpis_std[c] = (kpis[c] - kpi_norm["mean"][c]) / kpi_norm["std"][c]
    kpi_needed = family in ("vae_b", "vae_c")
    ds = CropDataset(store, view, stride=CROP, kpis=kpis_std if kpi_needed else None,
                     kpi_cols=cond_cols if kpi_needed else ())
    meta = crop_meta(ds)
    E = embed_all(model, ds, dev, kpi_needed=kpi_needed)
    E_lift = embed_all(model, ds, dev, transform=lambda x: black_level_lift(x, LIFT), kpi_needed=kpi_needed)
    groups = meta.group_id.to_numpy()

    Yk = meta.merge(kpis, on=["group_id", "y", "x"], how="left")[list(K.GATED_COLS)]
    ist = pd.read_csv(OUT / "imaging_stats" / "per_crop.csv")
    chans = sorted(set(ds.ch))
    wide = []
    for c in chans:
        s = ist[ist.channel == c][["group_id", "y_half", "x_half", *IMG_STATS]]
        s.columns = ["group_id", "y", "x", *[f"{k}_c{c}" for k in IMG_STATS]]
        wide.append(s)
    Yi = meta.copy()
    for s in wide:
        Yi = Yi.merge(s, on=["group_id", "y", "x"], how="left")
    Yi = Yi.drop(columns=["group_id", "batch", "y", "x"])

    kr, ir = probe_r2(E, Yk, groups), probe_r2(E, Yi, groups)
    widths = {g: im.shape[1] for g, im in zip(store.fields.group_id, store.images)}
    res = dict(kpi_r2=float(np.mean(list(kr.values()))), img_r2=float(np.mean(list(ir.values()))),
               image_id_ratio=image_id_ratio(E, meta, widths), lift_shift=lift_shift(E, E_lift, meta),
               knn_batch_acc=knn_batch_acc(E, meta), n_crops=len(ds), emb_dim=E.shape[1])
    res.update(recon_metrics(model, ds, dev, kpi_needed) if family.startswith("vae")
               else dict(recon_kpi_err=np.nan, psnr=np.nan, ssim=np.nan))
    res.update({f"kpi_r2__{k}": v for k, v in kr.items()})
    res.update({f"img_r2__{k}": v for k, v in ir.items()})
    return res, E, meta


SELECT = {"kpi_r2": False, "img_r2": True, "image_id_ratio": True, "lift_shift": True, "recon_kpi_err": True}


KPI_R2_FLOOR = 0.0
KPI_R2_REL_FLOOR = 0.75  # eligible runs keep >= 75% of the best KPI R^2 in the table


def selection_score(lb: pd.DataFrame) -> pd.Series:
    """Rank-average (1 = best) over available selection metrics; NaN metrics are skipped per row.
    The nuisance metrics only break ties among representations that carry the microstructure signal:
    runs with kpi_r2 <= 0 or below 75% of the table's best kpi_r2 rank below every eligible run
    (otherwise a weak-KPI embedding wins on being insensitive to everything)."""
    ranks = pd.DataFrame({m: lb[m].rank(ascending=asc) for m, asc in SELECT.items() if m in lb})
    score = ranks.mean(axis=1, skipna=True)
    if "kpi_r2" in lb:
        floor = max(KPI_R2_FLOOR, KPI_R2_REL_FLOOR * float(lb["kpi_r2"].max()))
        score = score + np.where(lb["kpi_r2"] > floor, 0.0, len(lb) + 1.0)
    return score
