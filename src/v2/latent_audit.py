"""Phase 5 (field-level inference) + Phase 6 (latent confounder audit) on saved embeddings.

Input: <run>/embeddings_eval.npz (all 31 fields, eval grid). All probes are grouped by field
(leave-one-field-out), so no crop of a test field is ever seen in training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.v2.common import OUT
from src.v2.evaluate import IMG_STATS, image_id_ratio, probe_r2
from src.v2.kpi_adapter import GATED_COLS

BASELINE = "Batch_1"
IMG_R2_LIMIT, IMAGE_ID_LIMIT = 0.5, 2.0  # brief: round 2 triggers


def load(run_dir: Path):
    z = np.load(run_dir / "embeddings_eval.npz", allow_pickle=True)
    meta = pd.DataFrame({k: z[k] for k in ("group_id", "batch", "y", "x")})
    return z["E"], meta


def targets(meta: pd.DataFrame, chans=(0, 1, 2)):
    kp = meta.merge(pd.read_csv(OUT / "kpis" / "crop_kpis_eval.csv"), on=["group_id", "y", "x"], how="left")
    ist = pd.read_csv(OUT / "imaging_stats" / "per_crop.csv")
    im = meta[["group_id", "y", "x"]].copy()
    for c in chans:
        s = ist[ist.channel == c][["group_id", "y_half", "x_half", *IMG_STATS]]
        s.columns = ["group_id", "y", "x", *[f"{k}_c{c}" for k in IMG_STATS]]
        im = im.merge(s, on=["group_id", "y", "x"], how="left")
    return kp[list(GATED_COLS)], im.drop(columns=["group_id", "y", "x"])


def lobo_knn_batch(E, meta):
    """Leave-one-batch-out is undefined for batch prediction; report grouped-by-field kNN batch
    accuracy plus per-held-out-batch accuracy of a field-grouped 5-fold classifier."""
    g = meta.group_id.to_numpy()
    pred = cross_val_predict(make_pipeline(StandardScaler(), KNeighborsClassifier(10)), E, meta.batch,
                             groups=g, cv=GroupKFold(5))
    acc = (pred == meta.batch.to_numpy())
    return float(acc.mean()), meta.assign(ok=acc).groupby("batch").ok.mean().to_dict()


def field_inference(E, meta, n_perm=2000, n_boot=2000, seed=0):
    """Image-level (field-level) inference: each field is one unit (crops aggregated to the mean).
    Statistic = distance between batch centroid and baseline centroid. p from permuting field labels;
    CI from bootstrapping fields within batch. Conformal novelty: per-field score = distance to
    baseline centroid; p = rank among leave-one-out baseline scores."""
    rng = np.random.default_rng(seed)
    Z = StandardScaler().fit_transform(E)
    F = pd.DataFrame(Z).groupby(meta.group_id.to_numpy()).mean()
    fb = meta.drop_duplicates("group_id").set_index("group_id").batch.loc[F.index]
    X, b = F.to_numpy(), fb.to_numpy()
    base = X[b == BASELINE]
    out = {}
    for B in sorted(set(b) - {BASELINE}):
        idx = np.where((b == BASELINE) | (b == B))[0]
        lab = b[idx] == B
        stat = lambda L: np.linalg.norm(X[idx][L].mean(0) - X[idx][~L].mean(0))  # noqa: E731
        obs = stat(lab)
        perm = np.array([stat(rng.permutation(lab)) for _ in range(n_perm)])
        boot = []
        xb, xB = X[b == BASELINE], X[b == B]
        for _ in range(n_boot):
            boot.append(np.linalg.norm(xB[rng.integers(0, len(xB), len(xB))].mean(0) - xb[rng.integers(0, len(xb), len(xb))].mean(0)))
        out[B] = dict(dist=float(obs), p_perm=float((1 + (perm >= obs).sum()) / (1 + n_perm)),
                      ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], n_fields=int(lab.sum()))
    loo = np.array([np.linalg.norm(base[i] - np.delete(base, i, 0).mean(0)) for i in range(len(base))])
    c = base.mean(0)
    conf = {gid: float((1 + (loo >= np.linalg.norm(X[i] - c)).sum()) / (1 + len(loo)))
            for i, gid in enumerate(F.index) if b[i] != BASELINE}
    return out, conf


def pc_correlations(E, Yk, Yi, meta, n_pc=10):
    P = PCA(n_pc, random_state=0).fit_transform(StandardScaler().fit_transform(E))
    V = pd.concat([Yk.reset_index(drop=True), Yi.reset_index(drop=True),
                   pd.get_dummies(meta.batch).astype(float).reset_index(drop=True)], axis=1)
    return pd.DataFrame({f"PC{i + 1}": V.corrwith(pd.Series(P[:, i]), method="spearman") for i in range(n_pc)})


def umap_plots(E, Yk, Yi, meta, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    U = umap.UMAP(random_state=0, n_neighbors=30).fit_transform(StandardScaler().fit_transform(E))
    cols = {"batch": meta.batch.astype("category").cat.codes, "field": meta.group_id.astype("category").cat.codes,
            **{c: Yk[c] for c in Yk}, **{c: Yi[c] for c in Yi if c.split("_c")[0] in ("p1", "dyn_range", "noise_sigma", "sharpness", "gmm_mu1")}}
    n = len(cols); nc = 5; nr = int(np.ceil(n / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(3.2 * nc, 3 * nr))
    for a, (k, v) in zip(ax.flat, cols.items()):
        a.scatter(U[:, 0], U[:, 1], c=v, s=2, cmap="tab20" if k == "field" else "viridis"); a.set_title(k, fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    for a in ax.flat[n:]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def active_units(E, thr=0.01):
    return int((E.var(0) > thr).sum())


def audit(run_dir: Path, out_dir: Path, umap=True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(run_dir / "config.json"))
    from src.v2.data import VIEWS
    chans = tuple(sorted(set(VIEWS[cfg["view"]])))
    E, meta = load(run_dir)
    Yk, Yi = targets(meta, chans)
    g = meta.group_id.to_numpy()
    kr, ir = probe_r2(E, Yk, g), probe_r2(E, Yi, g)
    # batch probe (grouped by field) and field-ID (left/right halves, within batch)
    pb = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), E, meta.batch,
                           groups=g, cv=LeaveOneGroupOut())
    from src.v2.data import heldout_split
    w = heldout_split().set_index("group_id").width.to_dict()
    knn_acc, knn_by = lobo_knn_batch(E, meta)
    fi, conf = field_inference(E, meta)
    res = dict(hash=run_dir.name, family=cfg["family"], view=cfg["view"], kpi_r2=kr, img_r2=ir,
               img_r2_max=max(ir.values()), img_r2_mean=float(np.mean(list(ir.values()))),
               batch_probe_acc=float((pb == meta.batch.to_numpy()).mean()),
               image_id_ratio=image_id_ratio(E, meta.reset_index(drop=True), w),
               knn_batch_acc=knn_acc, knn_batch_by_batch=knn_by, field_inference=fi, conformal_p=conf,
               active_units=active_units(E) if cfg["family"].startswith("vae") else None)
    res["round2_trigger"] = bool(res["img_r2_max"] > IMG_R2_LIMIT or res["image_id_ratio"] > IMAGE_ID_LIMIT)
    pc_correlations(E, Yk, Yi, meta).round(3).to_csv(out_dir / "pc_correlations.csv")
    if umap:
        umap_plots(E, Yk, Yi, meta, out_dir / "umap.png")
    (out_dir / "audit.json").write_text(json.dumps(res, indent=2, default=float))
    return res
