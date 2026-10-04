"""Sparse-autoencoder (SAE) analysis of a representation: which directions of the embedding are
material (KPI-aligned) and which are imaging (brightness / noise / sharpness / batch).

A TopK SAE (Gao et al. 2024: encoder ReLU + keep the k largest activations, tied-norm decoder) is
fitted on the standardised per-crop embedding of a run (``embeddings_eval.npz``, the non-overlapping
eval grid of all labelled fields). Every learned feature is then characterised by
  * how often it fires and on which batches (one-vs-rest AUC of its activation),
  * Spearman correlation with the gated KPIs of the crop (material) and with the per-crop imaging
    statistics p1 / noise / sharpness per detector (imaging),
  * its top-activating crops (gallery).
A feature is called *imaging* when the crops it fires on are, activation-weighted, > 0.5 SD away from
the average in some imaging statistic and that effect is larger than for any KPI; *material* in the
opposite case; *mixed/unexplained* otherwise; *rare* when it fires on < 10 crops. The share of
total activation mass in each class is the summary number; a representation that 'does not overfit
on imaging' should put most of its mass in material features and have few batch-selective ones.
Also works on classifier penultimate features (``--cls``): the out-of-fold features of the five
folds of a config are pooled first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.v2.common import OUT, load_half_raw, harm_method, CROP
from src.v2.evaluate import IMG_STATS
from src.v2.kpi_adapter import GATED_COLS
from src.v2.latent_audit import load, targets

BATCHES = ("Batch_1", "Batch_2", "Batch_3")
IMG_KEYS = ("p1", "noise_sigma", "sharpness")  # black level, noise, focus: the known SEM effects (p99 tracks Si content)


class TopKSAE(nn.Module):
    def __init__(self, d: int, m: int, k: int):
        super().__init__()
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.k = k
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.t())
            self.dec.weight.div_(self.dec.weight.norm(dim=0, keepdim=True))

    def encode(self, x):
        a = F.relu(self.enc(x - self.b_pre))
        v, i = a.topk(self.k, dim=1)
        return torch.zeros_like(a).scatter_(1, i, v)

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z) + self.b_pre, z


def fit_sae(X: np.ndarray, m: int, k: int, steps: int = 4000, lr: float = 2e-3, seed: int = 0) -> tuple[TopKSAE, dict]:
    torch.manual_seed(seed)
    x = torch.from_numpy(X).float()
    sae = TopKSAE(X.shape[1], m, k)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = len(x)
    for s in range(steps):
        idx = torch.randint(0, n, (min(256, n),))
        xb = x[idx]
        rec, z = sae(xb)
        loss = F.mse_loss(rec, xb)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sae.dec.weight.div_(sae.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-6))
    with torch.no_grad():
        rec, z = sae(x)
        fvu = float(((rec - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
    return sae, {"fvu": fvu, "r2_recon": 1 - fvu, "dead_features": int((z.abs().sum(0) == 0).sum())}


def characterise(Z: np.ndarray, meta: pd.DataFrame, T: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """Per feature: firing rate, batch one-vs-rest AUC, and the activation-weighted standardised mean
    of every target over the crops where the feature fires (an effect size in SD units: 'crops this
    feature selects have a Si fraction 1.2 SD above average'). Spearman over all crops is kept for
    reference but a sparse feature is characterised by what it selects, not by a global correlation."""
    y = meta.batch.map(BATCHES.index).to_numpy()
    img_cols = [f"{k}_c{c}" for k in IMG_KEYS for c in (0, 1, 2) if f"{k}_c{c}" in T]
    kpi_cols = [c for c in GATED_COLS if c in T]
    Ts = (T - T.mean()) / (T.std() + 1e-9)
    rows = []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        on = z > 0
        r = dict(feature=j, freq=float(on.mean()), mass=float(z.sum()))
        if on.sum() < 10:
            r["kind"] = "rare"; rows.append(r); continue
        for k, b in enumerate(BATCHES):
            r[f"auc_{b}"] = float(roc_auc_score(y == k, z))
        r["batch_selectivity"] = float(max(abs(r[f"auc_{b}"] - 0.5) for b in BATCHES) * 2)
        w = z[on] / z[on].sum()
        best_k, best_i = (None, 0.0), (None, 0.0)
        for c in kpi_cols + img_cols:
            v = Ts[c].to_numpy()[on]
            ok = ~np.isnan(v)
            d = float((w[ok] * v[ok]).sum() / w[ok].sum()) if ok.any() else 0.0
            r[f"d_{c}"] = d
            r[f"rho_{c}"] = float(spearmanr(z, T[c], nan_policy="omit").statistic)
            if c in kpi_cols and abs(d) > abs(best_k[1]):
                best_k = (c, d)
            if c in img_cols and abs(d) > abs(best_i[1]):
                best_i = (c, d)
        r["best_kpi"], r["best_kpi_d"] = best_k
        r["best_img"], r["best_img_d"] = best_i
        ak, ai = abs(best_k[1]), abs(best_i[1])
        r["kind"] = "imaging" if (ai > thr and ai > ak) else "material" if (ak > thr and ak >= ai) else "mixed"
        rows.append(r)
    return pd.DataFrame(rows)


def gallery(Z: np.ndarray, meta: pd.DataFrame, feats: pd.DataFrame, harm: str, path: Path, n_feat: int = 12, n_top: int = 6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = feats[feats.kind != "rare"].sort_values("mass", ascending=False).head(n_feat)
    fig, ax = plt.subplots(len(sel), n_top, figsize=(1.6 * n_top, 1.75 * len(sel)))
    cache = {}
    for r, (_, f) in zip(ax, sel.iterrows()):
        top = np.argsort(-Z[:, int(f.feature)])[:n_top]
        for a, i in zip(r, top):
            g, yy, xx = meta.group_id[i], int(meta.y[i]), int(meta.x[i])
            b, s = g.split("/")
            if g not in cache:
                cache[g] = load_half_raw(b, s, harm)[..., 0]
            a.imshow(cache[g][yy:yy + CROP, xx:xx + CROP], cmap="gray", vmin=0, vmax=255)
            a.set_xticks([]); a.set_yticks([]); a.set_title(b.replace("Batch_", "B"), fontsize=7)
        lab = f"f{int(f.feature)} {f.kind}\n{f.best_kpi} {f.best_kpi_d:+.2f}\n{f.best_img} {f.best_img_d:+.2f}\nB-sel {f.batch_selectivity:.2f}"
        r[0].set_ylabel(lab, fontsize=6.5)
    fig.suptitle(path.stem, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def analyse_embedding(E: np.ndarray, meta: pd.DataFrame, out: Path, harm: str, expansion: int = 8, k: int = 8, tag: str = ""):
    out.mkdir(parents=True, exist_ok=True)
    mu, sd = E.mean(0), E.std(0) + 1e-6
    X = (E - mu) / sd
    sae, fit = fit_sae(X, m=expansion * X.shape[1], k=k)
    with torch.no_grad():
        Z = sae.encode(torch.from_numpy(X).float()).numpy()
    kp, im = targets(meta)
    T = pd.concat([kp.reset_index(drop=True), im.reset_index(drop=True)], axis=1)
    feats = characterise(Z, meta, T)
    feats.to_csv(out / f"sae_features{tag}.csv", index=False)
    mass = feats.groupby("kind").mass.sum() / feats.mass.sum()
    summ = {**fit, "n_features": int(Z.shape[1]), "k": k, "d": int(E.shape[1]),
            **{f"mass_{kk}": float(v) for kk, v in mass.items()},
            **{f"n_{kk}": int(v) for kk, v in feats.kind.value_counts().items()},
            "n_batch_selective": int((feats.batch_selectivity > 0.5).sum()),
            "mass_batch_selective": float(feats.mass[feats.batch_selectivity > 0.5].sum() / feats.mass.sum()),
            "top_imaging": feats[feats.kind == "imaging"].sort_values("mass", ascending=False).head(5)[["feature", "best_img", "best_img_d", "mass"]].to_dict("records"),
            "top_material": feats[feats.kind == "material"].sort_values("mass", ascending=False).head(5)[["feature", "best_kpi", "best_kpi_d", "mass"]].to_dict("records")}
    (out / f"sae_summary{tag}.json").write_text(json.dumps(summ, indent=1, default=float))
    gallery(Z, meta, feats, harm, out / f"sae_gallery{tag}.png")
    return summ, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run hashes under outputs/v2/runs (or --cls hashes under cls_runs)")
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    rows = []
    for h in a.runs:
        d = OUT / "runs" / h
        c = json.loads((d / "config.json").read_text())
        E, meta = load(d)
        summ, _ = analyse_embedding(E, meta, OUT / "sae" / h, harm_method(c.get("harmonise")), a.expansion, a.k)
        rows.append({"hash": h, "family": c.get("family"), "harmonise": c.get("harmonise"), **{k: v for k, v in summ.items() if not isinstance(v, list)}})
        print(h, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}))
    pd.DataFrame(rows).to_csv(OUT / "sae" / "sae_runs.csv", index=False)


if __name__ == "__main__":
    main()
