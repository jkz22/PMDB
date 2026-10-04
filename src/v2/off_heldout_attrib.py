"""Reasoning of the binary off-detector on the 3 held-out sites: fold-ensemble P(off), occlusion-patch and
Grad-CAM maps (averaged over the 5 fold models) projected on the Si / graphite / pore segmentation and
correlated with local brightness / contrast / edge density / high-frequency noise. Writes
outputs/v2/off/heldout_attrib/<route>_<site>.png galleries, summary.png and summary.csv."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.v2 import kpi_adapter as K
from src.v2.attribution import gradcam, local_stats, occlusion, phase_share, upsample
from src.v2.classify import Classifier, classes_of
from src.v2.common import CROP, NM_HALF, OUT, grid
from src.v2.off_explain import load_binary_runs, route_of
from src.v2.sae_ablate import HELDOUT, _route_array, heldout_crops

TRUTH = {"3e122cbj": "Batch_2 (off)", "fn0mhxef": "Batch_1 (off)", "xrv9xvzb": "Batch_3 (baseline)"}
OUTD = OUT / "off" / "heldout_attrib"
N_ATTR = 10
N_SHOW = 4


def load_models(route: str, arch: str):
    runs = load_binary_runs()
    folds = [f for f in next(v for k, v in runs.items() if k == (route, arch)) if (f[2] / "final.pt").exists() and not f[0].get("exclude")]
    assert len(folds) == 5, [f[2].name for f in folds]
    models = []
    for c, _, d in folds:
        m = Classifier(c["arch"], n_cls=len(classes_of(c))); m.load_state_dict(torch.load(d / "final.pt", map_location="cpu"))
        models.append(m.eval())
    return models, folds[0][0]


class Ens(torch.nn.Module):
    """Fold ensemble presented as one model (mean logits) so occlusion/Grad-CAM run once per crop."""
    def __init__(self, models):
        super().__init__(); self.ms = torch.nn.ModuleList(models); self.arch = models[0].arch
        self.m = models[0].m  # gradcam hooks layer4 of the first fold model; CAMs of the others are averaged below
    def forward(self, x):
        return torch.stack([m(x) for m in self.ms]).mean(0)


def cam_ens(models, x, cls):
    return np.mean([gradcam(m, x, cls) for m in models], 0)


def main(route="extreme", arch="resnet18_imnet"):
    OUTD.mkdir(parents=True, exist_ok=True)
    models, cfg = load_models(route, arch)
    ens = Ens(models)
    print("models loaded", flush=True)
    X, meta = heldout_crops(cfg)
    print("crops", X.shape, flush=True)
    OFF = 0  # class order ("Batch_1+2", "Batch_3")
    with torch.no_grad():
        P = torch.cat([torch.softmax(ens(torch.from_numpy(X[i:i + 32]).permute(0, 3, 1, 2).float()), 1) for i in range(0, len(X), 32)]).numpy()
    meta["p_off"] = P[:, OFF]
    rows, gal = [], {}
    for site in HELDOUT:
        raw, _ = _route_array(site, "none")
        seg = K.segment(raw[..., 0].astype(np.float32), NM_HALF)
        print(site, "segmented", flush=True)
        masks = {"si": seg.si, "graphite": seg.graphite, "pore": seg.pore}
        idx = np.where((meta.site == site).to_numpy())[0]
        pred_site = OFF if meta.p_off.iloc[idx].mean() > 0.5 else 1 - OFF
        # most typical crops for the site call: highest P(predicted class), spread over the field
        order = idx[np.argsort(-P[idx, pred_site])]
        show = order[:N_SHOW]
        rng = np.random.default_rng(0)
        attr_idx = np.unique(np.concatenate([show, rng.choice(idx, size=min(N_ATTR, len(idx)), replace=False)]))
        for i in attr_idx:  # attribution on a subsample of crops: occlusion x 5-fold ensemble is CPU-heavy
            x = torch.from_numpy(X[i]).permute(2, 0, 1).float()
            y0, x0 = int(meta.y.iloc[i]), int(meta.x.iloc[i])
            occ = occlusion(ens, x, pred_site)
            print(site, int(i), "occ", flush=True)
            cam = cam_ens(models, x, pred_site)
            cam_p = upsample(cam, occ.shape[0]) if cam.shape != occ.shape else cam
            bse_in = x[0].numpy()
            bse_raw = raw[y0:y0 + CROP, x0:x0 + CROP, 0].astype(np.float32) / 255.0
            ls = local_stats(bse_raw)
            mk = {k: v[y0:y0 + CROP, x0:x0 + CROP] for k, v in masks.items()}
            occ_up, cam_up = upsample(occ), upsample(cam)
            row = dict(route=route, site=site, truth=TRUTH[site], y=y0, x=x0, p_off=float(P[i, OFF]), pred="off" if pred_site == OFF else "baseline",
                       occ_total=float(np.clip(occ, 0, None).sum()), occ_cam_corr=float(spearmanr(occ.ravel(), cam_p.ravel()).correlation))
            for name, s in ls.items():
                row[f"occ_corr_{name}"] = float(spearmanr(occ.ravel(), s.ravel()).correlation)
                row[f"cam_corr_{name}"] = float(spearmanr(cam_p.ravel(), s.ravel()).correlation)
            row.update({f"occ_{k}": v for k, v in phase_share(occ_up, mk).items()})
            row.update({f"cam_{k}": v for k, v in phase_share(cam_up, mk).items()})
            rows.append(row)
            if i in show:
                gal[i] = (bse_raw, bse_in, occ_up, cam_up, mk, row)
        # gallery
        fig, ax = plt.subplots(len(show), 5, figsize=(16, 3.3 * len(show)), squeeze=False)
        for r, i in enumerate(show):
            bse_raw, bse_in, occ_up, cam_up, mk, row = gal[i]
            ph = np.zeros(bse_raw.shape + (3,)); ph[mk["si"]] = (1, 0.9, 0.2); ph[mk["graphite"]] = (0.4, 0.4, 0.4); ph[mk["pore"]] = (0.1, 0.3, 1)
            ax[r, 0].imshow(bse_raw, cmap="gray", vmin=0, vmax=1); ax[r, 0].set_title(f"{site} raw BSE  y={row['y']} x={row['x']}", fontsize=8)
            ax[r, 1].imshow(bse_in, cmap="gray", vmin=0, vmax=1); ax[r, 1].set_title(f"model input ({route})  P(off)={row['p_off']:.2f}", fontsize=8)
            ax[r, 2].imshow(bse_raw, cmap="gray", vmin=0, vmax=1); ax[r, 2].imshow(occ_up, cmap="inferno", alpha=0.55)
            ax[r, 2].set_title(f"occlusion for '{row['pred']}'  Si×{row['occ_ratio_si']:.1f} gr×{row['occ_ratio_graphite']:.1f} pore×{row['occ_ratio_pore']:.1f}", fontsize=8)
            ax[r, 3].imshow(bse_raw, cmap="gray", vmin=0, vmax=1); ax[r, 3].imshow(cam_up, cmap="inferno", alpha=0.55)
            ax[r, 3].set_title(f"Grad-CAM (5-fold mean)  edges r={row['cam_corr_edges']:.2f} noise r={row['cam_corr_noise']:.2f}", fontsize=8)
            ax[r, 4].imshow(ph); ax[r, 4].set_title("Si (yellow) graphite (grey) pore (blue)", fontsize=8)
            for a in ax[r]: a.axis("off")
        fig.suptitle(f"{site}: truth {TRUTH[site]}; ResNet-18 {route} off-detector says {'OFF' if pred_site == OFF else 'BASELINE'} "
                     f"(site P(off)={meta.p_off.iloc[idx].mean():.2f}, {int((P[idx, OFF] > 0.5).sum())}/{len(idx)} crops off)", fontsize=11)
        fig.tight_layout(); fig.savefig(OUTD / f"{route}_{site}.png", dpi=85); plt.close(fig)
        print(site, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUTD / f"{route}_crops.csv", index=False)
    cols = ["occ_ratio_si", "occ_ratio_graphite", "occ_ratio_pore", "cam_ratio_si", "cam_ratio_graphite", "cam_ratio_pore",
            "occ_corr_brightness", "occ_corr_contrast", "occ_corr_edges", "occ_corr_noise", "cam_corr_brightness", "cam_corr_contrast", "cam_corr_edges", "cam_corr_noise", "occ_total", "p_off"]
    S = df.groupby("site")[cols].median().round(3); S["route"] = route
    S.to_csv(OUTD / f"{route}_summary.csv")
    print(S.T.to_string())
    # summary figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ph = S[["occ_ratio_si", "occ_ratio_graphite", "occ_ratio_pore"]]; ph.columns = ["Si", "graphite", "pore"]
    ph.plot.bar(ax=ax[0], color=["#e6c229", "#777777", "#3060d0"]); ax[0].axhline(1, c="k", lw=0.8); ax[0].set_ylabel("attribution share / area share"); ax[0].set_title("Occlusion attribution by phase (1 = proportional to area)")
    tx = S[["occ_corr_brightness", "occ_corr_contrast", "occ_corr_edges", "occ_corr_noise"]]; tx.columns = ["brightness", "contrast", "edges", "HF noise"]
    tx.plot.bar(ax=ax[1]); ax[1].axhline(0, c="k", lw=0.8); ax[1].set_ylabel("Spearman r (per 32 px patch)"); ax[1].set_title("Occlusion attribution vs local raw-image statistics")
    for a in ax: a.tick_params(axis="x", rotation=0)
    fig.suptitle(f"ResNet-18 {route} off-detector, held-out sites (median over all crops)"); fig.tight_layout(); fig.savefig(OUTD / f"{route}_summary.png", dpi=110)


if __name__ == "__main__":
    main(*sys.argv[1:])
