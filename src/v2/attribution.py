"""Why does a batch classifier decide what it decides?

For one finished classification run (``cls_runs/<hash>/final.pt``) this scores every held-out-fold
crop with two attribution maps and projects them onto what a materials scientist can name:

* **occlusion**: drop in the predicted-class probability when a 32 px patch is replaced by the crop
  mean (model-agnostic, works for ResNet / EfficientNet / DINOv2);
* **Grad-CAM** on the last convolutional map (CNNs) or on the patch tokens (DINOv2);
* **phase projection**: share of positive attribution falling on Kevin's Si / graphite / pore masks
  (``pmdb.segment`` on the field's BSE channel) relative to the phase's area share — ratio > 1 means
  the classifier looks at that phase more than its area would predict;
* **texture correlates**: Spearman correlation of the per-patch attribution with local brightness,
  local contrast (std), edge density (gradient magnitude) and high-frequency noise — these separate
  "it looks at structure" from "it looks at grey level / sharpness / noise".

Outputs in the run dir: ``attribution.csv`` (one row per crop), ``attribution_summary.json`` and
``attrib_<batch>_<correct|wrong>.png`` galleries for the most confident crops.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import spearmanr

from src.v2 import kpi_adapter as K
from src.v2.classify import (BATCHES, CLS_RUNS, Classifier, _labels, classes_of, device, full_cfg, manifest,
                             stratified_group_folds)
from src.v2.common import CROP, NM_HALF, load_half_raw
from src.v2.data import CropDataset, FieldStore, VIEWS

PATCH = 32
N_TOP = 6


def load_run(d: Path):
    c = json.loads((d / "config.json").read_text())
    model = Classifier(c["arch"], n_cls=len(classes_of(c)))
    model.load_state_dict(torch.load(d / "final.pt", map_location="cpu"))
    return model.eval(), full_cfg(c)


def test_fold(c: dict) -> pd.DataFrame:
    m = manifest()
    m["fold"] = stratified_group_folds(m.group_id.to_numpy(), m.batch.to_numpy(), c["n_folds"], c["seed"])
    return m[m.fold == c["fold"]]


@torch.no_grad()
def predict(model, x):
    return torch.softmax(model(x).float(), 1)


@torch.no_grad()
def occlusion(model, x: torch.Tensor, cls: int, patch: int = PATCH) -> np.ndarray:
    """(H/patch, W/patch) drop in p(cls) when each patch is replaced by the crop's channel means."""
    n = x.shape[-1] // patch
    base = predict(model, x[None])[0, cls].item()
    xs = x[None].repeat(n * n, 1, 1, 1)
    mean = x.mean((1, 2), keepdim=True)
    for k in range(n * n):
        i, j = divmod(k, n)
        xs[k, :, i * patch:(i + 1) * patch, j * patch:(j + 1) * patch] = mean
    p = torch.cat([predict(model, xs[a:a + 64])[:, cls] for a in range(0, n * n, 64)])
    return (base - p).reshape(n, n).cpu().numpy()


def gradcam(model, x: torch.Tensor, cls: int) -> np.ndarray:
    """Grad-CAM on the last conv map (ResNet layer4 / EfficientNet conv_head / smp stage 5) or on the
    DINOv2 patch tokens; returned at the feature-map resolution, ReLU'd and max-normalised."""
    feats = {}

    def hook(_, __, out):
        feats["a"] = out
        out.register_hook(lambda g: feats.__setitem__("g", g))

    arch = model.arch
    if arch.startswith("resnet18"):
        h = model.m.layer4.register_forward_hook(hook)
    elif arch == "effb4_imnet":
        h = model.m.conv_head.register_forward_hook(hook)
    elif arch == "effb4_micronet":  # smp encoder: forward stops after the last block (no _conv_head)
        h = model.m._blocks[-1].register_forward_hook(hook)
    else:  # dinov2: last encoder block output tokens
        h = model.bb.encoder.layer[-1].register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = model(x[None].requires_grad_(False))
        out[0, cls].backward()
    h.remove()
    a, g = feats["a"], feats["g"]
    if isinstance(a, tuple):
        a, g = a[0], g if not isinstance(g, tuple) else g[0]
    if a.dim() == 3:  # (1, 1+N, C) tokens -> (n, n)
        a, g = a[:, 1:], g[:, 1:]
        w = g.mean(1, keepdim=True)
        cam = (w * a).sum(-1)[0]
        n = int(round(cam.numel() ** 0.5))
        cam = cam.reshape(n, n)
    else:
        w = g.mean((2, 3), keepdim=True)
        cam = (w * a).sum(1)[0]
    cam = torch.relu(cam).detach().float().cpu().numpy()
    return cam / (cam.max() + 1e-8)


def upsample(m: np.ndarray, size: int = CROP) -> np.ndarray:
    return ndimage.zoom(m, size / m.shape[0], order=1)


def local_stats(bse: np.ndarray, patch: int = PATCH) -> dict[str, np.ndarray]:
    """Per-patch brightness, contrast, edge density and high-frequency noise of the BSE crop."""
    n = bse.shape[0] // patch
    g = np.hypot(ndimage.sobel(bse, 0), ndimage.sobel(bse, 1))
    hf = bse - ndimage.gaussian_filter(bse, 1.0)
    blk = lambda a: a.reshape(n, patch, n, patch).transpose(0, 2, 1, 3).reshape(n, n, -1)
    return {"brightness": blk(bse).mean(-1), "contrast": blk(bse).std(-1),
            "edges": blk(g).mean(-1), "noise": blk(hf).std(-1)}


def phase_share(heat: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    pos = np.clip(heat, 0, None)
    tot = pos.sum() + 1e-8
    out = {}
    for k, m in masks.items():
        area = m.mean()
        share = pos[m].sum() / tot
        out[f"share_{k}"] = float(share)
        out[f"area_{k}"] = float(area)
        out[f"ratio_{k}"] = float(share / area) if area > 1e-4 else np.nan
    return out


def analyse(d: Path, n_top: int = N_TOP, max_crops: int | None = None) -> dict:
    dev = device()
    model, c = load_run(d)
    model.to(dev)
    te_f = test_fold(c)
    store = FieldStore(te_f, c["input"], harmonise=c["harmonise"])
    ds = CropDataset(store, c["view"], stride=CROP)
    y_te = _labels(te_f, c)[torch.tensor([i for i, _, _ in ds.index])].numpy()
    masks_by_field = {}
    for gid in te_f.group_id:
        b, s = gid.split("/")
        m = K.segment(load_half_raw(b, s)[..., 0], NM_HALF)
        masks_by_field[gid] = {"si": m.si, "graphite": m.graphite, "pore": m.pore}
    rows, store_maps = [], {}
    idx = range(len(ds)) if max_crops is None else range(min(len(ds), max_crops))
    for i in idx:
        it = ds[i]
        x = it["x"].to(dev)
        fi, y0, x0 = ds.index[i]
        gid = ds.group[i]
        p = predict(model, x[None])[0].cpu().numpy()
        pred = int(p.argmax())
        occ = occlusion(model, x, pred)
        cam = gradcam(model, x, pred)
        cam_p = upsample(cam, occ.shape[0]) if cam.shape != occ.shape else cam
        bse = x[VIEWS[c["view"]].index(0) if 0 in VIEWS[c["view"]] else 0].cpu().numpy()
        ls = local_stats(bse)
        mk = {k: v[y0:y0 + CROP, x0:x0 + CROP] for k, v in masks_by_field[gid].items()}
        occ_up, cam_up = upsample(occ), upsample(cam)
        row = {"group_id": gid, "batch": ds.batch[i], "y": y0, "x": x0, "y_true": int(y_te[i]), "y_pred": pred,
               "p_pred": float(p[pred]), "correct": bool(pred == y_te[i]),
               "occ_total": float(np.clip(occ, 0, None).sum()), "occ_max": float(occ.max()),
               "occ_cam_corr": float(spearmanr(occ.ravel(), cam_p.ravel()).correlation)}
        for name, s in ls.items():
            row[f"occ_corr_{name}"] = float(spearmanr(occ.ravel(), s.ravel()).correlation)
            row[f"cam_corr_{name}"] = float(spearmanr(cam_p.ravel(), s.ravel()).correlation)
        row.update({f"occ_{k}": v for k, v in phase_share(occ_up, mk).items()})
        row.update({f"cam_{k}": v for k, v in phase_share(cam_up, mk).items()})
        rows.append(row)
        store_maps[i] = (bse, occ_up, cam_up, mk)
    df = pd.DataFrame(rows)
    df.to_csv(d / "attribution.csv", index=False)
    summary = summarise(df)
    (d / "attribution_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    try:
        galleries(d, df, store_maps, n_top)
    except Exception as e:  # plotting must not fail the analysis
        print("gallery failed:", e, file=sys.stderr)
    return summary


def summarise(df: pd.DataFrame) -> dict:
    out = {"n_crops": len(df), "crop_acc": float(df.correct.mean())}
    cols = [c for c in df.columns if c.startswith(("occ_corr_", "cam_corr_", "occ_ratio_", "cam_ratio_", "occ_share_"))]
    out["all"] = df[cols].median().to_dict()
    for b in BATCHES:
        sub = df[(df.batch == b)]
        if len(sub):
            out[b] = {"n": len(sub), "acc": float(sub.correct.mean()), **sub[cols].median().to_dict()}
    conf = df[df.p_pred > 0.8]
    out["confident"] = {"n": len(conf), "acc": float(conf.correct.mean()) if len(conf) else np.nan,
                        **conf[cols].median().to_dict()}
    return out


def galleries(d: Path, df: pd.DataFrame, maps: dict, n_top: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for b in BATCHES:
        for correct in (True, False):
            sub = df[(df.batch == b) & (df.correct == correct)].sort_values("p_pred", ascending=False)
            sub = sub.groupby("group_id").head(2).head(n_top)  # at most 2 crops per field, so the gallery spans fields
            if sub.empty:
                continue
            fig, ax = plt.subplots(len(sub), 4, figsize=(12, 3 * len(sub)), squeeze=False)
            for r, (i, row) in enumerate(sub.iterrows()):
                bse, occ, cam, mk = maps[i]
                ph = np.zeros(bse.shape + (3,))
                ph[mk["si"]] = (1, 0.9, 0.2); ph[mk["graphite"]] = (0.4, 0.4, 0.4); ph[mk["pore"]] = (0.1, 0.3, 1)
                ax[r, 0].imshow(bse, cmap="gray", vmin=0, vmax=1)
                ax[r, 0].set_title(f"{row.group_id} true {row.batch[-1]} pred {row.y_pred + 1} p={row.p_pred:.2f}", fontsize=8)
                ax[r, 1].imshow(bse, cmap="gray", vmin=0, vmax=1); ax[r, 1].imshow(occ, cmap="inferno", alpha=0.55)
                ax[r, 1].set_title(f"occlusion  Si×{row.occ_ratio_si:.1f} gr×{row.occ_ratio_graphite:.1f} pore×{row.occ_ratio_pore:.1f}", fontsize=8)
                ax[r, 2].imshow(bse, cmap="gray", vmin=0, vmax=1); ax[r, 2].imshow(cam, cmap="inferno", alpha=0.55)
                ax[r, 2].set_title(f"Grad-CAM  edges r={row.cam_corr_edges:.2f} bright r={row.cam_corr_brightness:.2f}", fontsize=8)
                ax[r, 3].imshow(ph); ax[r, 3].set_title("Si (yellow) graphite (grey) pore (blue)", fontsize=8)
                for a in ax[r]:
                    a.axis("off")
            fig.tight_layout()
            fig.savefig(d / f"attrib_{b}_{'correct' if correct else 'wrong'}.png", dpi=80)
            plt.close(fig)


if __name__ == "__main__":
    d = Path(sys.argv[1]) if "/" in sys.argv[1] else CLS_RUNS / sys.argv[1]
    s = analyse(d, max_crops=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(json.dumps({k: v for k, v in s.items() if k in ("n_crops", "crop_acc", "all", "confident")}, indent=1, default=float))
