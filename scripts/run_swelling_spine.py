#!/usr/bin/env python3
"""Frozen pretrained spine -> swelling targets: can a generic vision embedding read the lithiation geometry?

Mirrors the v2 representation protocol (256 px crops at 50 nm/px, stride 128, frozen backbone, ridge
probe, splits by field) but the targets are the functional swelling test computed on each crop's masks:
SOC-1 constrained share (growth landing on graphite) and pore loss. Must-beat baseline: ridge on the five
mask fractions of the same crop (Si, graphite, pore fraction, Si-graphite contact, Si object count).

    python scripts/run_swelling_spine.py extract [--spines dinov2,micronet]   # targets + cached features
    python scripts/run_swelling_spine.py evaluate                              # LOSO probes -> results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from pmdb import functional, segment as segment_mod  # noqa: E402
from pmdb.io import list_sites, load_site  # noqa: E402
from pmdb.kpis.crossphase import si_graphite_contact  # noqa: E402

OUT = REPO_ROOT / "outputs" / "spine"
HELDOUT_DATA = REPO_ROOT / "data_heldout"
HELDOUT_CACHE = REPO_ROOT / "cache_heldout"
CROP, STRIDE, SOC = 256, 128, 1.0
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BASELINE = ["si0_frac", "graphite_frac", "pore_frac", "contact", "n_si_objects"]
TARGETS = ["into_graphite", "pore_loss"]
MICRONET_URL = ("https://huggingface.co/jstuckner/microscopy-efficientnet-b4-imagenet-micronet/resolve/main/"
                "efficientnet-b4_imagenet-micronet_weights.pth")


def grid(h: int, w: int, size: int, stride: int) -> list[tuple[int, int]]:
    ys = list(range(0, h - size + 1, stride)) or [0]
    xs = list(range(0, w - size + 1, stride)) or [0]
    return [(y, x) for y in ys for x in xs]


def percentile_norm(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, (0.5, 99.5))
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def crop_targets(masks: segment_mod.Masks, y: int, x: int) -> dict[str, float]:
    m = masks.crop(slice(y, y + CROP), slice(x, x + CROP))
    dom = m.fraction_space
    n = max(int(dom.sum()), 1)
    sw = functional.swelling_test(m, SOC)
    from scipy import ndimage
    n_obj = int(ndimage.label(m.si, structure=np.ones((3, 3)))[1])
    return {"si0_frac": float((m.si & dom).sum() / n), "graphite_frac": float((m.graphite & dom).sum() / n),
            "pore_frac": float((m.pore & dom).sum() / n), "contact": si_graphite_contact(m.si, m.graphite),
            "n_si_objects": n_obj, "into_graphite": sw["into_graphite"], "pore_loss": sw["pore_loss"],
            "swollen_si_frac": sw["si_frac"]}


class Spine:
    def __init__(self, name: str):
        self.name = name
        if name == "dinov2":
            self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        elif name == "micronet":
            import segmentation_models_pytorch as smp
            self.m = smp.encoders.get_encoder("efficientnet-b4", weights=None)
            sd = torch.hub.load_state_dict_from_url(MICRONET_URL, map_location="cpu", progress=False)
            sd = {k: v for k, v in sd.items() if not k.startswith("_fc.")}
            missing, unexpected = torch.nn.Module.load_state_dict(self.m, sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"MicroNet weights mismatch: {missing[:3]} {unexpected[:3]}")
        else:
            raise ValueError(name)
        self.m.eval().requires_grad_(False)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> np.ndarray:
        x = (x.repeat(1, 3, 1, 1) - IMNET_MEAN) / IMNET_STD
        if self.name == "dinov2":
            x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
            out = self.m.forward_features(x)
            f = torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], 1)
        else:
            stages = self.m(x)[-3:]
            f = torch.cat([t for s in stages for t in (s.mean((2, 3)), s.std((2, 3)))], 1)
        return f.numpy()


def site_crops(batch: str, site: str, heldout: bool) -> tuple[np.ndarray, list[dict]]:
    kw = {"data_root": HELDOUT_DATA, "cache_root": HELDOUT_CACHE} if heldout else {}
    raw = load_site(batch, site, resolution="half", normalise="none", **kw)
    bse = np.asarray(raw.image[..., 0], dtype=np.float64)
    masks = segment_mod.segment_bse(bse, raw.nm_per_px, params=None)
    img = percentile_norm(bse)
    rows, tiles = [], []
    for y, x in grid(*bse.shape, CROP, STRIDE):
        rows.append({"batch": batch, "site": site, "key": f"{batch}/{site}", "heldout": heldout, "y": y, "x": x,
                     **crop_targets(masks, y, x)})
        tiles.append(img[y:y + CROP, x:x + CROP])
    return np.stack(tiles)[:, None], rows


def extract(spines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    models = {s: Spine(s) for s in spines}
    sites = [(r.batch, r.site, False) for r in list_sites().itertuples()] + \
            [(r.batch, r.site, True) for r in list_sites(HELDOUT_DATA).itertuples()]
    rows, feats = [], {s: [] for s in spines}
    t0 = time.time()
    for i, (b, s, h) in enumerate(sites):
        tiles, r = site_crops(b, s, h)
        rows += r
        xt = torch.from_numpy(tiles)
        for name, m in models.items():
            feats[name].append(np.concatenate([m.embed(xt[j:j + 32]) for j in range(0, len(xt), 32)]))
        print(f"[{i + 1}/{len(sites)}] {b}/{s} crops={len(r)} t={time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "crop_targets.csv", index=False)
    for name in spines:
        np.save(OUT / f"features_{name}.npy", np.concatenate(feats[name]).astype(np.float32))


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def loso_probe(X: np.ndarray, y: np.ndarray, keys: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.full_like(y, np.nan)
    for k in np.unique(keys):
        te = keys == k
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        model.fit(X[~te], y[~te])
        pred[te] = model.predict(X[te])
    return pred


def evaluate() -> None:
    T = pd.read_csv(OUT / "crop_targets.csv")
    spines = [p.stem.replace("features_", "") for p in OUT.glob("features_*.npy")]
    F = {s: np.load(OUT / f"features_{s}.npy") for s in spines}
    lab = (~T["heldout"]).to_numpy()
    sets = {"baseline_masks": T[BASELINE].fillna({"contact": 0.0}).to_numpy(float)}  # no Si -> no contact
    for s in spines:
        sets[s] = F[s]
        sets[f"{s}+masks"] = np.hstack([F[s], sets["baseline_masks"]])
    results, site_rows, heldout_rows = {}, [], []
    for tgt in TARGETS + ["si0_frac"]:
        ok = lab & np.isfinite(T[tgt].to_numpy())
        y, keys, batch = T.loc[ok, tgt].to_numpy(), T.loc[ok, "key"].to_numpy(), T.loc[ok, "batch"].to_numpy()
        results[tgt] = {"n_crops": int(ok.sum()), "n_sites": int(len(np.unique(keys)))}
        for name, X in sets.items():
            if tgt == "si0_frac" and "masks" in name:
                continue
            p = loso_probe(X[ok], y, keys)
            df = pd.DataFrame({"key": keys, "batch": batch, "y": y, "p": p}).groupby(["key", "batch"]).mean().reset_index()
            results[tgt][name] = {"crop_r2": _r2(y, p), "site_r2": _r2(df["y"].to_numpy(), df["p"].to_numpy()),
                                  "site_spearman": float(df[["y", "p"]].corr("spearman").iloc[0, 1])}
            for r in df.itertuples():
                site_rows.append({"target": tgt, "features": name, "key": r.key, "batch": r.batch, "y": r.y, "p": r.p})
            if tgt in TARGETS:
                from sklearn.linear_model import RidgeCV
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                full = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(X[ok], y)
                ho = (~lab) & np.isfinite(T[tgt].to_numpy())
                hp = pd.DataFrame({"site": T.loc[ho, "site"], "y": T.loc[ho, tgt], "p": full.predict(X[ho])})
                for r in hp.groupby("site").mean().reset_index().itertuples():
                    heldout_rows.append({"target": tgt, "features": name, "site": r.site, "y_true": r.y, "y_pred": r.p})
            print(tgt, name, {k: round(v, 3) for k, v in results[tgt][name].items()}, flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(site_rows).to_csv(OUT / "site_predictions.csv", index=False)
    pd.DataFrame(heldout_rows).to_csv(OUT / "heldout_predictions.csv", index=False)
    plot(pd.DataFrame(site_rows), results)


def plot(S: pd.DataFrame, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats = [f for f in S["features"].unique()]
    fig, axes = plt.subplots(len(TARGETS), len(feats), figsize=(3.2 * len(feats), 3.2 * len(TARGETS)), squeeze=False)
    col = {"Batch_1": "tab:blue", "Batch_2": "tab:orange", "Batch_3": "tab:green"}
    for i, tgt in enumerate(TARGETS):
        for j, f in enumerate(feats):
            ax = axes[i, j]
            d = S[(S["target"] == tgt) & (S["features"] == f)]
            for b, g in d.groupby("batch"):
                ax.scatter(g["y"], g["p"], s=18, c=col[b], label=b)
            lo, hi = d[["y", "p"]].min().min(), d[["y", "p"]].max().max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            r = results[tgt][f]
            ax.set_title(f"{f}\nsite R2 {r['site_r2']:.2f}, crop R2 {r['crop_r2']:.2f}", fontsize=9)
            ax.set_xlabel(f"{tgt} (site mean, observed)", fontsize=8)
            if j == 0:
                ax.set_ylabel("LOSO prediction", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Frozen spine vs mask-fraction baseline -> SOC-1 swelling targets (leave-one-site-out)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "spine_vs_baseline.png", dpi=130)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "evaluate"])
    ap.add_argument("--spines", default="dinov2,micronet")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.spines.split(","))
    else:
        evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
