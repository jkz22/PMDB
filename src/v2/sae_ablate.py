"""SAE feature ablation: find the imaging-artefact features of a representation, switch them off, and see
what happens to the batch prediction, the KPI content and the three held-out sites.

Per run (outputs/v2/sae_ablation/<hash>/):
  variants.csv      LOFO batch probe (field accuracy per batch, predicted shares), KPI ridge R2 and imaging-stat
                    ridge R2 for: the raw embedding, its SAE reconstruction (control), and the reconstruction
                    with imaging / imaging+mixed / material (control) / random-matched (control) features zeroed.
  per_field.csv     per-field P(batch) before and after the imaging ablation, and whether the decision flips.
  heldout.csv       the 3 organiser held-out sites embedded with the run's model: P(batch) from a probe fitted on
                    all 31 labelled fields, before / after ablation, next to their imaging statistics.
  heldout_features.csv  mean activation of every imaging feature per held-out site vs per labelled batch.
  feature_maps.png  (VAE-C) input-gradient saliency of the top imaging and top material features on their
                    top-activating crops, over the BSE crop: where in the image the feature looks.
The SAE is refitted with the same seed/settings as src.v2.sae, so features match sae_features.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter, laplace
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.v2.common import CROP, OUT, REPO, grid, harm_method
from src.v2.data import VIEWS, normalise_percentile, naive_transform, phase_labels
from src.v2.kpi_adapter import GATED_COLS, kpi_cols
from src.v2.latent_audit import load, targets
from src.v2.sae import BATCHES, IMG_KEYS, characterise, fit_sae

HELDOUT = ("3e122cbj", "fn0mhxef", "xrv9xvzb")
ABL = OUT / "sae_ablation"


# ----------------------------------------------------------------------------------------------- probes
def lofo_probs(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    y = meta.batch.map(BATCHES.index).to_numpy()
    g = meta.group_id.to_numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
    P = cross_val_predict(pipe, X, y, groups=g, cv=LeaveOneGroupOut(), method="predict_proba")
    f = pd.DataFrame(P, columns=[f"P_{b}" for b in BATCHES]).assign(field=g, y=y).groupby("field").mean()
    f["batch"] = [BATCHES[int(v)] for v in f.y]
    f["pred"] = [BATCHES[i] for i in f[[f"P_{b}" for b in BATCHES]].to_numpy().argmax(1)]
    return f.drop(columns="y")


def summarise_probs(f: pd.DataFrame) -> dict:
    out = {"field_acc": float((f.pred == f.batch).mean()),
           "bal_acc": float(np.mean([(f.pred[f.batch == b] == b).mean() for b in BATCHES]))}
    for b in BATCHES:
        out[f"recall_{b}"] = float((f.pred[f.batch == b] == b).mean())
        out[f"share_{b}"] = float((f.pred == b).mean())
    return out


def lofo_r2(X: np.ndarray, Y: pd.DataFrame, groups: np.ndarray) -> dict:
    out = {}
    for c in Y.columns:
        y = Y[c].to_numpy(float)
        ok = ~np.isnan(y)
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        p = cross_val_predict(pipe, X[ok], y[ok], groups=groups[ok], cv=LeaveOneGroupOut())
        out[c] = float(r2_score(y[ok], p))
    return out


# ----------------------------------------------------------------------------------------------- SAE
class Dict_:
    def __init__(self, E: np.ndarray, expansion=8, k=8):
        self.mu, self.sd = E.mean(0), E.std(0) + 1e-6
        X = (E - self.mu) / self.sd
        self.sae, self.fit = fit_sae(X, m=expansion * X.shape[1], k=k)

    def std(self, E):
        return (E - self.mu) / self.sd

    def encode(self, E) -> np.ndarray:
        with torch.no_grad():
            return self.sae.encode(torch.from_numpy(self.std(E)).float()).numpy()

    def decode(self, Z) -> np.ndarray:
        with torch.no_grad():
            X = self.sae.dec(torch.from_numpy(Z).float()) + self.sae.b_pre
        return X.numpy() * self.sd + self.mu

    def ablate(self, E, feats) -> np.ndarray:
        Z = self.encode(E)
        Z[:, list(feats)] = 0.0
        return self.decode(Z)


def feature_sets(feats: pd.DataFrame, seed=0) -> dict[str, list[int]]:
    img = feats.feature[feats.kind == "imaging"].tolist()
    mixed = feats.feature[feats.kind == "mixed"].tolist()
    mat = feats.feature[feats.kind == "material"].tolist()
    rng = np.random.default_rng(seed)
    pool = feats[feats.kind != "rare"]
    # control: random non-rare features with the same total activation mass as the imaging set
    target = feats.mass[feats.feature.isin(img)].sum()
    order = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    rand, acc = [], 0.0
    for _, r in order.iterrows():
        if acc >= target:
            break
        rand.append(int(r.feature)); acc += r.mass
    strong = feats[(feats.kind == "imaging") & (feats.best_img_d.abs() > 1.0)].feature.tolist()
    # 'pure': large imaging effect with no material effect (|d_kpi| < 0.3 SD) -- the only set that is an imaging
    # artefact by construction; 'imaging' also contains features where sharpness/p1 merely co-vary with graphite
    pure = feats[(feats.kind != "rare") & (feats.best_img_d.abs() > 0.5) & (feats.best_kpi_d.abs() < 0.3)].feature.tolist()
    return {"imaging": img, "imaging_strong": strong, "imaging_pure": pure, "imaging+mixed": img + mixed,
            "material": mat, "random_matched": rand}


# ----------------------------------------------------------------------------------------------- held-out
def _route_array(site: str, harm: str) -> tuple[np.ndarray, np.ndarray | None]:
    root = REPO / "cache_heldout"
    if harm == "none":
        return np.load(root / "half" / f"Batch_heldout__{site}.npz")["image"], None
    if harm.startswith("clean"):
        tag, kind = harm.split("_", 1)
        z = np.load(root / tag / kind / "half" / f"Batch_heldout__{site}.npz")
        im, v = z["image"].copy(), z["valid"]
        im[~v] = 0
        return im, v.all(-1)
    return np.load(root / "harmonised" / harm / "half" / f"Batch_heldout__{site}.npz")["image"], None


def crop_imaging_stats(a: np.ndarray) -> dict:
    a = a.astype(np.float32)
    p1, p99 = np.percentile(a, [1, 99])
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    from scipy.ndimage import convolve
    h, w = a.shape
    noise = float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(convolve(a, k)[1:-1, 1:-1]).sum())
    lapvar = float(laplace(a)[1:-1, 1:-1].var())
    return dict(p1=float(p1), p99=float(p99), noise_sigma=noise, sharpness=lapvar / max(float(p99 - p1), 1.0) ** 2,
                zero_frac=float((a == 0).mean()))


def load_model(run_dir: Path, cfg: dict):
    from src.v2.models import build
    fam = cfg["family"]
    model = build(fam, n_kpi=len(kpi_cols(cfg.get("kpi_set"))), vae_mask=cfg["vae_mask"], mae_mask=cfg["mae_mask"],
                  phase_mask=cfg.get("phase_mask", "none"))
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)["model"])
    return model.eval()


def heldout_crops(cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    harm = harm_method(cfg["harmonise"])
    ch = list(VIEWS[cfg["view"]])
    crops, rows = [], []
    for s in HELDOUT:
        im, valid = _route_array(s, harm)
        x = normalise_percentile(im.astype(np.float32)) if cfg["input"] == "norm" else im.astype(np.float32) / 255.0
        if cfg["input"] == "naive":
            x[phase_labels(im[..., 0].astype(np.float32)) == 0] = 0.0
        for k, (y, xx) in enumerate(grid(*im.shape[:2], CROP, CROP)):
            if valid is not None and valid[y:y + CROP, xx:xx + CROP].mean() < 0.9:
                continue
            c = x[y:y + CROP, xx:xx + CROP][..., ch]
            if cfg["input"] == "naive":
                c = naive_transform(torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1)[None],
                                    torch.Generator().manual_seed(1_000_003 * k + 17))[0].permute(1, 2, 0).numpy()
            crops.append(c)
            rows.append(dict(site=s, y=y, x=xx, **crop_imaging_stats(im[y:y + CROP, xx:xx + CROP, 0])))
    return np.stack(crops), pd.DataFrame(rows)


def embed(model, crops: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([model.embed(torch.from_numpy(c).permute(0, 3, 1, 2).float())
                          for c in np.array_split(crops, max(1, len(crops) // 32))]).numpy()


# ----------------------------------------------------------------------------------------------- feature maps
def feature_maps(model, D: Dict_, feats: pd.DataFrame, E: np.ndarray, meta: pd.DataFrame, cfg: dict, path: Path,
                 n_feat=4, n_top=3):
    """Input-gradient saliency of SAE feature j (pre-TopK ReLU activation) wrt the crop, for its top-activating
    crops. VAE only (deterministic encoder mean)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.v2.data import FieldStore
    Z = D.encode(E)
    sel = pd.concat([feats[feats.kind == "imaging"].sort_values("mass", ascending=False).head(n_feat),
                     feats[feats.kind == "material"].sort_values("mass", ascending=False).head(n_feat)])
    fields = meta[["group_id"]].drop_duplicates()
    fields = fields.assign(batch=fields.group_id.str.split("/").str[0], site=fields.group_id.str.split("/").str[1])
    store = FieldStore(fields, cfg["input"], harmonise=cfg["harmonise"])
    gidx = {g: i for i, g in enumerate(store.fields.group_id)}
    ch = list(VIEWS[cfg["view"]])
    mu_t, sd_t = torch.from_numpy(D.mu).float(), torch.from_numpy(D.sd).float()
    fig, ax = plt.subplots(len(sel), 2 * n_top, figsize=(2.0 * 2 * n_top, 2.1 * len(sel)))
    for r, (_, f) in zip(ax, sel.iterrows()):
        j = int(f.feature)
        top = np.argsort(-Z[:, j])[:n_top]
        for t, i in enumerate(top):
            g, y, x = meta.group_id[i], int(meta.y[i]), int(meta.x[i])
            im = store.images[gidx[g]][y:y + CROP, x:x + CROP][..., ch]
            xt = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].float().requires_grad_(True)
            mu = model.encode(xt)[0]
            a = torch.relu(D.sae.enc((mu - mu_t) / sd_t - D.sae.b_pre))[0, j]
            a.backward()
            sal = gaussian_filter(xt.grad[0].abs().sum(0).numpy(), 3)
            bse = im[..., 0]
            r[2 * t].imshow(bse, cmap="gray", vmin=0, vmax=1); r[2 * t].set_title(g.split("/")[0].replace("Batch_", "B") + f" a={Z[i, j]:.1f}", fontsize=7)
            r[2 * t + 1].imshow(bse, cmap="gray", vmin=0, vmax=1); r[2 * t + 1].imshow(sal, cmap="inferno", alpha=0.55)
            for a_ in (r[2 * t], r[2 * t + 1]):
                a_.set_xticks([]); a_.set_yticks([])
        r[0].set_ylabel(f"f{j} {f.kind}\n{f.best_kpi} {f.best_kpi_d:+.2f}\n{f.best_img} {f.best_img_d:+.2f}", fontsize=6.5)
    fig.suptitle(f"{path.parent.name}: SAE feature saliency (top imaging rows, then top material rows)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def analyse(run: str, maps: bool = True) -> dict:
    run_dir = OUT / "runs" / run
    cfg = json.load(open(run_dir / "config.json"))
    out = ABL / run
    out.mkdir(parents=True, exist_ok=True)
    E, meta = load(run_dir)
    meta = meta.reset_index(drop=True)
    g = meta.group_id.to_numpy()
    D = Dict_(E)
    kp, im = targets(meta)
    T = pd.concat([kp.reset_index(drop=True), im.reset_index(drop=True)], axis=1)
    feats = characterise(D.encode(E), meta, T)
    feats.to_csv(out / "sae_features.csv", index=False)
    sets = feature_sets(feats)
    img_cols = [f"{k}_c{c}" for k in IMG_KEYS for c in (0, 1, 2) if f"{k}_c{c}" in T]
    variants = {"orig": E, "sae_recon": D.decode(D.encode(E))}
    for name, fs in sets.items():
        variants[f"abl_{name}"] = D.ablate(E, fs)
    rows, probs = [], {}
    for name, X in variants.items():
        f = lofo_probs(X, meta)
        probs[name] = f
        r2k = lofo_r2(X, T[list(GATED_COLS)], g)
        r2i = lofo_r2(X, T[img_cols], g)
        rows.append(dict(run=run, family=cfg["family"], harmonise=harm_method(cfg["harmonise"]), variant=name,
                         n_feat=len(sets.get(name.replace("abl_", ""), [])) if name.startswith("abl_") else 0,
                         mass_frac=float(feats.mass[feats.feature.isin(sets.get(name.replace("abl_", ""), []))].sum() / feats.mass.sum()) if name.startswith("abl_") else 0.0,
                         **summarise_probs(f), kpi_r2=float(np.mean(list(r2k.values()))), **{f"r2_{k}": v for k, v in r2k.items()},
                         imaging_r2=float(np.mean(list(r2i.values()))),
                         **{f"r2_{k}": float(np.mean([r2i[f"{k}_c{c}"] for c in (0, 1, 2) if f"{k}_c{c}" in r2i])) for k in IMG_KEYS}))
    var = pd.DataFrame(rows)
    var.to_csv(out / "variants.csv", index=False)
    pf = probs["orig"].join(probs["abl_imaging"], rsuffix="_abl").drop(columns="batch_abl")
    pf["flip"] = pf.pred != pf.pred_abl
    pf["dP_Batch_3"] = pf["P_Batch_3_abl"] - pf["P_Batch_3"]
    pf.to_csv(out / "per_field.csv")

    # held-out sites
    model = load_model(run_dir, cfg)
    crops, hm = heldout_crops(cfg)
    Eh = embed(model, crops)
    y = meta.batch.map(BATCHES.index).to_numpy()
    hrows = []
    for name in ("orig", "abl_imaging", "abl_imaging_pure", "abl_imaging+mixed"):
        Xtr = variants[name]
        Xh = Eh if name == "orig" else D.ablate(Eh, sets[name.replace("abl_", "")])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")).fit(Xtr, y)
        P = clf.predict_proba(Xh)
        for s in HELDOUT:
            m = (hm.site == s).to_numpy()
            p = P[m].mean(0)
            hrows.append(dict(run=run, variant=name, site=s, n_crops=int(m.sum()), pred=BATCHES[int(p.argmax())],
                              **{f"P_{b}": float(v) for b, v in zip(BATCHES, p)},
                              **{k: float(hm[k][m].mean()) for k in ("p1", "p99", "noise_sigma", "sharpness", "zero_frac")}))
    pd.DataFrame(hrows).to_csv(out / "heldout.csv", index=False)
    # which imaging features the held-out sites light up, against the labelled batches
    Zl, Zh = D.encode(E), D.encode(Eh)
    img = sets["imaging"]
    act = pd.DataFrame({**{b: Zl[y == k][:, img].mean(0) for k, b in enumerate(BATCHES)},
                        **{s: Zh[(hm.site == s).to_numpy()][:, img].mean(0) for s in HELDOUT}}, index=img)
    act = act.join(feats.set_index("feature")[["mass", "best_img", "best_img_d", "best_kpi", "best_kpi_d", "batch_selectivity"]])
    act.sort_values("mass", ascending=False).to_csv(out / "heldout_features.csv")
    if maps and cfg["family"].startswith("vae"):
        feature_maps(model, D, feats, E, meta, cfg, out / "feature_maps.png")
    summ = {"run": run, "family": cfg["family"], "harmonise": harm_method(cfg["harmonise"]), **D.fit,
            "n_imaging": len(img), "mass_imaging": float(feats.mass[feats.feature.isin(img)].sum() / feats.mass.sum()),
            "variants": var.set_index("variant")[["field_acc", "bal_acc", "recall_Batch_1", "recall_Batch_2", "recall_Batch_3", "kpi_r2", "imaging_r2"]].round(3).to_dict("index"),
            "n_flips": int(pf.flip.sum()), "heldout": {r["site"]: (r["pred"], round(r["P_Batch_3"], 2)) for r in hrows if r["variant"] == "orig"},
            "heldout_abl": {r["site"]: (r["pred"], round(r["P_Batch_3"], 2)) for r in hrows if r["variant"] == "abl_imaging"}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args()
    for r in a.runs:
        s = analyse(r, maps=not a.no_maps)
        print(json.dumps(s, default=float))
    allv = pd.concat([pd.read_csv(p) for p in ABL.glob("*/variants.csv")])
    allv.to_csv(ABL / "variants_all.csv", index=False)
    allh = pd.concat([pd.read_csv(p) for p in ABL.glob("*/heldout.csv")])
    allh.to_csv(ABL / "heldout_all.csv", index=False)


if __name__ == "__main__":
    main()
