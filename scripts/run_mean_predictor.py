#!/usr/bin/env python3
"""The simplest possible predictor: nearest batch mean (nearest centroid).

Each field is summarised by one small vector; the batch mean of that vector is computed from the training
fields only; a field is assigned to the batch whose mean is closest (Euclidean after standardising on the
training fold). Leave-one-site-out, label-permutation null. Flavours of "the mean of the image":

  grey_raw          raw uint8 BSE: mean, std, p1, p99 + 16-bin histogram               (reads the Batch 3 artefact?)
  grey_harm         same on hybrid-harmonised BSE
  grey_depth_raw    raw BSE mean grey in 15 depth bands (the image averaged along x)
  grey_depth_harm   same, harmonised
  overlay_frac      segmentation overlay means: Si / graphite / pore / binder fraction
  overlay_depth     Si and graphite depth profiles from the overlay (15 bands, relative to field mean)
  grey_harm_*       decomposition of grey_harm: moments only / std only / histogram only;
                    graphite_only / pore_only = the harmonised grey distribution *inside one phase* (pure imaging:
                    noise, residual gain, quantisation - no composition can enter)
  fingerprint16     the 16 fingerprint features, nearest centroid instead of the robust NB

Writes outputs/meanpred/{site_features.csv, results.json, loso.csv, heldout.csv, figures/accuracy.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmdb import segment as segment_mod  # noqa: E402
from pmdb.io import list_sites, load_site  # noqa: E402
from pmdb.kpis.fields import band_profile  # noqa: E402

O = ROOT / "outputs"
OUT = O / "meanpred"
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
HELD = ["3e122cbj", "fn0mhxef", "xrv9xvzb"]
NB = 15
HIST_BINS = 16


def site_features(batch: str, site: str) -> dict:
    kw = {"data_root": ROOT / "data_heldout", "cache_root": ROOT / "cache_heldout"} if batch == "Batch_heldout" else {}
    raw = load_site(batch, site, resolution="half", normalise="none", **kw)
    harm = load_site(batch, site, resolution="half", normalise="fixed", harmonise="hybrid", **kw)
    row = {"batch": batch, "site": site}
    for tag, img in (("raw", raw.image[..., 0].astype(float)), ("harm", 255.0 * harm.image[..., 0].astype(float))):
        row.update({f"grey_{tag}_mean": img.mean(), f"grey_{tag}_std": img.std(),
                    f"grey_{tag}_p1": np.percentile(img, 1), f"grey_{tag}_p99": np.percentile(img, 99)})
        h, _ = np.histogram(img, bins=HIST_BINS, range=(0, 256))
        row.update({f"grey_{tag}_hist{i}": v for i, v in enumerate(h / h.sum())})
        edges = np.linspace(0, img.shape[0], NB + 1).round().astype(int)
        row.update({f"grey_depth_{tag}_b{i}": img[a:b].mean() for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))})
    m = segment_mod.segment(raw)
    adm = m.fraction_space
    hb = 255.0 * harm.image[..., 0].astype(float)
    for ph in ("graphite", "pore"):
        px = hb[getattr(m, ph)]
        row.update({f"grey_harm_{ph}_mean": px.mean(), f"grey_harm_{ph}_std": px.std(),
                    f"grey_harm_{ph}_iqr": np.subtract(*np.percentile(px, [75, 25]))})
        h, _ = np.histogram(px, bins=HIST_BINS, range=(0, 256))
        row.update({f"grey_harm_{ph}_hist{i}": v for i, v in enumerate(h / h.sum())})
    n = adm.sum()
    row.update({"overlay_si": m.si.sum() / n, "overlay_graphite": m.graphite.sum() / n, "overlay_pore": m.pore.sum() / n})
    row["overlay_binder"] = 1.0 - row["overlay_si"] - row["overlay_graphite"] - row["overlay_pore"]
    for ph in ("si", "graphite"):
        prof = band_profile(getattr(m, ph), adm, n_bands=NB)
        rel = prof / prof.mean()
        row.update({f"overlay_depth_{ph}_b{i}": v for i, v in enumerate(rel)})
    return row


def families(cols: list[str]) -> dict[str, list[str]]:
    g = lambda p: [c for c in cols if c.startswith(p)]  # noqa: E731
    return {
        "grey_raw": g("grey_raw_"),
        "grey_harm": g("grey_harm_"),
        "grey_depth_raw": g("grey_depth_raw_"),
        "grey_depth_harm": g("grey_depth_harm_"),
        "grey_harm_moments": ["grey_harm_mean", "grey_harm_std", "grey_harm_p1", "grey_harm_p99"],
        "grey_harm_std_only": ["grey_harm_std"],
        "grey_harm_hist": g("grey_harm_hist"),
        "grey_harm_graphite_only": g("grey_harm_graphite_"),
        "grey_harm_pore_only": g("grey_harm_pore_"),
        "overlay_frac": ["overlay_si", "overlay_graphite", "overlay_pore", "overlay_binder"],
        "overlay_depth": g("overlay_depth_"),
    }


def nearest_centroid_loso(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), int)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Z = (X - mu) / sd
        cents = np.stack([Z[tr & (y == k)].mean(0) for k in range(3)])
        pred[i] = np.argmin(((Z[i] - cents) ** 2).sum(1))
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reuse-features", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    if a.reuse_features:
        F = pd.read_csv(OUT / "site_features.csv", dtype={"batch": str, "site": str})
    else:
        sites = [(r.batch, r.site) for r in list_sites().itertuples()] + [("Batch_heldout", s) for s in HELD]
        with ProcessPoolExecutor(a.workers) as ex:
            F = pd.DataFrame(list(ex.map(site_features, *zip(*sites))))
        F.to_csv(OUT / "site_features.csv", index=False)
    fams = families(F.columns.tolist())
    fp = pd.concat([pd.read_csv(O / "fingerprint" / f, dtype={"site": str}) for f in ("features.csv", "heldout_features.csv")])
    fpc = [c for c in fp.columns if c not in ("batch", "site")]
    F = F.merge(fp[["site"] + fpc], on="site", how="left")
    fams["fingerprint16"] = fpc
    L = F[F.batch.isin(BATCHES)].reset_index(drop=True)
    H = F[F.batch == "Batch_heldout"].reset_index(drop=True)
    y = np.array([BATCHES.index(b) for b in L.batch])
    rng = np.random.default_rng(0)
    perms = [rng.permutation(y) for _ in range(a.n_perm)]
    res, loso, ho = {}, [], []
    for fam, cols in fams.items():
        X = L[cols].to_numpy(float)
        X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
        pred = nearest_centroid_loso(X, y)
        acc = float((pred == y).mean())
        null = np.array([(nearest_centroid_loso(X, p) == p).mean() for p in perms])
        b3_recall = float((pred[y == 2] == 2).mean())
        res[fam] = {"n_features": len(cols), "loso_accuracy": acc, "perm_p": float((1 + (null >= acc).sum()) / (a.n_perm + 1)),
                    "null_mean": float(null.mean()), "batch3_recall": b3_recall,
                    "recall": {b: float((pred[y == k] == k).mean()) for k, b in enumerate(BATCHES)}}
        loso.append(pd.DataFrame({"family": fam, "site": L.site, "true": [BATCHES[k] for k in y], "pred": [BATCHES[k] for k in pred]}))
        # held-out: centroids from all labelled fields
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - mu) / sd
        cents = np.stack([Z[y == k].mean(0) for k in range(3)])
        Xh = H[cols].to_numpy(float)
        Xh = np.where(np.isnan(Xh), np.nanmedian(X, 0), Xh)
        D = (((Xh - mu) / sd)[:, None, :] - cents[None]) ** 2
        d = np.sqrt(D.sum(2))
        for i, s in enumerate(H.site):
            ho.append({"family": fam, "site": s, "call": BATCHES[int(d[i].argmin())],
                       **{f"dist_{b}": float(v) for b, v in zip(BATCHES, d[i])}})
        print(f"{fam:16s} p={len(cols):3d}  LOSO acc {acc:.3f}  perm p {res[fam]['perm_p']:.3f}  "
              f"recall B1/B2/B3 {res[fam]['recall']['Batch_1']:.2f}/{res[fam]['recall']['Batch_2']:.2f}/{b3_recall:.2f}", flush=True)
    pd.concat(loso).to_csv(OUT / "loso.csv", index=False)
    HO = pd.DataFrame(ho)
    HO.to_csv(OUT / "heldout.csv", index=False)
    print(HO.pivot(index="site", columns="family", values="call").to_string())
    (OUT / "results.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(res)
    acc = [res[k]["loso_accuracy"] for k in names]
    nm = [res[k]["null_mean"] for k in names]
    ax.bar(range(len(names)), acc, color=["#c44" if n.endswith("raw") else "#48a" for n in names])
    ax.plot(range(len(names)), nm, "k_", ms=18, label="permutation null mean")
    ax.axhline(0.677, color="k", ls="--", lw=1, label="fingerprint NB (0.68)")
    ax.axhline(0.548, color="grey", ls=":", lw=1, label="majority (0.55)")
    for i, k in enumerate(names):
        ax.text(i, acc[i] + 0.01, f"p={res[k]['perm_p']:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("LOSO accuracy")
    ax.set_title("Nearest batch mean: what 'the mean of the image' reads", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "accuracy.png", dpi=150)
    return 0


if __name__ == "__main__":
    sys.exit(main())
