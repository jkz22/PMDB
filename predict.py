"""Incoming-batch QC against the approved baseline (BASELINE, default Batch_3).

KPI verdict (default): gated teammate KPIs per field at 50 nm/px, compared with the baseline
field distribution (z-score and baseline range) -> PASS / REVIEW per field and KPI.

    python predict.py --sites Batch_3/hawkfj64 Batch_3/hzumfsms
    python predict.py --npz path/to/new_field.npz               # uint8 (H, W, 3) array under key 'img'

Representation (adds novelty from a learned embedding):

    python predict.py --sites Batch_3/hawkfj64 --representation best        # leaderboard rank 1
    python predict.py --sites Batch_3/hawkfj64 --representation <run_hash>

Representation novelty = conformal p-value of the field's mean embedding distance to the
baseline centroid, against leave-one-out baseline fields (small p = unlike baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2 import kpi_adapter as K
from src.v2.common import BASELINE, CROP, NM_HALF, OUT, grid, load_half_raw, manifest

Z_REVIEW = 3.0


def field_kpis(img: np.ndarray, key: str) -> dict:
    return K.kpis_from_masks(K.segment(img[..., 0], NM_HALF), NM_HALF, key, ids=K.CROP_KPIS)


def baseline_kpis() -> pd.DataFrame:
    cache = OUT / "kpis" / f"baseline_field_kpis_{BASELINE}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    m = manifest()
    rows = [{"group_id": g, **field_kpis(load_half_raw(b, s), g)}
            for b, s, g in m[m.batch == BASELINE][["batch", "site", "group_id"]].itertuples(index=False)]
    df = pd.DataFrame(rows)[["group_id", *K.GATED_COLS]]
    df.to_csv(cache, index=False)
    return df


def kpi_verdict(name: str, img: np.ndarray, base: pd.DataFrame) -> list[dict]:
    k = field_kpis(img, name)
    out = []
    for c in K.GATED_COLS:
        mu, sd = base[c].mean(), base[c].std(ddof=1)
        z = (k[c] - mu) / sd if sd > 0 and np.isfinite(k[c]) else np.nan
        inside = base[c].min() <= k[c] <= base[c].max()
        out.append(dict(field=name, kpi=c, value=k[c], baseline_mean=mu, baseline_sd=sd, z=z,
                        in_baseline_range=bool(inside),
                        verdict="REVIEW" if (not np.isfinite(z)) or abs(z) > Z_REVIEW else "PASS"))
    return out


def resolve_run(rep: str) -> Path:
    runs = OUT / "runs"
    if rep == "best":
        lb = pd.read_csv(OUT / "leaderboard.csv")
        rep = lb.sort_values("selection_rank").iloc[0]["hash"]
    d = runs / rep
    if not (d / "embeddings_eval.npz").exists():
        sys.exit(f"run {rep} not found locally under {runs} (pull it from the pmdb-v2 Modal volume)")
    return d


def representation_novelty(name: str, img: np.ndarray, run_dir: Path) -> dict:
    import torch

    from src.v2.data import VIEWS, normalise_percentile
    from src.v2.models import build
    cfg = json.load(open(run_dir / "config.json"))
    fam = cfg["family"]
    if fam == "vae_b":
        raise SystemExit("VAE-B (KPI-conditioned) needs crop KPIs as encoder input; choose another representation")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # inference only
    model = build(fam) if fam.startswith("ots_") else build(fam, n_kpi=len(K.kpi_cols(cfg.get("kpi_set"))), vae_mask=cfg["vae_mask"],
                                                           mae_mask=cfg["mae_mask"])
    if not fam.startswith("ots_"):
        model.load_state_dict(torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)["model"])
    model.to(dev).eval()
    if cfg.get("harmonise"):
        raise SystemExit("harmonised runs need per-image GMM stats of the new field; not supported here yet")
    x = normalise_percentile(img) if cfg["input"] == "norm" else img.astype(np.float32) / 255.0
    x = x[..., list(VIEWS[cfg["view"]])]
    crops = np.stack([x[y:y + CROP, xx:xx + CROP] for y, xx in grid(*x.shape[:2], CROP, CROP)])
    with torch.no_grad():
        E = torch.cat([model.embed(torch.from_numpy(c).permute(0, 3, 1, 2).to(dev)).float().cpu()
                       for c in np.array_split(crops, max(1, len(crops) // 64))]).numpy()
    z = np.load(run_dir / "embeddings_eval.npz", allow_pickle=True)
    Eb, gb = z["E"], z["group_id"]
    mu, sd = Eb.mean(0), Eb.std(0) + 1e-6
    F = pd.DataFrame((Eb - mu) / sd).groupby(gb).mean()
    base = F[[g.startswith(BASELINE + "/") for g in F.index]].to_numpy()
    f = ((E - mu) / sd).mean(0)
    loo = np.array([np.linalg.norm(base[i] - np.delete(base, i, 0).mean(0)) for i in range(len(base))])
    d = float(np.linalg.norm(f - base.mean(0)))
    p = float((1 + (loo >= d).sum()) / (1 + len(loo)))
    return dict(field=name, representation=run_dir.name, family=fam, view=cfg["view"], n_crops=len(crops),
                dist_to_baseline=d, conformal_p=p, verdict="REVIEW" if p <= 1 / (1 + len(loo)) else "PASS")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*", default=[], help="batch/site ids from cache/half")
    ap.add_argument("--npz", nargs="*", default=[], help="uint8 (H,W,3) [BSE, Inlens, SE_type] at 50 nm/px, key 'img'")
    ap.add_argument("--representation", default=None, help="'best' or a run hash under outputs/v2/runs")
    ap.add_argument("--out", default=None, help="optional CSV path for the KPI verdict table")
    a = ap.parse_args(argv)
    fields = [(s, load_half_raw(*s.split("/"))) for s in a.sites] + [(p, np.load(p)["img"]) for p in a.npz]
    if not fields:
        ap.error("give --sites and/or --npz")
    base = baseline_kpis()
    kv = pd.DataFrame([r for n, im in fields for r in kpi_verdict(n, im, base)])
    print(f"KPI verdict (commit {K.kpi_commit_hash()[:7]}, gated KPIs only, |z|>{Z_REVIEW} -> REVIEW)")
    print(kv.round(4).to_string(index=False))
    if a.out:
        kv.to_csv(a.out, index=False)
    if a.representation:
        run_dir = resolve_run(a.representation)
        rv = pd.DataFrame([representation_novelty(n, im, run_dir) for n, im in fields])
        print("\nRepresentation novelty vs baseline")
        print(rv.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
