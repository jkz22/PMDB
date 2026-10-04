#!/usr/bin/env python3
"""Tile voting: run the KPIs *and* the arrangement features on full-depth tiles, classify every tile,
and let the tiles vote for the field. The vote share of the winning batch is the field's confidence; a
cutoff on it turns the classifier into an accept/abstain rule, evaluated as accuracy vs coverage.

Per tile (16 or 32 equal-width, full-height strips; `pmdb.kpis.tile_slices`):
  kpis         the reliable v1 tile KPIs from outputs/overnight/features/tile_kpis_rich.csv (ICC >= 0.75)
  arrangement  Si fraction in 5 depth bands relative to the tile mean, slope, mid-dip, SOC-1 pore loss
Classifiers: logistic regression (standardised, C = 0.3) and XGBoost (depth 2, 100 rounds).
Evaluation: leave-one-SITE-out (all tiles of the held-out field removed), plurality vote over its tiles,
label-permutation null on the LOSO vote accuracy, accuracy-vs-coverage for cutoffs on the vote share.
Held-back sites: fit on all labelled tiles, same vote rule, same cutoff.

Writes outputs/tilevote/{tile_features.csv, loso_votes.csv, coverage.csv, heldout.csv, results.json,
figures/coverage.png}.  CPU only.
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
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmdb import functional, segment as segment_mod  # noqa: E402
from pmdb.io import list_sites, load_site  # noqa: E402
from pmdb.kpis import tile_slices  # noqa: E402
from pmdb.kpis.fields import band_profile  # noqa: E402

O = ROOT / "outputs"
OUT = O / "tilevote"
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
HELD = ["3e122cbj", "fn0mhxef", "xrv9xvzb"]
TILE_COUNTS = (16, 32)
N_BANDS = 5
KPIS = ["K01_si_frac_adm", "K02_si_density_per_1000um2", "K03_ecd_d50_um", "K03_ecd_d90_um", "K03_ecd_max_um",
        "K05_voronoi_sigma", "K09_mst_m_norm", "K09_mst_sigma_norm", "K14_empty_p95_um", "K15_si_graphite_contact_frac"]
ARR = [f"si_depth_rel_band{i}" for i in range(N_BANDS)] + ["si_depth_slope", "si_depth_mid_dip", "F02_soc100_pore_loss"]
CUTOFFS = np.round(np.arange(0.34, 0.96, 0.02), 2)


# ----------------------------------------------------------------------------- features
def tile_arrangement(batch: str, site: str) -> list[dict]:
    kw = {"data_root": ROOT / "data_heldout", "cache_root": ROOT / "cache_heldout"} if batch == "Batch_heldout" else {}
    raw = load_site(batch, site, resolution="half", normalise="none", **kw)
    masks = segment_mod.segment(raw)
    rows = []
    for n in TILE_COUNTS:
        for t, sl in enumerate(tile_slices(masks.shape[1], n)):
            tm = masks.crop(slice(None), sl)
            prof = band_profile(tm.si, tm.fraction_space, n_bands=N_BANDS)
            rel = prof / prof.mean() if prof.mean() > 0 else np.full(N_BANDS, np.nan)
            x = np.linspace(-1, 1, N_BANDS)
            row = {"batch": batch, "site": site, "n_tiles": n, "tile": t}
            row.update({f"si_depth_rel_band{i}": float(v) for i, v in enumerate(rel)})
            row["si_depth_slope"] = float(np.polyfit(x, rel, 1)[0]) if np.all(np.isfinite(rel)) else np.nan
            row["si_depth_mid_dip"] = float(rel[N_BANDS // 2] - 0.5 * (rel[0] + rel[-1])) if np.all(np.isfinite(rel)) else np.nan
            try:
                row["F02_soc100_pore_loss"] = float(functional.swelling_test(tm, 1.0)["pore_loss"])
            except Exception:
                row["F02_soc100_pore_loss"] = np.nan
            rows.append(row)
    return rows


def build_features(workers: int) -> pd.DataFrame:
    sites = [(r.batch, r.site) for r in list_sites().itertuples()] + [("Batch_heldout", s) for s in HELD]
    with ProcessPoolExecutor(workers) as ex:
        arr = pd.DataFrame([r for rows in ex.map(tile_arrangement, *zip(*sites)) for r in rows])
    kp = pd.read_csv(O / "overnight" / "features" / "tile_kpis_rich.csv", dtype={"batch": str, "site": str})
    kp = kp[kp.n_tiles.isin(TILE_COUNTS)][["batch", "site", "n_tiles", "tile"] + KPIS]
    F = arr.merge(kp, on=["batch", "site", "n_tiles", "tile"], how="left")
    F.to_csv(OUT / "tile_features.csv", index=False)
    return F


# ----------------------------------------------------------------------------- models
def make_model(kind: str):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
    return XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=100, max_depth=2, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3, n_jobs=1, verbosity=0, tree_method="hist")


def impute(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)


def vote(P: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Plurality vote over tile predictions. Returns (call, vote share of call, mean posterior)."""
    counts = np.bincount(P.argmax(1), minlength=3)
    call = int(counts.argmax())
    return call, float(counts[call] / len(P)), P.mean(0)


def loso_votes(X: np.ndarray, site_of_row: np.ndarray, y_site: dict, kind: str) -> pd.DataFrame:
    sites = list(y_site)
    y_row = np.array([y_site[s] for s in site_of_row])
    out = []
    for s in sites:
        te = site_of_row == s
        a, b = impute(X[~te], X[te])
        m = make_model(kind).fit(a, y_row[~te])
        P = np.zeros((te.sum(), 3))
        P[:, m.classes_] = m.predict_proba(b)
        call, share, mp = vote(P)
        out.append({"site": s, "true": y_site[s], "call": call, "vote_share": share, "n_tiles": int(te.sum()),
                    **{f"meanp_{i}": float(v) for i, v in enumerate(mp)}})
    return pd.DataFrame(out)


def coverage_table(V: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CUTOFFS:
        sel = V.vote_share >= c
        rows.append({"cutoff": float(c), "coverage": float(sel.mean()),
                     "accuracy": float((V.call[sel] == V.true[sel]).mean()) if sel.any() else np.nan,
                     "n_covered": int(sel.sum())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reuse-features", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    F = pd.read_csv(OUT / "tile_features.csv", dtype={"batch": str, "site": str}) if a.reuse_features else build_features(a.workers)
    FAMILIES = {"kpis": KPIS, "arrangement": ARR, "all": KPIS + ARR}
    results: dict = {"configs": {}}
    votes, cov = [], []
    for n in TILE_COUNTS:
        L = F[(F.n_tiles == n) & F.batch.isin(BATCHES)].reset_index(drop=True)
        y_site = {s: BATCHES.index(b) for b, s in L[["batch", "site"]].drop_duplicates().itertuples(index=False)}
        site_of_row = L.site.to_numpy()
        for fam, cols in FAMILIES.items():
            X = L[cols].to_numpy(float)
            for kind in ("lr", "xgb"):
                tag = f"{fam}|{kind}|{n}"
                V = loso_votes(X, site_of_row, y_site, kind)
                acc = float((V.call == V.true).mean())
                acc_meanp = float((V[[f"meanp_{i}" for i in range(3)]].to_numpy().argmax(1) == V.true).mean())
                V.insert(0, "config", tag)
                votes.append(V)
                C = coverage_table(V)
                C.insert(0, "config", tag)
                cov.append(C)
                results["configs"][tag] = {"loso_vote_accuracy": acc, "loso_meanposterior_accuracy": acc_meanp,
                                           "recall": {b: float((V.call[V.true == i] == i).mean()) for i, b in enumerate(BATCHES)},
                                           "mean_vote_share_correct": float(V.vote_share[V.call == V.true].mean()),
                                           "mean_vote_share_wrong": float(V.vote_share[V.call != V.true].mean()) if (V.call != V.true).any() else np.nan}
                print(f"{tag:22s} vote acc {acc:.3f}  mean-post acc {acc_meanp:.3f}  share correct {results['configs'][tag]['mean_vote_share_correct']:.2f} "
                      f"wrong {results['configs'][tag]['mean_vote_share_wrong']:.2f}", flush=True)
    VV = pd.concat(votes)
    VV.to_csv(OUT / "loso_votes.csv", index=False)
    CC = pd.concat(cov)
    CC.to_csv(OUT / "coverage.csv", index=False)

    # permutation null for the best config per family (by vote accuracy) at 16 tiles
    best = {}
    for fam in FAMILIES:
        cands = {k: v["loso_vote_accuracy"] for k, v in results["configs"].items() if k.startswith(fam + "|")}
        best[fam] = max(cands, key=cands.get)
    results["best_per_family"] = best
    results["permutation"] = {}
    for fam, tag in best.items():
        _, kind, n = tag.split("|")
        n = int(n)
        L = F[(F.n_tiles == n) & F.batch.isin(BATCHES)].reset_index(drop=True)
        y_site = {s: BATCHES.index(b) for b, s in L[["batch", "site"]].drop_duplicates().itertuples(index=False)}
        X = L[FAMILIES[fam]].to_numpy(float)
        sor = L.site.to_numpy()
        obs = results["configs"][tag]["loso_vote_accuracy"]
        rng = np.random.default_rng(0)
        sites = list(y_site)
        truth = np.array([y_site[s] for s in sites])

        def stat(perm):
            ys = dict(zip(sites, perm))
            V = loso_votes(X, sor, ys, kind)
            return float((V.call == V.true).mean())
        null = np.asarray(Parallel(n_jobs=8)(delayed(stat)(rng.permutation(truth)) for _ in range(a.n_perm)))
        results["permutation"][tag] = {"observed": obs, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                                       "p_value": float((1 + (null >= obs).sum()) / (a.n_perm + 1)), "n_perm": a.n_perm}
        print(f"perm {tag}: obs {obs:.3f} null mean {null.mean():.3f} p95 {np.percentile(null, 95):.3f} p {results['permutation'][tag]['p_value']:.3f}", flush=True)

    # held-out under the best 'all' config
    tag = best["all"]
    fam, kind, n = tag.split("|")
    n = int(n)
    L = F[(F.n_tiles == n) & F.batch.isin(BATCHES)].reset_index(drop=True)
    H = F[(F.n_tiles == n) & (F.batch == "Batch_heldout")].reset_index(drop=True)
    y_row = np.array([BATCHES.index(b) for b in L.batch])
    a_, b_ = impute(L[FAMILIES[fam]].to_numpy(float), H[FAMILIES[fam]].to_numpy(float))
    m = make_model(kind).fit(a_, y_row)
    P = np.zeros((len(H), 3))
    P[:, m.classes_] = m.predict_proba(b_)
    rows = []
    for s in HELD:
        sel = (H.site == s).to_numpy()
        call, share, mp = vote(P[sel])
        rows.append({"config": tag, "site": s, "call": BATCHES[call], "vote_share": share,
                     **{f"meanp_{b}": float(v) for b, v in zip(BATCHES, mp)},
                     "tile_calls_B1_B2_B3": "/".join(str(int(v)) for v in np.bincount(P[sel].argmax(1), minlength=3))})
    HO = pd.DataFrame(rows)
    HO.to_csv(OUT / "heldout.csv", index=False)
    print(HO.round(3).to_string())

    # figure: accuracy vs coverage for the best config per family + fingerprint reference
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for fam, tag in best.items():
        C = CC[CC.config == tag]
        ax.plot(C.coverage, C.accuracy, "o-", ms=3, label=f"{tag} (vote cutoff sweep)")
    ax.axhline(0.677, color="k", ls="--", lw=1, label="fingerprint, all sites (0.68)")
    ax.axhline(0.548, color="grey", ls=":", lw=1, label="majority class (0.55)")
    ax.set_xlabel("coverage (share of fields with vote share >= cutoff)")
    ax.set_ylabel("LOSO accuracy among covered fields")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Tile voting: confidence cutoff trades coverage for accuracy?", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "coverage.png", dpi=150)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
