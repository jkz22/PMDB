"""Overnight run 2: stability suite for the merged fingerprint model.

Two diagnostics of how stable the three held-out assignments are, with the model
code and its defaults untouched (constants are varied per-config in-process only):

A. Jackknife: drop 1-3 random labelled sites, refit, re-predict the 3 held-out
   sites. Drop-1 (31 subsets) and drop-2 (465) are enumerated exhaustively;
   drop-3 samples --n-drop3 distinct subsets (default 1000, seed 0).
B. Hyperparameter grid over SCALE_SHRINK x MAX_Z x G_OBS_BINS: LOO accuracy and
   held-out assignments per config. Recorded for tomorrow's single evaluation
   decision; nothing is selected here.

    .venv/Scripts/python scripts/overnight_stability.py

Writes to outputs/overnight/stability/:
    jackknife_resamples.csv  one row per (resample, held-out site)
    jackknife_summary.csv    per held-out site x drop-k: assignment shares
    hyperparam_grid.csv      one row per (config, held-out site) + LOO metrics
    run_log.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import fingerprint as fp  # noqa: E402

DEFAULTS = {"SCALE_SHRINK": fp.SCALE_SHRINK, "MAX_Z": fp.MAX_Z, "G_OBS_BINS": fp.G_OBS_BINS}

GRID_SCALE_SHRINK = (0.25, 0.4, 0.5, 0.6, 0.75)
GRID_MAX_Z = (5.0, 10.0, 20.0)
GRID_G_OBS_BINS = {
    "default4": ((0.5, 2.0), (2.0, 4.0), (4.0, 7.0), (7.0, 10.0)),
    "fine6": ((0.5, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 5.5), (5.5, 7.5), (7.5, 10.0)),
    "coarse3": ((0.5, 3.0), (3.0, 7.0), (7.0, 10.0)),
}


def load_inputs(kpis_dir: Path, heldout_dir: Path):
    X = fp.read_feature_inputs(kpis_dir / "curves.csv", kpis_dir / "tile_kpis.csv")
    y = pd.Series(X.index.get_level_values("batch"), index=X.index, name="batch")
    H = fp.read_feature_inputs(heldout_dir / "curves.csv", heldout_dir / "tile_kpis.csv")
    return X, y, H


def predict_heldout(X, y, H) -> pd.DataFrame:
    model = fp.fit(X, y)
    return fp.predict(model, H)


def check_n_drop3(n_drop3: int, n_sites: int) -> None:
    cap = math.comb(n_sites, 3)
    if not 0 <= n_drop3 <= cap:
        raise SystemExit(f"--n-drop3 must be between 0 and {cap} "
                         f"(distinct 3-site subsets of {n_sites} sites), got {n_drop3}")


def jackknife(X, y, H, n_drop3: int, seed: int) -> pd.DataFrame:
    n = len(X)
    subsets: list[tuple[int, tuple[int, ...]]] = []
    subsets += [(1, (i,)) for i in range(n)]
    subsets += [(2, c) for c in itertools.combinations(range(n), 2)]
    check_n_drop3(n_drop3, n)
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    while len(seen) < n_drop3:
        seen.add(tuple(sorted(rng.choice(n, size=3, replace=False).tolist())))
    subsets += [(3, c) for c in sorted(seen)]

    rows = []
    t0 = time.time()
    for r, (k, drop) in enumerate(subsets):
        keep = np.ones(n, dtype=bool)
        keep[list(drop)] = False
        pred = predict_heldout(X.iloc[keep], y.iloc[keep], H)
        dropped = ";".join(f"{b}/{s}" for b, s in X.index[list(drop)])
        for (batch, site), p in pred.iterrows():
            rows.append({
                "resample": r, "drop_k": k, "dropped": dropped, "site": site,
                "assigned": p["assigned"], "credibility": p["credibility"],
                "confidence": p["confidence"], "ood": p["ood"],
                **{c: p[c] for c in pred.columns if c.startswith(("p_", "score_"))},
            })
        if (r + 1) % 200 == 0:
            print(f"  jackknife {r + 1}/{len(subsets)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarise_jackknife(res: pd.DataFrame, batches: list[str]) -> pd.DataFrame:
    rows = []
    for (site, k), grp in res.groupby(["site", "drop_k"]):
        row = {"site": site, "drop_k": k, "n_resamples": len(grp),
               "ood_share": float(grp["ood"].mean()),
               "mean_credibility": float(grp["credibility"].mean()),
               "mean_confidence": float(grp["confidence"].mean())}
        for b in batches:
            row[f"share_{b}"] = float((grp["assigned"] == b).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["site", "drop_k"]).reset_index(drop=True)


def hyperparam_grid(kpis_dir: Path, heldout_dir: Path) -> pd.DataFrame:
    rows = []
    configs = list(itertools.product(GRID_SCALE_SHRINK, GRID_MAX_Z, GRID_G_OBS_BINS))
    t0 = time.time()
    try:
        for i, (shrink, max_z, bins_name) in enumerate(configs):
            fp.SCALE_SHRINK = shrink
            fp.MAX_Z = max_z
            fp.G_OBS_BINS = GRID_G_OBS_BINS[bins_name]
            X, y, H = load_inputs(kpis_dir, heldout_dir)  # features depend on G_OBS_BINS
            _, metrics = fp.loo_evaluate(X, y)
            pred = predict_heldout(X, y, H)
            for (batch, site), p in pred.iterrows():
                rows.append({
                    "scale_shrink": shrink, "max_z": max_z, "g_obs_bins": bins_name,
                    "is_default": (shrink == DEFAULTS["SCALE_SHRINK"]
                                   and max_z == DEFAULTS["MAX_Z"] and bins_name == "default4"),
                    "loo_accuracy": metrics["accuracy"],
                    **{f"recall_{b}": v for b, v in metrics["recall"].items()},
                    "site": site, "assigned": p["assigned"],
                    "credibility": p["credibility"], "confidence": p["confidence"],
                    "ood": p["ood"],
                    **{c: p[c] for c in pred.columns if c.startswith("p_")},
                })
            print(f"  grid {i + 1}/{len(configs)} shrink={shrink} max_z={max_z} "
                  f"bins={bins_name} loo={metrics['accuracy']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        fp.SCALE_SHRINK = DEFAULTS["SCALE_SHRINK"]
        fp.MAX_Z = DEFAULTS["MAX_Z"]
        fp.G_OBS_BINS = DEFAULTS["G_OBS_BINS"]
    return pd.DataFrame(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kpis-dir", default="outputs/kpis")
    ap.add_argument("--heldout-dir", default="outputs/heldout/kpis")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "overnight" / "stability"))
    ap.add_argument("--n-drop3", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    kdir, hdir = ROOT / args.kpis_dir, ROOT / args.heldout_dir
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    X, y, H = load_inputs(kdir, hdir)
    batches = sorted(y.unique())
    print(f"{len(X)} labelled sites, {len(H)} held-out, {X.shape[1]} features", flush=True)

    print("jackknife...", flush=True)
    res = jackknife(X, y, H, n_drop3=args.n_drop3, seed=args.seed)
    res.to_csv(out / "jackknife_resamples.csv", index=False)
    summary = summarise_jackknife(res, batches)
    summary.to_csv(out / "jackknife_summary.csv", index=False)
    print(summary.to_string(), flush=True)

    print("hyperparameter grid...", flush=True)
    grid = hyperparam_grid(kdir, hdir)
    grid.to_csv(out / "hyperparam_grid.csv", index=False)

    run_log = {
        "git_commit": git_commit(),
        "command": " ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "total_seconds": round(time.time() - t_start, 1),
        "n_jackknife_resamples": int(res["resample"].nunique()),
        "n_drop3": args.n_drop3,
        "seed": args.seed,
        "grid": {"scale_shrink": list(GRID_SCALE_SHRINK), "max_z": list(GRID_MAX_Z),
                 "g_obs_bins": {k: [list(b) for b in v] for k, v in GRID_G_OBS_BINS.items()}},
        "defaults": {"SCALE_SHRINK": DEFAULTS["SCALE_SHRINK"], "MAX_Z": DEFAULTS["MAX_Z"],
                     "G_OBS_BINS": [list(b) for b in DEFAULTS["G_OBS_BINS"]]},
    }
    (out / "run_log.json").write_text(json.dumps(run_log, indent=2))
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
