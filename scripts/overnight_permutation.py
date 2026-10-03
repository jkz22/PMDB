"""Overnight run 3: 10,000-label-permutation test of the merged fingerprint pipeline.

Mirrors pmdb.fingerprint.permutation_test exactly (same observed statistic, same rng
sequence for a given seed, same p-value estimator) but runs in chunks with a JSON
checkpoint after each chunk, so progress is visible and a crash resumes cheaply by
re-running. Model code and defaults are untouched.

    .venv/Scripts/python scripts/overnight_permutation.py --n-perm 10000

Writes to outputs/overnight/permutation/:
    permutation_10k.json   observed accuracy, p-value, null summary (updated per chunk)
    null_accuracies.csv    the full null distribution (one accuracy per permutation)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import fingerprint as fp  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kpis-dir", default="outputs/kpis")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "overnight" / "permutation"))
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250)
    args = ap.parse_args(argv)

    kdir = ROOT / args.kpis_dir
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    X = fp.read_feature_inputs(kdir / "curves.csv", kdir / "tile_kpis.csv")
    y = pd.Series(X.index.get_level_values("batch"), index=X.index).astype(str)
    batches = sorted(y.unique())
    codes = np.asarray([batches.index(b) for b in y])
    x = X.to_numpy(dtype=float)
    n_batches = len(batches)

    observed = float((fp._loo_assignments(x, codes, n_batches) == codes).mean())
    print(f"{len(X)} sites, observed LOO accuracy {observed:.4f}", flush=True)

    rng = np.random.default_rng(args.seed)
    null = np.empty(args.n_perm)
    for k in range(args.n_perm):
        cp = rng.permutation(codes)
        null[k] = (fp._loo_assignments(x, cp, n_batches) == cp).mean()
        done = k + 1
        if done % args.chunk == 0 or done == args.n_perm:
            d = null[:done]
            p = (1.0 + float((d >= observed).sum())) / (done + 1.0)
            result = {
                "git_commit": git_commit(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
                "elapsed_seconds": round(time.time() - t_start, 1),
                "n_perm_done": done,
                "n_perm_target": args.n_perm,
                "seed": args.seed,
                "observed_accuracy": observed,
                "p_value": p,
                "null_mean": float(d.mean()),
                "null_sd": float(d.std(ddof=1)) if done > 1 else None,
                "null_p95": float(np.percentile(d, 95)),
                "null_max": float(d.max()),
                "n_null_ge_observed": int((d >= observed).sum()),
            }
            (out / "permutation_10k.json").write_text(json.dumps(result, indent=2))
            pd.DataFrame({"perm": np.arange(done), "accuracy": d}).to_csv(
                out / "null_accuracies.csv", index=False)
            print(f"  {done}/{args.n_perm} perms, p = {p:.5f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    print(f"done: p = {result['p_value']:.5f} after {args.n_perm} permutations "
          f"in {(time.time() - t_start) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
