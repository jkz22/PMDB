"""Run the batch fingerprint model: LOO validation, permutation test, held-out assignment.

    python scripts/run_fingerprint.py
    python scripts/run_fingerprint.py --heldout-dir outputs/heldout/kpis

Writes to --out-dir (default outputs/fingerprint/):
    evaluation.json          LOO accuracy, confusion, permutation test, feature list
    features.csv             per labelled site: the fingerprint feature table
    loo_predictions.csv      per labelled site: true batch, assigned, p per batch
    heldout_features.csv     per held-out site: the fingerprint feature table
    heldout_predictions.csv  per held-out site: assigned, credibility, confidence, OOD flag
    heldout_explain.csv      per held-out site x feature: evidence vs each batch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import fingerprint as fp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kpis-dir", default="outputs/kpis",
                    help="directory with curves.csv and tile_kpis.csv for labelled sites")
    ap.add_argument("--heldout-dir", default=None,
                    help="directory with curves.csv and tile_kpis.csv for unlabelled sites")
    ap.add_argument("--out-dir", default="outputs/fingerprint")
    ap.add_argument("--ood-alpha", type=float, default=fp.DEFAULT_OOD_ALPHA)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kdir = ROOT / args.kpis_dir
    X = fp.read_feature_inputs(kdir / "curves.csv", kdir / "tile_kpis.csv")
    y = pd.Series(X.index.get_level_values("batch"), index=X.index, name="batch")
    print(f"{len(X)} labelled sites, {X.shape[1]} fingerprint features")

    loo_pred, metrics = fp.loo_evaluate(X, y, ood_alpha=args.ood_alpha)
    majority = y.value_counts(normalize=True).max()
    print(f"LOO accuracy: {metrics['accuracy']:.3f} "
          f"(majority-class baseline {majority:.3f})")
    print("confusion (true -> assigned):")
    for bt, row in metrics["confusion"].items():
        print(f"  {bt}: {row}")

    perm = fp.permutation_test(X, y, n_perm=args.n_perm, seed=args.seed)
    print(f"permutation test: p = {perm['p_value']:.4f} "
          f"(null mean {perm['null_mean']:.3f}, {perm['n_perm']} perms)")

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    X.to_csv(out / "features.csv")
    loo_pred.to_csv(out / "loo_predictions.csv")

    model = fp.fit(X, y)
    evaluation = {
        "features": model.features,
        "n_sites": int(len(X)),
        "majority_baseline": float(majority),
        "loo": metrics,
        "permutation": perm,
        "ood_alpha": args.ood_alpha,
        "inputs": {"kpis_dir": args.kpis_dir, "heldout_dir": args.heldout_dir},
    }
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2))

    if args.heldout_dir:
        hdir = ROOT / args.heldout_dir
        H = fp.read_feature_inputs(hdir / "curves.csv", hdir / "tile_kpis.csv")
        H.to_csv(out / "heldout_features.csv")
        pred = fp.predict(model, H, ood_alpha=args.ood_alpha)
        pred.to_csv(out / "heldout_predictions.csv")
        fp.explain(model, H).to_csv(out / "heldout_explain.csv", index=False)
        print("\nheld-out assignments:")
        cols = ["assigned", "credibility", "confidence", "ood"] + \
               [f"p_{b}" for b in model.batches]
        print(pred[cols].to_string())

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
