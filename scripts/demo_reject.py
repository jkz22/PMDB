"""Live demo: a drifting 'Batch 4' shipment gets rejected as out-of-distribution.

    python scripts/demo_reject.py

Fits the fingerprint model on all labelled sites, then takes a real Batch 3
(baseline) site and drifts its spatial-arrangement features step by step toward
heavy sedimentation: Si sinking to the bottom bands, mid-range order changing,
within-site heterogeneity growing. At each step the model reports per-batch
conformal p-values. Early steps stay accepted (assigned to a batch with honest
credibility); once the site is less typical than every labelled site of every
batch, the verdict flips to REJECT - the accept/reject decision a manufacturer
needs when an unknown batch N arrives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import fingerprint as fp  # noqa: E402


def drift(x: pd.Series, t: float) -> pd.Series:
    """Drift a feature vector toward heavy sedimentation by factor t in [0, 1+]."""
    d = x.copy()
    # Si migrates to the bottom: bottom bands up, top bands down
    d["si_depth_rel_band0"] *= 1 - 0.5 * t
    d["si_depth_rel_band1"] *= 1 - 0.3 * t
    d["si_depth_rel_band3"] *= 1 + 0.4 * t
    d["si_depth_rel_band4"] *= 1 + 0.8 * t
    d["si_depth_slope"] = d["si_depth_rel_band4"] - d["si_depth_rel_band0"]
    d["si_depth_mid_dip"] = d["si_depth_rel_band2"] - (
        d["si_depth_rel_band0"] + d["si_depth_rel_band4"]) / 2
    # clustering coarsens: pair correlation recovers later
    for c in d.index:
        if c.startswith(("gx_", "gz_")):
            d[c] *= 1 - 0.35 * t
    # the electrode becomes less homogeneous
    d["k15_contact_tilestd"] *= 1 + 4.0 * t
    return d


def main() -> int:
    fdir = ROOT / "outputs" / "kpis"
    X = fp.read_feature_inputs(fdir / "curves.csv", fdir / "tile_kpis.csv")
    y = pd.Series(X.index.get_level_values("batch"), index=X.index)
    model = fp.fit(X, y)

    # start from the most typical Batch 3 site (highest conformal p for Batch 3)
    p3 = fp.conformal_p(model, X)["Batch_3"]
    is_b3 = X.index.get_level_values("batch") == "Batch_3"
    base = X.loc[p3[is_b3].idxmax()]
    print(f"incoming shipment drifts from the Batch 3 baseline site "
          f"{p3[is_b3].idxmax()[1]} toward heavy sedimentation:\n")
    header = f"{'drift':>6} {'verdict':<28} {'credibility':>11} " + " ".join(
        f"{'p(' + b.replace('_', ' ') + ')':>12}" for b in model.batches)
    print(header)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        probe = pd.DataFrame([drift(base, t)], index=pd.MultiIndex.from_tuples(
            [("Batch_N", f"drift_{t:.2f}")], names=["batch", "site"]))
        pred = fp.predict(model, probe).iloc[0]
        verdict = ("REJECT: out of distribution" if pred["ood"]
                   else f"accept as {pred['assigned'].replace('_', ' ')}")
        ps = " ".join(f"{pred[f'p_{b}']:>12.3f}" for b in model.batches)
        print(f"{t:>6.2f} {verdict:<28} {pred['credibility']:>11.3f} {ps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
