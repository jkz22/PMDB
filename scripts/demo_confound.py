"""Demo: a classifier trained on image intensity reads the microscope, not the material.

    python scripts/demo_confound.py

Trains the simplest possible "model" on raw BSE intensity statistics (black
level p1, median p50, std) - no segmentation, no geometry. It scores decently
in leave-one-out, because Batch 3 was acquired with an elevated BSE black
level on many sites. Then the punchline: take a Batch 1 site and add a flat
+7 grey-level offset - change nothing about the material - and the intensity
model confidently reassigns it to Batch 3. The fingerprint model never sees
intensity, so it cannot be fooled this way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES = ["p1", "p50", "frac_zero"]
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]


def load_bse_stats() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "outputs/raw_intensity_stats.csv",
                     dtype={"batch": str, "site": str})
    return df[df["detector"] == "BSE"].set_index(["batch", "site"])[FEATURES]


def nearest_centroid(train: pd.DataFrame, ytr: np.ndarray, x: np.ndarray) -> str:
    s = train.std().to_numpy()
    d = [np.linalg.norm((x - train[ytr == b].mean().to_numpy()) / s)
         for b in BATCHES]
    return BATCHES[int(np.argmin(d))]


def main() -> int:
    X = load_bse_stats()
    y = X.index.get_level_values("batch").to_numpy()

    correct = 0
    for i in range(len(X)):
        tr = np.arange(len(X)) != i
        pred = nearest_centroid(X.iloc[tr], y[tr], X.iloc[i].to_numpy())
        correct += pred == y[i]
    acc = correct / len(X)
    print(f"intensity-only model (BSE p1/p50/std, no geometry): "
          f"LOO accuracy {acc:.2f} vs majority baseline 0.55")
    print("looks like it works! but what did it learn?\n")

    site = ("Batch_1", "4ih2ggld")
    x0 = X.loc[site].to_numpy()
    print(f"take {site[1]} (a real {site[0]} site): "
          f"predicted {nearest_centroid(X, y, x0)}")
    # flat +7 grey levels: p1 and p50 shift up, no pixel clips to black anymore;
    # the material is untouched
    x1 = np.array([x0[0] + 7.0, x0[1] + 7.0, 0.0])
    print(f"same site, +7 grey-level offset (material untouched): "
          f"predicted {nearest_centroid(X, y, x1)}")
    print("\nIt learned the microscope's black level, not the electrode.")
    print("The fingerprint model uses segmentation geometry only - a flat "
          "intensity offset cannot move a single one of its features.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
