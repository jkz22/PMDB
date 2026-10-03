"""Preliminary leave-one-site-out outlier ranking (numbers in docs/reports/preliminary-site-outliers.md).

Score: RMS robust z over the verdict KPIs, median/MAD from the other 30 sites.
Confidence: conformal p = (1 + #calibration scores >= score) / n, calibration by nested leave-one-out.
Not the final pipeline: the two top sites mask each other in the calibration set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb.outliers import VERDICT_FAMILIES, robust_z  # noqa: E402


def score(ref: np.ndarray, x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(robust_z(ref, x[None])[0] ** 2)))


def main() -> None:
    d = pd.read_csv(ROOT / "outputs/kpis/site_kpis.csv")
    cols = [c for _k, m in VERDICT_FAMILIES.values() for c in m]
    X = d[cols].values.astype(float)
    n = len(d)
    s = np.array([score(np.delete(X, i, 0), X[i]) for i in range(n)])
    p = []
    for i in range(n):
        R = np.delete(X, i, 0)
        cal = np.array([score(np.delete(R, j, 0), R[j]) for j in range(n - 1)])
        p.append((1 + np.sum(cal >= s[i])) / n)
    out = pd.DataFrame({"batch": d.batch, "site": d.site, "rms_z": s.round(3), "p_conformal": np.round(p, 3)})
    out = out.sort_values("rms_z", ascending=False)
    dest = ROOT / "outputs/outliers/prelim_site_ranking.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
