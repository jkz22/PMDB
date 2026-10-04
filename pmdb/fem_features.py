"""FEM site-feature pieces shared by the D21 (fingerprint) and D22 (XGBoost) tests.

All fitted quantities (OLS residual, Kruskal-Wallis selection, Spearman de-duplication) take
training rows only; callers pass the training fold explicitly.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

FRAMES = {"s0.5": 5, "s1.0": 10}
S1_FRAME = 10
# D21 exclusions: si_yield_frac, Si stress / pressure / J magnitudes, sxx_mean, surface_rough
EXCLUDE_RE = re.compile(r"vm_si|p_si|J_si|sxx_mean|surface_rough|si_yield")
N_SELECT = 2
RHO_MAX = 0.9
ARRAY_KEYS = ("index", "leo", "kpi", "swell", "spread", "k01", "cand")


def load_fem(path, heldout: bool) -> pd.DataFrame:
    """Per-site table (index site): swelling@s=1, si_stress_spread@s=1 and the raw
    candidate metrics at s=0.5 and s=1.0 (orientation sym), D21 exclusions removed."""
    sc = pd.read_csv(path, dtype={"batch": str, "site": str})
    sc = sc[(sc["orientation"] == "sym") & (sc["heldout"].astype(str) == str(heldout))]
    metrics = [c for c in sc.columns[9:] if not EXCLUDE_RE.search(c)]
    parts = []
    for tag, fr in FRAMES.items():
        g = sc[sc["frame"] == fr].set_index("site")
        part = g[metrics].copy()
        part.columns = [f"{c}@{tag}" for c in metrics]
        parts.append(part)
    out = pd.concat(parts, axis=1)
    g1 = sc[sc["frame"] == S1_FRAME].set_index("site")
    out.insert(0, "swell", g1["swelling"])
    out.insert(1, "si_stress_spread", (g1["q75_vm_si"] - g1["q25_vm_si"]) / g1["q50_vm_si"])
    return out


def subset(d, idx):
    return {k: (v[idx] if k in ARRAY_KEYS else v) for k, v in d.items()}


def swell_resid_columns(tr: dict, te: dict):
    """A1 physics features (train/test): swelling minus its OLS fit on K01 (fitted on the
    training rows only) and si_stress_spread. Returns (train cols, test cols, names)."""
    b1, b0 = np.polyfit(tr["k01"], tr["swell"], 1)
    r_tr = tr["swell"] - (b0 + b1 * tr["k01"])
    r_te = te["swell"] - (b0 + b1 * te["k01"])
    return (np.column_stack([r_tr, tr["spread"]]), np.column_stack([r_te, te["spread"]]),
            ["swell_resid", "si_stress_spread"])


def kw_pvalues(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Kruskal-Wallis p per column (NaN for constant columns); equals scipy.stats.kruskal."""
    n = len(x)
    ranks = rankdata(x, axis=0)
    groups = np.unique(codes)
    h = np.full(x.shape[1], -3.0 * (n + 1))
    for g in groups:
        m = codes == g
        h = h + 12.0 / (n * (n + 1)) * ranks[m].sum(axis=0) ** 2 / m.sum()
    tie = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        _, t = np.unique(x[:, j], return_counts=True)
        tie[j] = 1.0 - (t ** 3 - t).sum() / (n ** 3 - n)
    p = np.full(x.shape[1], np.nan)
    ok = tie > 0
    p[ok] = chi2.sf(h[ok] / tie[ok], len(groups) - 1)
    return p


def select_features(cand_tr: np.ndarray, codes_tr: np.ndarray, k: int = N_SELECT,
                    rho_max: float = RHO_MAX):
    """Top-k columns by KW p on the training rows, greedily skipping any candidate with
    |Spearman| > rho_max to an already chosen one (training rows only).
    Returns (chosen column indices, their p, skipped (col, rho, kept_col) tuples)."""
    p = kw_pvalues(cand_tr, codes_tr)
    order = [j for j in np.argsort(np.where(np.isnan(p), np.inf, p), kind="stable")
             if np.isfinite(p[j])]
    ranks = rankdata(cand_tr, axis=0)
    chosen, skipped = [], []
    for j in order:
        dup = None
        for c in chosen:
            rho = np.corrcoef(ranks[:, j], ranks[:, c])[0, 1]
            if abs(rho) > rho_max:
                dup = (int(j), float(rho), int(c))
                break
        if dup:
            skipped.append(dup)
            continue
        chosen.append(int(j))
        if len(chosen) == k:
            break
    return chosen, p[chosen], skipped
