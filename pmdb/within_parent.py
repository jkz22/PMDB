"""Within-parent batch contrasts: which fingerprint features differ consistently between batches for crops of
the same parent image. See docs/patch_mil.md (Feedback round 1)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pmdb import patch_lopo as plo

PAIRS = (("Batch_1", "Batch_2"), ("Batch_1", "Batch_3"), ("Batch_2", "Batch_3"))


def parent_contrasts(F: pd.DataFrame, label: pd.Series, parent: pd.Series) -> pd.DataFrame:
    """Long table [parent_id, pair, feature, n_a, n_b, diff] with diff = mean_a - mean_b, for each parent
    holding both batches of a pair."""
    lab, par = np.asarray(label), np.asarray(parent)
    rows = []
    for p in pd.unique(par):
        in_p = par == p
        for a, b in PAIRS:
            ma, mb = in_p & (lab == a), in_p & (lab == b)
            if not ma.any() or not mb.any():
                continue
            d = F[ma].mean() - F[mb].mean()
            for f, v in d.items():
                rows.append({"parent_id": p, "pair": f"{a}-{b}", "feature": f, "n_a": int(ma.sum()),
                             "n_b": int(mb.sum()), "diff": float(v)})
    return pd.DataFrame(rows, columns=["parent_id", "pair", "feature", "n_a", "n_b", "diff"])


def sign_test_p(n_pos: int, n_neg: int) -> float:
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    k = min(n_pos, n_neg)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def summarise_contrasts(C: pd.DataFrame, sd: pd.Series) -> pd.DataFrame:
    rows = []
    for (pair, f), g in C.groupby(["pair", "feature"], sort=False):
        d = g["diff"].to_numpy()
        n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
        n = len(d)
        med = float(np.median(d))
        cons = max(n_pos, n_neg) / n
        rows.append({"feature": f, "pair": pair, "n_parents": n, "n_pos": n_pos, "n_neg": n_neg,
                     "p_sign": sign_test_p(n_pos, n_neg), "median_diff": med,
                     "median_diff_sd": med / float(sd[f]), "consistency": cons,
                     "is_signal": bool(n >= 3 and cons == 1.0)})
    S = pd.DataFrame(rows)
    S["_abs"] = S["median_diff_sd"].abs()
    S = S.sort_values(["pair", "is_signal", "n_parents", "_abs"], ascending=[True, False, False, False],
                      kind="stable").drop(columns="_abs").reset_index(drop=True)
    return S


def signal_sentence(S: pd.DataFrame, max_items: int = 3) -> str:
    sig = S[S["is_signal"]].copy()
    if sig.empty:
        return ""
    sig["_abs"] = sig["median_diff_sd"].abs()
    sig = sig.sort_values(["n_parents", "_abs"], ascending=False, kind="stable").head(max_items)
    items = []
    for r in sig.itertuples():
        a, b = (x.split("_")[1] for x in r.pair.split("-"))
        items.append(f"{plo.feature_phrase(r.feature)} is {'higher' if r.median_diff > 0 else 'lower'} in "
                     f"Batch {a} than in Batch {b} ({r.n_parents} of {r.n_parents} source images)")
    return ("Differences that hold between crops of the same source image assigned to different batches: "
            + "; ".join(items) + ".")
