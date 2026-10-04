"""Hunt the feature the artificial batches were built from.

The organiser said the batches are artificial groupings of crops based on "some feature they
calculated or observed", and the parent images are split across batches. If that feature is
(close to) a scalar we already compute, a single-feature two-cutpoint rule should reproduce
the 34 labels far better than chance. We test every scalar we have:

- v1 KPIs, functional (swelling) features, fingerprint16, grey/imaging statistics
  (clean-pipeline summary: noise, edge width, gain, black level, defect-mask fractions;
  raw intensity stats per detector; nearest-mean site features).

For each feature and each of the three batch orderings (which batch is the middle class) the
best two-cutpoint accuracy is found exhaustively. Significance of the best feature is judged
against a max-statistic permutation null (labels permuted, the maximum over all features and
orderings recomputed each time), so the search itself is accounted for. Two variants:

- raw: the rule on the site value (a feature computed on the crop);
- parent-centred: the rule on value minus the parent mean (a feature *ranked within* each
  parent image), which is how one would split crops of one image into groups.

Labels: 31 labelled + 3 released held-out truths (outputs/heldout_labels.csv). Nothing is
fitted for prediction; this is a descriptive search. Writes outputs/parents/feature_hunt*.csv/.md.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
O = ROOT / "outputs"
OUT = O / "parents"
B = ["Batch_1", "Batch_2", "Batch_3"]
ORDERS = {"B1<B2<B3": (0, 1, 2), "B1<B3<B2": (0, 2, 1), "B2<B1<B3": (1, 0, 2)}  # middle class distinct
N_PERM = 1000
MIN_UNIQUE = 6


def rd(p: str) -> pd.DataFrame:
    return pd.read_csv(O / p, dtype={"batch": str, "site": str})


def table() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    pg = pd.read_csv(O / "parent_groups.csv", dtype=str).set_index("site")
    truth = pd.read_csv(O / "heldout_labels.csv", dtype=str).set_index("site")["batch"]
    y = pg["batch"].copy()
    y.loc[truth.index] = truth
    parts = []
    for lab, held in (("kpis/site_kpis.csv", "heldout/kpis/site_kpis.csv"),
                      ("functional/site_functional.csv", "functional/heldout_site_functional.csv"),
                      ("fingerprint/features.csv", "fingerprint/heldout_features.csv"),
                      ("clean/summary.csv", "clean_heldout/summary.csv")):
        d = pd.concat([rd(lab), rd(held)]).set_index("site")
        parts.append(d.select_dtypes("number"))
    mp = rd("meanpred/site_features.csv").set_index("site").select_dtypes("number")
    parts.append(mp[[c for c in mp.columns if not c.startswith("grey_raw_hist") and not c.startswith("grey_harm_hist")
                     and "hist" not in c]])
    ri = pd.concat([rd("raw_intensity_stats.csv"), rd("heldout/raw_intensity_stats.csv")])
    ri["detector"] = ri["detector"].replace({"ETD": "SE_type", "SE": "SE_type"})
    piv = ri.pivot_table(index="site", columns="detector", values=["mean", "std", "p1", "p99", "frac_zero", "frac_255"])
    piv.columns = [f"raw_{d}_{v}" for v, d in piv.columns]
    parts.append(piv)
    X = pd.concat(parts, axis=1).loc[pg.index]
    X = X.loc[:, ~X.columns.duplicated()]
    keep = [c for c in X.columns if X[c].notna().sum() >= 30 and X[c].nunique() >= MIN_UNIQUE]
    return X[keep], y.loc[X.index], pg.loc[X.index, "parent_id"]


def best_two_cut(v: np.ndarray, rank: np.ndarray) -> tuple[float, float, float]:
    """Max accuracy of rule rank = 0 if v<=c1, 1 if c1<v<=c2, 2 otherwise (ties/NaN count wrong)."""
    ok = np.isfinite(v)
    v, rank = v[ok], rank[ok]
    n_all = len(ok)
    order = np.argsort(v, kind="stable")
    r = rank[order]
    n = len(r)
    low = np.concatenate([[0], np.cumsum(r == 0)])          # low[i] = #rank0 among first i
    mid = np.concatenate([[0], np.cumsum(r == 1)])
    high_tot = (r == 2).sum()
    high = high_tot - np.concatenate([[0], np.cumsum(r == 2)])
    vs = v[order]
    allowed = np.ones(n + 1, bool)                          # a cut may only fall between distinct values
    allowed[1:n] = vs[1:] > vs[:-1]
    best, bi, bj = -1, 0, 0
    for i in np.flatnonzero(allowed):                       # first i sorted are "low"
        acc = low[i] + (mid[i:] - mid[i]) + high[i:]        # j from i..n
        acc = np.where(allowed[i:], acc, -1)
        j = int(np.argmax(acc)) + i
        if acc[j - i] > best:
            best, bi, bj = acc[j - i], i, j
    c1 = vs[bi - 1] if bi > 0 else -np.inf
    c2 = vs[bj - 1] if bj > 0 else -np.inf
    return best / n_all, float(c1), float(c2)


def hunt(X: pd.DataFrame, codes: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        v = X[c].to_numpy(float)
        for name, perm in ORDERS.items():
            rank = np.asarray(perm)[codes]
            acc, c1, c2 = best_two_cut(v, rank)
            rows.append((c, name, acc, c1, c2))
    return pd.DataFrame(rows, columns=["feature", "order", "acc", "cut1", "cut2"])


def max_stat(X: pd.DataFrame, codes: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    V = X.to_numpy(float)
    out = np.empty(N_PERM)
    for k in range(N_PERM):
        cp = rng.permutation(codes)
        m = 0.0
        for j in range(V.shape[1]):
            for perm in ORDERS.values():
                m = max(m, best_two_cut(V[:, j], np.asarray(perm)[cp])[0])
        out[k] = m
    return out


def within_parent_concordance(x: pd.Series, codes: np.ndarray, parents: pd.Series, perm: tuple) -> tuple[int, int]:
    """Pairs of sites from the same parent with different batches: how many have the feature
    ordered the same way as the batch rank?"""
    rank = np.asarray(perm)[codes]
    conc = tot = 0
    df = pd.DataFrame({"x": x.to_numpy(float), "r": rank, "p": parents.to_numpy()})
    for _, g in df.groupby("p"):
        for a, b in itertools.combinations(range(len(g)), 2):
            ra, rb = g.r.iloc[a], g.r.iloc[b]
            xa, xb = g.x.iloc[a], g.x.iloc[b]
            if ra == rb or not (np.isfinite(xa) and np.isfinite(xb)):
                continue
            tot += 1
            conc += int((xa - xb) * (ra - rb) > 0)
    return conc, tot


def wp_pairs(parents: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ia, ib = [], []
    for _, idx in pd.Series(np.arange(len(parents)), index=parents.index).groupby(parents.values):
        for a, b in itertools.combinations(idx.to_numpy(), 2):
            ia.append(a); ib.append(b)
    return np.asarray(ia), np.asarray(ib)


def wp_concordance_all(V: np.ndarray, rank: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Per feature: fraction of within-parent pairs with different batch ranks whose feature order agrees (ties 0.5)."""
    dr = np.sign(rank[ia] - rank[ib])
    use = dr != 0
    dx = np.sign(V[ia[use]] - V[ib[use]])            # (pairs, features)
    s = dx * dr[use][:, None]
    return np.where(np.isnan(s), 0.5, (s + 1) / 2).sum(0) / use.sum()


def wp_maxstat(V: np.ndarray, codes: np.ndarray, parents: pd.Series, ia, ib, seed: int = 1) -> np.ndarray:
    """Null: labels permuted *within* each parent (parent composition fixed); max over features and orders."""
    rng = np.random.default_rng(seed)
    pidx = [np.flatnonzero(parents.values == g) for g in pd.unique(parents.values)]
    out = np.empty(N_PERM)
    for k in range(N_PERM):
        cp = codes.copy()
        for ix in pidx:
            cp[ix] = rng.permutation(cp[ix])
        out[k] = max(wp_concordance_all(V, np.asarray(perm)[cp], ia, ib).max() for perm in ORDERS.values())
    return out


def md_table(df: pd.DataFrame, cols: list[str], fmt: dict) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(getattr(r, c)) for c in cols) + " |")
    return lines


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, parents = table()
    codes = np.array([B.index(b) for b in y])
    maj = np.bincount(codes).max() / len(codes)
    Xc = X - X.groupby(parents.values).transform("mean")
    Xc = Xc.loc[:, Xc.nunique() >= MIN_UNIQUE]
    res = {}
    for tag, M in (("raw", X), ("parent_centred", Xc)):
        h = hunt(M, codes)
        null = max_stat(M, codes)
        h["p_maxstat"] = [(1 + (null >= a).sum()) / (N_PERM + 1) for a in h.acc]
        cw = [within_parent_concordance(M[f], codes, parents, ORDERS[o]) for f, o in zip(h.feature, h.order)]
        h["wp_concordant"], h["wp_pairs"] = zip(*cw)
        h = h.sort_values("acc", ascending=False).reset_index(drop=True)
        h.to_csv(OUT / f"feature_hunt_{tag}.csv", index=False)
        res[tag] = (h, null)

    lines = ["# Feature hunt: can one scalar + two cutpoints reproduce the 34 batch labels?", "",
             f"{len(X.columns)} scalar features, 3 batch orderings, 34 sites (31 labelled + 3 released truths), "
             f"majority-class accuracy {maj:.3f}. p is a max-statistic permutation p ({N_PERM} label permutations, "
             "maximum over all features and orderings each time), so it already pays for the search.",
             "`wp` = within-parent pairs of sites with different batches whose feature ordering agrees with the batch ordering.", ""]
    for tag, (h, null) in res.items():
        lines += [f"## {tag}", "",
                  f"Null (max over features) : median {np.median(null):.3f}, 95th pct {np.percentile(null, 95):.3f}, max {null.max():.3f}.",
                  f"Best observed: {h.acc.iloc[0]:.3f} ({h.feature.iloc[0]}, {h.order.iloc[0]}), p = {h.p_maxstat.iloc[0]:.3f}.", ""]
        fmt = {"acc": "{:.3f}", "p_maxstat": "{:.3f}", "cut1": "{:.3g}", "cut2": "{:.3g}"}
        cols = ["feature", "order", "acc", "p_maxstat", "wp_concordant", "wp_pairs", "cut1", "cut2"]
        lines += md_table(h.head(20), cols, fmt)
        lines += ["", f"### {tag}: ranked by within-parent concordance (the split that was actually designed)", ""]
        lines += md_table(h.sort_values(["wp_concordant", "acc"], ascending=False).head(20), cols, fmt)
        lines.append("")
    ia, ib = wp_pairs(parents)
    V = X.to_numpy(float)
    rows = []
    for name, perm in ORDERS.items():
        conc = wp_concordance_all(V, np.asarray(perm)[codes], ia, ib)
        rows += [(f, name, c) for f, c in zip(X.columns, conc)]
    W = pd.DataFrame(rows, columns=["feature", "order", "wp_frac"])
    null = wp_maxstat(V, codes, parents, ia, ib)
    W["p_maxstat"] = [(1 + (null >= c).sum()) / (N_PERM + 1) for c in W.wp_frac]
    W = W.sort_values("wp_frac", ascending=False).reset_index(drop=True)
    W.to_csv(OUT / "feature_hunt_within_parent.csv", index=False)
    n_pairs = int((np.sign(codes[ia] - codes[ib]) != 0).sum())
    lines += ["## within-parent concordance (the designed split: labels permuted within parents)", "",
              f"{n_pairs} within-parent pairs with different batches. Statistic = fraction of those pairs whose feature "
              "ordering matches the batch ordering; null = labels shuffled within each parent, max over all features and orderings.",
              f"Null: median {np.median(null):.3f}, 95th pct {np.percentile(null, 95):.3f}, max {null.max():.3f}. "
              f"Best observed: {W.wp_frac.iloc[0]:.3f} ({W.feature.iloc[0]}, {W.order.iloc[0]}), p = {W.p_maxstat.iloc[0]:.3f}.", ""]
    lines += md_table(W.head(25), ["feature", "order", "wp_frac", "p_maxstat"], {"wp_frac": "{:.3f}", "p_maxstat": "{:.3f}"})
    lines.append("")
    (OUT / "feature_hunt.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
