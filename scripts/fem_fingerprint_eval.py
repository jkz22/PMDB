"""D21 pre-registered FEM-on-fingerprint test (see .claude/plans/fem-swelling.md, D21).

    python scripts/fem_fingerprint_eval.py                       # tag "main"
    python scripts/fem_fingerprint_eval.py --fem-site-curves outputs/fem/edge5/site_curves.csv --tag edge5

Arms (leave-one-site-out over the 31 labelled sites, everything refit per fold):
    A0  Leo's 16 fingerprint features (must reproduce 21/31)
    A1  A0 + swell_resid (in-fold OLS of swelling@s=1 on K01) + si_stress_spread (s=1)
    A2  A0 + top-2 raw FEM site metrics chosen inside each fold (training sites only)
Decision rule (D21): an FEM arm adds value iff LOO correct >= 23/31 and the label-
permutation p (>= 1000 perms, whole pipeline incl. in-fold selection rerun) <= 0.05;
both pass -> higher accuracy, tie -> A1; otherwise A0 stays the classifier of record.
With --tag other than "main" the run is a sensitivity analysis (A3), never the final arm.

Writes outputs/fem_fingerprint/<tag>/: metrics.csv, loo_predictions.csv, permutation.json,
selected_features.csv, heldout_predictions.csv, verdict.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import fingerprint as fp  # noqa: E402

FRAMES = {"s0.5": 5, "s1.0": 10}
S1_FRAME = 10
# D21 exclusions: si_yield_frac, Si stress / pressure / J magnitudes, sxx_mean, surface_rough
EXCLUDE_RE = re.compile(r"vm_si|p_si|J_si|sxx_mean|surface_rough|si_yield")
N_SELECT = 2
RHO_MAX = 0.9
MIN_CORRECT = 23
ALPHA = 0.05
ARMS = ("A0", "A1", "A2")
LEO_N_CORRECT = 21
ARRAY_KEYS = ("index", "leo", "swell", "spread", "k01", "cand")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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


def load_inputs(fem_path):
    rd = dict(dtype={"batch": str, "site": str})
    leo = pd.read_csv(ROOT / "outputs/fingerprint/features.csv", **rd).set_index(["batch", "site"])
    leo_h = pd.read_csv(ROOT / "outputs/fingerprint/heldout_features.csv", **rd).set_index(["batch", "site"])
    k = pd.read_csv(ROOT / "outputs/kpis/site_kpis.csv", dtype={"site": str}).set_index("site")["K01_si_frac_adm"]
    kh = pd.read_csv(ROOT / "outputs/heldout/kpis/site_kpis.csv", dtype={"site": str}).set_index("site")["K01_si_frac_adm"]
    fem = load_fem(fem_path, False)
    fem_h = load_fem(fem_path, True)

    def pack(L, F, K):
        sites = L.index.get_level_values("site")
        cand = F.drop(columns=["swell", "si_stress_spread"]).reindex(sites)
        return {"index": np.array(list(L.index), dtype=object).reshape(-1, 2),
                "leo": L.to_numpy(float),
                "swell": F["swell"].reindex(sites).to_numpy(float),
                "spread": F["si_stress_spread"].reindex(sites).to_numpy(float),
                "k01": K.reindex(sites).to_numpy(float),
                "cand": cand.to_numpy(float), "cand_names": list(cand.columns)}

    tr, te = pack(leo, fem, k), pack(leo_h, fem_h, kh)
    # the finite-candidate screen is applied inside each fold (training rows only), see fold_matrices
    for d in (tr, te):
        for name in ("swell", "spread", "k01"):
            if not np.isfinite(d[name]).all():
                raise ValueError(f"non-finite {name}")
    y = pd.Series(leo.index.get_level_values("batch"), index=leo.index, name="batch")
    return tr, te, y, list(leo.columns)


def subset(d, idx):
    return {k: (v[idx] if k in ARRAY_KEYS else v) for k, v in d.items()}


def _mi(index_arr):
    return pd.MultiIndex.from_tuples([tuple(r) for r in index_arr], names=["batch", "site"])


# ---------------------------------------------------------------------------
# In-fold feature construction
# ---------------------------------------------------------------------------

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


def fold_matrices(arm: str, tr: dict, te: dict, codes_tr: np.ndarray):
    """Training and test feature matrices for one fold; every fitted quantity (OLS,
    selection) uses the training rows only."""
    if arm == "A0":
        return tr["leo"], te["leo"], [], None
    if arm == "A1":
        b1, b0 = np.polyfit(tr["k01"], tr["swell"], 1)
        r_tr = tr["swell"] - (b0 + b1 * tr["k01"])
        r_te = te["swell"] - (b0 + b1 * te["k01"])
        return (np.column_stack([tr["leo"], r_tr, tr["spread"]]),
                np.column_stack([te["leo"], r_te, te["spread"]]),
                ["swell_resid", "si_stress_spread"], None)
    if arm == "A2":
        ok = np.flatnonzero(np.isfinite(tr["cand"]).all(axis=0))  # training-rows-only availability screen
        sel, p, skipped = select_features(tr["cand"][:, ok], codes_tr)
        chosen = [int(ok[j]) for j in sel]
        names = [tr["cand_names"][j] for j in chosen]
        return (np.column_stack([tr["leo"], tr["cand"][:, chosen]]),
                np.column_stack([te["leo"], te["cand"][:, chosen]]), names,
                {"chosen": chosen, "p": p, "skipped": skipped})
    raise ValueError(arm)


def fast_assign(xtr, codes_tr, xte, n_batches):
    """Leo's training-fit assignment (lowest score), numpy only."""
    center, scale, ok, mu, sb = fp._fit_core(xtr, codes_tr, n_batches)
    z = np.clip((xte[:, ok] - center[ok]) / scale[ok], -fp.MAX_Z, fp.MAX_Z)
    return fp._scores(z, mu, sb).argmin(axis=1)


def loo_codes(arm, tr, codes, n_batches):
    n = len(codes)
    out = np.empty(n, dtype=int)
    for i in range(n):
        m = np.arange(n) != i
        xtr, xte, _, _ = fold_matrices(arm, subset(tr, m), subset(tr, ~m), codes[m])
        out[i] = fast_assign(xtr, codes[m], xte, n_batches)[0]
    return out


def _perm_chunk(args):
    arm, tr, codes, n_batches, seed, ks = args
    out = []
    for k in ks:
        cp = np.random.default_rng([seed, int(k)]).permutation(codes)
        out.append((loo_codes(arm, tr, cp, n_batches) == cp).mean())
    return out


def permutation_arm(arm, tr, codes, n_batches, n_perm, seed, workers=None):
    """Label-permutation test of the whole LOO pipeline (selection rerun per permutation).
    Permutation k is seeded by (seed, k), so results do not depend on the worker count."""
    obs = float((loo_codes(arm, tr, codes, n_batches) == codes).mean())
    workers = workers or os.cpu_count() or 1
    chunks = [(arm, tr, codes, n_batches, seed, c) for c in np.array_split(np.arange(n_perm), workers * 4)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        null = np.array([v for part in ex.map(_perm_chunk, chunks) for v in part])
    print(f"  {arm}: {n_perm} permutations in {time.time() - t0:.0f}s on {workers} workers", flush=True)
    return {"observed_accuracy": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1.0 + (null >= obs - 1e-12).sum()) / (n_perm + 1.0)),
            "n_perm": int(n_perm), "seed": int(seed)}


# ---------------------------------------------------------------------------
# Full (Leo predict) LOO and held-out scoring
# ---------------------------------------------------------------------------

def frames(arm, tr, te, codes_tr, leo_cols):
    xtr, xte, names, info = fold_matrices(arm, tr, te, codes_tr)
    cols = list(leo_cols) + names
    return (pd.DataFrame(xtr, index=_mi(tr["index"]), columns=cols),
            pd.DataFrame(xte, index=_mi(te["index"]), columns=cols), names, info)


def loo_full(arm, tr, y, batches, leo_cols):
    codes = np.array([batches.index(b) for b in y])
    n = len(codes)
    rows, sel = [], []
    for i in range(n):
        m = np.arange(n) != i
        Xtr, Xte, names, info = frames(arm, subset(tr, m), subset(tr, ~m), codes[m], leo_cols)
        pred = fp.predict(fp.fit(Xtr, y[m]), Xte)
        pred.insert(0, "true", y.iloc[i])
        rows.append(pred)
        if info:
            for r, (j, pv) in enumerate(zip(info["chosen"], info["p"]), 1):
                sel.append({"left_out_site": tr["index"][i][1], "rank": r,
                            "feature": tr["cand_names"][j], "kw_p_train": float(pv)})
    return pd.concat(rows), sel


def summarize(pred, batches):
    correct = pred["assigned"] == pred["true"]
    rec = {b: float(correct[pred["true"] == b].mean()) for b in batches}
    conf = {bt: {ba: int(((pred["true"] == bt) & (pred["assigned"] == ba)).sum()) for ba in batches}
            for bt in batches}
    return {"n_correct": int(correct.sum()), "n": int(len(pred)), "accuracy": float(correct.mean()),
            "balanced_accuracy": float(np.mean(list(rec.values()))), "recall": rec, "confusion": conf}


def decide(results: dict, sensitivity: bool = False) -> tuple[str, dict]:
    """Apply the D21 rule mechanically. Sensitivity runs (tag != main) are never promoted: A0 stays the arm."""
    passes = {a: bool(results[a]["n_correct"] >= MIN_CORRECT and results[a]["perm_p"] <= ALPHA)
              for a in ("A1", "A2")}
    if sensitivity:
        chosen = "A0"
    elif passes["A1"] and passes["A2"]:
        chosen = "A2" if results["A2"]["n_correct"] > results["A1"]["n_correct"] else "A1"
    elif passes["A1"]:
        chosen = "A1"
    elif passes["A2"]:
        chosen = "A2"
    else:
        chosen = "A0"
    return chosen, passes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fem-site-curves", default="outputs/fem/site_curves.csv")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.n_perm < 1000:
        ap.error("D21 requires >= 1000 permutations")

    tr, te, y, leo_cols = load_inputs(ROOT / args.fem_site_curves)
    batches = sorted(y.unique())
    codes = np.array([batches.index(b) for b in y])
    print(f"{len(y)} labelled sites, {len(leo_cols)} Leo features, "
          f"{len(tr['cand_names'])} candidate raw FEM metrics, tag={args.tag}")

    leo_loo = pd.read_csv(ROOT / "outputs/fingerprint/loo_predictions.csv",
                          dtype={"batch": str, "site": str}).set_index(["batch", "site"])
    preds, sels, results, perms = {}, {}, {}, {}
    for arm in ARMS:
        pred, sel = loo_full(arm, tr, y, batches, leo_cols)
        fast = loo_codes(arm, tr, codes, len(batches))
        assert (np.array(batches)[fast] == pred["assigned"].to_numpy()).all(), \
            f"{arm}: fast path disagrees with Leo predict"
        preds[arm], sels[arm] = pred, sel
        results[arm] = summarize(pred, batches)
        if arm == "A0":
            assert results["A0"]["n_correct"] == LEO_N_CORRECT, results["A0"]["n_correct"]
            assert (pred["assigned"].reindex(leo_loo.index) == leo_loo["assigned"]).all(), \
                "A0 per-site predictions differ from outputs/fingerprint/loo_predictions.csv"
            print("A0 reproduces Leo: 21/31 and identical per-site predictions")
        perms[arm] = permutation_arm(arm, tr, codes, len(batches), args.n_perm, args.seed)
        assert abs(perms[arm]["observed_accuracy"] - results[arm]["accuracy"]) < 1e-12
        results[arm]["perm_p"] = perms[arm]["p_value"]
        print(f"{arm}: {results[arm]['n_correct']}/31 acc {results[arm]['accuracy']:.3f} "
              f"bal {results[arm]['balanced_accuracy']:.3f} perm p {perms[arm]['p_value']:.4f}")

    chosen, passes = decide(results, sensitivity=args.tag != "main")
    print(f"verdict: chosen arm {chosen} (pass: {passes})")

    # held-out scoring after the rule; in-fold quantities refit on all 31 labelled sites
    held = []
    for arm in sorted({chosen, "A0"}):
        Xtr, Xte, names, _ = frames(arm, tr, te, codes, leo_cols)
        p = fp.predict(fp.fit(Xtr, y), Xte)
        p.insert(0, "arm", arm)
        p.insert(1, "extra_features", ";".join(names))
        held.append(p)
    held = pd.concat(held)

    out = ROOT / "outputs/fem_fingerprint" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    mrows = []
    for arm in ARMS:
        r = results[arm]
        row = {"arm": arm, "n_correct": r["n_correct"], "n": r["n"], "accuracy": r["accuracy"],
               "balanced_accuracy": r["balanced_accuracy"], "perm_p": r["perm_p"],
               "null_mean": perms[arm]["null_mean"], "null_p95": perms[arm]["null_p95"],
               "passes_rule": passes.get(arm, "reference")}
        row.update({f"recall_{b}": v for b, v in r["recall"].items()})
        row.update({f"conf_{bt}_as_{ba}": v for bt, d in r["confusion"].items() for ba, v in d.items()})
        mrows.append(row)
    pd.DataFrame(mrows).to_csv(out / "metrics.csv", index=False)
    pd.concat({a: preds[a] for a in ARMS}, names=["arm"]).reset_index().to_csv(
        out / "loo_predictions.csv", index=False)
    (out / "permutation.json").write_text(json.dumps(perms, indent=2))
    sel = pd.DataFrame(sels["A2"])
    sel.to_csv(out / "selected_features.csv", index=False)
    held.reset_index().to_csv(out / "heldout_predictions.csv", index=False)

    freq = sel.groupby("feature").size().sort_values(ascending=False)
    sens = "" if args.tag == "main" else (
        f"**Sensitivity run (A3, tag `{args.tag}`, table `{args.fem_site_curves}`): never the final arm.**")
    lines = [f"# D21 verdict (tag `{args.tag}`)", "", sens, "",
             f"Rule: FEM arm adds value iff LOO correct >= {MIN_CORRECT}/31 and permutation "
             f"p <= {ALPHA} ({args.n_perm} perms, seed {args.seed}, in-fold selection rerun).", "",
             "| arm | correct | accuracy | balanced acc | perm p | null mean | passes |",
             "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        r = results[arm]
        lines.append(f"| {arm} | {r['n_correct']}/31 | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | "
                     f"{r['perm_p']:.4f} | {perms[arm]['null_mean']:.3f} | {passes.get(arm, 'reference')} |")
    lines += ["", f"**Verdict: classifier of record = {chosen}**"
              + (" (no FEM arm passed the rule)" if chosen == "A0" else ""), "",
              "A2 selected features (count over 31 folds):", ""]
    lines += [f"- {f}: {c}" for f, c in freq.items()]
    lines += ["", "Held-out predictions:", "", "```",
              held[["arm", "assigned", "credibility", "confidence", "ood"]].reset_index().to_string(index=False),
              "```", ""]
    (out / "verdict.md").write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
