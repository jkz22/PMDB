"""Sibling-crop information vs the fingerprint naive Bayes, LOSO on Modal.

    modal run scripts/modal_sibling_prior.py              # 200 label permutations
    modal run scripts/modal_sibling_prior.py --n-perm 10  # smoke test
    # new test sites: list them with their parent group, features in the fingerprint format
    modal run scripts/modal_sibling_prior.py --heldout-only \
        --heldout-sites my_sites.csv --heldout-features my_features.csv

The 31 labelled + 3 held-out sites are crops of ~15 parent electrode images
(organiser info, PARENTS below). Pre-stated models, evaluated leave-one-site-out:

- nb:        fingerprint robust naive Bayes (pmdb.fingerprint), material features only.
- sib_vote:  majority batch of the test crop's labelled siblings; tie / no siblings -> nb.
- nb_prior:  posterior_b ∝ exp(-T * score_b) * prior_b, score = fp._scores (mean
             per-feature Laplace NLL), prior_b = (n_sib_b + a) / (n_sib + 3a).
             Primary T = 16 (number of features: turns the mean NLL back into the
             naive-Bayes sum), a = 1 (Laplace).
- nb_prior_t4: sensitivity, T = 4, a = 1. The 16 features are strongly dependent
             (5 band fractions with fixed mean, slope/dip are linear combinations of
             them, 4+4 neighbouring g(r) bins), roughly 4 independent groups
             (depth profile, g_x, g_z, k15), so T = 16 overcounts the evidence.

Sibling information exploits how the dataset was built (crop provenance), not
electrode material. Under leave-one-parent-out a test crop has no labelled
siblings by construction, so sib_vote and nb_prior collapse to nb there; only
LOSO is evaluated. The permutation test shuffles batch labels over the 31 sites
(seeds 0..n_perm-1) with the parent grouping fixed and reruns every model.

LOSO metrics also include the organiser rubric (P(predicted) > 0.5 = high;
correct high 2, correct low 1, wrong low 1, wrong high 0). nb probability =
softmax(-16 * score) (T = number of features); sib_vote P = sibling vote share.

Writes outputs/sibling_prior/{evaluation.json, loso_predictions.csv,
heldout_predictions.csv, purity.csv, null_permutations.csv}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.4", "scipy==1.14.1", "scikit-image==0.25.2", "pandas==2.1.4",
        "Pillow==12.3.0", "tifffile==2025.5.10", "imagecodecs==2025.3.30",
    )
    .add_local_python_source("pmdb")
)
app = modal.App("pmdb-sibling-prior", image=image)
FN_KW = dict(cpu=1.0, memory=2048, timeout=1800)

PARENTS = {
    "G1612": ["ptg8lmto", "xgj4xftb"],
    "G1780": ["iv6g2oq0"],
    "G1880": ["uhdslk0o"],
    "G1904": ["0grcilhi", "hawkfj64", "mgxahqnk"],
    "G2048": ["avn74qx1", "3806gxp0"],
    "G2060": ["71vgq3fw", "tuy3zymq", "x7u69zsw", "kbdh4tri"],
    "G2068SE": ["rxax5ozo", "x77cy643", "utfgcjfa", "vc2whyaq"],
    "G2080": ["ffwubibz", "r17byphk", "cfe5vt7s"],
    "G2088": ["ufdvpb81", "hzumfsms", "9luzk4jm"],
    "G2148": ["f1vzngrs", "epqdaau9"],
    "G2156": ["fzrt2k6r", "b3esycq1"],
    "G2272": ["pl8uabbv", "i9jiqjwl"],
    "G2316": ["4ih2ggld", "5n1q8atc"],
}
#: Held-out sites and their parent group: CSV with columns `site,parent`. A parent
#: not in PARENTS is allowed (no labelled siblings, so sibling models fall back to nb).
HELDOUT_SITES_CSV = "outputs/sibling_prior/heldout_sites.csv"
HELDOUT_FEATURES_CSV = "outputs/fingerprint/heldout_features.csv"
SETTINGS = {"nb_prior": (16.0, 1.0), "nb_prior_t4": (4.0, 1.0)}  # name: (T, alpha)
MODELS = ("nb", "sib_vote", *SETTINGS)


# ---------------------------------------------------------------------------
# Core (numpy; runs remotely)
# ---------------------------------------------------------------------------

def _nb_scores(x_tr, c_tr, x_te, k):
    import numpy as np
    from pmdb import fingerprint as fp
    center, scale, ok, mu, sb = fp._fit_core(x_tr, c_tr, k)
    z = np.clip((x_te[:, ok] - center[ok]) / scale[ok], -fp.MAX_Z, fp.MAX_Z)
    return fp._scores(z, mu, sb)


def _sib_counts(sib_codes, k):
    import numpy as np
    return np.bincount(np.asarray(sib_codes, dtype=int), minlength=k).astype(float)


def _combine(score, counts, k):
    """Per-model (assigned, posterior) from one site's NB scores and sibling counts."""
    import numpy as np
    out = {}
    nb_post = np.exp(-16.0 * (score - score.min()))
    out["nb"] = (int(score.argmin()), nb_post / nb_post.sum())
    n = counts.sum()
    top = np.flatnonzero(counts == counts.max())
    if n > 0 and len(top) == 1:
        out["sib_vote"] = (int(top[0]), counts / n)
    else:
        out["sib_vote"] = (int(score.argmin()), out["nb"][1])
    for name, (T, a) in SETTINGS.items():
        prior = (counts + a) / (n + k * a)
        lp = -T * (score - score.min()) + np.log(prior)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        out[name] = (int(p.argmax()), p)
    return out


def _loso(x, codes, parent_ids, k):
    """LOSO per model: assigned codes (n,) and posteriors (n, k)."""
    import numpy as np
    n = len(x)
    idx = np.arange(n)
    res = {m: (np.empty(n, dtype=int), np.empty((n, k))) for m in MODELS}
    for i in range(n):
        tr = idx != i
        score = _nb_scores(x[tr], codes[tr], x[i:i + 1], k)[0]
        sib = tr & (parent_ids == parent_ids[i])
        for m, (a, p) in _combine(score, _sib_counts(codes[sib], k), k).items():
            res[m][0][i] = a
            res[m][1][i] = p
    return res


def _bacc(codes, assigned, k):
    return float(sum((assigned[codes == c] == c).mean() for c in range(k)) / k)


@app.function(**FN_KW)
def observed(x: list, codes: list, parent_ids: list, k: int,
             xh: list, h_sib_codes: list) -> dict:
    import numpy as np
    x, codes, parent_ids, xh = map(np.asarray, (x, codes, parent_ids, xh))
    res = _loso(x, codes, parent_ids, k)
    out = {"loso": {m: {"assigned": a.tolist(), "post": p.tolist()} for m, (a, p) in res.items()}}
    hs = _nb_scores(x, codes, xh, k)
    out["heldout_scores"] = hs.tolist()
    out["heldout"] = []
    for j in range(len(xh)):
        comb = _combine(hs[j], _sib_counts(h_sib_codes[j], k), k)
        out["heldout"].append({m: {"assigned": a, "post": p.tolist()} for m, (a, p) in comb.items()})
    return out


@app.function(**FN_KW)
def null_bacc(x: list, codes: list, parent_ids: list, k: int, seeds: list) -> list:
    import numpy as np
    x, codes, parent_ids = map(np.asarray, (x, codes, parent_ids))
    rows = []
    for seed in seeds:
        cp = np.random.default_rng(seed).permutation(codes)
        res = _loso(x, cp, parent_ids, k)
        rows.append({"seed": int(seed),
                     **{m: _bacc(cp, a, k) for m, (a, _) in res.items()},
                     **{f"acc_{m}": float((a == cp).mean()) for m, (a, _) in res.items()}})
    return rows


# ---------------------------------------------------------------------------
# Local orchestration
# ---------------------------------------------------------------------------

def _metrics(codes, assigned, batches):
    import numpy as np
    k = len(batches)
    return {
        "accuracy": float((assigned == codes).mean()),
        "balanced_accuracy": _bacc(codes, assigned, k),
        "recall": {b: float((assigned[codes == c] == c).mean()) for c, b in enumerate(batches)},
        "confusion": {bt: {ba: int(((codes == ct) & (assigned == ca)).sum())
                           for ca, ba in enumerate(batches)} for ct, bt in enumerate(batches)},
    }


def _purity(sites, codes, parent_of, batches):
    import numpy as np
    import pandas as pd
    rows = []
    for i, s in enumerate(sites):
        sib = [j for j, t in enumerate(sites) if t != s and parent_of[t] == parent_of[s]]
        cnt = np.bincount(codes[sib], minlength=len(batches)) if sib else np.zeros(len(batches), int)
        top = np.flatnonzero(cnt == cnt.max())
        maj = batches[top[0]] if sib and len(top) == 1 else ("none" if not sib else "tie")
        rows.append({"site": s, "parent": parent_of[s], "batch": batches[codes[i]],
                     "n_siblings": len(sib), "sibling_batches": ";".join(batches[codes[j]] for j in sib),
                     "sibling_majority": maj, "majority_matches": maj == batches[codes[i]]})
    return pd.DataFrame(rows)


def _rubric(codes, assigned, post):
    """Organiser rubric per site: high = P(predicted) > 0.5; correct high 2, correct low 1,
    wrong low 1, wrong high 0. Plus accuracy within P(predicted) bins."""
    import numpy as np
    pp = post[np.arange(len(assigned)), assigned]
    ok = assigned == codes
    high = pp > 0.5
    score = np.where(ok, np.where(high, 2, 1), np.where(high, 0, 1))
    bins = {"<0.5": pp < 0.5, "0.5-0.7": (pp >= 0.5) & (pp <= 0.7), ">0.7": pp > 0.7}
    return {"mean_score": float(score.mean()), "n_high": int(high.sum()),
            "bins": {k: {"n": int(m.sum()), "accuracy": float(ok[m].mean()) if m.any() else None}
                     for k, m in bins.items()}}


@app.local_entrypoint()
def main(n_perm: int = 200, chunk: int = 25, heldout_sites: str = HELDOUT_SITES_CSV,
         heldout_features: str = HELDOUT_FEATURES_CSV, heldout_only: bool = False):
    """heldout_sites: CSV `site,parent`; heldout_features: fingerprint features of those
    sites (index batch,site). heldout_only: skip LOSO metrics/permutations, write only
    heldout_predictions.csv."""
    import numpy as np
    import pandas as pd

    out_dir = ROOT / "outputs/sibling_prior"
    out_dir.mkdir(parents=True, exist_ok=True)
    X = pd.read_csv(ROOT / "outputs/fingerprint/features.csv", index_col=[0, 1])
    hs_df = pd.read_csv(ROOT / heldout_sites, dtype=str)
    H = pd.read_csv(ROOT / heldout_features, index_col=[0, 1])
    H = H.reset_index("batch", drop=True).loc[hs_df["site"]]
    sites = list(X.index.get_level_values("site"))
    hsites = list(hs_df["site"])
    parent_of = {s: p for p, ss in PARENTS.items() for s in ss}
    assert sorted(parent_of) == sorted(sites), "PARENTS must cover exactly the 31 labelled sites"
    assert not set(hsites) & set(sites), "held-out sites must not be labelled sites"
    h_parent = dict(zip(hs_df["site"], hs_df["parent"]))

    batches = sorted(X.index.get_level_values("batch").unique())
    k = len(batches)
    codes = np.array([batches.index(b) for b in X.index.get_level_values("batch")])
    pnames = sorted(PARENTS)
    parent_ids = np.array([pnames.index(parent_of[s]) for s in sites])
    x = X.to_numpy(float)
    xh = H[X.columns].to_numpy(float)
    h_sib = [[int(codes[j]) for j, s in enumerate(sites) if parent_of[s] == h_parent[h]] for h in hsites]

    obs = observed.remote(x.tolist(), codes.tolist(), parent_ids.tolist(), k, xh.tolist(), h_sib)

    hrows = []
    for j, h in enumerate(hsites):
        sib = [s for s in sites if parent_of[s] == h_parent[h]]
        row = {"site": h, "parent": h_parent[h], "siblings": ";".join(sib),
               "sibling_batches": ";".join(batches[codes[sites.index(s)]] for s in sib)}
        for c, b in enumerate(batches):
            row[f"nb_score_{b}"] = obs["heldout_scores"][j][c]
        for m in MODELS:
            r = obs["heldout"][j][m]
            row[f"{m}_assigned"] = batches[r["assigned"]]
            for c, b in enumerate(batches):
                row[f"{m}_post_{b}"] = r["post"][c]
            row[f"{m}_p_assigned"] = r["post"][r["assigned"]]
            row[f"{m}_confidence"] = "high" if r["post"][r["assigned"]] > 0.5 else "low"
        hrows.append(row)
    pd.DataFrame(hrows).to_csv(out_dir / "heldout_predictions.csv", index=False)
    print(pd.DataFrame(hrows)[["site"] + [f"{m}_assigned" for m in MODELS]].to_string(index=False))
    if heldout_only:
        return

    seeds = list(range(n_perm))
    chunks = [seeds[i:i + chunk] for i in range(0, n_perm, chunk)]
    args = [(x.tolist(), codes.tolist(), parent_ids.tolist(), k, c) for c in chunks]
    null = pd.DataFrame([r for rows in null_bacc.starmap(args) for r in rows]).sort_values("seed")
    null.to_csv(out_dir / "null_permutations.csv", index=False)

    purity = _purity(sites, codes, parent_of, batches)
    purity.to_csv(out_dir / "purity.csv", index=False)
    with_sib = purity[purity["n_siblings"] > 0]

    ev = {"n_sites": len(sites), "n_perm": n_perm, "seeds": f"0..{n_perm - 1}",
          "settings": {m: {"T": T, "alpha": a} for m, (T, a) in SETTINGS.items()},
          "nb_probability": "softmax(-16 * score), T = number of features",
          "purity": {"n_with_siblings": int(len(with_sib)),
                     "majority_matches": int(with_sib["majority_matches"].sum()),
                     "ties": int((with_sib["sibling_majority"] == "tie").sum()),
                     "fraction": float(with_sib["majority_matches"].mean()),
                     "no_siblings": purity.loc[purity["n_siblings"] == 0, "site"].tolist()},
          "models": {}}
    pred = pd.DataFrame({"batch": X.index.get_level_values("batch"), "site": sites,
                         "parent": [parent_of[s] for s in sites]})
    for m in MODELS:
        a = np.asarray(obs["loso"][m]["assigned"])
        p = np.asarray(obs["loso"][m]["post"])
        met = _metrics(codes, a, batches)
        nb_null = null[m].to_numpy()
        met["perm_p_bacc"] = float((1 + (nb_null >= met["balanced_accuracy"]).sum()) / (n_perm + 1))
        met["null_bacc_mean"] = float(nb_null.mean())
        met["null_bacc_p95"] = float(np.percentile(nb_null, 95))
        met["rubric"] = _rubric(codes, a, p)
        ev["models"][m] = met
        pred[f"{m}_assigned"] = [batches[c] for c in a]
        for c, b in enumerate(batches):
            pred[f"{m}_post_{b}"] = p[:, c]
    pred.to_csv(out_dir / "loso_predictions.csv", index=False)
    (out_dir / "evaluation.json").write_text(json.dumps(ev, indent=2))

    for m in MODELS:
        e = ev["models"][m]
        print(f"{m:12s} acc={e['accuracy']:.3f} bacc={e['balanced_accuracy']:.3f} "
              f"p={e['perm_p_bacc']:.4f} rubric={e['rubric']}")
    print("purity", ev["purity"])
