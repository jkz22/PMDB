"""Leave-one-parent-out (LOPO) vs leave-one-site-out (LOSO) evaluation on Modal.

    modal run scripts/modal_lopo.py                 # 200 label permutations (seeds 0..199)
    modal run scripts/modal_lopo.py --n-perm 4      # smoke test
    modal run scripts/modal_lopo.py::rubric         # organiser-rubric scores (needs predictions.csv)

The 31 labelled sites are crops of ~13 parent electrode images (organiser info,
recovered from acquisition fingerprints). LOSO leaves a crop out while its
siblings stay in training; LOPO leaves out every labelled crop of one parent
(13 folds), which is the honest protocol for unseen electrodes.

Models (all pre-stated before running, no tuning):
  nb             baseline fingerprint robust naive Bayes (pmdb.fingerprint), 16 features.
  nb_parentavg   variant (a): training crops are collapsed to one row per
                 (parent, batch) pair (feature-wise median) before the pooled
                 standardisation and per-batch params are fitted, so a parent with
                 4 crops does not outweigh a parent with 1. Test crops scored as-is.
  nb_parentctr   variant (b): label-free within-parent centring. Every crop's
                 features minus the median over all crops of its parent (labelled
                 crops; for a held-out crop, its labelled siblings plus itself).
                 Applied identically to train and test crops; uses parent
                 membership only, never labels. Under LOPO the test crops are
                 centred against their own (also held-out) siblings. Single-crop
                 parents become all-zero (no information) -- a known cost.
  rf_kpi         two-stage random forest, KPI arm (pmdb.classify.model.TwoStage on
                 outputs/classifier/kpi_tiles6.csv), tile-level fit, site = tile mean.

Permutation test: site-level label shuffles with np.random.default_rng(seed),
seeds 0..n_perm-1, same site order and seeds for every model and protocol; the
statistic is balanced accuracy; p = (1 + #{null >= obs}) / (n_perm + 1).

Writes outputs/lopo/{evaluation.json, predictions.csv, heldout_predictions.csv, results_table.md}.
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
        "Pillow==12.3.0", "tifffile==2025.5.10", "imagecodecs==2025.3.30", "scikit-learn==1.3.2",
    )
    .add_local_python_source("pmdb")
)
app = modal.App("pmdb-lopo", image=image)
FN_KW = dict(cpu=1.0, memory=2048, timeout=3600)

BATCHES = ["Batch_1", "Batch_2", "Batch_3"]

#: Parent electrode image -> labelled crops (organiser-confirmed grouping).
PARENTS: dict[str, list[str]] = {
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
HELDOUT_PARENT = {"fn0mhxef": "G2048", "xrv9xvzb": "G2088", "3e122cbj": "G2316"}

NB_MODELS = ("nb", "nb_parentavg", "nb_parentctr")
MODELS = NB_MODELS + ("rf_kpi",)
PROTOCOLS = ("loso", "lopo")


# ---------------------------------------------------------------------------
# Core (numpy; runs remotely)
# ---------------------------------------------------------------------------

def _folds(protocol, site_parent):
    import numpy as np
    keys = np.arange(len(site_parent)) if protocol == "loso" else np.asarray(site_parent)
    return [np.flatnonzero(keys == k) for k in dict.fromkeys(keys.tolist())]


def _collapse(x, codes, parents):
    """One row per (parent, batch): feature-wise median."""
    import numpy as np
    pairs = sorted(set(zip(parents, codes.tolist())))
    xs = np.array([np.median(x[(parents == p) & (codes == c)], axis=0) for p, c in pairs])
    return xs, np.array([c for _, c in pairs])


def _nb_scores(model, x_tr, c_tr, p_tr, x_te):
    """Per-batch NB scores (mean neg. Laplace log-lik over kept features) and the kept-feature count."""
    import numpy as np
    from pmdb import fingerprint as fp
    if model == "nb_parentavg":
        x_tr, c_tr = _collapse(x_tr, c_tr, p_tr)
    center, scale, ok, mu, sb = fp._fit_core(x_tr, c_tr, len(BATCHES))
    z = np.clip((x_te[:, ok] - center[ok]) / scale[ok], -fp.MAX_Z, fp.MAX_Z)
    return fp._scores(z, mu, sb), int(ok.sum())


def _nb_fit_score(model, x_tr, c_tr, p_tr, x_te):
    return _nb_scores(model, x_tr, c_tr, p_tr, x_te)[0].argmin(1)


def _assign(model, protocol, data, codes):
    import numpy as np
    parents = np.asarray(data["site_parent"])
    out = np.empty(len(codes), dtype=int)
    if model in NB_MODELS:
        x = np.asarray(data["x_ctr" if model == "nb_parentctr" else "x"], float)
        for te in _folds(protocol, data["site_parent"]):
            tr = np.setdiff1d(np.arange(len(codes)), te)
            out[te] = _nb_fit_score(model, x[tr], codes[tr], parents[tr], x[te])
        return out
    # rf_kpi: tile-level two-stage RF, site prediction from mean tile probabilities
    import pandas as pd
    from pmdb.classify.model import TwoStage, combine
    feats = data["rf_features"]
    tiles = pd.DataFrame(np.asarray(data["rf_x"], float), columns=feats)
    tsite = np.asarray(data["rf_site"])
    tiles["heldout"] = False
    for te in _folds(protocol, data["site_parent"]):
        trm = ~np.isin(tsite, te)
        tr_tiles = tiles[trm].copy()
        tr_tiles["batch"] = [BATCHES[codes[s]] for s in tsite[trm]]
        m = TwoStage(feats).fit(tr_tiles)
        for s in te:
            X = tiles.loc[tsite == s, feats].to_numpy(float)
            p = float(m.stage1.predict_proba(X)[:, list(m.stage1.classes_).index(1)].mean())
            q = float(m.stage2.predict_proba(X)[:, list(m.stage2.classes_).index(1)].mean())
            out[s] = BATCHES.index(combine(p, q)["predicted"])
    return out


def _bacc(codes, a):
    return float(sum((a[codes == c] == c).mean() for c in range(len(BATCHES))) / len(BATCHES))


T_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def _softmax_neg(score, t):
    import numpy as np
    a = -t * score
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)


def _nb_oof(model, protocol, x, codes, parents):
    """Out-of-fold NB score matrix (n x batches) and kept-feature count per row."""
    import numpy as np
    S, nk = np.empty((len(codes), len(BATCHES))), np.empty(len(codes), dtype=int)
    for te in _folds(protocol, parents.tolist()):
        tr = np.setdiff1d(np.arange(len(codes)), te)
        S[te], nk[te] = _nb_scores(model, x[tr], codes[tr], parents[tr], x[te])
    return S, nk


def _probs(model, protocol, data):
    """Out-of-fold class probabilities. NB: {'fixed': softmax(-n_features * score),
    'cal': softmax(-T * score) with T chosen per outer fold by min log-loss of an inner
    same-protocol CV on the training sites only}. RF: two-stage combined probabilities."""
    import numpy as np
    codes = np.asarray(data["codes"])
    parents = np.asarray(data["site_parent"])
    n = len(codes)
    if model in NB_MODELS:
        x = np.asarray(data["x_ctr" if model == "nb_parentctr" else "x"], float)
        S, nk = _nb_oof(model, protocol, x, codes, parents)
        fixed = np.vstack([_softmax_neg(S[i:i + 1], nk[i]) for i in range(n)])
        cal, t_used = np.empty_like(fixed), np.empty(n)
        for te in _folds(protocol, parents.tolist()):
            tr = np.setdiff1d(np.arange(n), te)
            Si, _ = _nb_oof(model, protocol, x[tr], codes[tr], parents[tr])
            # inner folds may lack a batch in training: _fit_core then yields NaN medians; skip those rows
            good = np.isfinite(Si).all(1)
            ll = [-np.mean(np.log(_softmax_neg(Si[good], t)[np.arange(good.sum()), codes[tr][good]] + 1e-12))
                  for t in T_GRID]
            t = T_GRID[int(np.argmin(ll))]
            cal[te], t_used[te] = _softmax_neg(S[te], t), t
        return {"fixed": fixed.tolist(), "cal": cal.tolist(), "T_cal": t_used.tolist(),
                "T_fixed": nk.tolist()}
    import pandas as pd
    from pmdb.classify.model import TwoStage, combine
    feats = data["rf_features"]
    tiles = pd.DataFrame(np.asarray(data["rf_x"], float), columns=feats)
    tsite = np.asarray(data["rf_site"])
    tiles["heldout"] = False
    P = np.empty((n, len(BATCHES)))
    for te in _folds(protocol, data["site_parent"]):
        trm = ~np.isin(tsite, te)
        tr_tiles = tiles[trm].copy()
        tr_tiles["batch"] = [BATCHES[codes[s]] for s in tsite[trm]]
        m = TwoStage(feats).fit(tr_tiles)
        for s in te:
            X = tiles.loc[tsite == s, feats].to_numpy(float)
            p = float(m.stage1.predict_proba(X)[:, list(m.stage1.classes_).index(1)].mean())
            q = float(m.stage2.predict_proba(X)[:, list(m.stage2.classes_).index(1)].mean())
            c = combine(p, q)
            P[s] = [c[f"P_{b}"] for b in BATCHES]
    return {"rf": P.tolist()}


@app.function(**FN_KW)
def probs(model: str, protocol: str, data: dict) -> dict:
    return _probs(model, protocol, data)


@app.function(**FN_KW)
def run(model: str, protocol: str, data: dict, seeds: list | None) -> dict:
    import numpy as np
    codes = np.asarray(data["codes"])
    if seeds is None:
        return {"assigned": _assign(model, protocol, data, codes).tolist()}
    null = []
    for seed in seeds:
        cp = np.random.default_rng(seed).permutation(codes)
        null.append(_bacc(cp, _assign(model, protocol, data, cp)))
    return {"null": null}


@app.function(**FN_KW)
def heldout(model: str, data: dict) -> list:
    """Full fit on all 31 labelled sites, assignment of the 3 held-out sites (NB models)."""
    import numpy as np
    codes = np.asarray(data["codes"])
    key = "x_ctr" if model == "nb_parentctr" else "x"
    return _nb_fit_score(model, np.asarray(data[key], float), codes, np.asarray(data["site_parent"]),
                         np.asarray(data["h" + key], float)).tolist()


# ---------------------------------------------------------------------------
# Local data assembly (light CSV reads)
# ---------------------------------------------------------------------------

def _build_data():
    import numpy as np
    import pandas as pd
    from pmdb.classify.features import build_arm_table, load_kpi_tiles6

    X = pd.read_csv(ROOT / "outputs/fingerprint/features.csv", index_col=[0, 1])
    H = pd.read_csv(ROOT / "outputs/fingerprint/heldout_features.csv", index_col=[0, 1])
    sites = X.index.get_level_values("site").tolist()
    site2parent = {s: p for p, ss in PARENTS.items() for s in ss}
    assert sorted(site2parent) == sorted(sites) and len(site2parent) == 31, "PARENTS must cover the 31 labelled sites"
    assert len(PARENTS) == 13
    hsites = H.index.get_level_values("site").tolist()
    assert sorted(hsites) == sorted(HELDOUT_PARENT)
    site_parent = [site2parent[s] for s in sites]
    codes = [BATCHES.index(b) for b in X.index.get_level_values("batch")]

    # label-free within-parent centring (labelled crops; held-out crop vs labelled siblings + itself)
    xa = X.to_numpy(float)
    pmed = {p: np.median(xa[[i for i, q in enumerate(site_parent) if q == p]], axis=0) for p in PARENTS}
    x_ctr = xa - np.array([pmed[p] for p in site_parent])
    ha = H.to_numpy(float)
    hx_ctr = []
    for i, s in enumerate(hsites):
        sib = xa[[j for j, q in enumerate(site_parent) if q == HELDOUT_PARENT[s]]]
        hx_ctr.append(ha[i] - np.median(np.vstack([sib, ha[i:i + 1]]), axis=0))

    kpi = load_kpi_tiles6(ROOT / "outputs/classifier/kpi_tiles6.csv")
    table, feats = build_arm_table("KPI", kpi, None)
    lab = table[~table["heldout"]]
    sidx = {s: i for i, s in enumerate(sites)}
    assert set(lab["site"]) == set(sites)
    assert all(BATCHES[codes[sidx[s]]] == b for s, b in zip(lab["site"], lab["batch"]))

    data = {
        "sites": sites, "site_parent": site_parent, "codes": codes,
        "x": xa.tolist(), "x_ctr": x_ctr.tolist(), "hx": ha.tolist(), "hx_ctr": np.array(hx_ctr).tolist(),
        "rf_features": feats, "rf_x": lab[feats].to_numpy(float).tolist(),
        "rf_site": [sidx[s] for s in lab["site"]],
    }
    return data, hsites


def _metrics(codes, a, null):
    import numpy as np
    k = len(BATCHES)
    bacc = _bacc(codes, a)
    null = np.asarray(null)
    return {
        "accuracy": float((a == codes).mean()),
        "balanced_accuracy": bacc,
        "recall": {b: float((a[codes == c] == c).mean()) for c, b in enumerate(BATCHES)},
        "confusion": {bt: {ba: int(((codes == i) & (a == j)).sum()) for j, ba in enumerate(BATCHES)}
                      for i, bt in enumerate(BATCHES)},
        "n_perm": int(len(null)),
        "perm_null_mean_bacc": float(null.mean()) if len(null) else None,
        "perm_null_p95_bacc": float(np.percentile(null, 95)) if len(null) else None,
        "perm_p_bacc": float((1 + (null >= bacc).sum()) / (len(null) + 1)),
    }


@app.local_entrypoint()
def main(n_perm: int = 200, nb_chunk: int = 50, rf_chunk: int = 2, out_dir: str = "outputs/lopo"):
    import numpy as np
    import pandas as pd

    data, hsites = _build_data()
    codes = np.asarray(data["codes"])
    configs = [(m, p) for m in MODELS for p in PROTOCOLS]

    obs_calls = {c: run.spawn(c[0], c[1], data, None) for c in configs}
    held_calls = {m: heldout.spawn(m, data) for m in NB_MODELS}
    perm_args, perm_keys = [], []
    for m, p in configs:
        ch = rf_chunk if m == "rf_kpi" else nb_chunk
        for s in range(0, n_perm, ch):
            perm_args.append((m, p, data, list(range(s, min(s + ch, n_perm)))))
            perm_keys.append((m, p))
    null = {c: [] for c in configs}
    for key, r in zip(perm_keys, run.starmap(perm_args, order_outputs=True)):
        null[key].extend(r["null"])
    obs = {c: np.asarray(f.get()["assigned"]) for c, f in obs_calls.items()}
    held = {m: f.get() for m, f in held_calls.items()}

    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    ev = {"n_perm": n_perm, "seeds": f"0..{n_perm - 1}", "batches": BATCHES, "parents": PARENTS,
          "results": {}}
    pred = pd.DataFrame({"site": data["sites"], "parent": data["site_parent"],
                         "true": [BATCHES[c] for c in codes]})
    rows = ["| model | protocol | acc | bacc | recall B1/B2/B3 | perm null mean | perm p (bacc) |",
            "|---|---|---|---|---|---|---|"]
    for m, p in configs:
        r = _metrics(codes, obs[(m, p)], null[(m, p)])
        ev["results"][f"{m}/{p}"] = r
        pred[f"{m}_{p}"] = [BATCHES[c] for c in obs[(m, p)]]
        rec = "/".join(f"{v:.2f}" for v in r["recall"].values())
        rows.append(f"| {m} | {p.upper()} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {rec} | "
                    f"{r['perm_null_mean_bacc']:.3f} | {r['perm_p_bacc']:.3f} |")
    hp = pd.DataFrame({"site": hsites, "parent": [HELDOUT_PARENT[s] for s in hsites],
                       **{m: [BATCHES[c] for c in held[m]] for m in NB_MODELS}})
    (out / "evaluation.json").write_text(json.dumps(ev, indent=2))
    pred.to_csv(out / "predictions.csv", index=False)
    hp.to_csv(out / "heldout_predictions.csv", index=False)
    (out / "results_table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(hp.to_string(index=False))
    print(f"wrote {out}")


def _rubric(codes, P, assigned=None):
    """Organiser rubric: high (P(pred) > 0.5) correct 2, high wrong 0, low 1 either way."""
    import numpy as np
    P = np.asarray(P)
    a = P.argmax(1) if assigned is None else np.asarray(assigned)
    pp = P[np.arange(len(a)), a]
    hi, ok = pp > 0.5, a == codes
    score = np.where(hi, np.where(ok, 2, 0), 1)
    bins = {"<0.5": pp < 0.5, "0.5-0.7": (pp >= 0.5) & (pp <= 0.7), ">0.7": pp > 0.7}
    return {
        "rubric_mean": float(score.mean()),
        "acc": float(ok.mean()),
        "n_high": int(hi.sum()),
        "acc_high": float(ok[hi].mean()) if hi.any() else None,
        "acc_low": float(ok[~hi].mean()) if (~hi).any() else None,
        "reliability": {k: {"n": int(m.sum()), "mean_p": float(pp[m].mean()) if m.any() else None,
                            "acc": float(ok[m].mean()) if m.any() else None} for k, m in bins.items()},
    }


@app.local_entrypoint()
def rubric(out_dir: str = "outputs/lopo"):
    """Organiser-rubric expected score for every model x protocol (observed only, no permutations)."""
    import numpy as np
    import pandas as pd

    data, _ = _build_data()
    codes = np.asarray(data["codes"])
    configs = [(m, p) for m in MODELS for p in PROTOCOLS]
    res = dict(zip(configs, probs.starmap([(m, p, data) for m, p in configs])))
    out = ROOT / out_dir
    pred = pd.read_csv(out / "predictions.csv")
    ev = {"rule": "high iff P(predicted) > 0.5; high&correct 2, high&wrong 0, low 1; "
                  "reliability pooled over out-of-fold predictions",
          "nb_prob": "fixed: softmax(-T*score), T = number of kept features (16); "
                     "cal: T in T_GRID chosen per outer fold by min inner same-protocol CV log-loss on training sites",
          "T_grid": list(T_GRID), "results": {}}
    rows = ["| model | probs | protocol | acc | rubric mean (max 2) | n high | acc high | acc low | "
            "rel <0.5 n/acc | rel 0.5-0.7 n/acc | rel >0.7 n/acc |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    for (m, p), r in res.items():
        assigned = np.array([BATCHES.index(b) for b in pred[f"{m}_{p}"]])
        for kind, P in r.items():
            if kind.startswith("T_"):
                continue
            P = np.asarray(P)
            if m in NB_MODELS:
                assert (P.argmax(1) == assigned).all(), f"{m}/{p}: argmax P != assigned"
            # NB assignment is argmin score == argmax P for any T > 0; RF uses its own decision rule
            rb = _rubric(codes, P, assigned)
            if kind == "cal":
                rb["T_cal_values"] = sorted(set(r["T_cal"]))
            ev["results"][f"{m}/{kind}/{p}"] = rb
            pred[f"{m}_{kind}_{p}_Ppred"] = P[np.arange(len(assigned)), assigned]
            rel = " | ".join(f"{v['n']}/{fmt(v['acc'])}" for v in rb["reliability"].values())
            rows.append(f"| {m} | {kind} | {p.upper()} | {rb['acc']:.3f} | {rb['rubric_mean']:.3f} | "
                        f"{rb['n_high']} | {fmt(rb['acc_high'])} | {fmt(rb['acc_low'])} | {rel} |")
    (out / "rubric.json").write_text(json.dumps(ev, indent=2))
    pred.to_csv(out / "predictions_rubric.csv", index=False)
    (out / "rubric_table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
