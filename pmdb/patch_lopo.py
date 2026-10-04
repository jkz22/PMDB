"""Leave-one-parent-out (LOPO) evaluation, patch + fingerprint ensemble, final held-out calls.

Parent membership is used ONLY to exclude sibling crops from CV folds / banks; it never enters a
probability, call or confidence. See docs/patch_mil.md.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pmdb import fingerprint as fp
from pmdb import patch_mil as pm

BATCHES = pm.BATCHES

FEATURE_PHRASES = {
    "si_depth_slope": "the top-to-bottom trend in Si fraction",
    "si_depth_mid_dip": "Si depletion at mid-depth",
    "k15_contact_tilestd": "variability of Si-graphite contact across the image",
}

FORBIDDEN = ("model", "fingerprint", "ensemble", "knn", "embedding", "score", "agree", "stratum",
             "lopo", "probabilit", "parent", "sibling")


# --------------------------------------------------------------------------- numerics

def normalise_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p / p.sum(axis=-1, keepdims=True)


def ensemble(p_patch: np.ndarray, q_fp: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(p_patch, dtype=float) + np.asarray(q_fp, dtype=float))


def _pred_frame(pr: pd.DataFrame) -> pd.DataFrame:
    p = pr[[f"p_{b}" for b in BATCHES]].to_numpy(dtype=float)
    q = normalise_p(p)
    out = pd.DataFrame({"fp_call": pr["assigned"].to_numpy(), "fp_ood": pr["ood"].to_numpy(dtype=bool),
                        "fp_confidence": pr["confidence"].to_numpy(dtype=float)}, index=pr.index)
    for j, b in enumerate(BATCHES):
        out[f"p_fp_{b}"] = p[:, j]
        out[f"q_fp_{b}"] = q[:, j]
    return out


def fingerprint_cv(X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> pd.DataFrame:
    groups = np.asarray(groups)
    parts = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = fp.fit(X[tr], y[tr])
        assert m.batches == list(BATCHES)
        parts.append(_pred_frame(fp.predict(m, X[te])))
    return pd.concat(parts).loc[X.index]


def fingerprint_heldout(X: pd.DataFrame, y: pd.Series, groups_lab: np.ndarray, H: pd.DataFrame,
                        groups_h: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups_lab, groups_h = np.asarray(groups_lab), np.asarray(groups_h)
    preds, expl = [], []
    for h in range(len(H)):
        tr = groups_lab != groups_h[h]
        m = fp.fit(X[tr], y[tr])
        assert m.batches == list(BATCHES)
        row = H.iloc[[h]]
        preds.append(_pred_frame(fp.predict(m, row)))
        e = fp.explain(m, row)
        e["site"] = H.index[h][1] if isinstance(H.index[h], tuple) else H.index[h]
        expl.append(e)
    return pd.concat(preds), pd.concat(expl, ignore_index=True)


def patch_cv(D, patch_site, site_labels, groups) -> list[pm.SiteScore]:
    return pm.lopo_scores(D, patch_site, site_labels, groups, len(BATCHES))


def stratum_flags(correct: np.ndarray, agree: np.ndarray, groups: np.ndarray) -> np.ndarray:
    correct, agree, groups = np.asarray(correct, float), np.asarray(agree, bool), np.asarray(groups)
    out = np.zeros(len(correct), dtype=bool)
    for i in range(len(correct)):
        m = (groups != groups[i]) & (agree == agree[i])
        out[i] = bool(m.any() and correct[m].mean() > 0.5)
    return out


def stratum_table(correct, agree) -> dict:
    correct, agree = np.asarray(correct, float), np.asarray(agree, bool)
    out = {}
    for name, m in (("agree", agree), ("disagree", ~agree)):
        out[name] = {"n": int(m.sum()), "accuracy": float(correct[m].mean()) if m.any() else None}
    return out


def heldout_flag(correct, agree, query_agree: bool) -> bool:
    correct, agree = np.asarray(correct, float), np.asarray(agree, bool)
    m = agree == bool(query_agree)
    return bool(m.any() and correct[m].mean() > 0.5)


def expected_rubric(correct: np.ndarray, high: np.ndarray) -> float:
    correct, high = np.asarray(correct, bool), np.asarray(high, bool)
    return float(np.mean(np.where(high, np.where(correct, 2, 0), 1)))


def cv_feasible(labels: np.ndarray, groups: np.ndarray) -> bool:
    labels, groups = np.asarray(labels), np.asarray(groups)
    for g in np.unique(groups):
        rest = groups != g
        for b in np.unique(labels):
            if len(np.unique(groups[(labels == b) & rest])) < 2:
                return False
    return True


# --------------------------------------------------------------------------- CV

def _run_cv(D, patch_site, site_labels, groups, X, y):
    site_labels = np.asarray(site_labels)
    sc = patch_cv(D, patch_site, site_labels, groups)
    fpf = fingerprint_cv(X, y, groups)
    top = np.stack([s.pooled["top10"] for s in sc])
    p_patch = pm.softmax_conf(top)
    q_fp = fpf[[f"q_fp_{b}" for b in BATCHES]].to_numpy()
    p_ens = ensemble(p_patch, q_fp)
    patch_call = top.argmin(axis=1)
    fp_call = np.array([BATCHES.index(c) for c in fpf["fp_call"]])
    ens_call = p_ens.argmax(axis=1)
    df = pd.DataFrame({
        "batch": [i[0] for i in X.index], "site": [i[1] for i in X.index],
        "true": [BATCHES[c] for c in site_labels],
        "patch_call": [BATCHES[c] for c in patch_call], "fp_call": [BATCHES[c] for c in fp_call],
        "ens_call": [BATCHES[c] for c in ens_call], "agree": patch_call == fp_call,
        "fp_ood": fpf["fp_ood"].to_numpy(), "fp_confidence": fpf["fp_confidence"].to_numpy(),
    })
    for j, b in enumerate(BATCHES):
        df[f"p_patch_{b}"] = p_patch[:, j]
    for j, b in enumerate(BATCHES):
        df[f"q_fp_{b}"] = q_fp[:, j]
    for j, b in enumerate(BATCHES):
        df[f"p_ens_{b}"] = p_ens[:, j]
    df["correct_patch"] = patch_call == site_labels
    df["correct_fp"] = fp_call == site_labels
    df["correct_ens"] = ens_call == site_labels
    return df, sc


def run_cv(D, patch_site, site_labels, groups, X, y) -> pd.DataFrame:
    return _run_cv(D, patch_site, site_labels, groups, X, y)[0]


def lopo_permutation(D, patch_site, site_labels, groups, X, n_perm=1000, seed=0) -> dict:
    site_labels = np.asarray(site_labels)
    rng = np.random.default_rng(seed)
    null = {"patch": [], "fingerprint": [], "ensemble": []}
    rejected = 0
    while len(null["patch"]) < n_perm:
        perm = rng.permutation(site_labels)
        if not cv_feasible(perm, groups):
            rejected += 1
            if rejected > 100 * n_perm:
                raise RuntimeError("permutation feasibility rejection limit")
            continue
        df = run_cv(D, patch_site, perm, groups, X, pd.Series([BATCHES[c] for c in perm], index=X.index))
        for k, col in (("patch", "correct_patch"), ("fingerprint", "correct_fp"), ("ensemble", "correct_ens")):
            null[k].append(df[col].mean())
    obs = run_cv(D, patch_site, site_labels, groups, X, pd.Series([BATCHES[c] for c in site_labels], index=X.index))
    out = {}
    for k, col in (("patch", "correct_patch"), ("fingerprint", "correct_fp"), ("ensemble", "correct_ens")):
        o = float(obs[col].mean())
        nl = np.asarray(null[k])
        out[k] = {"observed_accuracy": o, "null_mean": float(nl.mean()),
                  "null_p95": float(np.percentile(nl, 95)),
                  "p_value": float((1.0 + (nl >= o - 1e-12).sum()) / (n_perm + 1.0))}
    out.update({"n_perm": int(n_perm), "seed": int(seed), "n_rejected": int(rejected),
                "scheme": "site labels permuted, parent groups fixed, infeasible draws (a batch with <2 "
                          "training parents in some fold) redrawn"})
    return out


# --------------------------------------------------------------------------- explanations

def feature_phrase(name: str) -> str:
    m = re.fullmatch(r"si_depth_rel_band(\d+)", name)
    if m:
        return f"Si fraction in depth band {int(m.group(1)) + 1} of 5 (1 = top)"
    m = re.fullmatch(r"gx_([^_]+)_([^_]+)", name)
    if m:
        return f"graphite clustering along the layer at {m.group(1)}-{m.group(2)} um spacing"
    m = re.fullmatch(r"gz_([^_]+)_([^_]+)", name)
    if m:
        return f"graphite clustering through the depth at {m.group(1)}-{m.group(2)} um spacing"
    return FEATURE_PHRASES.get(name, name)


def image_third(rows: np.ndarray, n_rows: int) -> str:
    rows = np.asarray(rows)
    th = np.minimum(2, rows * 3 // max(n_rows, 1))
    counts = np.bincount(th, minlength=3)
    return ("top", "middle", "bottom")[int(np.argmax(counts))]


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def make_explanation(site: str, k: int, high: bool, expl: pd.DataFrame, patch_call: int, frac: float,
                     anom_rows: np.ndarray, n_rows: int, b3_median_frac: float) -> str:
    e = expl.copy()
    e = e.sort_values("dev_Batch_3", ascending=False, kind="stable").head(3)
    items, resembles = [], []
    for _, r in e.iterrows():
        dev = [r[f"dev_{b}"] for b in BATCHES]
        res = int(np.argmin(dev))
        direction = "higher" if r["z"] > r["center_Batch_3"] else "lower"
        items.append((feature_phrase(r["feature"]), direction, res))
    parts = [f"({i}) {ph} is {d} than in Batch 3, typical of Batch {res + 1}"
             for i, (ph, d, res) in enumerate(items, 1)]
    if frac <= 0:
        where = "none do"
    else:
        third = image_third(anom_rows, n_rows)
        th = np.minimum(2, np.asarray(anom_rows) * 3 // max(n_rows, 1))
        if np.bincount(th, minlength=3).max() / len(anom_rows) > 0.5:
            where = (f"mostly in the {third} third of the image "
                     f"(see outputs/patch_mil/figures/heldout_{site}.png)")
        else:
            where = "spread across the whole image"
    parts.append(f"(4) local microstructure appearance: {frac:.0%} of the image's 11.2 um areas look unlike "
                 f"any Batch 3 image (Batch 3 images: typically {b3_median_frac:.0%}), {where}; "
                 f"overall it looks most like Batch {k + 1}")
    all_items = [(ph, res) for ph, _, res in items] + [("the local microstructure appearance", patch_call)]
    if high:
        conf = (f"Confidence is high because the depth-profile, graphite-arrangement and local-appearance "
                f"evidence consistently point to Batch {k + 1}.")
    else:
        with_k = [ph for ph, res in all_items if res == k]
        other: dict[int, list[str]] = {}
        for ph, res in all_items:
            if res != k:
                other.setdefault(res, []).append(ph)
        if not other:
            conf = ("Confidence is low because the differences from the other batches are small relative "
                    "to normal variation between images of the same batch.")
        else:
            lead = (f"{_join(with_k)} point to Batch {k + 1}" if with_k
                    else f"nothing clearly points to Batch {k + 1}")
            rest = "; ".join(f"{_join(v)} look like Batch {o + 1}" for o, v in sorted(other.items()))
            conf = f"Confidence is low because the evidence is mixed: {lead}, but {rest}."
    return (f"{site}: Batch {k + 1} ({'high' if high else 'low'} confidence). Compared with Batch 3 "
            f"(supplier baseline): " + "; ".join(parts) + ". " + conf)


# --------------------------------------------------------------------------- assembly

def build_lopo_outputs(D, patch_site, site_labels, sites: pd.DataFrame, parents: pd.DataFrame, X, H, Dh,
                       patch_site_h, heldout_ids, coords_h, n_perm: int = 1000, seed: int = 0) -> dict:
    nb = len(BATCHES)
    site_labels = np.asarray(site_labels)
    n_sites = len(site_labels)
    sites = sites.reset_index(drop=True)
    keys = list(zip(sites["batch"], sites["site"]))

    par = parents.drop_duplicates(["batch", "site"]).set_index(["batch", "site"])["parent_id"]
    lab_par = par.reindex(pd.MultiIndex.from_tuples(keys))
    assert not lab_par.isna().any(), "labelled site without parent_id"
    codes, uniques = pd.factorize(lab_par.to_numpy())
    code_of = {u: i for i, u in enumerate(uniques)}
    assert set(X.index) == set(keys), "fingerprint features do not match the labelled sites"
    X = X.loc[keys]
    missing = [s for s in heldout_ids if ("Batch_heldout", s) not in H.index]
    if missing:
        raise ValueError(f"no fingerprint features for held-out {missing}; run "
                         "python scripts/run_fingerprint.py --heldout-dir outputs/heldout/kpis")
    H = H.loc[[("Batch_heldout", s) for s in heldout_ids]]
    y = pd.Series([BATCHES[c] for c in site_labels], index=X.index)
    h_par = [par.get(("Batch_heldout", s)) for s in heldout_ids]
    h_codes = np.array([code_of.get(p, -1) if p is not None else -1 for p in h_par])

    lopo, lopo_sc = _run_cv(D, patch_site, site_labels, codes, X, y)
    loso, loso_sc = _run_cv(D, patch_site, site_labels, np.arange(n_sites), X, y)
    for df in (lopo, loso):
        g = codes if df is lopo else np.arange(n_sites)
        df["parent_id"] = lab_par.to_numpy()
        df["flag_high"] = stratum_flags(df["correct_ens"], df["agree"], g) & ~df["fp_ood"].to_numpy()
    lopo["fold"], loso["fold"] = "lopo", "loso"
    preds = pd.concat([lopo, loso], ignore_index=True)

    def met(df, col):
        return pm.classification_metrics(site_labels, np.array([BATCHES.index(c) for c in df[col]]))

    cols = {"patch": "patch_call", "fingerprint": "fp_call", "ensemble": "ens_call"}
    ev_loso = {k: met(loso, c) for k, c in cols.items()}
    ev_lopo = {k: met(lopo, c) for k, c in cols.items()}
    perm = lopo_permutation(D, patch_site, site_labels, codes, X, n_perm=n_perm, seed=seed)

    agree = lopo["agree"].to_numpy()
    ood = lopo["fp_ood"].to_numpy()
    c_p, c_f, c_e = (lopo[c].to_numpy() for c in ("correct_patch", "correct_fp", "correct_ens"))
    fpc = lopo["fp_confidence"].to_numpy() >= 0.5
    fl_e = lopo["flag_high"].to_numpy()
    fl_p = stratum_flags(c_p, agree, codes)
    fl_f = stratum_flags(c_f, agree, codes) & ~ood
    ones = np.ones(n_sites, bool)

    def acc_n(c, m):
        return {"n": int(m.sum()), "accuracy": float(c[m].mean()) if m.any() else None}

    diag = {"fingerprint_by_fp_confidence": {"ge_0.5": acc_n(c_f, fpc), "lt_0.5": acc_n(c_f, ~fpc)},
            "ensemble_agree_x_fp_confidence": {
                f"{'agree' if a else 'disagree'}_fpconf_{'ge' if f else 'lt'}_0.5": acc_n(c_e, (agree == a) & (fpc == f))
                for a in (True, False) for f in (True, False)},
            "note": "diagnostic only; not used in the confidence flag"}
    rubric = {"ensemble_flag": expected_rubric(c_e, fl_e), "ensemble_all_high": expected_rubric(c_e, ones),
              "all_low": expected_rubric(c_e, ~ones),
              "patch_flag": expected_rubric(c_p, fl_p), "patch_all_high": expected_rubric(c_p, ones),
              "fingerprint_flag": expected_rubric(c_f, fl_f), "fingerprint_all_high": expected_rubric(c_f, ones),
              "fp_conf_flag": expected_rubric(c_f, fpc & ~ood)}
    evaluation = {
        "n_sites": int(n_sites), "n_parents": int(len(uniques)),
        "parents": {u: [s for (b, s), p in zip(keys, lab_par) if p == u] for u in uniques},
        "loso": ev_loso, "lopo": ev_lopo, "permutation": perm,
        "confidence": {
            "rule": "high iff ensemble LOPO accuracy over the same agreement stratum (patch call == "
                    "fingerprint call) of labelled sites of other parents > 0.5; forced low if the "
                    "fingerprint marks the site OOD",
            "strata": stratum_table(c_e, agree), "fp_confidence_diagnostic": diag},
        "rubric_expected_score": rubric,
        "fingerprint_probability": "conformal p-values normalised to sum 1 (heuristic, not a posterior)",
        "parents_used_as_evidence": False,
    }

    # held-out
    b3_med = float(np.median([(s.u[:, 2] > pm.ANOM_U).mean() for s, t in zip(loso_sc, site_labels) if t == 2]))
    hsc = pm.heldout_scores(Dh, patch_site_h, len(heldout_ids), D, patch_site, site_labels, nb, codes, h_codes)
    fph, expl = fingerprint_heldout(X, y, codes, H, h_codes)
    rows = []
    for h, sid in enumerate(heldout_ids):
        top = hsc[h].pooled["top10"]
        p_patch = pm.softmax_conf(top)
        q = fph[[f"q_fp_{b}" for b in BATCHES]].to_numpy()[h]
        p_e = ensemble(p_patch, q)
        k = int(np.argmax(p_e))
        pc = int(np.argmin(top))
        fc = BATCHES.index(fph["fp_call"].iloc[h])
        ag = pc == fc
        high = heldout_flag(c_e, agree, ag) and not bool(fph["fp_ood"].iloc[h])
        sel = patch_site_h == h
        c = coords_h[sel]
        anom = hsc[h].u[:, 2] > pm.ANOM_U
        frac = float(anom.mean())
        rows.append({
            "site": sid, "assigned": BATCHES[k], "confidence_flag": "high" if high else "low",
            **{f"p_ens_{b}": float(p_e[j]) for j, b in enumerate(BATCHES)},
            "patch_call": BATCHES[pc], "fingerprint_call": BATCHES[fc],
            "explanation": make_explanation(sid, k, high, expl[expl["site"] == sid], pc, frac,
                                            c[anom, 0], int(c[:, 0].max()) + 1, b3_med),
        })
    final = pd.DataFrame(rows)[["site", "assigned", "confidence_flag", "p_ens_Batch_1", "p_ens_Batch_2",
                                "p_ens_Batch_3", "patch_call", "fingerprint_call", "explanation"]]
    return {"evaluation": evaluation, "lopo_predictions": preds, "final_heldout": final}
