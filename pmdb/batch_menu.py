"""Model menu for the batch call: labels incl. organiser truths, parent-centred fingerprint, LOPO menu CV,
pre-registered selection and test-site prediction. See docs/patch_mil.md (Feedback round 1).

Builds on pmdb.patch_lopo / pmdb.patch_mil / pmdb.fingerprint without modifying them. Parent membership is used
only for fold/bank exclusion and for centring (labels are never read there); never as a call or flag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pmdb import fingerprint as fp
from pmdb import patch_lopo as plo
from pmdb import patch_mil as pm

ROOT = Path(__file__).resolve().parents[1]
BATCHES = pm.BATCHES
OPTIONS = ("fingerprint", "fingerprint_centred", "patch", "ensemble", "ensemble_centred")
N_COMPONENTS = {"fingerprint": 1, "fingerprint_centred": 1, "patch": 1, "ensemble": 2, "ensemble_centred": 2}
# option -> (call column, correct column, flag column)
OPTION_COLS = {
    "fingerprint": ("fp_call", "correct_fp", "flag_fingerprint"),
    "fingerprint_centred": ("fpc_call", "correct_fpc", "flag_fingerprint_centred"),
    "patch": ("patch_call", "correct_patch", "flag_patch"),
    "ensemble": ("ens_call", "correct_ens", "flag_ensemble"),
    "ensemble_centred": ("ensc_call", "correct_ensc", "flag_ensemble_centred"),
}
FP_TAU = pm.SOFTMAX_TAU  # same temperature as the patch softmax_conf
SINGLETON_RULE = ("A site that is the only member of its parent in the pool is excluded from centred-model "
                  "training and predicted with the uncentred fingerprint (its centred row is identically zero "
                  "and carries no information).")
CAVEAT = ("Selection is the maximum of 5 LOPO rubric estimates on 34 sites from 13 parent images; the winning "
          "estimate is optimistically biased (winner's curse). Differences smaller than about one SE are not "
          "meaningful. The 3 held-out sites enter LOPO as ordinary labelled sites; nothing was tuned on them.")


# --------------------------------------------------------------------------- labels / features

def labelled_sites(root: Path | str | None = None) -> pd.DataFrame:
    """[batch, site, label]: 31 rows of cache/half/manifest.csv (label = batch), then the rows of
    outputs/heldout_labels.csv as batch='Batch_heldout', label=<truth>."""
    root = Path(root) if root is not None else ROOT
    man = pd.read_csv(root / "cache" / "half" / "manifest.csv", dtype={"site": str})
    lab = pd.DataFrame({"batch": man["batch"], "site": man["site"], "label": man["batch"]})
    hl = pd.read_csv(root / "outputs" / "heldout_labels.csv", dtype={"site": str})
    hman = pd.read_csv(root / "cache_heldout" / "half" / "manifest.csv", dtype={"site": str})
    missing = sorted(set(hl["site"]) - set(hman["site"]))
    if missing:
        raise ValueError(f"labels for sites not in cache_heldout/half/manifest.csv: {missing}")
    bad = sorted(set(hl["batch"]) - set(BATCHES))
    if bad:
        raise ValueError(f"labels not in {BATCHES}: {bad}")
    h = pd.DataFrame({"batch": "Batch_heldout", "site": hl["site"], "label": hl["batch"]})
    return pd.concat([lab, h], ignore_index=True)


def feature_table(keys: list[tuple[str, str]], root: Path | str | None = None) -> pd.DataFrame:
    """features.csv + the Modal-computed held-out features (outputs/heldout_features_modal.csv), index
    (batch, site), reindexed to keys."""
    root = Path(root) if root is not None else ROOT
    parts = [pd.read_csv(root / "outputs" / "fingerprint" / "features.csv", dtype={"site": str}),
             pd.read_csv(root / "outputs" / "heldout_features_modal.csv", dtype={"site": str})]
    F = pd.concat(parts, ignore_index=True).set_index(["batch", "site"])
    idx = pd.MultiIndex.from_tuples([tuple(k) for k in keys], names=["batch", "site"])
    missing = [k for k in idx if k not in F.index]
    if missing:
        raise ValueError(f"no fingerprint features for {missing}")
    return F.loc[idx]


# --------------------------------------------------------------------------- numerics

def fp_prob(scores: np.ndarray, tau: float = FP_TAU) -> np.ndarray:
    """softmax(-scores / tau) over the last axis; argmax equals the fingerprint call (argmin of scores)."""
    s = np.asarray(scores, dtype=float)
    z = np.exp(-(s - s.min(axis=-1, keepdims=True)) / tau)
    return z / z.sum(axis=-1, keepdims=True)


def _pred_frame2(pr: pd.DataFrame) -> pd.DataFrame:
    out = plo._pred_frame(pr)
    sc = pr[[f"score_{b}" for b in BATCHES]].to_numpy(dtype=float)
    pf = fp_prob(sc)
    for j, b in enumerate(BATCHES):
        out[f"s_fp_{b}"] = sc[:, j]
    for j, b in enumerate(BATCHES):
        out[f"pf_{b}"] = pf[:, j]
    return out


def centre_by_parent(F: pd.DataFrame, parent: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    g = np.asarray(parent)
    Xc = F - F.groupby(g).transform("mean")
    size = pd.Series(g, index=F.index).map(pd.Series(g).value_counts())
    return Xc, (size == 1)


def _fp_cv(X, y, groups) -> pd.DataFrame:
    groups = np.asarray(groups)
    parts = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = fp.fit(X[tr], y[tr])
        assert m.batches == list(BATCHES)
        parts.append(_pred_frame2(fp.predict(m, X[te])))
    return pd.concat(parts).loc[X.index]


def fingerprint_centred_cv(Xc, y, groups, singleton: np.ndarray, fallback: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-parent-out fingerprint on parent-centred features. Singleton rows are excluded from training
    and take the fallback (uncentred) rows. A fold whose fit fails falls back to the uncentred rows
    (recorded in `fpc_fallback`)."""
    groups, singleton = np.asarray(groups), np.asarray(singleton, dtype=bool)
    out = fallback.copy()
    out["fpc_fallback"] = False
    for g in np.unique(groups):
        te = (groups == g) & ~singleton
        if not te.any():
            continue
        tr = (groups != g) & ~singleton
        try:
            m = fp.fit(Xc[tr], y[tr])
            assert m.batches == list(BATCHES)
            pr = _pred_frame2(fp.predict(m, Xc[te]))
        except (ValueError, AssertionError, KeyError, np.linalg.LinAlgError):
            out.loc[te, "fpc_fallback"] = True
            continue
        out.loc[pr.index, pr.columns] = pr
    return out.loc[Xc.index]


def global_flags(correct, groups) -> np.ndarray:
    correct, groups = np.asarray(correct, float), np.asarray(groups)
    out = np.zeros(len(correct), dtype=bool)
    for i in range(len(correct)):
        m = groups != groups[i]
        out[i] = bool(m.any() and correct[m].mean() > 0.5)
    return out


def menu_cv(D, patch_site, site_labels, groups, X, Xc, y, singleton) -> tuple[pd.DataFrame, list]:
    site_labels, groups = np.asarray(site_labels), np.asarray(groups)
    singleton = np.asarray(singleton, dtype=bool)
    df, sc = plo._run_cv(D, patch_site, site_labels, groups, X, y)
    fpf = _fp_cv(X, y, groups)
    fpc = fingerprint_centred_cv(Xc, y, groups, singleton, fpf)
    top = np.stack([s.pooled["top10"] for s in sc])
    p_patch = pm.softmax_conf(top)
    patch_idx = top.argmin(axis=1)
    fp_idx = np.array([BATCHES.index(c) for c in fpf["fp_call"]])
    fpc_idx = np.array([BATCHES.index(c) for c in fpc["fp_call"]])
    pf = fpf[[f"pf_{b}" for b in BATCHES]].to_numpy()
    pfc = fpc[[f"pf_{b}" for b in BATCHES]].to_numpy()
    # conformal-normalised ensemble of the lopo run kept for comparison; menu ensembles use fp_prob
    df["ens_call_conformal"] = df["ens_call"]
    p_ens = plo.ensemble(p_patch, pf)
    p_ensc = plo.ensemble(p_patch, pfc)
    ens_idx, ensc_idx = p_ens.argmax(axis=1), p_ensc.argmax(axis=1)
    df["ens_call"] = [BATCHES[c] for c in ens_idx]
    for j, b in enumerate(BATCHES):
        df[f"pf_{b}"] = pf[:, j]
        df[f"p_ens_{b}"] = p_ens[:, j]
        df[f"q_fpc_{b}"] = fpc[f"q_fp_{b}"].to_numpy()
        df[f"pfc_{b}"] = pfc[:, j]
        df[f"p_ensc_{b}"] = p_ensc[:, j]
    df["fpc_call"] = [BATCHES[c] for c in fpc_idx]
    df["ensc_call"] = [BATCHES[c] for c in ensc_idx]
    df["fpc_ood"] = fpc["fp_ood"].to_numpy(dtype=bool)
    df["fpc_fallback"] = fpc["fpc_fallback"].to_numpy(dtype=bool)
    df["agree_fpc"] = patch_idx == fpc_idx
    df["correct_fp"] = fp_idx == site_labels
    df["correct_fpc"] = fpc_idx == site_labels
    df["correct_ens"] = ens_idx == site_labels
    df["correct_ensc"] = ensc_idx == site_labels
    df["parent_code"] = groups
    df["singleton"] = singleton
    fp_ood, fpc_ood = df["fp_ood"].to_numpy(), df["fpc_ood"].to_numpy()
    df["flag_fingerprint"] = global_flags(df["correct_fp"], groups) & ~fp_ood
    df["flag_fingerprint_centred"] = global_flags(df["correct_fpc"], groups) & ~fpc_ood
    df["flag_patch"] = global_flags(df["correct_patch"], groups)
    df["flag_ensemble"] = plo.stratum_flags(df["correct_ens"], df["agree"], groups) & ~fp_ood
    df["flag_ensemble_centred"] = plo.stratum_flags(df["correct_ensc"], df["agree_fpc"], groups) & ~fpc_ood
    return df, sc


def menu_summary(df: pd.DataFrame, site_labels) -> dict:
    site_labels = np.asarray(site_labels)
    out: dict = {}
    for opt in OPTIONS:
        call, corr, flag = OPTION_COLS[opt]
        pred = np.array([BATCHES.index(c) for c in df[call]])
        met = pm.classification_metrics(site_labels, pred)
        c, f = df[corr].to_numpy(bool), df[flag].to_numpy(bool)
        pts = np.where(f, np.where(c, 2.0, 0.0), 1.0)
        out[opt] = {
            "accuracy": float(met["accuracy"]), "balanced_accuracy": float(met["balanced_accuracy"]),
            "metrics": met, "rubric": plo.expected_rubric(c, f),
            "rubric_all_high": plo.expected_rubric(c, np.ones(len(c), bool)),
            "rubric_se": float(pts.std(ddof=1) / np.sqrt(len(pts))), "n_high": int(f.sum()),
            "accuracy_when_high": float(c[f].mean()) if f.any() else None,
            "n_components": N_COMPONENTS[opt]}
    out["all_low"] = 1.0
    out["n_fpc_fallback"] = int(df["fpc_fallback"].sum())
    out["caveat"] = CAVEAT
    out["singleton_rule"] = SINGLETON_RULE
    return out


def select_option(summary: dict) -> dict:
    order = {o: i for i, o in enumerate(OPTIONS)}

    def key(score_name):
        return lambda o: (-round(summary[o][score_name], 9), N_COMPONENTS[o], order[o])

    best = sorted(OPTIONS, key=key("rubric"))[0]
    if summary[best]["rubric"] <= 1.0 + 1e-12:
        opt = sorted(OPTIONS, key=key("accuracy"))[0]
        return {"option": opt, "confidence_mode": "all_low", "rubric": 1.0,
                "rule": "no option beats the all-low rubric 1.0 on LOPO: every confidence is low; option = "
                        "highest LOPO accuracy (ties: fewer components, then menu order)"}
    return {"option": best, "confidence_mode": "rule", "rubric": float(summary[best]["rubric"]),
            "rule": "max LOPO expected rubric with the option's flag rule (ties: fewer components, then menu order)"}


# --------------------------------------------------------------------------- explanations

def evidence_items(expl: pd.DataFrame, patch_call: int) -> list[tuple[str, int]]:
    e = expl.sort_values("dev_Batch_3", ascending=False, kind="stable").head(3)
    items = [(plo.feature_phrase(r["feature"]), int(np.argmin([r[f"dev_{b}"] for b in BATCHES])))
             for _, r in e.iterrows()]
    return items + [("the local microstructure appearance", int(patch_call))]


def _verb(items: list[str], plural: str, singular: str) -> str:
    return singular if len(items) == 1 else plural


def confidence_sentence(k: int, high: bool, items: list[tuple[str, int]]) -> str:
    with_k = [ph for ph, r in items if r == k]
    other: dict[int, list[str]] = {}
    for ph, r in items:
        if r != k:
            other.setdefault(r, []).append(ph)
    rest = "; ".join(f"{plo._join(v)} {_verb(v, 'look', 'looks')} like Batch {o + 1}"
                     for o, v in sorted(other.items()))
    lead = (f"{plo._join(with_k)} {_verb(with_k, 'point', 'points')} to Batch {k + 1}" if with_k
            else f"nothing clearly points to Batch {k + 1}")
    if len(with_k) == len(items):
        if high:
            return f"Confidence is high: all four lines of evidence consistently point to Batch {k + 1}."
        return ("Confidence is low because the differences from the other batches are small relative "
                "to normal variation between images of the same batch.")
    if high:
        return f"Confidence is high: {lead}, while {rest}."
    return f"Confidence is low because the evidence is mixed: {lead}, but {rest}."


def make_explanation(site, k, high, expl, patch_call, frac, anom_rows, n_rows, b3_med) -> tuple[str, int]:
    """plo.make_explanation with the confidence sentence rebuilt from the evidence. Returns (text, n items for k)."""
    base = plo.make_explanation(site, k, high, expl, patch_call, frac, anom_rows, n_rows, b3_med)
    head = base.rsplit(" Confidence is", 1)[0]
    items = evidence_items(expl, patch_call)
    return head + " " + confidence_sentence(k, high, items), sum(r == k for _, r in items)


# --------------------------------------------------------------------------- test-site prediction

def _fp_rows(Xtr, ytr, rows: pd.DataFrame):
    m = fp.fit(Xtr, ytr)
    assert m.batches == list(BATCHES)
    return _pred_frame2(fp.predict(m, rows)), fp.explain(m, rows)


def predict_test(D, patch_site, site_labels, groups, X, Xc, y, singleton, menu_df, sc_lopo,
                 Dh, patch_site_h, test_keys, test_codes, H, Hc, singleton_h, coords_h, selection,
                 test_parent_ids=None, p_probe=None) -> pd.DataFrame:
    site_labels, groups = np.asarray(site_labels), np.asarray(groups)
    singleton, test_codes = np.asarray(singleton, bool), np.asarray(test_codes)
    singleton_h = np.asarray(singleton_h, bool)
    n = len(test_keys)
    hsc = pm.heldout_scores(Dh, patch_site_h, n, D, patch_site, site_labels, len(BATCHES), groups, test_codes)
    b3_med = float(np.median([(s.u[:, 2] > pm.ANOM_U).mean() for s, t in zip(sc_lopo, site_labels) if t == 2]))
    rows = []
    for h in range(n):
        site = test_keys[h][1] if isinstance(test_keys[h], tuple) else test_keys[h]
        top = hsc[h].pooled["top10"]
        p_patch = pm.softmax_conf(top)
        pc = int(np.argmin(top))
        tr = groups != test_codes[h]
        fph, expl = _fp_rows(X[tr], y[tr], H.iloc[[h]])
        fc = BATCHES.index(fph["fp_call"].iloc[0])
        pf = fph[[f"pf_{b}" for b in BATCHES]].to_numpy()[0]
        fpc_ood = bool(fph["fp_ood"].iloc[0])
        if singleton_h[h]:
            pfc, fcc = pf, fc
        else:
            try:
                fpcf, _ = _fp_rows(Xc[tr & ~singleton], y[tr & ~singleton], Hc.iloc[[h]])
                pfc = fpcf[[f"pf_{b}" for b in BATCHES]].to_numpy()[0]
                fcc = BATCHES.index(fpcf["fp_call"].iloc[0])
                fpc_ood = bool(fpcf["fp_ood"].iloc[0])
            except (ValueError, AssertionError, KeyError, np.linalg.LinAlgError):
                pfc, fcc = pf, fc
        p_ens, p_ensc = plo.ensemble(p_patch, pf), plo.ensemble(p_patch, pfc)
        fp_ood = bool(fph["fp_ood"].iloc[0])
        opt = selection["option"]
        if opt != "probe":
            call, corr, _ = OPTION_COLS[opt]
        if opt == "probe":
            prob = np.asarray(p_probe[h])
            k = int(np.argmax(prob))
            high = float(prob.max()) >= 0.5
        elif opt == "fingerprint":
            prob, k = pf, fc
            high = plo_global(menu_df[corr]) and not fp_ood
        elif opt == "fingerprint_centred":
            prob, k = pfc, fcc
            high = plo_global(menu_df[corr]) and not fpc_ood
        elif opt == "patch":
            prob, k = p_patch, pc
            high = plo_global(menu_df[corr])
        elif opt == "ensemble":
            prob, k = p_ens, int(np.argmax(p_ens))
            high = plo.heldout_flag(menu_df[corr], menu_df["agree"], pc == fc) and not fp_ood
        else:
            prob, k = p_ensc, int(np.argmax(p_ensc))
            high = plo.heldout_flag(menu_df[corr], menu_df["agree_fpc"], pc == fcc) and not fpc_ood
        if selection.get("confidence_mode") == "all_low":
            high = False
        sel = patch_site_h == h
        c = coords_h[sel]
        anom = hsc[h].u[:, 2] > pm.ANOM_U
        text, n_for = make_explanation(site, k, bool(high), expl, pc, float(anom.mean()), c[anom, 0],
                                       int(c[:, 0].max()) + 1, b3_med)
        rows.append({
            "site": site, "assigned": BATCHES[k], "confidence": "high" if high else "low", "option": opt,
            **{f"p_{b}": float(prob[j]) for j, b in enumerate(BATCHES)},
            **{f"reg_{m}_{b}": float(v[j]) for m, v in (("fingerprint", pf), ("patch_knn", p_patch),
                                                        ("probe", np.asarray(p_probe[h])))
               for j, b in enumerate(BATCHES)},
            "patch_call": BATCHES[pc], "fingerprint_call": BATCHES[fc], "fingerprint_centred_call": BATCHES[fcc],
            "n_evidence_for_call": int(n_for),
            "parent_id": test_parent_ids[h] if test_parent_ids is not None else "",
            "explanation": text})
    return pd.DataFrame(rows)


def plo_global(correct) -> bool:
    return bool(np.asarray(correct, float).mean() > 0.5)
