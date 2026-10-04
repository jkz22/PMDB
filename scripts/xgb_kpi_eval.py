"""D22 pre-registered XGBoost-on-screened-KPIs test (see .claude/plans/fem-swelling.md, D22).

    modal run modal_eval.py --tag main        # permutation nulls on Modal (once per tag)
    python scripts/xgb_kpi_eval.py            # tag "main": arms X1, X2, X3
    modal run modal_eval.py --tag edge5
    python scripts/xgb_kpi_eval.py --tag edge5 --fem-site-curves outputs/fem/edge5/site_curves.csv

Observed leave-one-site-out runs locally; the >= 1000 label-permutation nulls come from
outputs/xgb_kpi/<tag>/permutation_null.json (written by modal_eval.py; --check-local K
recomputes the first K permutations of every arm locally and reports how many are identical).
Rule (D22): an arm BEATS the fingerprint (21/31) iff LOO correct >= 22/31 and perm p <= 0.05.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb.fem_features import subset  # noqa: E402
from pmdb.xgb_kpi import (ALPHA, FINGERPRINT_CORRECT, MIN_CORRECT, decide,  # noqa: E402
                          fit_predict, fold_matrices, load_inputs, loo_codes, perm_chunk,
                          perm_summary)

TAG_ARMS = {"main": ("X1", "X2", "X3"), "edge5": ("X2e", "X3e")}
DEFAULT_FEM = {"main": "outputs/fem/site_curves.csv", "edge5": "outputs/fem/edge5/site_curves.csv"}


def mi(index_arr):
    return pd.MultiIndex.from_tuples([tuple(r) for r in index_arr], names=["batch", "site"])


def loo_full(arm, tr, codes, batches, kpi_cols):
    """LOO probabilities, per-fold selected features (X3) and nothing else; refit per fold."""
    n, nb = len(codes), len(batches)
    proba = np.empty((n, nb))
    sel = []
    for i in range(n):
        m = np.arange(n) != i
        xtr, xte, _, info = fold_matrices(arm, subset(tr, m), subset(tr, ~m), codes[m])
        proba[i] = fit_predict(xtr, codes[m], xte, nb)[0][0]
        if info:
            for r, (j, pv) in enumerate(zip(info["chosen"], info["p"]), 1):
                sel.append({"arm": arm, "left_out_site": tr["index"][i][1], "rank": r,
                            "feature": tr["cand_names"][j], "kw_p_train": float(pv)})
    return proba, sel


def summarize(true_codes, pred_codes, batches):
    correct = pred_codes == true_codes
    rec = {b: float(correct[true_codes == k].mean()) for k, b in enumerate(batches)}
    conf = {bt: {ba: int(((true_codes == i) & (pred_codes == j)).sum()) for j, ba in enumerate(batches)}
            for i, bt in enumerate(batches)}
    return {"n_correct": int(correct.sum()), "n": int(len(correct)), "accuracy": float(correct.mean()),
            "balanced_accuracy": float(np.mean(list(rec.values()))), "recall": rec, "confusion": conf}


def heldout_table(arm, tr, te, codes, batches, names_cols):
    xtr, xte, extra, _ = fold_matrices(arm, tr, te, codes)
    proba, model = fit_predict(xtr, codes, xte, len(batches))
    df = pd.DataFrame(proba, index=mi(te["index"]), columns=[f"p_{b}" for b in batches])
    df.insert(0, "arm", arm)
    df.insert(1, "extra_features", ";".join(extra))
    df.insert(2, "assigned", [batches[k] for k in proba.argmax(axis=1)])
    df.insert(3, "confidence", proba.max(axis=1))
    imp = pd.DataFrame({"arm": arm, "feature": list(names_cols) + extra,
                        "gain_importance": model.feature_importances_})
    return df, imp


def check_null_matches_table(raw: dict, fem: Path, default_rel: str) -> None:
    """Refuse nulls generated from a different FEM table than the one being evaluated."""
    import hashlib

    want = raw.get("fem_sha256")
    if want is None:  # legacy null file: only trustworthy for the default table
        if Path(fem).resolve() != (ROOT / default_rel).resolve():
            raise SystemExit(f"null file has no table hash and {fem} is not the default {default_rel}; "
                             "regenerate it with modal_eval.py")
        return
    if hashlib.sha256(Path(fem).read_bytes()).hexdigest() != want:
        raise SystemExit(f"permutation null was built from {raw.get('fem_table')} (sha256 mismatch with {fem}); "
                         "p-values would be invalid: rerun modal_eval.py for this table")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="main", choices=sorted(TAG_ARMS))
    ap.add_argument("--fem-site-curves", default=None)
    ap.add_argument("--check-local", type=int, default=0,
                    help="recompute the first K permutations of every arm locally and report agreement")
    args = ap.parse_args()
    arms = TAG_ARMS[args.tag]
    fem = ROOT / (args.fem_site_curves or DEFAULT_FEM[args.tag])
    sens = args.tag != "main"

    tr, te, y, kpi_cols = load_inputs(fem)
    batches = sorted(y.unique())
    codes = np.array([batches.index(b) for b in y])
    nb = len(batches)
    print(f"{len(y)} labelled sites, {len(kpi_cols)} KPIs, tag={args.tag}, arms={arms}")

    null_file = ROOT / "outputs/xgb_kpi" / args.tag / "permutation_null.json"
    if not null_file.exists():
        raise SystemExit(f"{null_file} missing: run `modal run modal_eval.py --tag {args.tag}` first")
    raw = json.loads(null_file.read_text())
    check_null_matches_table(raw, fem, DEFAULT_FEM[args.tag])
    seed = raw["seed"]
    if raw["n_perm"] < 1000:
        raise SystemExit("D22 requires >= 1000 permutations")

    fp_loo = pd.read_csv(ROOT / "outputs/fingerprint/loo_predictions.csv",
                         dtype={"batch": str, "site": str}).set_index(["batch", "site"])
    fp_pred = fp_loo["assigned"].reindex(mi(tr["index"])).to_numpy()
    fp_correct = fp_pred == np.array(y)
    assert int(fp_correct.sum()) == FINGERPRINT_CORRECT, int(fp_correct.sum())

    probas, sels, results, perms, agree = {}, [], {}, {}, []
    for arm in arms:
        proba, sel = loo_full(arm, tr, codes, batches, kpi_cols)
        pred = proba.argmax(axis=1)
        assert (pred == loo_codes(arm, tr, codes, nb)).all()
        probas[arm] = proba
        sels += sel
        results[arm] = summarize(codes, pred, batches)
        null = np.array(raw["null"][arm], float)
        assert len(null) == raw["n_perm"] and np.isfinite(null).all(), arm
        perms[arm] = perm_summary(null, results[arm]["accuracy"], seed)
        results[arm]["perm_p"] = perms[arm]["p_value"]
        if args.check_local:
            ks = list(range(args.check_local))
            local = perm_chunk(arm, tr, codes, nb, seed, ks)
            same = int(np.isclose(local, null[ks], atol=0, rtol=0).sum())
            print(f"{arm}: first {args.check_local} permutations: {same} identical to the local serial run "
                  f"(max |diff| {np.abs(np.array(local) - null[ks]).max():.3f}; Modal x86/Linux vs local "
                  "float differences can flip a borderline split)")
        xc = pred == codes
        for i in range(len(codes)):
            agree.append({"arm": arm, "batch": tr["index"][i][0], "site": tr["index"][i][1],
                          "true": y.iloc[i], "xgb_assigned": batches[pred[i]],
                          "fingerprint_assigned": fp_pred[i],
                          "category": ("both right" if xc[i] and fp_correct[i] else
                                       "only XGB right" if xc[i] else
                                       "only fingerprint right" if fp_correct[i] else "both wrong")})
        print(f"{arm}: {results[arm]['n_correct']}/31 acc {results[arm]['accuracy']:.3f} "
              f"bal {results[arm]['balanced_accuracy']:.3f} perm p {perms[arm]['p_value']:.4f}")

    chosen, passes = decide(results)
    print(f"verdict: {chosen} (pass: {passes})")

    # held-out after the rule: chosen arm + X1 reference (sensitivity tag: every arm in the tag)
    hold_arms = list(arms) if sens else sorted({a for a in (chosen, "X1") if a != "fingerprint"})
    held, imps = [], []
    for arm in hold_arms:
        h, imp = heldout_table(arm, tr, te, codes, batches, kpi_cols)
        held.append(h)
        imps.append(imp)
    held = pd.concat(held)

    out = ROOT / "outputs/xgb_kpi" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    mrows = []
    for arm in arms:
        r = results[arm]
        row = {"arm": arm, "n_correct": r["n_correct"], "n": r["n"], "accuracy": r["accuracy"],
               "balanced_accuracy": r["balanced_accuracy"], "perm_p": r["perm_p"],
               "null_mean": perms[arm]["null_mean"], "null_p95": perms[arm]["null_p95"],
               "passes_rule": passes[arm]}
        row.update({f"recall_{b}": v for b, v in r["recall"].items()})
        row.update({f"conf_{bt}_as_{ba}": v for bt, d in r["confusion"].items() for ba, v in d.items()})
        mrows.append(row)
    fb = summarize(codes, np.array([batches.index(b) for b in fp_pred]), batches)
    ref = {"arm": "fingerprint_A0", "n_correct": fb["n_correct"], "n": fb["n"], "accuracy": fb["accuracy"],
           "balanced_accuracy": fb["balanced_accuracy"], "passes_rule": "reference"}
    ref.update({f"recall_{b}": v for b, v in fb["recall"].items()})
    ref.update({f"conf_{bt}_as_{ba}": v for bt, d in fb["confusion"].items() for ba, v in d.items()})
    mrows.append(ref)
    pd.DataFrame(mrows).to_csv(out / "metrics.csv", index=False)

    lo = []
    for arm in arms:
        d = pd.DataFrame(probas[arm], columns=[f"p_{b}" for b in batches])
        d.insert(0, "arm", arm)
        d.insert(1, "batch", [r[0] for r in tr["index"]])
        d.insert(2, "site", [r[1] for r in tr["index"]])
        d.insert(3, "true", y.to_numpy())
        d.insert(4, "assigned", [batches[k] for k in probas[arm].argmax(axis=1)])
        lo.append(d)
    pd.concat(lo).to_csv(out / "loo_predictions.csv", index=False)
    (out / "permutation.json").write_text(json.dumps(perms, indent=2))
    sel = pd.DataFrame(sels, columns=["arm", "left_out_site", "rank", "feature", "kw_p_train"])
    sel.to_csv(out / "selected_features.csv", index=False)
    pd.DataFrame(agree).to_csv(out / "agreement_vs_fingerprint.csv", index=False)
    held.reset_index().to_csv(out / "heldout_predictions.csv", index=False)
    pd.concat(imps).sort_values(["arm", "gain_importance"], ascending=[True, False]).to_csv(
        out / "feature_importance.csv", index=False)

    ag = pd.DataFrame(agree).groupby(["arm", "category"]).size().unstack(fill_value=0)
    for c in ("both right", "only XGB right", "only fingerprint right", "both wrong"):
        if c not in ag:
            ag[c] = 0
    ag = ag[["both right", "only XGB right", "only fingerprint right", "both wrong"]]
    beats = [a for a in arms if passes[a]]
    lines = [f"# D22 verdict (tag `{args.tag}`)", "",
             (f"**Sensitivity run (tag `{args.tag}`, table `{fem.relative_to(ROOT)}`): never the final arm.**"
              if sens else ""), "",
             f"Rule: an XGB arm BEATS the fingerprint iff LOO correct >= {MIN_CORRECT}/31 (fingerprint "
             f"{FINGERPRINT_CORRECT}/31) and permutation p <= {ALPHA} ({raw['n_perm']} perms, seed {seed}, "
             "all in-fold steps rerun, Modal). Highest correct wins, tie -> fewer features.", "",
             "| arm | correct | accuracy | balanced acc | perm p | null mean | beats fingerprint |",
             "|---|---|---|---|---|---|---|"]
    for arm in arms:
        r = results[arm]
        lines.append(f"| {arm} | {r['n_correct']}/31 | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | "
                     f"{r['perm_p']:.4f} | {perms[arm]['null_mean']:.3f} | {passes[arm]} |")
    lines.append(f"| fingerprint A0 | {fb['n_correct']}/31 | {fb['accuracy']:.3f} | "
                 f"{fb['balanced_accuracy']:.3f} | - | - | reference |")
    if beats:
        lines += ["", f"**Verdict: {chosen} beats the fingerprint ({results[chosen]['n_correct']}/31 vs "
                      f"{FINGERPRINT_CORRECT}/31)**" + (" (sensitivity only)" if sens else "")]
    else:
        lines += ["", "**Verdict: no XGB arm beats the fingerprint (21/31); the fingerprint stays "
                      "classifier of record**"]
    lines += ["", "Per-site agreement with the fingerprint (LOO):", "", "```", ag.to_string(), "```", "",
              "Confusion (true rows -> assigned):", ""]
    for arm in arms:
        lines += [f"{arm}: " + json.dumps(results[arm]["confusion"]), ""]
    if len(sel):
        lines += ["Selected FEM features (count over 31 folds):", ""]
        lines += [f"- {a} {f}: {c}" for (a, f), c in
                  sel.groupby(["arm", "feature"]).size().sort_values(ascending=False).items()]
    lines += ["", "Held-out predictions:", "", "```",
              held[["arm", "assigned", "confidence"]].reset_index().to_string(index=False), "```", ""]
    (out / "verdict.md").write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0


def load_inputs(fem):
    from pmdb.xgb_kpi import load_inputs
    return load_inputs(fem)


if __name__ == "__main__":
    sys.exit(main())
