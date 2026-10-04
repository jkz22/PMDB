"""Sibling-anchored batch calls from the pore-floor dark level.

Hypothesis (docs/parents.md on the functional-morphology branch): the organiser's per-crop grouping
feature tracks the SE-detector pore-floor dark level, ordered B1 < B2 < B3.  Two predictors:

* cross-parent: exhaustive two-cutpoint rule, evaluated leave-one-parent-out (LOPO);
* sibling-anchored: a crop is placed relative to the labelled crops of its *own parent image*
  (same acquisition, so instrument offsets cancel); the cross-parent rule is a weak fallback vote.

Writes outputs/sibling_anchor/<feature>/{labelled_eval.csv,test_predictions.csv,summary.json,summary.md}.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "sibling_anchor"
CLASSES = ["Batch_1", "Batch_2", "Batch_3"]
TRUTH_HELDOUT = {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}


def load(feature: str) -> pd.DataFrame:
    cl = pd.concat([pd.read_csv(ROOT / f) for f in
                    ["outputs/clean/summary.csv", "outputs/clean_heldout/summary.csv", "outputs/clean_test/summary.csv"]])
    sites = pd.read_csv(ROOT / "outputs/stitching/sites.csv")
    sites["label"] = [TRUTH_HELDOUT.get(s, b if b.startswith("Batch_") and b[-1].isdigit() else None)
                      for s, b in zip(sites.site, sites.batch)]
    df = sites.merge(cl[["site", feature]], on="site", how="left").rename(columns={feature: "f"})
    return df[["site", "batch", "parent", "label", "f"]]


def two_cut_fit(f: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    vals = np.unique(f)
    mids = (vals[:-1] + vals[1:]) / 2
    best = (-1.0, -np.inf, np.inf)
    for c1, c2 in itertools.combinations(mids, 2):
        pred = np.where(f <= c1, 0, np.where(f <= c2, 1, 2))
        acc = (pred == y).mean()
        if acc > best[0]:
            best = (acc, c1, c2)
    return best


def two_cut_predict(f: float, c1: float, c2: float) -> int:
    return 0 if f <= c1 else (1 if f <= c2 else 2)


def within_pure_sd(lab: pd.DataFrame) -> float:
    res = []
    for _, g in lab.groupby("parent"):
        if g.label.nunique() == 1 and len(g) > 1:
            res.extend((g.f - g.f.mean()).tolist())
    return float(np.std(res, ddof=1))


def anchored_probs(f: float, sibs: pd.DataFrame, delta: float, fallback: int | None, w_fallback: float = 0.5) -> np.ndarray:
    w = np.zeros(3)
    for _, s in sibs.iterrows():
        k = CLASSES.index(s.label)
        d = f - s.f
        if d > delta:
            k = min(k + 1, 2)
        elif d < -delta:
            k = max(k - 1, 0)
        w[k] += 1.0 / (1.0 + abs(d) / delta)
    if fallback is not None:
        w[fallback] += w_fallback
    return w / w.sum()


def main(feature: str = "SE_type_D") -> None:
    OUT = OUT_ROOT / feature
    OUT.mkdir(parents=True, exist_ok=True)
    df = load(feature)
    lab = df[df.label.notna()].reset_index(drop=True)
    y = lab.label.map(CLASSES.index).to_numpy()
    delta = within_pure_sd(lab)

    # cross-parent rule, LOPO
    lopo_pred = np.zeros(len(lab), int)
    for p in lab.parent.unique():
        tr = lab.parent != p
        _, c1, c2 = two_cut_fit(lab.f[tr].to_numpy(), y[tr])
        for i in np.where(~tr)[0]:
            lopo_pred[i] = two_cut_predict(lab.f[i], c1, c2)
    acc_full, c1_full, c2_full = two_cut_fit(lab.f.to_numpy(), y)

    # sibling-anchored, each labelled site predicted from its siblings only (its own label hidden)
    rows = []
    for i, r in lab.iterrows():
        sibs = lab[(lab.parent == r.parent) & (lab.site != r.site)]
        tr = lab.parent != r.parent
        _, c1, c2 = two_cut_fit(lab.f[tr].to_numpy(), y[tr])
        fb = two_cut_predict(r.f, c1, c2)
        if len(sibs):
            pr = anchored_probs(r.f, sibs, delta, fb)
            mode = "anchored"
        else:
            pr = np.eye(3)[fb]
            mode = "fallback"
        rows.append(dict(site=r.site, parent=r.parent, truth=r.label, f=r.f, mode=mode, n_sibs=len(sibs),
                         mixed_parent=sibs.label.nunique() > 1 or (len(sibs) and (sibs.label != r.label).any()),
                         lopo_cut_call=CLASSES[lopo_pred[i]], anchored_call=CLASSES[int(pr.argmax())],
                         p_B1=pr[0], p_B2=pr[1], p_B3=pr[2]))
    ev = pd.DataFrame(rows)
    ev.to_csv(OUT / "labelled_eval.csv", index=False)

    # test sites
    test = df[df.label.isna()].reset_index(drop=True)
    trows = []
    for _, r in test.iterrows():
        sibs = lab[lab.parent == r.parent]
        fb = two_cut_predict(r.f, c1_full, c2_full)
        pr = anchored_probs(r.f, sibs, delta, fb) if len(sibs) else np.eye(3)[fb]
        call = CLASSES[int(pr.argmax())]
        sib_txt = "; ".join(f"{s.site} ({s.label[-1]}) {s.f:+.2f}" for _, s in sibs.iterrows())
        trows.append(dict(site=r.site, parent=r.parent, f=r.f, assigned=call,
                          confidence="high" if pr.max() >= 0.6 else "low", p_B1=pr[0], p_B2=pr[1], p_B3=pr[2],
                          cut_rule_call=CLASSES[fb], siblings=sib_txt))
    tp = pd.DataFrame(trows)
    tp.to_csv(OUT / "test_predictions.csv", index=False)

    def acc(pred, truth):
        return float((pred == truth).mean())

    def recalls(pred, truth):
        return {c: float((pred[truth == c] == c).mean()) for c in CLASSES}

    anch = ev[ev["mode"] == "anchored"]
    mixed = anch[anch.mixed_parent]
    summ = {
        "feature": feature, "delta_within_pure_parent_sd": delta, "n_labelled": len(lab),
        "majority": float(np.bincount(y).max() / len(y)),
        "cut_rule_full_fit": {"acc": acc_full, "c1": c1_full, "c2": c2_full},
        "cut_rule_lopo": {"acc": acc(ev.lopo_cut_call, ev.truth), "recall": recalls(ev.lopo_cut_call, ev.truth)},
        "anchored_all": {"n": len(anch), "acc": acc(anch.anchored_call, anch.truth), "recall": recalls(anch.anchored_call, anch.truth)},
        "anchored_mixed_parents": {"n": len(mixed), "acc": acc(mixed.anchored_call, mixed.truth), "recall": recalls(mixed.anchored_call, mixed.truth)},
        "lopo_cut_on_same_sites": {"all": acc(anch.lopo_cut_call, anch.truth), "mixed": acc(mixed.lopo_cut_call, mixed.truth)},
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))

    md = [f"# Sibling-anchored calls from `{feature}` (pore-floor dark level)\n",
          f"Labelled sites {len(lab)} (31 + 3 released truths), majority {summ['majority']:.3f}. "
          f"delta (within pure-parent SD of {feature}) = {delta:.2f} DN.\n",
          "| predictor | n | acc | recall B1/B2/B3 |", "|---|---|---|---|",
          f"| two-cut rule, full fit (cuts {c1_full:.2f}, {c2_full:.2f}) | {len(lab)} | {acc_full:.3f} | - |",
          f"| two-cut rule, LOPO | {len(lab)} | {summ['cut_rule_lopo']['acc']:.3f} | " + "/".join(f"{v:.2f}" for v in summ['cut_rule_lopo']['recall'].values()) + " |",
          f"| sibling-anchored (sites with labelled siblings) | {len(anch)} | {summ['anchored_all']['acc']:.3f} | " + "/".join(f"{v:.2f}" for v in summ['anchored_all']['recall'].values()) + " |",
          f"| sibling-anchored, mixed parents only | {len(mixed)} | {summ['anchored_mixed_parents']['acc']:.3f} | " + "/".join(f"{v:.2f}" for v in summ['anchored_mixed_parents']['recall'].values()) + " |",
          f"| LOPO cut rule on the same mixed-parent sites | {len(mixed)} | {summ['lopo_cut_on_same_sites']['mixed']:.3f} | - |",
          "\n## Test-site calls\n",
          "| site | parent | D | assigned | conf | p(B1) | p(B2) | p(B3) | cut-rule | labelled siblings (label) D |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tp.iterrows():
        md.append(f"| {r.site} | {r.parent} | {r.f:+.2f} | {r.assigned} | {r.confidence} | {r.p_B1:.2f} | {r.p_B2:.2f} | {r.p_B3:.2f} | {r.cut_rule_call} | {r.siblings} |")
    md += ["\n## Per-site labelled evaluation\n", "| site | parent | truth | D | mixed | LOPO cut | anchored | p(B1) | p(B2) | p(B3) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ev.sort_values(["parent", "f"]).iterrows():
        md.append(f"| {r.site} | {r.parent} | {r.truth} | {r.f:+.2f} | {'y' if r.mixed_parent else ''} | {r.lopo_cut_call} | {r.anchored_call} | {r.p_B1:.2f} | {r.p_B2:.2f} | {r.p_B3:.2f} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    print(tp[["site", "parent", "f", "assigned", "confidence", "p_B1", "p_B2", "p_B3", "cut_rule_call"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "SE_type_D")
