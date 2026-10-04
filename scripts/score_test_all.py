"""Test day: run every registered model on the test-day sites and write the tables.

    python scripts/score_test_all.py

Needs (see docs/patch_mil.md, Test day runbook): outputs/test/features.csv and outputs/test/kpis/site_kpis.csv
(modal_test_prep.py), outputs/fem/free_lateral_test/site_curves.csv (modal_fem.py full_free settings on
Batch_test + scripts/fem_collect.py --edge), outputs/test/menu_predictions.csv (modal_patch_mil.py --mode test).

Models: fem_a1 (selected, outputs/menu/selection.json; refit on the 31 labelled sites exactly as its held-out
run) and the 3 menu options (fingerprint, fingerprint_centred, patch). Writes outputs/test/
all_models_predictions.csv (long), all_models_wide.csv, final_predictions.csv and submission.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import fem_fingerprint_eval as ffe  # noqa: E402

from pmdb import fingerprint as fp  # noqa: E402
from pmdb import patch_lopo as plo  # noqa: E402

BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
OUT = ROOT / "outputs/test"
FEM_LABELLED = ROOT / "outputs/fem/free_lateral/site_curves.csv"
TEST = (OUT / "features.csv", OUT / "kpis/site_kpis.csv", ROOT / "outputs/fem/free_lateral_test/site_curves.csv")
MENU = ("fingerprint", "fingerprint_centred", "patch")
# organiser-facing names of each line of evidence (no model vocabulary, see plo.FORBIDDEN)
EVIDENCE = {"fingerprint": "the Si depth distribution and graphite clustering",
            "fingerprint_centred": "the same measures relative to adjacent images of the same electrode",
            "patch": "the local microstructure appearance"}
PHRASES = {"swell_resid": "simulated lithiation swelling beyond what its Si content predicts",
           "si_stress_spread": "the spread of simulated stress across the Si particles"}


def fem_a1(test=None):
    """fem_a1 refit on all 31 labelled sites (as fem_fingerprint_eval.py held-out scoring), predict `test`."""
    tr, te, y, leo_cols = ffe.load_inputs(FEM_LABELLED, test)
    codes = np.array([BATCHES.index(b) for b in y])
    Xtr, Xte, _, _ = ffe.frames("A1", tr, te, codes, leo_cols)
    m = fp.fit(Xtr, y)
    return fp.predict(m, Xte), fp.explain(m, Xte)


def phrase(f: str) -> str:
    return PHRASES.get(f, plo.feature_phrase(f)).removeprefix("the ")


def explain(site: str, k: int, high: bool, expl: pd.DataFrame, others: dict[str, str]) -> str:
    """How the site differs from Batch 3 (top 3 features by distance from the Batch 3 centre) + confidence."""
    e = expl.sort_values("dev_Batch_3", ascending=False, kind="stable").head(3)
    diffs = [f"{'higher' if r['z'] > r['center_Batch_3'] else 'lower'} {phrase(r['feature'])}"
             for _, r in e.iterrows()]
    head = (f"Assigned to Batch {k + 1}. Compared with the Batch 3 supplier baseline this image shows "
            f"{plo._join(diffs)}." if k != 2 else
            f"Assigned to Batch 3: it sits within the normal Batch 3 range; its largest departures "
            f"({plo._join(diffs)}) are within the spread between Batch 3 images.")
    agree = [EVIDENCE[o] for o, c in others.items() if c == BATCHES[k]]
    other = [f"{EVIDENCE[o]} looks like Batch {c.split('_')[1]}" for o, c in others.items() if c != BATCHES[k]]
    if high:
        tail = f"Confidence is high: {plo._join(agree)} also point to Batch {k + 1}."
    elif agree:
        tail = (f"Confidence is low because the evidence is mixed: {plo._join(agree)} also point to "
                f"Batch {k + 1}, but {plo._join(other)}.")
    else:
        tail = f"Confidence is low because the evidence is mixed: {plo._join(other)}."
    text = f"{head} {tail}"
    for w in plo.FORBIDDEN:
        assert w not in text.lower(), (w, text)
    return text


def main() -> None:
    sel = json.loads((ROOT / "outputs/menu/selection.json").read_text())
    assert sel["option"] == "fem_a1", sel["option"]
    # reproduce the registered held-out calls before trusting the test path
    held, _ = fem_a1()
    reg = sel["registry"]["fem_a1"]["heldout_calls"]
    assert dict(zip(held.index.get_level_values("site"), held["assigned"])) == reg, held["assigned"]

    pred, expl = fem_a1(TEST)
    menu = pd.read_csv(OUT / "menu_predictions.csv", dtype={"site": str}).set_index("site")
    sites = list(pred.index.get_level_values("site"))
    assert sorted(sites) == sorted(menu.index), (sites, list(menu.index))

    long, final = [], []
    for i, site in enumerate(sites):
        m = menu.loc[site]
        calls = {o: m[f"{o}_call"] for o in MENU}
        k = BATCHES.index(pred["assigned"].iloc[i])
        ood = bool(pred["ood"].iloc[i])
        n_agree = sum(c == BATCHES[k] for c in calls.values())
        high = (not ood) and n_agree >= 2
        p = {f"p_{b}": float(pred[f"p_{b}"].iloc[i]) for b in BATCHES}
        long.append({"site": site, "parent_id": m["parent_id"], "model": "fem_a1", "selected": True,
                     "assigned": BATCHES[k], "confidence": "high" if high else "low", **p, "ood": ood,
                     # fem_a1 p_* are conformal p-values (fp.predict), not probabilities
                     "credibility": float(pred["credibility"].iloc[i]),
                     "conformal_confidence": float(pred["confidence"].iloc[i])})
        for o in MENU:
            long.append({"site": site, "parent_id": m["parent_id"], "model": o, "selected": False,
                         "assigned": calls[o], "confidence": m[f"{o}_confidence"],
                         **{f"p_{b}": float(m[f"p_{o}_{b}"]) for b in BATCHES}, "ood": np.nan})
        e = expl[expl["site_index"] == pred.index[i]]
        final.append({"site": site, "assigned": BATCHES[k], "confidence": "high" if high else "low",
                      "model": "fem_a1", **p, "n_menu_agree": n_agree, "ood": ood,
                      "parent_id": m["parent_id"], "explanation": explain(site, k, high, e, calls)})

    long = pd.DataFrame(long)
    long.to_csv(OUT / "all_models_predictions.csv", index=False, float_format="%.4f")
    wide = long.pivot(index="site", columns="model", values="assigned")[["fem_a1", *MENU]]
    wide.insert(0, "parent_id", long.groupby("site")["parent_id"].first())
    wide["n_models_agree_with_fem_a1"] = (wide[list(MENU)].to_numpy() == wide[["fem_a1"]].to_numpy()).sum(1)
    wide.reset_index().to_csv(OUT / "all_models_wide.csv", index=False)
    final = pd.DataFrame(final)
    final.to_csv(OUT / "final_predictions.csv", index=False, float_format="%.4f")
    md = ["| site | batch | confidence | explanation |", "|---|---|---|---|"]
    md += [f"| {r.site} | Batch {r.assigned.split('_')[1]} | {r.confidence} | {r.explanation} |"
           for r in final.itertuples()]
    (OUT / "submission.md").write_text("\n".join(md) + "\n")
    print(wide.to_string())
    print(final[["site", "assigned", "confidence", "n_menu_agree", "ood"]].to_string(index=False))


if __name__ == "__main__":
    main()
