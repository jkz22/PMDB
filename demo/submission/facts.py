#!/usr/bin/env python3
"""Every number used in the submission materials, read from outputs/. Single source of truth.

    python demo/submission/facts.py            # print facts as JSON
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from demo_confound import BATCHES, load_bse_stats, nearest_centroid  # noqa: E402

O = ROOT / "outputs"
SOURCES = [
    "fingerprint/evaluation.json", "fingerprint/heldout_predictions.csv",
    "overnight/permutation/permutation_10k.json", "overnight/stability/jackknife_summary.csv",
    "overnight/stability/hyperparam_grid.csv", "raw_intensity_stats.csv", "fem/run_log.json",
    "fem/validation.csv", "kpis/site_kpis.csv",
]
# directories whose metrics.csv are pre-registered challengers to the fingerprint (passes_rule column)
CHALLENGERS = {"fem_fingerprint": "FEM features added to the fingerprint (D21)",
               "xgb_kpi": "XGBoost on screened KPIs (+FEM) (D22)"}
KNOWN_OUTPUT_DIRS = {"classifier", "clean", "clean_heldout", "clips", "fem", "fem_fingerprint", "fingerprint", "gallery",
                     "harmonisation", "harmonisation_ext", "heldout", "kpis", "overlays", "overnight", "pooling_checks",
                     "xgb_kpi"}


def read_json_loose(path: Path):
    return json.loads(re.sub(r"\bNaN\b|-?Infinity", "null", path.read_text()))


def fingerprint():
    ev = json.loads((O / "fingerprint/evaluation.json").read_text())
    perm = json.loads((O / "overnight/permutation/permutation_10k.json").read_text())
    n = ev["loo"]["n"]
    return {
        "fp_n": n, "fp_correct": round(ev["loo"]["accuracy"] * n), "fp_acc": ev["loo"]["accuracy"],
        "fp_features": len(ev["features"]), "majority_baseline": ev["majority_baseline"],
        "perm_n": perm["n_perm_done"], "perm_p": perm["p_value"], "perm_null_p95": perm["null_p95"],
    }


def heldout():
    calls = []
    jk = pd.read_csv(O / "overnight/stability/jackknife_summary.csv")
    hp = pd.read_csv(O / "overnight/stability/hyperparam_grid.csv")
    for r in csv.DictReader((O / "fingerprint/heldout_predictions.csv").open()):
        s = jk[jk.site == r["site"]]
        share = s[f"share_{r['assigned']}"]
        h = hp[hp.site == r["site"]]
        calls.append({"site": r["site"], "assigned": r["assigned"], "credibility": float(r["credibility"]),
                      "confidence": float(r["confidence"]), "ood": r["ood"] == "True",
                      "jk_share_min": float(share.min()), "jk_share_max": float(share.max()),
                      "hp_agree": int((h.assigned == r["assigned"]).sum()), "hp_configs": len(h)})
    jk_refits = int(jk.groupby("site").n_resamples.sum().iloc[0])
    return {"heldout": calls, "jk_refits": jk_refits,
            "jk_stable_min": min(c["jk_share_min"] for c in calls),
            "hp_configs": calls[0]["hp_configs"] if calls else 0}


def confound():
    X = load_bse_stats()
    y = X.index.get_level_values("batch").to_numpy()
    correct = sum(nearest_centroid(X.iloc[np.arange(len(X)) != i], y[np.arange(len(X)) != i], X.iloc[i].to_numpy()) == y[i]
                  for i in range(len(X)))
    def flips(delta):
        others = [k for k in X.index if k[0] != "Batch_3"]
        hit = sum(nearest_centroid(X, y, np.array([X.loc[k].iloc[0] + delta, X.loc[k].iloc[1] + delta, 0.0])) == "Batch_3"
                  for k in others)
        return hit, len(others)
    f1, n_other = flips(1.0)
    f7, _ = flips(7.0)
    return {"intensity_acc": correct / len(X), "flip1": f1, "flip7": f7, "flip_total": n_other}


def fem():
    log = read_json_loose(O / "fem/run_log.json")
    val = pd.read_csv(O / "fem/validation.csv")
    k = pd.read_csv(O / "kpis/site_kpis.csv")
    lab = k[k.batch != "Batch_heldout"]
    m = val[val.batch != "Batch_heldout"].merge(lab, on=["batch", "site"])
    r = np.corrcoef(m.K01_si_frac_adm, m.swelling_sym)[0, 1]
    groups = [g.K01_si_frac_adm.to_numpy() for _, g in lab.groupby("batch")]
    return {"fem_runs": len(log["cases"]), "fem_sites": len(val), "fem_missing": len(log["missing_cases"]),
            "fem_cost_usd": log["ledger_total_usd"], "fem_r2": r * r, "si_kw_p": kruskal(*groups).pvalue,
            "swell_median": {b: float(val[val.batch == b].swelling_sym.median()) for b in BATCHES}}


def challengers():
    arms = []
    for d, label in CHALLENGERS.items():
        for f in sorted((O / d).glob("*/metrics.csv")):
            for r in csv.DictReader(f.open()):
                if r.get("passes_rule") in ("True", "False"):
                    arms.append({"family": d, "label": label, "tag": f.parent.name, "arm": r["arm"],
                                 "correct": int(r["n_correct"]), "n": int(r["n"]),
                                 "perm_p": float(r["perm_p"]) if r["perm_p"] else None,
                                 "passes": r["passes_rule"] == "True"})
    xgb = [a for a in arms if a["family"] == "xgb_kpi" and a["tag"] == "main"]
    return {"challengers": arms, "any_challenger_passes": any(a["passes"] for a in arms),
            "best_challenger_correct": max((a["correct"] for a in arms), default=None),
            "xgb_kpi_only_correct": next((a["correct"] for a in xgb if a["arm"] == "X1"), None),
            "xgb_best_correct": max((a["correct"] for a in xgb), default=None)}


def provenance():
    files = SOURCES + [str(p.relative_to(O)) for d in CHALLENGERS for p in sorted((O / d).glob("*/metrics.csv"))]
    return {"sources": {f: hashlib.sha1((O / f).read_bytes()).hexdigest()[:12] for f in files},
            "unreviewed_output_dirs": sorted(p.name for p in O.iterdir() if p.is_dir() and p.name not in KNOWN_OUTPUT_DIRS)}


def collect():
    facts = {}
    for part in (fingerprint, heldout, confound, fem, challengers, provenance):
        facts.update(part())
    return facts


if __name__ == "__main__":
    print(json.dumps(collect(), indent=1))
