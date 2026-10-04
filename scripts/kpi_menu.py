"""KPI-PCA model and fp/patch ensembles added to the frozen LOPO menu (offline, no menu files modified).

    python scripts/kpi_menu.py

Writes outputs/menu/kpi_menu_predictions.csv, outputs/menu/kpi_menu_summary.csv and the standalone model's
LOPO predictions outputs/kpis/kpi_pca/lopo_predictions.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from batch_classifier import feature_sets, load_combined  # noqa: E402

from pmdb import batch_menu as bm  # noqa: E402
from pmdb import kpi_pca  # noqa: E402
from pmdb import patch_lopo as plo  # noqa: E402

B = kpi_pca.BATCHES
HELDOUT = ("3e122cbj", "fn0mhxef", "xrv9xvzb")


def calls(p):
    return np.array(B)[np.asarray(p).argmax(1)]


def main() -> None:
    menu = pd.read_csv(ROOT / "outputs" / "menu" / "menu_predictions.csv")
    k, h = ROOT / "outputs" / "kpis", ROOT / "outputs" / "heldout" / "kpis"
    names = ("site_kpis.csv", "materials_site_kpis.csv", "anna_site_kpis.csv")
    train = load_combined(*[k / n for n in names])
    test = load_combined(*[h / n for n in names])
    cols = [c for c in feature_sets(train)["combined"] if train[c].notna().any()]
    kp = pd.concat([train[["site"] + cols], test[["site"] + cols]], ignore_index=True)
    assert kp["site"].is_unique
    df = menu.merge(kp, on="site", how="inner", validate="1:1")
    assert len(df) == 34 and len(menu) == 34

    y, sites, groups = df["true"].to_numpy(), df["site"].to_numpy(), df["parent_code"].to_numpy()
    p_kpi = kpi_pca.lopo_proba(df[cols].to_numpy(float), y, sites, groups)
    pf = df[[f"pf_{b}" for b in B]].to_numpy(float)
    pp = df[[f"p_patch_{b}" for b in B]].to_numpy(float)
    probs = {"kpi_pca": p_kpi, "fp_kpi": 0.5 * (pf + p_kpi), "ensemble3": (pp + pf + p_kpi) / 3,
             "fingerprint": pf, "patch": pp, "ensemble": 0.5 * (pp + pf)}
    call = {o: calls(p) for o, p in probs.items()}
    assert (call["ensemble"] == df["ens_call"].to_numpy()).all()
    correct = {o: c == y for o, c in call.items()}
    ood = df["fp_ood"].to_numpy(bool)

    flag = {
        "kpi_pca": bm.global_flags(correct["kpi_pca"], groups),
        "fp_kpi": plo.stratum_flags(correct["fp_kpi"], call["fingerprint"] == call["kpi_pca"], groups) & ~ood,
        "ensemble3": plo.stratum_flags(correct["ensemble3"], (call["patch"] == call["fingerprint"])
                                       & (call["fingerprint"] == call["kpi_pca"]), groups) & ~ood,
        "fingerprint": df["flag_fingerprint"].to_numpy(bool),
        "patch": df["flag_patch"].to_numpy(bool),
        "ensemble": df["flag_ensemble"].to_numpy(bool),
    }

    rows, held = [], []
    for o in probs:
        c, f = correct[o], flag[o]
        score = np.where(f, np.where(c, 2, 0), 1)
        bacc = np.mean([np.mean(call[o][y == b] == b) for b in B])
        rows.append({"option": o, "accuracy": c.mean(), "balanced_accuracy": bacc,
                     "expected_rubric": plo.expected_rubric(c, f), "rubric_se": score.std() / np.sqrt(len(score)),
                     "n_high": int(f.sum())})
        for i in np.where(np.isin(sites, HELDOUT))[0]:
            held.append({"option": o, "site": sites[i], "true": y[i], "call": call[o][i],
                         "max_prob": probs[o][i].max(), "flag": bool(f[i])})
    summ = pd.DataFrame(rows)

    out = df[["site", "true", "parent_code"]].copy()
    for name, key in (("p_kpi", "kpi_pca"), ("p_fpkpi", "fp_kpi"), ("p_ens3", "ensemble3")):
        for i, b in enumerate(B):
            out[f"{name}_{b}"] = probs[key][:, i]
    for o in ("kpi_pca", "fp_kpi", "ensemble3"):
        out[f"call_{o}"], out[f"correct_{o}"], out[f"flag_{o}"] = call[o], correct[o], flag[o]
    out.to_csv(ROOT / "outputs" / "menu" / "kpi_menu_predictions.csv", index=False)
    summ.to_csv(ROOT / "outputs" / "menu" / "kpi_menu_summary.csv", index=False)

    # Standalone KPI-PCA model in the shared per-model schema (site, assigned, confidence, p_*),
    # LOPO predictions for all 34 labelled sites; `high` is its global flag rule.
    model = df[["site", "true"]].assign(assigned=call["kpi_pca"], confidence=p_kpi.max(1), high=flag["kpi_pca"])
    for i, b in enumerate(B):
        model[f"p_{b}"] = p_kpi[:, i]
    (ROOT / "outputs" / "kpis" / "kpi_pca").mkdir(exist_ok=True)
    model.to_csv(ROOT / "outputs" / "kpis" / "kpi_pca" / "lopo_predictions.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print()
    print(pd.DataFrame(held).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
