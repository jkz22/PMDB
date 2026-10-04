#!/usr/bin/env python3
"""Cross-session crosswalk: (1) every held-out call side by side, (2) geometric swelling test vs FEM.

    python scripts/crosswalk.py -> outputs/crosswalk/{heldout_consensus.csv, geometric_vs_fem.csv,
                                   figures/geometric_vs_fem.png, summary.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
OUT = O / "crosswalk"
SITES = ["3e122cbj", "fn0mhxef", "xrv9xvzb"]


def _site(k: str) -> str:
    return str(k).split("/")[-1].strip("')")


def consensus() -> pd.DataFrame:
    rows = []
    fp = pd.read_csv(O / "fingerprint" / "heldout_predictions.csv")
    for r in fp.itertuples():
        rows.append({"site": r.site, "method": "fingerprint (16 curve features, conformal NB)", "session": "fingerprint",
                     "call": r.assigned, "confidence": r.confidence, "note": f"credibility {r.credibility:.2f}, p_B3 {r.p_Batch_3:.2f}"})
    for name, path in (("fingerprint + stretch S01-S04", "stretch"), ("fingerprint + functional F", "functional")):
        d = pd.read_csv(O / path / "heldout_predictions.csv")
        for r in d.itertuples():
            rows.append({"site": _site(r.key), "method": name, "session": "functional-morphology", "call": r.assigned,
                         "confidence": r.confidence, "note": f"credibility {r.credibility:.2f}, p_B3 {r.p_Batch_3:.2f}"})
    rf = pd.read_csv(O / "classifier" / "heldout_predictions.csv")
    for r in rf.itertuples():
        rows.append({"site": r.site, "method": f"two-stage RF, {r.arm} arm" + (" (final)" if r.final else ""),
                     "session": "classifier/FEM", "call": r.predicted, "confidence": r.confidence,
                     "note": f"P(B3) {r.p_b3:.2f}, {r.stage1_votes_not_b3}/{r.n_tiles} tiles not-B3"})
    hj = pd.read_csv(O / "acceptance" / "heldout_joint.csv")
    for r in hj[hj.family == "functional"].itertuples():
        call = "outside Batch 3" if r.pct_among_B3_loo >= 100 else "inside Batch 3"
        rows.append({"site": r.site, "method": "joint functional score vs Batch 3 (one-class)", "session": "functional-morphology",
                     "call": call, "confidence": np.nan, "note": f"score {r.score:.2f}, {r.pct_among_B3_loo:.0f}th pct of B3"})
    df = pd.DataFrame(rows)
    return df


def geometric_vs_fem() -> tuple[pd.DataFrame, dict]:
    fem = pd.read_csv(O / "fem" / "site_curves.csv")
    fem = fem[(fem.s == 1.0) & (fem.orientation == "sym")].copy()
    fem["key"] = fem.batch + "/" + fem.site
    F = pd.concat([pd.read_csv(O / "functional" / "site_functional.csv"),
                   pd.read_csv(O / "functional" / "heldout_site_functional.csv")])
    F["key"] = F.batch + "/" + F.site
    K = pd.concat([pd.read_csv(O / "kpis" / "site_kpis.csv"), pd.read_csv(O / "heldout" / "kpis" / "site_kpis.csv")])
    K["key"] = K.batch + "/" + K.site
    M = fem[["key", "batch", "site", "swelling", "porosity_rel_change", "pore_closed_frac", "surface_rough"]].merge(
        F[["key", "F02_soc100_pore_loss", "F02_soc100_into_graphite", "F02_soc100_into_pore", "F02_soc100_si_frac"]],
        on="key").merge(K[["key", "K01_si_frac_adm"]], on="key")
    pairs = [("F02_soc100_pore_loss", "pore_closed_frac"), ("F02_soc100_pore_loss", "porosity_rel_change"),
             ("K01_si_frac_adm", "swelling"), ("K01_si_frac_adm", "porosity_rel_change"),
             ("F02_soc100_into_graphite", "surface_rough"), ("F02_soc100_into_graphite", "porosity_rel_change")]
    summ = {}
    for a, b in pairs:
        rho, p = spearmanr(M[a], M[b])
        summ[f"{a}~{b}"] = {"spearman": float(rho), "p": float(p)}

    def partial(y, x, ctrl):
        X = np.column_stack([np.ones(len(M)), M[ctrl], M[x]])
        beta, *_ = np.linalg.lstsq(X, M[y].to_numpy(), rcond=None)
        res_y = M[y] - np.column_stack([np.ones(len(M)), M[ctrl]]) @ np.linalg.lstsq(
            np.column_stack([np.ones(len(M)), M[ctrl]]), M[y].to_numpy(), rcond=None)[0]
        res_x = M[x] - np.column_stack([np.ones(len(M)), M[ctrl]]) @ np.linalg.lstsq(
            np.column_stack([np.ones(len(M)), M[ctrl]]), M[x].to_numpy(), rcond=None)[0]
        return float(spearmanr(res_y, res_x)[0]), float(spearmanr(res_y, res_x)[1])

    for y in ("pore_closed_frac", "porosity_rel_change"):
        for x in ("F02_soc100_pore_loss", "F02_soc100_into_graphite"):
            r, p = partial(y, x, "K01_si_frac_adm")
            summ[f"partial[{y}~{x}|K01]"] = {"spearman": r, "p": p}
    summ["n_sites"] = int(len(M))
    return M, summ


def plot(M: pd.DataFrame, summ: dict) -> None:
    col = {"Batch_1": "tab:blue", "Batch_2": "tab:orange", "Batch_3": "tab:green", "Batch_heldout": "k"}
    panels = [("F02_soc100_pore_loss", "pore_closed_frac", "geometric pore loss (SOC 1)", "FEM pore cells closed (s = 1)"),
              ("F02_soc100_pore_loss", "porosity_rel_change", "geometric pore loss (SOC 1)", "FEM relative porosity change"),
              ("K01_si_frac_adm", "swelling", "K01 Si fraction", "FEM surface swelling"),
              ("F02_soc100_into_graphite", "surface_rough", "geometric constrained share", "FEM surface roughness")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (a, b, la, lb) in zip(axes, panels):
        for bt, g in M.groupby("batch"):
            ax.scatter(g[a], g[b], c=col[bt], s=22, label=bt, marker="x" if bt == "Batch_heldout" else "o")
        s = summ[f"{a}~{b}"]
        ax.set_title(f"Spearman {s['spearman']:+.2f} (p = {s['p']:.1e})", fontsize=9)
        ax.set_xlabel(la, fontsize=9)
        ax.set_ylabel(lb, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("13-second geometric swelling test vs FEM production run, 34 sites (sym orientation, s = 1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "geometric_vs_fem.png", dpi=130)


def main() -> int:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    C = consensus()
    C.to_csv(OUT / "heldout_consensus.csv", index=False)
    piv = C[C.call.str.startswith("Batch")].pivot_table(index="site", columns="method", values="call", aggfunc="first")
    print(piv.T.to_string())
    M, summ = geometric_vs_fem()
    M.to_csv(OUT / "geometric_vs_fem.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    plot(M, summ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
