"""Score every held-out call on this branch against the released labels.

Labels were released by the organiser in the Modelling-session chat on 2026-10-04
(3e122cbj = Batch_2, fn0mhxef = Batch_1, xrv9xvzb = Batch_3). They are only used
here, after the fact, to score; nothing in `data_heldout/` is touched and no model
is refitted.

Writes outputs/decision/heldout_scored.csv and heldout_scored.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "decision"

TRUTH = {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}
SITES = list(TRUTH)
B = ("Batch_1", "Batch_2", "Batch_3")


def _loso_recall(true: pd.Series, pred: pd.Series) -> dict[str, float]:
    return {f"loso_recall_{b[-1]}": float((pred[true == b] == b).mean()) for b in B}


def collect() -> pd.DataFrame:
    rows: list[dict] = []

    def add(method: str, kind: str, calls: dict[str, str], loso: dict | None = None, src: str = ""):
        r = {"method": method, "kind": kind, "source": src}
        r.update({f"call_{s}": calls.get(s, "") for s in SITES})
        r["n_correct"] = sum(calls.get(s) == TRUTH[s] for s in SITES)
        if loso:
            r.update(loso)
        rows.append(r)

    # 1. consensus table (methods on main + this branch)
    C = pd.read_csv(ROOT / "outputs/crosswalk/heldout_consensus.csv")
    for m, g in C.groupby("method", sort=False):
        calls = dict(zip(g.site, g.call))
        if m.startswith("joint functional"):
            continue
        kind = "arrangement" if ("fingerprint" in m or "FEM" in m) else "scalar KPIs"
        add(m, kind, calls, src="outputs/crosswalk/heldout_consensus.csv")

    # 2. XGBoost families
    X = pd.read_csv(ROOT / "outputs/xgb/heldout_predictions.csv")
    L = pd.read_csv(ROOT / "outputs/xgb/loso_predictions.csv")
    for fam, g in X.groupby("family", sort=False):
        l = L[L.family == fam]
        kind = "scalar KPIs" if fam == "reliable_kpis" else "arrangement"
        add(f"XGBoost {fam}", kind, dict(zip(g.site, g.assigned)),
            {"loso_acc": float((l.true == l.assigned).mean()), **_loso_recall(l.true, l.assigned)},
            "outputs/xgb/heldout_predictions.csv")

    # 3. fingerprint LOSO recall for the NB row
    F = pd.read_csv(ROOT / "outputs/fingerprint/loo_predictions.csv")
    fp = next(r for r in rows if r["method"].startswith("fingerprint (16"))
    fp.update({"loso_acc": float((F.true == F.assigned).mean()), **_loso_recall(F.true, F.assigned)})

    # 4. tile vote
    T = pd.read_csv(ROOT / "outputs/tilevote/heldout.csv")
    V = pd.read_csv(ROOT / "outputs/tilevote/loso_votes.csv")
    for cfg, g in T.groupby("config", sort=False):
        v = V[V.config == cfg]
        tv = pd.Series(B)[v.true].reset_index(drop=True); cv = pd.Series(B)[v.call].reset_index(drop=True)
        add(f"tile vote {cfg}", "tiles (scalar + per-tile depth)", dict(zip(g.site, g.call)),
            {"loso_acc": float((tv == cv).mean()), **_loso_recall(tv, cv)}, "outputs/tilevote/heldout.csv")

    # 5. nearest batch mean
    M = pd.read_csv(ROOT / "outputs/meanpred/heldout.csv")
    ML = pd.read_csv(ROOT / "outputs/meanpred/loso.csv")
    for fam, g in M.groupby("family", sort=False):
        l = ML[ML.family == fam]
        kind = "grey level (acquisition)" if fam.startswith("grey") else ("arrangement" if fam == "fingerprint16" else "scalar KPIs")
        add(f"nearest mean {fam}", kind, dict(zip(g.site, g.call)),
            {"loso_acc": float((l.true == l.pred).mean()), **_loso_recall(l.true, l.pred)},
            "outputs/meanpred/heldout.csv")

    # 6. the shipped decision card
    D = pd.read_csv(ROOT / "outputs/decision/heldout_decisions.csv")
    add("decision card (consensus call)", "ensemble", dict(zip(D.site, D.call)), src="outputs/decision/heldout_decisions.csv")

    # 7. binary off/baseline readings (truth: 3e off, fn off, x baseline)
    off_truth = {"3e122cbj": "off", "fn0mhxef": "off", "xrv9xvzb": "baseline"}
    J = pd.read_csv(ROOT / "outputs/acceptance/heldout_joint.csv")
    for fam, g in J.groupby("family", sort=False):
        calls = {s: ("off" if p >= 95 else "baseline") for s, p in zip(g.site, g.pct_among_B3_loo)}
        rows.append({"method": f"one-class {fam} score (95th pct of Batch 3)", "kind": "binary off/baseline",
                     **{f"call_{s}": calls[s] for s in SITES},
                     "n_correct": sum(calls[s] == off_truth[s] for s in SITES),
                     "source": "outputs/acceptance/heldout_joint.csv"})
    dev = D.set_index("site")["n_kpis_off"]
    calls = {s: ("off" if dev[s] > 0 else "baseline") for s in SITES}
    rows.append({"method": "noise-aware KPI deviation (any KPI off)", "kind": "binary off/baseline",
                 **{f"call_{s}": calls[s] for s in SITES},
                 "n_correct": sum(calls[s] == off_truth[s] for s in SITES),
                 "source": "outputs/decision/heldout_decisions.csv"})
    rows.append({"method": "Modelling session binary off-detector (CNN, every route)", "kind": "binary off/baseline",
                 "call_3e122cbj": "off", "call_fn0mhxef": "off", "call_xrv9xvzb": "baseline", "n_correct": 3,
                 "source": "Modelling session chat, 2026-10-04 11:04 UTC"})

    out = pd.DataFrame(rows)
    out.insert(1, "truth", " / ".join(TRUTH[s] for s in SITES))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    S = collect()
    S.to_csv(OUT / "heldout_scored.csv", index=False)
    three = S[S.kind != "binary off/baseline"]
    n_methods = len(three)
    dist = three.n_correct.value_counts().reindex([0, 1, 2, 3], fill_value=0)
    exp = {k: n_methods * binom.pmf(k, 3, 1 / 3) for k in range(4)}
    lines = ["# Held-out calls scored against the released labels", "",
             "Truth (organiser, Modelling-session chat 2026-10-04): **3e122cbj = Batch 2, fn0mhxef = Batch 1, xrv9xvzb = Batch 3.**", "",
             "| method | kind | 3e122cbj | fn0mhxef | xrv9xvzb | correct | LOSO acc | LOSO recall B1/B2/B3 |", "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        def c(s):
            v = r[f"call_{s}"]
            t = TRUTH[s] if r.kind != "binary off/baseline" else {"3e122cbj": "off", "fn0mhxef": "off", "xrv9xvzb": "baseline"}[s]
            return f"**{v}** ✓" if v == t else v
        acc = "" if pd.isna(r.get("loso_acc", np.nan)) else f"{r.loso_acc:.2f}"
        rec = "" if pd.isna(r.get("loso_recall_1", np.nan)) else "/".join(f"{r[f'loso_recall_{k}']:.2f}" for k in "123")
        lines.append(f"| {r.method} | {r.kind} | {c('3e122cbj')} | {c('fn0mhxef')} | {c('xrv9xvzb')} | {r.n_correct}/3 | {acc} | {rec} |")
    lines += ["", f"Three-class methods: {n_methods}. Distribution of correct calls (observed vs expected if every method guessed uniformly):", "",
              "| correct | observed | expected at chance |", "|---|---|---|"]
    for k in range(4):
        lines.append(f"| {k}/3 | {int(dist[k])} | {exp[k]:.1f} |")
    lines += ["", f"Mean correct per three-class method: {three.n_correct.mean():.2f} (chance 1.00)."]
    (OUT / "heldout_scored.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
