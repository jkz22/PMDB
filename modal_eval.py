"""Modal runner for the D22 XGBoost-on-KPIs permutation tests (CPU only, no volumes).

    modal run modal_eval.py --tag main     # arms X1, X2, X3
    modal run modal_eval.py --tag edge5    # arms X2e, X3e (sensitivity)

Writes outputs/xgb_kpi/<tag>/permutation_null.json (null accuracies per arm, permutation k
seeded by default_rng([seed, k])) and ledgers every chunk call in outputs/modal/fem/ledger.csv.
scripts/xgb_kpi_eval.py then reloads that file.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
CPU, MEMORY_MB, TIMEOUT_S = 1.0, 1024, 3600
ARMS = {"main": ("X1", "X2", "X3"), "edge5": ("X2e", "X3e")}
FEM_TABLE = {"main": "outputs/fem/site_curves.csv", "edge5": "outputs/fem/edge5/site_curves.csv"}

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4", "pandas==2.1.4", "scipy==1.14.1", "scikit-learn==1.3.2", "xgboost==3.2.0",
                 # pmdb/__init__ imports pmdb.io, which needs these
                 "scikit-image==0.25.2", "imagecodecs==2025.3.30", "tifffile==2025.5.10")
    .add_local_python_source("pmdb")
)
app = modal.App("pmdb-eval", image=image)


@app.function(cpu=CPU, memory=MEMORY_MB, timeout=TIMEOUT_S)
def perm_chunk(arm, tr, codes, n_classes, seed, ks):
    import time as _t

    from pmdb.xgb_kpi import perm_chunk as run

    t0 = _t.time()
    vals = run(arm, tr, codes, n_classes, seed, ks)
    return {"arm": arm, "ks": [int(k) for k in ks], "acc": vals, "wall_s": _t.time() - t0}


@app.local_entrypoint()
def main(tag: str = "main", n_perm: int = 1000, n_chunks: int = 50, seed: int = 0):
    import numpy as np

    import modal_fem
    from pmdb.xgb_kpi import load_inputs

    if tag not in ARMS:
        raise SystemExit(f"unknown tag {tag}")
    arms = ARMS[tag]
    tr, _, y, _ = load_inputs(ROOT / FEM_TABLE[tag])
    batches = sorted(y.unique())
    codes = np.array([batches.index(b) for b in y])
    chunks = [c.tolist() for c in np.array_split(np.arange(n_perm), n_chunks)]
    calls = [(a, tr, codes, len(batches), seed, ks) for a in arms for ks in chunks]

    worst = len(calls) * modal_fem.cost_usd(TIMEOUT_S, CPU, MEMORY_MB)
    total = modal_fem.ledger_total() + worst + modal_fem.DEV_ALLOWANCE_USD
    print(f"{len(calls)} calls; ledger {modal_fem.ledger_total():.2f} + worst case {worst:.2f} "
          f"+ allowance -> {total:.2f} (cap {modal_fem.CAP_USD})")
    if total > modal_fem.CAP_USD:
        raise SystemExit("BUDGET REFUSAL")

    t0 = time.time()
    res = list(perm_chunk.starmap(calls))
    wall = time.time() - t0
    utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    modal_fem.append_ledger([
        {"utc": utc, "mode": "xgb_perm", "tag": tag, "batch": "-", "site": r["arm"],
         "orientation": f"k{r['ks'][0]}-{r['ks'][-1]}", "attempt": 1, "cpu": CPU, "memory_mb": MEMORY_MB,
         "timeout_s": TIMEOUT_S, "wall_s": round(r["wall_s"], 1),
         "cost_usd": round(modal_fem.cost_usd(r["wall_s"], CPU, MEMORY_MB), 4), "status": "ok"}
        for r in res])

    null = {a: [None] * n_perm for a in arms}
    for r in res:
        for k, v in zip(r["ks"], r["acc"]):
            null[r["arm"]][k] = v
    out = ROOT / "outputs/xgb_kpi" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "permutation_null.json").write_text(json.dumps({"seed": seed, "n_perm": n_perm, "null": null}))
    cost = sum(modal_fem.cost_usd(r["wall_s"], CPU, MEMORY_MB) for r in res)
    print(f"done: wall {wall:.0f}s, summed container time {sum(r['wall_s'] for r in res):.0f}s, "
          f"ledgered cost ${cost:.2f}, ledger total ${modal_fem.ledger_total():.2f}")
