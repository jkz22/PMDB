"""Leave-one-parent-out (LOPO) evaluation of fingerprint arms A0 / A1 on the 34 labelled sites, the same
protocol as the batch menu (pmdb/batch_menu.py: groups = pmdb.parents.parent_groups, 3 held-out sites enter as
labelled sites). Uses only committed FEM site features. Writes outputs/fem_fingerprint/lopo/{metrics.json,predictions.csv}.

    python scripts/fem_a1_lopo.py [--fem-site-curves outputs/fem/free_lateral/site_curves.csv]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("fem_fingerprint_eval", ROOT / "scripts" / "fem_fingerprint_eval.py")
ffe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ffe)

from pmdb import fingerprint as fp  # noqa: E402
from pmdb.batch_menu import BATCHES, labelled_sites  # noqa: E402
from pmdb.fem_features import subset  # noqa: E402
from pmdb.parents import parent_groups  # noqa: E402


def pooled(tr: dict, te: dict) -> dict:
    out = {}
    for k, v in tr.items():
        out[k] = np.concatenate([v, te[k]]) if k in ffe.ARRAY_KEYS else v
    return out


def lopo(arm: str, pool: dict, y: pd.Series, groups: np.ndarray, leo_cols: list[str]) -> pd.DataFrame:
    codes = np.array([BATCHES.index(b) for b in y])
    rows = []
    for g in np.unique(groups):
        te_m = groups == g
        Xtr, Xte, _, _ = ffe.frames(arm, subset(pool, ~te_m), subset(pool, te_m), codes[~te_m], leo_cols)
        pred = fp.predict(fp.fit(Xtr, y[~te_m]), Xte)
        pred.insert(0, "true", y[te_m].to_numpy())
        pred.insert(0, "parent_id", g)
        rows.append(pred)
    return pd.concat(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fem-site-curves", default="outputs/fem/free_lateral/site_curves.csv")
    ap.add_argument("--out", default="outputs/fem_fingerprint/lopo")
    args = ap.parse_args()
    tr, te, _, leo_cols = ffe.load_inputs(ROOT / args.fem_site_curves)
    pool = pooled(tr, te)
    keys = [tuple(r) for r in pool["index"]]
    lab = labelled_sites().set_index(["batch", "site"])["label"]
    pg = parent_groups().set_index(["batch", "site"])["parent_id"]
    y = pd.Series([lab[k] for k in keys], index=ffe._mi(pool["index"]), name="batch")
    groups = np.array([pg[k] for k in keys])
    res, preds = {}, {}
    for arm in ("A0", "A1"):
        p = lopo(arm, pool, y, groups, leo_cols)
        preds[arm] = p
        c = p["assigned"].to_numpy() == p["true"].to_numpy()
        rec = [float(c[p["true"].to_numpy() == b].mean()) for b in BATCHES]
        res[arm] = {"lopo_correct": int(c.sum()), "lopo_n": int(len(p)), "accuracy": float(c.mean()),
                    "balanced_accuracy": float(np.mean(rec))}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"protocol": "leave-one-parent-out, 34 labelled sites",
                                                  "fem_site_curves": args.fem_site_curves, **res}, indent=2))
    pd.concat(preds, names=["arm"]).reset_index().to_csv(out / "predictions.csv", index=False)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
