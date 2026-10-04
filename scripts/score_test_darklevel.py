"""Apply the frozen dark-level label rule (docs/parents.md §3) to organiser test sites.

The rule is one feature with two cutpoints, fitted on the 34 labelled sites (31 + released truths):
    SE_type_D <= cut1 -> Batch_1,  cut1 < SE_type_D <= cut2 -> Batch_2,  else Batch_3.
Cuts are frozen here as midpoints between the adjacent labelled values around the hunt's cuts
(the hunt stores the upper value of the lower group, which puts one training site exactly on the cut).
Input: outputs/clean_test/summary.csv (written by the Modal test prep, see docs/patch_mil.md); with
--heldout the released held-out sites are scored instead as a dry run. Output: outputs/test/darklevel_predictions.csv.
This is a label reader (acquisition / pore-floor brightness), not a material model: see docs/parents.md §4.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
B = ["Batch_1", "Batch_2", "Batch_3"]
FEATURES = ["SE_type_D", "BSE_D"]


def labelled() -> pd.DataFrame:
    pg = pd.read_csv(O / "parent_groups.csv", dtype=str).set_index("site")
    truth = pd.read_csv(O / "heldout_labels.csv", dtype=str).set_index("site")["batch"]
    pg.loc[truth.index, "batch"] = truth
    s = pd.concat([pd.read_csv(O / "clean/summary.csv", dtype={"site": str}),
                   pd.read_csv(O / "clean_heldout/summary.csv", dtype={"site": str})]).set_index("site")
    return pg.join(s[FEATURES])


def midpoint_cuts(v: np.ndarray, cut1: float, cut2: float) -> tuple[float, float]:
    u = np.unique(v)
    def mid(c):
        i = np.searchsorted(u, c, side="right")
        return float((u[i - 1] + u[i]) / 2) if 0 < i < len(u) else float(c)
    return mid(cut1), mid(cut2)


def frozen_rules(lab: pd.DataFrame) -> dict[str, tuple[float, float]]:
    hunt = pd.read_csv(O / "parents/feature_hunt_raw.csv")
    rules = {}
    for f in FEATURES:
        row = hunt[(hunt.feature == f) & (hunt.order == "B1<B2<B3")].iloc[0]
        rules[f] = midpoint_cuts(lab[f].to_numpy(float), row.cut1, row.cut2)
    return rules


def apply(v: pd.Series, cuts: tuple[float, float]) -> pd.Series:
    return pd.Series(np.where(v <= cuts[0], B[0], np.where(v <= cuts[1], B[1], B[2])), index=v.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", action="store_true", help="dry run on the released held-out sites")
    args = ap.parse_args()
    lab = labelled()
    rules = frozen_rules(lab)
    for f, (c1, c2) in rules.items():
        acc = (apply(lab[f], (c1, c2)) == lab.batch).mean()
        print(f"{f}: Batch_1 <= {c1:.2f} < Batch_2 <= {c2:.2f} < Batch_3   (fit on 34 labelled: {acc:.3f})")
    if args.heldout:
        src = O / "clean_heldout/summary.csv"
    else:
        src = O / "clean_test/summary.csv"
        if not src.exists():
            raise SystemExit(f"{src} not found: run the test-day prep first (docs/patch_mil.md), or use --heldout")
    s = pd.read_csv(src, dtype={"site": str}).set_index("site")
    out = pd.DataFrame(index=s.index)
    for f, cuts in rules.items():
        out[f] = s[f].round(2)
        out[f"call_{f}"] = apply(s[f], cuts)
    out["call"] = out["call_SE_type_D"]
    out["agree"] = out["call_SE_type_D"] == out["call_BSE_D"]
    if args.heldout:
        out["truth"] = lab.loc[out.index, "batch"]
    (O / "test").mkdir(exist_ok=True)
    dst = O / "test" / ("darklevel_predictions_heldout.csv" if args.heldout else "darklevel_predictions.csv")
    out.to_csv(dst)
    print(out.to_string())
    print(f"-> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
