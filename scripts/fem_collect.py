"""Collector for the FEM runs. Step 7 adds only ``--g2`` (resolution check, run in Step 9).

``python scripts/fem_collect.py --g2 DIR100 DIR50`` compares the window results of a 100 nm and a 50 nm run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_G2 = ROOT / "outputs" / "modal" / "fem" / "g2.csv"
FRAMES = {0.5: 5, 1.0: 10}
# (name, getter, limit)
LIMIT_MAIN, LIMIT_VM = 0.10, 0.20
METRICS = [("swelling", lambda r: r["swelling"], LIMIT_MAIN),
           ("sxx_mean_MPa", lambda r: r["sxx_mean_MPa"], LIMIT_MAIN),
           ("porosity_change", lambda r: r["porosity_change"], LIMIT_MAIN),
           ("J_si_mean_minus_1", lambda r: r["J_si_mean"] - 1.0, LIMIT_MAIN),
           ("vm_si_p50_MPa", lambda r: r["vm_si_p50_MPa"], LIMIT_VM),
           ("vm_si_p95_MPa", lambda r: r["vm_si_p95_MPa"], LIMIT_VM)]


def _read(d: Path) -> dict[tuple, dict]:
    out = {}
    for rp in sorted(Path(d).rglob("*.json")):
        j = json.loads(rp.read_text())
        m = j["meta"]
        if m.get("error"):
            continue
        out[(m["batch"], m["site"], m["orientation"])] = {r["frame"]: r for r in j["site_rows"]}
    return out


def g2(dir100: Path, dir50: Path) -> int:
    a, b = _read(dir100), _read(dir50)
    rows = []
    for key in sorted(set(a) & set(b)):
        for s, fr in FRAMES.items():
            for name, get, limit in METRICS:
                va, vb = float(get(a[key][fr])), float(get(b[key][fr]))
                rel = abs(va - vb) / max(abs(vb), 1e-12)
                rows.append({"batch": key[0], "site": key[1], "orientation": key[2], "s": s, "metric": name,
                             "value_100nm": va, "value_50nm": vb, "rel_diff": rel, "limit": limit,
                             "pass": bool(rel <= limit)})
    df = pd.DataFrame(rows)
    OUT_G2.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_G2, index=False, float_format="%.6g")
    print(df.to_string(index=False))
    if df.empty:
        print("no common cases found")
        return 1
    return 0 if bool(df["pass"].all()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2", nargs=2, metavar=("DIR100", "DIR50"))
    args = ap.parse_args()
    if args.g2:
        return g2(Path(args.g2[0]), Path(args.g2[1]))
    ap.error("nothing to do (Step 8 adds the full tables)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
