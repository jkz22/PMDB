"""Print the markdown tables of docs/kpis/consolidation.md from the screen run outputs.

Reads `kpi_screen.csv` of the baseline and combined screen runs plus the combined
`screen_config.json`, and prints four blocks (`### A` counts, `### B` new KPIs, `### C`
redundancy clusters with a new KPI, `### D` like-for-like pairs). Writes nothing.

    python scripts/consolidation_tables.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb.screen import apply_sentinels, merge_kpi_tables, read_table  # noqa: E402

LIKE_FOR_LIKE = (
    ("mat_porosity", "D04_porosity_mean",
     "theirs: pore px / all px, whole image; ours: mean pore fraction over 5 depth bands"),
    ("mat_graphite_d50_um", "D03_graphite_ecd_d50_um",
     "theirs: watershed instances, border-touching excluded; ours: graphite object ECD D50"),
    ("mat_bright_fraction", "K01_si_frac_adm",
     "theirs: opened brightest Otsu class / all px; ours: Si area / non-artefact area"),
)
GATE_FAMILIES = ("degeneracy", "artefact", "reliability", "robustness", "redundant_with")


def _read_screen(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for c in ("deciding_gate", "untested_gates", "cluster_rep"):
        df[c] = df[c].fillna("")
    return df


def _rho(v: float) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v:+.3f}"


def _icc(v: float) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v:.2f}"


def _dash(s: str) -> str:
    return s if s else "-"


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _decision(r: pd.Series) -> str:
    return "keep" if r["decision"] == "keep" else f"drop ({r['deciding_gate']})"


def _family_counts(df: pd.DataFrame) -> list[int]:
    fam = df.loc[df["decision"] != "keep", "deciding_gate"].str.split(":").str[0]
    return [int((fam == g).sum()) for g in GATE_FAMILIES]


def block_a(base: pd.DataFrame, comb: pd.DataFrame, new: list[str]) -> list[str]:
    out = [_row(["run", "KPIs", "kept", "dropped", *GATE_FAMILIES]),
           _row(["---"] * (4 + len(GATE_FAMILIES)))]
    for name, df in (("baseline", base), ("combined", comb)):
        kept = int((df["decision"] == "keep").sum())
        out.append(_row([name, len(df), kept, len(df) - kept, *_family_counts(df)]))
    nk = int(comb.loc[comb["kpi"].isin(new), "decision"].eq("keep").sum())
    out += ["", f"New KPIs kept: {nk}/{len(new)}.", ""]
    b, c = base.set_index("kpi"), comb.set_index("kpi")
    changed = [k for k in b.index if k in c.index
               and (b.loc[k, "decision"] != c.loc[k, "decision"]
                    or b.loc[k, "deciding_gate"] != c.loc[k, "deciding_gate"])]
    if not changed:
        out.append("_No existing KPI changed decision._")
    else:
        out += [_row(["existing KPI", "baseline", "combined"]), _row(["---"] * 3)]
        out += [_row([k, _decision(b.loc[k]), _decision(c.loc[k])]) for k in changed]
    return out


def block_b(comb: pd.DataFrame, new: list[str], existing: list[str], rho: pd.DataFrame) -> list[str]:
    out = [_row(["new KPI", "decision", "deciding gate", "untested gates",
                 "closest existing KPI", "rho", "cluster rep"]), _row(["---"] * 7)]
    c = comb.set_index("kpi")
    for k in new:
        s = rho.loc[k, existing]
        s = s[s.notna()]
        if len(s):
            best = sorted(s.index, key=lambda e: (-abs(s[e]), e))[0]
            near, r = best, float(s[best])
        else:
            near, r = "n/a", float("nan")
        row = c.loc[k]
        out.append(_row([k, row["decision"], _dash(row["deciding_gate"]), _dash(row["untested_gates"]),
                         near, _rho(r), _dash(row["cluster_rep"])]))
    return out


def block_c(comb: pd.DataFrame, new: list[str], rho: pd.DataFrame) -> list[str]:
    sub = comb[comb["cluster_id"].notna()]
    out = [_row(["cluster", "KPI", "decision", "ICC", "mean abs rho to other members", "rule"]),
           _row(["---"] * 6)]
    n_rows = 0
    for cid, g in sub.groupby("cluster_id", sort=True):
        members = list(g["kpi"])
        if len(members) < 2 or not set(members) & set(new):
            continue
        rep = g["cluster_rep"].iloc[0]
        has_icc = bool(g["icc"].map(lambda v: pd.notna(v) and math.isfinite(v)).any())
        rule = "highest ICC" if has_icc else "medoid (no ICC in cluster)"
        ordered = [rep] + sorted(m for m in members if m != rep)
        gi = g.set_index("kpi")
        for m in ordered:
            others = [o for o in members if o != m]
            mar = float(rho.loc[m, others].abs().mean())
            out.append(_row([int(cid), m, gi.loc[m, "decision"], _icc(gi.loc[m, "icc"]),
                             f"{mar:.3f}" if math.isfinite(mar) else "n/a",
                             rule if m == rep else "-"]))
            n_rows += 1
    return out if n_rows else ["_No redundancy cluster contains a new KPI._"]


def block_d(rho: pd.DataFrame, thr: float) -> list[str]:
    out = [_row(["new KPI", "existing KPI", "rho", "verdict", "definition difference"]), _row(["---"] * 5)]
    for n, e, diff in LIKE_FOR_LIKE:
        r = float(rho.loc[n, e])
        verdict = "agree" if math.isfinite(r) and abs(r) >= thr else "disagree: check segmentation"
        out.append(_row([n, e, _rho(r), verdict, diff]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combined-dir", default="outputs/kpis/screen_combined")
    ap.add_argument("--baseline-dir", default="outputs/kpis/screen")
    a = ap.parse_args(argv)

    def res(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else ROOT / p

    cdir, bdir = res(a.combined_dir), res(a.baseline_dir)
    comb, base = _read_screen(cdir / "kpi_screen.csv"), _read_screen(bdir / "kpi_screen.csv")
    cfg = json.loads((cdir / "screen_config.json").read_text())
    sources = list(cfg["inputs"]["kpis"])
    frames = [read_table(res(s)) for s in sources]
    site, kpis, _ = merge_kpi_tables(frames, sources, cfg["config"]["ignore_cols"])
    clean = apply_sentinels(site, kpis, [tuple(s) for s in cfg["config"]["sentinels"]])
    rho = clean[kpis].corr(method="spearman")
    existing = list(base["kpi"])
    new = [k for k in kpis if k not in set(existing)]

    for tag, lines in (
        ("A", block_a(base, comb, new)),
        ("B", block_b(comb, new, existing, rho)),
        ("C", block_c(comb, new, rho)),
        ("D", block_d(rho, cfg["config"]["redundancy_rho"])),
    ):
        print(f"### {tag}")
        print("\n".join(lines))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
