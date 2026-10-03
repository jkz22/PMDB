"""Fill the generated blocks of docs/fem/method.md (and, in a later step, results.md). P28, D19.

Usage: python scripts/fem_docs.py --method [--results]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pmdb.fem.config import load_params  # noqa: E402
from pmdb.fem.materials import graphite_strains, lithiation_state, soc_fractions  # noqa: E402

PENDING = "_pending: generated after the full run_"
LIT_MD = REPO / "docs" / "fem" / "literature-review.md"
METHOD_MD = REPO / "docs" / "fem" / "method.md"
RUN_LOG = REPO / "outputs" / "fem" / "run_log.json"
LIT_LINK = "[literature review, parameter table](literature-review.md#1-final-parameter-table)"

# lit-review Parameter cell -> fem.yaml keys (P28)
PARAM_MAP = {
    "Si volumetric eigenstretch": ["si.beta"],
    "Si utilisation at 100% overall SOC, u_max": ["soc.u_max"],
    "Graphite lithiation at 100% overall SOC, y_max": ["soc.y_max"],
    "Overall SOC → per-phase normalised fraction": ["soc.s_star", "soc.si_ratio", "soc.gr_ratio"],
    "Si E(u)": ["si.E_MPa"],
    "Si ν(u)": ["si.nu"],
    "Si yield (flag only, v1 is elastic)": ["si.yield_MPa", "si.x_per_u"],
    "Graphite c-axis strain ε_zz(y)": ["graphite.eps_c_points"],
    "Graphite a-axis strain ε_xx(y)": ["graphite.eps_a_over_c"],
    "Graphite E, ν (isotropic)": ["graphite.E_MPa", "graphite.nu"],
    "Binder + CBD (unassigned solid)": ["binder.E_MPa", "binder.nu"],
    "Pore / artefact": ["pore.E_rel_binder", "pore.nu", "pore.closure_J"],
    "Resolution": ["mesh.res_nm"],
}
PHYSICS_ROWS = ["Kinematics", "Elastic energy (all phases)", "Graphite orientation", "Out-of-plane",
                "Current-collector edge", "Lateral edges", "Separator-side edge"]


def fill_blocks(path: Path, blocks: dict[str, str]) -> None:
    path = Path(path)
    text = path.read_text()
    present = set(re.findall(r"<!-- AUTO:([A-Za-z0-9_]+) -->", text))
    for name in blocks:
        if f"<!-- AUTO:{name} -->" not in text or f"<!-- /AUTO:{name} -->" not in text:
            raise ValueError(f"marker pair for block {name!r} missing in {path}")
    for name in present:
        if name not in blocks:
            raise ValueError(f"marker {name!r} in {path} has no entry in blocks")
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- AUTO:{re.escape(name)} -->).*?(<!-- /AUTO:{re.escape(name)} -->)", re.S)
        text = pat.sub(lambda m: m.group(1) + "\n" + body + "\n" + m.group(2), text, count=1)
    path.write_text(text)


def lit_param_rows(lit_md: Path = LIT_MD) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    in_table = False
    for line in Path(lit_md).read_text().splitlines():
        if line.startswith("## 1. Final parameter table"):
            in_table = True
            continue
        if in_table and line.startswith("#"):
            break
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 5 or cells[0] in ("Parameter",) or set(cells[0]) <= {"-"}:
                continue
            rows[cells[0]] = {"default": cells[1], "source": cells[3], "evidence": cells[4]}
    return rows


def _get(p: dict, dotted: str):
    v = p
    for k in dotted.split("."):
        v = v[k]
    return v


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _params_block(p: dict, lit: dict) -> str:
    rows = []
    for name, keys in PARAM_MAP.items():
        info = lit[name]  # KeyError if the lit review lost the row
        val = ", ".join(f"{k} = {_get(p, k)}" for k in keys)
        rows.append([name, val, info["source"], info["evidence"]])
    return f"Sources: {LIT_LINK}\n\n" + _md_table(["Parameter", "Value used", "Source", "Evidence strength"], rows)


def _physics_block(lit: dict) -> str:
    rows = [[n, lit[n]["default"], lit[n]["source"], lit[n]["evidence"]] for n in PHYSICS_ROWS]
    return _md_table(["Assumption", "Model choice", "Source", "Evidence strength"], rows)


def _soc_block(p: dict) -> str:
    rows = []
    beta = p["si"]["beta"]
    for s in np.linspace(0.0, 1.0, int(p["soc"]["frames"])):
        f_si, f_gr = soc_fractions(float(s), p)
        u, y = lithiation_state(float(s), p)
        _, eps_c = graphite_strains(y, p)
        rows.append([f"{s:.1f}", f"{f_si:.3f}", f"{f_gr:.3f}", f"{u:.3f}", f"{y:.3f}",
                     f"{1 + beta * u:.3f}", f"{eps_c:.3f}"])
    return _md_table(["s", "f_Si", "f_Gr", "u", "y", "J_Si", "ε_c"], rows)


def _gates_block(p: dict) -> str:
    g = p["gates"]
    lines = [f"- Stop window for the site swelling at full SOC: {g['swelling_stop']}",
             f"- Literature band (reported only): {g['swelling_lit_band']}",
             f"- Maximum sites with a failed solve: {g['max_failed_sites']}",
             "- Solver settings: " + ", ".join(f"`{k}` = {v}" for k, v in p["solver"].items())]
    rem = p.get("remediation") or []
    lines.append("- Remediation applied: " + ("; ".join(rem) if rem else "none applied"))
    return "\n".join(lines)


def _cost_block(run_log: dict | None) -> str:
    if not run_log:
        return PENDING
    lines = []
    cm = run_log.get("cost_model", {})
    if cm:
        lines.append("- Cost model: " + ", ".join(f"`{k}` = {v}" for k, v in cm.items()))
    prod = run_log.get("production", {})
    if prod:
        lines.append("- Production settings: " + ", ".join(f"`{k}` = {v}" for k, v in prod.items()))
    cases = [c for c in run_log.get("cases", []) if c.get("wall_s") is not None]
    if cases:
        wall = np.array([c["wall_s"] for c in cases], dtype=float)
        cost = np.array([c.get("cost_usd", np.nan) for c in cases], dtype=float)
        lines.append(f"- Per-case wall time (s): median {np.nanmedian(wall):.0f}, max {np.nanmax(wall):.0f}")
        lines.append(f"- Per-case cost (USD): median {np.nanmedian(cost):.2f}, max {np.nanmax(cost):.2f}")
    tot = run_log.get("ledger_total_usd", run_log.get("ledger_total"))
    if tot is not None:
        lines.append(f"- Ledger total (USD): {float(tot):.2f} of cap {cm.get('cap_usd', 'n/a')}")
    return "\n".join(lines) if lines else PENDING


def method_blocks(p: dict, run_log: dict | None) -> dict[str, str]:
    lit = lit_param_rows()
    return {"params": _params_block(p, lit), "physics": _physics_block(lit), "soc": _soc_block(p),
            "gates": _gates_block(p), "cost": _cost_block(run_log)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", action="store_true")
    ap.add_argument("--results", action="store_true")
    args = ap.parse_args(argv)
    if not (args.method or args.results):
        ap.error("give --method and/or --results")
    if args.method:
        run_log = json.loads(RUN_LOG.read_text()) if RUN_LOG.exists() else None
        fill_blocks(METHOD_MD, method_blocks(load_params(), run_log))
        print(f"filled {METHOD_MD}")
    if args.results:
        print("--results is implemented in the results-doc step (Step 10)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
