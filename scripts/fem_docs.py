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
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pmdb.fem.config import load_params  # noqa: E402
from pmdb.fem.features import METRIC_NAMES  # noqa: E402
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


# ------------------------------------------------------------------ results.md (Step 10, P28)
OUT_FEM = REPO / "outputs" / "fem"
RESULTS_MD = REPO / "docs" / "fem" / "results.md"
TESTS_DIR = REPO / "outputs" / "modal" / "fem" / "tests"
LABELLED = ("Batch_1", "Batch_2", "Batch_3")
BATCH_COLOURS = {"Batch_1": "tab:blue", "Batch_2": "tab:orange", "Batch_3": "tab:green"}


def _fmt(x, nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{float(x):.{nd}g}"


def _newest_pytest_line() -> str:
    files = sorted(TESTS_DIR.glob("*/pytest.txt")) if TESTS_DIR.exists() else []
    for f in reversed(files):
        for line in reversed(f.read_text().splitlines()):
            if re.search(r"\d+ passed", line):
                return line.strip()
    return "no pytest summary found"


def _batch_medians(v: pd.DataFrame, col: str) -> str:
    parts = []
    for b in (*LABELLED, "Batch_heldout"):
        sel = v.loc[v["batch"] == b, col].astype(float)
        parts.append(f"{b} {_fmt(np.nanmedian(sel))}" if len(sel) else f"{b} n/a")
    return ", ".join(parts)


def _plot_figures(sc: pd.DataFrame, v: pd.DataFrame, gates: dict, fig_dir: Path, gif_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = gates["swelling_stop"]
    blo, bhi = gates["swelling_lit_band"]
    sym = sc[sc["orientation"] == "sym"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (b, _s), g in sym.groupby(["batch", "site"]):
        g = g.sort_values("frame")
        if b == "Batch_heldout":
            ax.plot(g["s"], g["swelling"], "--", color="grey", lw=1)
        else:
            ax.plot(g["s"], g["swelling"], color=BATCH_COLOURS.get(b, "k"), lw=0.9, alpha=0.8)
    ax.axhspan(lo, hi, color="0.9", zorder=0)
    ax.axhspan(blo, bhi, color="gold", alpha=0.25, zorder=0)
    for b, c in BATCH_COLOURS.items():
        ax.plot([], [], color=c, label=b)
    ax.plot([], [], "--", color="grey", label="held-out")
    ax.set(xlabel="SOC s", ylabel="site swelling (sym)",
           title="Swelling vs SOC (grey: stop window, gold: literature band)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "swelling_vs_soc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    batches = [*LABELLED, "Batch_heldout"]
    rng = np.random.default_rng(0)
    for i, b in enumerate(batches):
        y = v.loc[v["batch"] == b, "swelling_sym"].astype(float).to_numpy()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y, s=18, color=BATCH_COLOURS.get(b, "grey"))
    ax.axhspan(blo, bhi, color="gold", alpha=0.25, zorder=0)
    ax.set_xticks(range(len(batches)), batches, rotation=15)
    ax.set(ylabel="swelling at s = 1 (sym)", title="Swelling by batch (gold: literature band)")
    fig.tight_layout()
    fig.savefig(fig_dir / "swelling_by_batch.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(11, 8))
    for r, b in enumerate(LABELLED):
        vb = v[v["batch"] == b].sort_values("swelling_sym")
        site = vb.iloc[(len(vb) - 1) // 2]["site"]
        gif = Image.open(gif_dir / f"{b}__{site}.gif")
        for c, fr in enumerate((0, 5, 10)):
            gif.seek(fr)
            axs[r, c].imshow(gif.convert("RGB"))
            axs[r, c].axis("off")
            axs[r, c].set_title(f"{b}/{site}  s = {fr / 10:.1f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "example_frames.png", dpi=150)
    plt.close(fig)


def results_blocks(out_dir: Path = OUT_FEM, pytest_txt: Path | None = None, figures: bool = True) -> dict[str, str]:
    from scipy.stats import kruskal

    out_dir = Path(out_dir)
    p = load_params()
    gates = p["gates"]
    sc = pd.read_csv(out_dir / "site_curves.csv")
    v = pd.read_csv(out_dir / "validation.csv")
    log = json.loads((out_dir / "run_log.json").read_text())
    g2p = out_dir / "g2.csv"
    g2 = pd.read_csv(g2p) if g2p.exists() else None
    lo, hi = gates["swelling_stop"]
    blo, bhi = gates["swelling_lit_band"]
    med = float(np.nanmedian(v["swelling_sym"].astype(float)))
    g4_ok = lo <= med <= hi
    lit_ok = blo <= med <= bhi
    full = [c for c in log.get("cases", []) if c.get("tag") == "full"]
    n_failed = int((v["failed_at_s"] < 1.0).sum())

    flags = []
    if g2 is None:
        flags.append("G2 (50 nm resolution check) not run: dropped by budget")
    elif not bool(g2["pass"].all()):
        flags.append(f"G2 failed: {int((~g2['pass']).sum())} of {len(g2)} comparisons exceed their limit; "
                     "production stays at 100 nm")
    if not g4_ok:
        flags.append(f"G4: median swelling_sym(s=1) {_fmt(med)} outside the stop window [{lo}, {hi}]")
    if n_failed > gates["max_failed_sites"]:
        flags.append(f"G5: {n_failed} sites with failed_at_s < 1 (max {gates['max_failed_sites']})")
    if "top" not in log.get("orientations_run", ["bottom", "top"]):
        flags.append("orientation `top` dropped by budget; `sym` rows are copies of `bottom`")
    flags += [f"missing case (all metrics NaN): {m}" for m in log.get("missing_cases", [])]
    flags += [f"production rerun (tag full_rerun) used for {m}" for m in log.get("rerun_cases", [])]
    flags += [f"remediation applied: {r}" for r in log.get("remediation", [])]
    flags_txt = "\n".join(f"- {f}" for f in flags) if flags else "none"

    val = [f"- G1: {pytest_txt.read_text().strip() if pytest_txt else _newest_pytest_line()}"]
    if g2 is not None:
        piv = g2.groupby("metric")["rel_diff"].max()
        val.append("- G2 (100 vs 50 nm, max relative difference over s = 0.5 and 1.0): "
                   + ", ".join(f"{k} {_fmt(x)}" for k, x in piv.items())
                   + f"; verdict {'pass' if bool(g2['pass'].all()) else 'FAIL'}")
    val.append(f"- Median swelling_sym at s = 1 over {len(v)} sites: {_fmt(med)} (stop window [{lo}, {hi}] -> G4 "
               f"{'pass' if g4_ok else 'FAIL'}; literature band [{blo}, {bhi}] lit_band_ok = {lit_ok})")
    val.append("- Median swelling_sym by batch: " + _batch_medians(v, "swelling_sym"))
    for col, label in (("sxx_ratio_vt2", "sxx_mean/(-10) (VT2, MPa scale)"),
                       ("porosity_change", "porosity_change (VT4)"),
                       ("porosity_rel_change", "porosity_rel_change (VT4)"), ("J_si_mean", "J_si_mean (VT5)")):
        val.append(f"- Median {label} by batch: " + _batch_medians(v, col))
    out_w = v[~v["gate_ok"].astype(bool)]
    val.append("- Sites outside the stop window: " + (
        ", ".join(f"{r.batch}/{r.site} ({_fmt(r.swelling_sym)})" for r in out_w.itertuples()) if len(out_w)
        else "none"))
    val.append("- Absolute stresses are not physical under the linear small-strain model (see Limitations).")

    conv = []
    for o in log.get("orientations_run", ["bottom", "top"]):
        cs = [c for c in full if c["orientation"] == o]
        if not cs:
            conv.append(f"- {o}: no cases")
            continue
        fa = np.array([np.nan if c["failed_at_s"] is None else c["failed_at_s"] for c in cs], dtype=float)
        nfail = int(np.sum(np.isfinite(fa) & (fa < 1.0)))
        st = {k: np.array([c[k] for c in cs], dtype=float) for k in ("n_substeps", "newton_its_total", "wall_s")}
        conv.append(f"- {o}: {len(cs)} cases, {nfail} with failed_at_s < 1; " + "; ".join(
            f"{k} median {_fmt(np.median(a))} max {_fmt(np.max(a))}" for k, a in st.items()))
    conv.append(f"- G5: {n_failed} sites with failed_at_s < 1 in either orientation (max "
                f"{gates['max_failed_sites']}) -> {'pass' if n_failed <= gates['max_failed_sites'] else 'FAIL'}")
    failed = v[v["failed_at_s"] < 1.0]
    conv.append("- Failed sites: " + (", ".join(
        f"{r.batch}/{r.site} failed_at_s {_fmt(r.failed_at_s)} first_pore_closure_s {_fmt(r.first_pore_closure_s)}"
        for r in failed.itertuples()) if len(failed) else "none"))
    conv.append("- Production mechanics is linear (one solve per frame); `n_substeps` counts those solves.")

    cost = [f"- Ledger total: ${_fmt(log['ledger_total_usd'], 4)} of the ${log['cost_model']['cap_usd']:.0f} cap",
            "- Per-mode subtotals (USD): " + ", ".join(
                f"{k} {x:.2f}" for k, x in log.get("ledger_per_mode_usd", {}).items())]
    cc = np.array([c["cost_usd"] for c in full if c.get("cost_usd") is not None], dtype=float)
    if cc.size:
        cost.append(f"- Production per-case cost (USD): median {np.median(cc):.3f}, max {np.max(cc):.3f}")

    names = ("swelling_vs_soc", "swelling_by_batch", "example_frames")
    if figures:
        _plot_figures(sc, v, gates, out_dir / "figures", out_dir / "gifs")
    figs = [f"![{n}](../../outputs/fem/figures/{n}.png)" for n in names]
    gifs = [f"- [{r.batch}/{r.site}](../../outputs/fem/gifs/{r.batch}__{r.site}.gif)" for r in v.itertuples()]
    figtxt = "\n\n".join(figs) + "\n\nGIFs (bottom orientation):\n\n" + "\n".join(gifs)

    s1 = sc[(sc["orientation"] == "sym") & (sc["frame"] == 10) & sc["batch"].isin(LABELLED)]
    rows = []
    for m in METRIC_NAMES:
        groups = [s1.loc[s1["batch"] == b, m].astype(float).dropna().to_numpy() for b in LABELLED]
        if any(len(g_) < 3 for g_ in groups):
            continue
        allv = np.concatenate(groups)
        if np.all(allv == allv[0]):
            continue
        h, pv = kruskal(*groups)
        rows.append((m, [float(np.median(g_)) for g_ in groups], float(h), float(pv)))
    rows.sort(key=lambda r: -r[2])
    table = _md_table(["metric", "median B1", "median B2", "median B3", "H", "p"],
                      [[m, *[_fmt(x, 4) for x in meds], _fmt(h, 4), _fmt(pv, 3)] for m, meds, h, pv in rows[:8]])
    diff = ("Descriptive, unadjusted, 31 sites. Kruskal-Wallis across Batch_1/2/3 on the 81 site metrics at s = 1 "
            "(orientation sym), top 8 by H.\n\n" + table)
    return {"flags": flags_txt, "validation": "\n".join(val), "convergence": "\n".join(conv),
            "cost": "\n".join(cost), "figures": figtxt, "batch_diff": diff}


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
        fill_blocks(RESULTS_MD, results_blocks())
        print(f"filled {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
