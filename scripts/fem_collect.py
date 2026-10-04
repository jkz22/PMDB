"""Collector for the FEM runs (Steps 7-9).

``python scripts/fem_collect.py --g2 DIR100 DIR50`` compares the window results of a 100 nm and a 50 nm run.
``python scripts/fem_collect.py --results outputs/modal/fem/results`` writes outputs/fem/{site_curves,tile_curves,
validation}.csv, run_log.json and the GIF copies (§2.1 of the plan).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pmdb.fem.config import load_params  # noqa: E402
from pmdb.fem.features import META_COLUMNS, METRIC_NAMES, fem_tile_slices, swelling_gate, symmetrise  # noqa: E402

FEM_DIR = ROOT / "outputs" / "modal" / "fem"
OUT = ROOT / "outputs" / "fem"
OUT_G2 = FEM_DIR / "g2.csv"
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
    if set(a) != set(b):
        for name, miss in (("100nm", sorted(set(b) - set(a))), ("50nm", sorted(set(a) - set(b)))):
            if miss:
                print(f"missing in {name}: {', '.join('/'.join(k) for k in miss)}")
        return 1
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


def _manifest() -> list[dict]:
    out = []
    for man in (ROOT / "cache" / "half" / "manifest.csv", ROOT / "cache_heldout" / "half" / "manifest.csv",
                ROOT / "cache_test" / "half" / "manifest.csv"):
        if not man.exists():  # cache_test only exists on test day
            continue
        with open(man, newline="") as fh:
            out += [{"batch": r["batch"], "site": r["site"], "width": int(r["width"])} for r in csv.DictReader(fh)]
    return sorted(out, key=lambda r: (r["batch"], r["site"]))


def _ledger() -> tuple[float, dict]:
    path = FEM_DIR / "ledger.csv"
    if not path.exists():
        return 0.0, {}
    df = pd.read_csv(path)
    return float(df["cost_usd"].sum()), {k: float(v) for k, v in df.groupby("mode")["cost_usd"].sum().items()}


def _pick(results_dir: Path, batch: str, site: str, orientation: str):
    """Best error-free result among tags full / full_rerun: largest failed_at_s (NaN = 1.0), ties to full."""
    best = None
    for tag in ("full", "full_rerun"):
        rp = results_dir / tag / orientation / f"{batch}__{site}.json"
        if not rp.exists():
            continue
        d = json.loads(rp.read_text())
        if d["meta"].get("error"):
            continue
        fa = d["meta"].get("failed_at_s")
        fa = 1.0 if fa is None or not np.isfinite(fa) else float(fa)
        if best is None or fa > best[0]:
            best = (fa, tag, d)
    return best


def _missing_rows(man: dict, orientation: str, p: dict, tiles: bool) -> list[dict]:
    px_um = p["mesh"]["res_nm"] / 1000.0
    width = man["width"] // 2
    sl = fem_tile_slices(width, px_um, p["features"]["n_tiles"], p["features"]["edge_um"])
    rows = []
    nf = int(p["soc"]["frames"])
    for i in range(nf):
        head = {"batch": man["batch"], "site": man["site"], "heldout": man["batch"] == "Batch_heldout",
                "orientation": orientation, "frame": i, "s": i / (nf - 1), "converged": False,
                "failed_at_s": 0.0, "first_pore_closure_s": float("nan")}
        nan = {k: float("nan") for k in METRIC_NAMES}
        if tiles:
            for t, s in enumerate(sl):
                rows.append({**head, "tile": t, "tile_x0_um": s.start * px_um, "tile_x1_um": s.stop * px_um, **nan})
        else:
            rows.append({**head, **nan})
    return rows


def _order(df: pd.DataFrame, tiles: bool) -> pd.DataFrame:
    cols = ["batch", "site", "heldout", "orientation"] + (["tile", "tile_x0_um", "tile_x1_um"] if tiles else []) + [
        "frame", "s", "converged", "failed_at_s", "first_pore_closure_s"] + list(METRIC_NAMES)
    df = df[cols].copy()
    df["_o"] = df["orientation"].map({"bottom": 0, "top": 1, "sym": 2})
    df = df.sort_values(["batch", "site", "_o"] + (["tile"] if tiles else []) + ["frame"]).drop(columns="_o")
    return df.reset_index(drop=True)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def collect(results_dir: Path) -> int:
    p = load_params()
    budget_path = FEM_DIR / "budget.json"
    budget = (json.loads(budget_path.read_text()) if budget_path.exists()
              else {"drop_top": False, "drop_stage2": False, "log": []})
    orientations = ["bottom"] if budget.get("drop_top") else ["bottom", "top"]
    site_rows, tile_rows, case_meta, missing, rerun, gif_src = [], [], [], [], [], []
    for man in _manifest():
        for o in orientations:
            pick = _pick(results_dir, man["batch"], man["site"], o)
            if pick is None:
                missing.append(f"{man['batch']}/{man['site']}/{o}")
                site_rows += _missing_rows(man, o, p, False)
                tile_rows += _missing_rows(man, o, p, True)
                continue
            _, tag, d = pick
            m = {k: v for k, v in d["meta"].items() if k != "substeps"}
            m["tag"] = tag
            case_meta.append(m)
            sub = FEM_DIR / "substeps" / tag / o
            sub.mkdir(parents=True, exist_ok=True)
            (sub / f"{man['batch']}__{man['site']}.json").write_text(json.dumps(d["meta"].get("substeps", [])))
            if tag == "full_rerun":
                rerun.append(f"{man['batch']}/{man['site']}/{o}")
            site_rows += d["site_rows"]
            tile_rows += d["tile_rows"]
            if o == "bottom":
                g = FEM_DIR / "gifs" / tag / f"{man['batch']}__{man['site']}.gif"
                if g.exists():
                    gif_src.append(g)
    keep = set(META_COLUMNS) | set(METRIC_NAMES)
    outs = []
    for rows, keys, tiles in ((site_rows, ["batch", "site", "frame"], False),
                              (tile_rows, ["batch", "site", "tile", "frame"], True)):
        df = pd.DataFrame(rows)
        df = df[[c for c in df.columns if c in keep]]
        if "top" in orientations:
            df = symmetrise(df, keys)
        else:
            sym = df[df["orientation"] == "bottom"].copy()
            sym["orientation"] = "sym"
            df = pd.concat([df, sym], ignore_index=True)
        outs.append(_order(df, tiles))
    sdf, tdf = outs
    OUT.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(OUT / "site_curves.csv", index=False, float_format="%.6g")
    tdf.to_csv(OUT / "tile_curves.csv", index=False, float_format="%.6g")

    vrows = []
    for (b, s), g in sdf.groupby(["batch", "site"], sort=True):
        def at(o, g=g):
            r = g[(g["orientation"] == o) & (g["frame"] == 10)]
            return r.iloc[0] if len(r) else None
        sym, bot, top = at("sym"), at("bottom"), at("top")
        gate = swelling_gate(float(sym["swelling"]), p)
        vrows.append({"batch": b, "site": s, "heldout": b == "Batch_heldout",
                      "swelling_bottom": float(bot["swelling"]),
                      "swelling_top": float(top["swelling"]) if top is not None else float("nan"),
                      "swelling_sym": gate["swelling_sym"], "gate_ok": gate["gate_ok"],
                      "lit_band_ok": gate["lit_band_ok"],
                      "sxx_ratio_vt2": float(sym["sxx_mean_MPa"]) / -10.0,
                      "porosity_change": float(sym["porosity_change"]),
                      "porosity_rel_change": float(sym["porosity_rel_change"]),
                      "J_si_mean": float(sym["J_si_mean"]), "si_yield_frac": float(sym["si_yield_frac"]),
                      "failed_at_s": float(sym["failed_at_s"]),
                      "first_pore_closure_s": float(sym["first_pore_closure_s"])})
    vdf = pd.DataFrame(vrows)
    vdf.to_csv(OUT / "validation.csv", index=False, float_format="%.6g")

    gdir = OUT / "gifs"
    gdir.mkdir(parents=True, exist_ok=True)
    for g in gif_src:
        shutil.copy2(g, gdir / g.name)
    if OUT_G2.exists():
        shutil.copy2(OUT_G2, OUT / "g2.csv")

    total, per_mode = _ledger()
    full_cases = [m for m in case_meta if m["tag"] == "full"]
    prod = {}
    if full_cases and (FEM_DIR / "ledger.csv").exists():
        led = pd.read_csv(FEM_DIR / "ledger.csv")
        led = led[(led["tag"] == "full") & (led["mode"] == "full") & (led["attempt"] == 1)]
        if len(led):
            prod = {k: float(led.iloc[0][k]) if k == "cpu" else int(led.iloc[0][k])
                    for k in ("cpu", "memory_mb", "timeout_s")}
    probes = sorted((FEM_DIR / "probe").glob("*.json")) if (FEM_DIR / "probe").exists() else []
    keys = ("batch", "site", "orientation", "tag", "n_cells", "wall_s", "cost_usd", "peak_rss_mb", "failed_at_s",
            "first_pore_closure_s", "n_substeps", "newton_its_total", "gif_error", "cpu", "memory_mb", "timeout_s",
            "H", "W")
    log = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "git_commit": _git_commit(),
           "command": "python scripts/fem_collect.py --results " + str(results_dir),
           "versions": case_meta[0].get("versions") if case_meta else None,
           "params": p, "params_hash": case_meta[0].get("params_hash") if case_meta else None,
           "image": "ghcr.io/fenics/dolfinx/dolfinx:v0.10.0",
           "orientations_run": orientations, "missing_cases": missing, "rerun_cases": rerun, "budget": budget,
           "remediation": p.get("remediation", []),
           "probe": json.loads(probes[-1].read_text()) if probes else None,
           "ledger_total_usd": total, "ledger_per_mode_usd": per_mode,
           "cost_model": {"cpu_rate": 0.0472, "mem_rate": 0.0080, "overhead_s": 120, "cap_usd": 180.0,
                          "exponent": 1.5},
           "production": prod, "cases": [{k: m.get(k) for k in keys} for m in case_meta]}
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2, default=str))

    med = float(np.nanmedian(vdf["swelling_sym"].astype(float)))
    gm = swelling_gate(med, p)
    g5 = int((vdf["failed_at_s"] < 1.0).sum())
    out_win = vdf[~vdf["gate_ok"]][["batch", "site", "swelling_sym"]].to_dict("records")
    print(f"rows: site_curves {len(sdf)} x {sdf.shape[1]}, tile_curves {len(tdf)} x {tdf.shape[1]}, "
          f"validation {len(vdf)}; GIFs {len(gif_src)}")
    print(f"median swelling_sym(s=1) = {med:.4f} gate_ok(G4)={gm['gate_ok']} lit_band_ok={gm['lit_band_ok']}")
    print(f"sites outside stop window: {out_win}")
    print(f"G5 (failed_at_s<1 sites) = {g5} (max {p['gates']['max_failed_sites']}); missing cases: {missing}")
    print(f"ledger total ${total:.2f}")
    return 0


def collect_edge(results_dir: Path, out_dir: Path) -> int:
    """Assemble outputs/fem/edge5/{site,tile}_curves.csv from the `refeature` per-case results (same schema)."""
    p = load_params()
    site_rows, tile_rows, missing = [], [], []
    for man in _manifest():
        for o in ("bottom", "top"):
            rp = results_dir / o / f"{man['batch']}__{man['site']}.json"
            d = json.loads(rp.read_text()) if rp.exists() else None
            if d is None or d["meta"].get("error"):
                missing.append(f"{man['batch']}/{man['site']}/{o}")
                site_rows += _missing_rows(man, o, p, False)
                tile_rows += _missing_rows(man, o, p, True)
            else:
                site_rows += d["site_rows"]
                tile_rows += d["tile_rows"]
    keep = set(META_COLUMNS) | set(METRIC_NAMES)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rows, keys, tiles, name in ((site_rows, ["batch", "site", "frame"], False, "site_curves.csv"),
                                    (tile_rows, ["batch", "site", "tile", "frame"], True, "tile_curves.csv")):
        df = pd.DataFrame(rows)
        df = symmetrise(df[[c for c in df.columns if c in keep]], keys)
        df = _order(df, tiles)
        df.to_csv(out_dir / name, index=False, float_format="%.6g")
        print(f"{name}: {len(df)} rows x {df.shape[1]}")
    print(f"missing cases: {missing}")
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", nargs=2, metavar=("RESULTS_DIR", "OUT_DIR"))
    ap.add_argument("--g2", nargs=2, metavar=("DIR100", "DIR50"))
    ap.add_argument("--results", metavar="DIR")
    args = ap.parse_args()
    if args.g2:
        return g2(Path(args.g2[0]), Path(args.g2[1]))
    if args.edge:
        return collect_edge(Path(args.edge[0]), Path(args.edge[1]))
    if args.results:
        return collect(Path(args.results))
    ap.error("give --g2 or --results")
    return 2


if __name__ == "__main__":
    sys.exit(main())
