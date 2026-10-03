"""FEM runner on Modal (D16, P20-P22, P26). Run: ``modal run modal_fem.py --mode probe|unit``.

All dolfinx code runs inside the pinned official dolfinx image; the local side only orchestrates,
keeps the cost ledger and enforces the budget cap.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import subprocess
import time
from pathlib import Path

import modal

DOLFINX_IMAGE = "ghcr.io/fenics/dolfinx/dolfinx:v0.10.0"
PY = "python"  # PATH python in the image: /dolfinx-env/bin/python
CPU_RATE = 0.0472  # USD per core-hour (Modal $0.0000131 per core-second)
MEM_RATE = 0.0080  # USD per GiB-hour (Modal $0.00000222 per GiB-second)
OVERHEAD_S = 120
DEV_ALLOWANCE_USD = 5.0
CAP_USD = 180.0
ROOT = Path(__file__).resolve().parent
FEM_DIR = ROOT / "outputs" / "modal" / "fem"
LEDGER = FEM_DIR / "ledger.csv"
BUDGET_JSON = FEM_DIR / "budget.json"
LEDGER_COLUMNS = ["utc", "mode", "tag", "batch", "site", "orientation", "attempt", "cpu", "memory_mb",
                  "timeout_s", "wall_s", "cost_usd", "status"]
PROBE_SITES = [("Batch_3", "vc2whyaq"), ("Batch_3", "71vgq3fw"), ("Batch_3", "kbdh4tri"),
               ("Batch_3", "tuy3zymq"), ("Batch_3", "x7u69zsw")]

image = (
    modal.Image.from_registry(DOLFINX_IMAGE)
    .run_commands(
        "python -c \"import numpy, sys; v = numpy.__version__; "
        "sys.exit(0 if v == '2.2.6' else 'image numpy ' + v + ' != 2.2.6')\"",
        "pip install uv",
        "uv pip install --python /dolfinx-env/bin/python numpy==2.2.6 scipy==1.14.1 scikit-image==0.25.2 pandas==2.2.3 "
        "matplotlib==3.10.3 pillow==11.3.0 pyyaml==6.0.2 tifffile==2025.5.10 imagecodecs==2025.3.30 "
        "pytest==8.3.5",
    )
    .env({"PMDB_CACHE": "/data",
          "PYTHONPATH": "/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib",
          "PETSC_ARCH": "linux-gnu-real64-32"})
    .add_local_file("pytest.ini", "/root/pytest.ini")
    .add_local_file("configs/fem/fem.yaml", "/root/configs/fem/fem.yaml")
    .add_local_file("tests/test_fem_unit.py", "/root/tests/test_fem_unit.py")
    .add_local_file("tests/test_fem_solver.py", "/root/tests/test_fem_solver.py")
    .add_local_python_source("pmdb")
)
data_vol = modal.Volume.from_name("pmdb-data").read_only()
out_vol = modal.Volume.from_name("pmdb-fem-out", create_if_missing=True)
app = modal.App("pmdb-fem", image=image)


# ------------------------------------------------------------------ local helpers
def _rate(cpu: float, memory_mb: float) -> float:
    return cpu * CPU_RATE + memory_mb / 1024.0 * MEM_RATE


def cost_usd(wall_s: float, cpu: float, memory_mb: float) -> float:
    return (wall_s + OVERHEAD_S) / 3600.0 * _rate(cpu, memory_mb)


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    with open(LEDGER, newline="") as fh:
        return float(sum(float(r["cost_usd"] or 0.0) for r in csv.DictReader(fh)))


def append_ledger(rows: list[dict]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LEDGER_COLUMNS})


def _load_budget() -> dict:
    if BUDGET_JSON.exists():
        return json.loads(BUDGET_JSON.read_text())
    return {"drop_top": False, "drop_stage2": False, "log": []}


def _worst(n_calls: int, cpu: float, memory_mb: int, timeout_s: int) -> float:
    return n_calls * (timeout_s + OVERHEAD_S) / 3600.0 * _rate(cpu, memory_mb)


def budget_check(mode: str, tag: str, items: list, cpu: float, memory_mb: int, timeout_s: int) -> str:
    """P22 rules (a) and (d). Returns "go" or "skip"; raises SystemExit(3) on a cap refusal.

    Rules (b) and (c) (tags ``full`` / ``full_rerun``) belong to the production modes (Step 8).
    """
    budget = _load_budget()
    if mode in ("probe", "unit", "bench", "window"):
        if mode == "window" and tag.startswith("stage2") and budget["drop_stage2"]:
            print("stage2 dropped by budget")
            return "skip"
        total = ledger_total() + _worst(len(items), cpu, memory_mb, timeout_s) + DEV_ALLOWANCE_USD
        if total > CAP_USD:
            print(f"BUDGET REFUSAL: ledger + worst case + allowance = {total:.2f} > {CAP_USD}")
            raise SystemExit(3)
        return "go"
    raise NotImplementedError(f"budget rule for mode {mode!r} is added with the production modes")


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ------------------------------------------------------------------ remote functions
@app.function(cpu=1.0, memory=4096, timeout=1800, retries=0, volumes={"/data": data_vol})
def probe() -> dict:
    import inspect
    import sys

    t0 = time.time()
    import dolfinx
    import dolfinx.fem.petsc
    import numpy
    import petsc4py
    import scipy
    import skimage
    from petsc4py import PETSc

    from pmdb.fem.geometry import labels_from_masks
    from pmdb.io import load_site
    from pmdb.segment import segment

    labels = {}
    for b, s in PROBE_SITES:
        lab = labels_from_masks(segment(load_site(b, s, resolution="half", normalise="none", cache_root="/data")))
        labels[f"{b}/{s}"] = {"shape": list(lab.shape), "data": lab.tobytes()}
    return {
        "python": sys.version, "dolfinx": dolfinx.__version__, "petsc4py": petsc4py.__version__,
        "numpy": numpy.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__,
        "mumps": bool(PETSc.Sys.hasExternalPackage("mumps")),
        "nonlinear_problem_signature": str(inspect.signature(dolfinx.fem.petsc.NonlinearProblem.__init__)),
        "nonlinear_problem_solve_doc": dolfinx.fem.petsc.NonlinearProblem.solve.__doc__,
        "labels": labels, "wall_s": time.time() - t0,
    }


@app.function(cpu=2.0, memory=4096, timeout=1800, retries=0)
def run_unit_tests(k: str = "") -> dict:
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    cmd = [PY, "-m", "pytest", "-q", "-rP", "tests/test_fem_unit.py", "tests/test_fem_solver.py"]
    if k:
        cmd += ["-k", k]
    pr = subprocess.run(cmd, cwd="/root", env=env, capture_output=True, text=True)
    return {"returncode": pr.returncode, "stdout": pr.stdout[-200000:], "stderr": pr.stderr[-200000:],
            "wall_s": time.time() - t0}


def _case_path(root: str, tag: str, orientation: str, batch: str, site: str) -> str:
    return f"{root}/{tag}/{orientation}/{batch}__{site}.json"


@app.function(cpu=4.0, memory=16384, timeout=21600, retries=0, volumes={"/data": data_vol, "/out": out_vol})
def run_case_remote(case: dict, tag: str, threads: int, crop_um: float = 0.0, res_nm: float = 0.0,
                    solver_overrides: dict | None = None) -> dict:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    batch, site, orientation = case["batch"], case["site"], case["orientation"]
    t0 = time.time()
    try:
        from pmdb.fem.run import run_case

        cache_root = "/data/heldout" if batch == "Batch_heldout" else "/data"
        res = run_case(batch, site, orientation, cache_root=cache_root, res_nm=res_nm or None,
                       crop_um=crop_um or None,
                       fields_path=Path(f"/out/fields/{tag}/{orientation}/{batch}__{site}.npz"),
                       render_gif=orientation == "bottom", solver_overrides=solver_overrides or None)
        res["meta"].update(tag=tag, threads=threads)
        out = {k: v for k, v in res.items() if k != "gif"}
        p = Path(_case_path("/out/results", tag, orientation, batch, site))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out))
        if res["gif"] is not None:
            g = Path(f"/out/gifs/{tag}/{batch}__{site}.gif")
            g.parent.mkdir(parents=True, exist_ok=True)
            g.write_bytes(res["gif"])
        out_vol.commit()
        return res
    except Exception as e:  # noqa: BLE001
        return {"meta": {"batch": batch, "site": site, "orientation": orientation,
                         "heldout": batch == "Batch_heldout", "wall_s": time.time() - t0,
                         "error": f"{type(e).__name__}: {e}"},
                "site_rows": [], "tile_rows": [], "gif": None}


@app.function(cpu=0.25, memory=1024, timeout=86400, retries=0, volumes={"/out": out_vol})
def run_cases(cases: list[dict], tag: str, cpu: float, memory: int, timeout: int, threads: int,
              crop_um: float = 0.0, res_nm: float = 0.0, solver_overrides: dict | None = None) -> dict:
    t_orch = time.time()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def run_pass(cs, cpu_, mem_, to_, attempt):
        fn = run_case_remote.with_options(cpu=cpu_, memory=mem_, timeout=to_)
        outs = list(fn.starmap([(c, tag, threads, crop_um, res_nm, solver_overrides) for c in cs],
                               return_exceptions=True))
        recs, failed = [], []
        for c, o in zip(cs, outs):
            rec = {"batch": c["batch"], "site": c["site"], "orientation": c["orientation"], "attempt": attempt,
                   "cpu": cpu_, "memory_mb": mem_, "timeout_s": to_}
            if isinstance(o, BaseException):
                rec.update(wall_s=float(to_), status="lost", error=f"{type(o).__name__}: {o}")
            elif o["meta"].get("error"):
                rec.update(wall_s=float(o["meta"].get("wall_s", to_)), status="error", error=o["meta"]["error"])
            else:
                rec.update(wall_s=float(o["meta"]["wall_s"]), status="ok")
            recs.append(rec)
            if rec["status"] != "ok":
                failed.append((c, rec))
        return recs, failed

    def write_status(name: str, payload: dict) -> None:
        d = Path(f"/out/status/{tag}")
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(json.dumps(payload))
        out_vol.commit()

    recs1, failed = run_pass(cases, cpu, memory, timeout, 1)
    write_status(f"{stamp}_a1.json", {"records": recs1})
    recs2 = []
    if failed:
        mem2 = int(math.ceil(1.5 * memory / 1024.0) * 1024)
        to2 = int(min(86400, math.ceil(1.5 * timeout)))
        recs2, failed2 = run_pass([c for c, _ in failed], cpu, mem2, to2, 2)
        for c, rec in failed2:
            stub = {"meta": {"batch": c["batch"], "site": c["site"], "orientation": c["orientation"],
                             "heldout": c["batch"] == "Batch_heldout", "error": rec["error"],
                             "status": rec["status"]}, "site_rows": [], "tile_rows": []}
            p = Path(_case_path("/out/results", tag, c["orientation"], c["batch"], c["site"]))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(stub))
    write_status(f"{stamp}_a2.json", {"records": recs2, "orchestrator": {
        "wall_s": time.time() - t_orch, "cpu": 0.25, "memory_mb": 1024, "timeout_s": 86400}})
    return {"first": recs1, "retry": recs2}


@app.function(cpu=4.0, memory=16384, timeout=3000, retries=0, volumes={"/data": data_vol})
def diag(runs: list[dict]) -> list[dict]:
    """Debug probe on the Stage-3 window: each run = {frames, orientation, overrides}."""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    import numpy as np

    from pmdb.fem.config import load_params
    from pmdb.fem.geometry import central_cols, coarsen_labels, labels_from_masks
    from pmdb.fem.materials import phase_properties
    from pmdb.fem.solver import BCSpec, simulate
    from pmdb.io import load_site
    from pmdb.segment import segment

    p = load_params()
    s = load_site("Batch_3", "vc2whyaq", resolution="half", normalise="none", cache_root="/data")
    lab = labels_from_masks(segment(s))
    lab = coarsen_labels(lab[:, central_cols(lab.shape[1], 40.0, s.nm_per_px)], 2)
    out = []
    for run in runs:
        pp = load_params()
        pp["solver"].update(run.get("overrides") or {})
        if run.get("pore_E"):
            pp["pore"]["E_rel_binder"] = run["pore_E"]
        fr = np.array(run["frames"], dtype=float)
        r = simulate(lab, 0.1, lambda x: phase_properties(x, pp), BCSpec(run.get("orientation", "bottom")),
                     pp["solver"], fr, extra_targets=())
        info = {"run": run, "failed_at_s": r.failed_at_s, "wall_s": r.wall_s,
                "substeps": [(x["s"], x["ds"], x["its"], x["reason"], x["fnorm"], x["error"], round(x["wall_s"], 1))
                             for x in r.substeps]}
        for i, f in enumerate(fr):
            if r.converged[i]:
                J = r.fields["J"][i]
                info[f"J_{f}"] = {n: [float(np.nanmin(J[lab == k])), float(np.nanmean(J[lab == k]))]
                                  for n, k in (("binder", 0), ("si", 1), ("gr", 2), ("pore", 3))
                                  if (lab == k).any()}
        out.append(info)
    return out


# ------------------------------------------------------------------ local sync / summarize
def _case_mode(tag: str) -> str:
    return "bench" if tag.startswith("bench") else ("full" if tag.startswith("full") else "window")


def _pull(vol, remote_dir: str, local_dir: Path, suffix: str) -> list[Path]:
    got = []
    try:
        entries = vol.listdir(remote_dir, recursive=True)
    except Exception:  # noqa: BLE001  (directory absent)
        return got
    base = remote_dir.strip("/")
    for e in entries:
        path = e.path.strip("/")
        if not path.endswith(suffix):
            continue
        dest = local_dir / path[len(base) + 1:]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"".join(vol.read_file(path)))
        got.append(dest)
    return got


def sync(tag: str) -> list[dict]:
    vol = modal.Volume.from_name("pmdb-fem-out")
    res = _pull(vol, f"/results/{tag}", FEM_DIR / "results" / tag, ".json")
    stat = _pull(vol, f"/status/{tag}", FEM_DIR / "status" / tag, ".json")
    _pull(vol, f"/gifs/{tag}", FEM_DIR / "gifs" / tag, ".gif")
    mode = _case_mode(tag)
    seen = set()
    if LEDGER.exists():
        with open(LEDGER, newline="") as fh:
            seen = {(r["mode"], r["tag"], r["utc"], r["batch"], r["site"], r["orientation"], r["attempt"])
                    for r in csv.DictReader(fh)}
    rows, ok_cost = [], {}
    for sf in sorted(stat):
        d = json.loads(sf.read_text())
        for rec in d.get("records", []):
            c = round(cost_usd(rec["wall_s"], rec["cpu"], rec["memory_mb"]), 4)
            if rec["status"] == "ok":
                ok_cost[(rec["batch"], rec["site"], rec["orientation"])] = c
            if (mode, tag, sf.stem, rec["batch"], rec["site"], rec["orientation"], str(rec["attempt"])) in seen:
                continue
            rows.append(dict(utc=sf.stem, mode=mode, tag=tag, batch=rec["batch"], site=rec["site"],
                             orientation=rec["orientation"], attempt=rec["attempt"], cpu=rec["cpu"],
                             memory_mb=rec["memory_mb"], timeout_s=rec["timeout_s"],
                             wall_s=round(rec["wall_s"], 1), cost_usd=c, status=rec["status"]))
        o = d.get("orchestrator")
        if o and ("orchestrator", tag, sf.stem, "", "", "", "1") not in seen:
            rows.append(dict(utc=sf.stem, mode="orchestrator", tag=tag, attempt=1, cpu=o["cpu"],
                             memory_mb=o["memory_mb"], timeout_s=o["timeout_s"], wall_s=round(o["wall_s"], 1),
                             cost_usd=round(cost_usd(o["wall_s"], o["cpu"], o["memory_mb"]), 4), status="ok"))
    if rows:
        append_ledger(rows)
    metas = []
    for rp in sorted(res):
        d = json.loads(rp.read_text())
        m = d["meta"]
        k = (m["batch"], m["site"], m["orientation"])
        if not m.get("error") and k in ok_cost and "cost_usd" not in m:
            m["cost_usd"] = ok_cost[k]
            rp.write_text(json.dumps(d))
        metas.append(m)
    return metas


def _load_results(tag: str) -> list[dict]:
    return [json.loads(rp.read_text()) for rp in sorted((FEM_DIR / "results" / tag).rglob("*.json"))]


def summarize(tag: str) -> list[dict]:
    """Print the per-case table and gates; returns the swelling gate dicts."""
    import numpy as np

    from pmdb.fem.config import load_params
    from pmdb.fem.features import swelling_gate

    p = load_params()
    by_site: dict[tuple, dict] = {}
    n_ok = n_err = n_lost = n_fail = 0
    walls = []
    for d in _load_results(tag):
        m = d["meta"]
        by_site.setdefault((m["batch"], m["site"]), {})[m["orientation"]] = d
        if m.get("error"):
            n_lost += m.get("status") == "lost"
            n_err += m.get("status") != "lost"
            print(f"{m['batch']}/{m['site']} {m['orientation']}: {m.get('status', 'error').upper()} {m['error']}")
            continue
        n_ok += 1
        walls.append(m["wall_s"])
        fa = m.get("failed_at_s")
        if fa is not None and np.isfinite(fa) and fa < 1.0:
            n_fail += 1
        rows = {r["frame"]: r for r in d["site_rows"]}
        t1 = " ".join(f"s{fr / 10:.1f}[sw={rows[fr]['swelling']:.4f} sxx={rows[fr]['sxx_mean_MPa']:.2f}]"
                      for fr in (5, 10))
        print(f"{m['batch']}/{m['site']} {m['orientation']}: wall={m['wall_s']:.0f}s cells={m['n_cells']} "
              f"substeps={m['n_substeps']} newton={m['newton_its_total']} "
              f"s/it={m['wall_s'] / max(m['newton_its_total'], 1):.2f} rss={m['peak_rss_mb']:.0f}MB "
              f"cost=${m.get('cost_usd', float('nan')):.3f} failed_at_s={fa} {t1}")
    gates = []
    for (b, s), o in sorted(by_site.items()):
        if "bottom" in o and "top" in o and not any(o[k]["meta"].get("error") for k in ("bottom", "top")):
            sw = (o["bottom"]["site_rows"][10]["swelling"] + o["top"]["site_rows"][10]["swelling"]) / 2.0
            g = swelling_gate(sw, p)
            g.update(batch=b, site=s)
            gates.append(g)
            print(f"{b}/{s}: swelling_sym(s=1) = {g['swelling_sym']:.4f} gate_ok={g['gate_ok']} "
                  f"lit_band_ok={g['lit_band_ok']}")
    tag_cost = 0.0
    if LEDGER.exists():
        with open(LEDGER, newline="") as fh:
            tag_cost = sum(float(r["cost_usd"] or 0) for r in csv.DictReader(fh) if r["tag"] == tag)
    mean_w = sum(walls) / len(walls) if walls else float("nan")
    print(f"n ok={n_ok} error={n_err} lost={n_lost}; failed_at_s<1: {n_fail}; total wall={sum(walls):.0f}s "
          f"mean wall={mean_w:.0f}s; tag cost=${tag_cost:.3f}; ledger total=${ledger_total():.3f}")
    return gates


# ------------------------------------------------------------------ entrypoint
def _local_labels(b: str, s: str):
    from pmdb.fem.geometry import labels_from_masks
    from pmdb.io import load_site
    from pmdb.segment import segment

    return labels_from_masks(segment(load_site(b, s, resolution="half", normalise="none")))


def _mode_probe() -> int:
    import numpy as np

    if budget_check("probe", "probe", [0], 1.0, 4096, 1800) == "skip":
        return 0
    res = probe.remote()
    ledger_row = dict(utc=_utc(), mode="probe", tag="probe", attempt=1, cpu=1.0, memory_mb=4096,
                      timeout_s=1800, wall_s=round(res["wall_s"], 1),
                      cost_usd=round(cost_usd(res["wall_s"], 1.0, 4096), 4), status="ok")
    append_ledger([ledger_row])
    diffs = {}
    for key, v in res["labels"].items():
        b, s = key.split("/")
        remote = np.frombuffer(v["data"], dtype=np.uint8).reshape(v["shape"])
        local = _local_labels(b, s)
        diffs[key] = float((remote != local).mean()) if remote.shape == local.shape else 1.0
    out = {k: v for k, v in res.items() if k != "labels"}
    out["label_diff_fraction"] = diffs
    d = FEM_DIR / "probe"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{_utc()}.json").write_text(json.dumps(out, indent=2))
    print("python", res["python"].split()[0], "| dolfinx", res["dolfinx"], "| petsc4py", res["petsc4py"],
          "| numpy", res["numpy"], "| scipy", res["scipy"], "| skimage", res["skimage"],
          "| mumps", res["mumps"])
    print("NonlinearProblem.__init__", res["nonlinear_problem_signature"])
    for k, v in diffs.items():
        print(("WARN " if v > 1e-4 else "") + f"label differing fraction {k}: {v:.3e}")
    print(f"probe wall {res['wall_s']:.1f}s")
    return 1 if (res["dolfinx"] != "0.10.0" or not res["mumps"]) else 0


def _mode_unit(k: str) -> int:
    if budget_check("unit", "unit", [0], 2.0, 4096, 1800) == "skip":
        return 0
    res = run_unit_tests.remote(k)
    append_ledger([dict(utc=_utc(), mode="unit", tag="unit", attempt=1, cpu=2.0, memory_mb=4096,
                        timeout_s=1800, wall_s=round(res["wall_s"], 1),
                        cost_usd=round(cost_usd(res["wall_s"], 2.0, 4096), 4),
                        status="ok" if res["returncode"] == 0 else "error")])
    d = FEM_DIR / "tests" / _utc()
    d.mkdir(parents=True, exist_ok=True)
    (d / "pytest.txt").write_text(res["stdout"] + "\n--- stderr ---\n" + res["stderr"])
    print("\n".join(res["stdout"].splitlines()[-40:]))
    for line in res["stdout"].splitlines():
        if "T6_REPORT" in line:
            print(line)
    print("pytest.txt:", d / "pytest.txt")
    return int(res["returncode"])


def _mode_window(orientation: str, sites: str, crop_um: float, res_nm: float, tag: str, cpu: float,
                 memory: int, timeout: int, solver_overrides: str) -> int:
    if not tag or not sites:
        raise SystemExit("window mode needs --tag and --sites")
    ors = ["bottom", "top"] if orientation == "both" else [orientation]
    cases = [{"batch": s.split("/")[0], "site": s.split("/")[1], "orientation": o}
             for s in sites.split(",") for o in ors]
    if budget_check("window", tag, cases, cpu, memory, timeout) == "skip":
        return 0
    ov = json.loads(solver_overrides) if solver_overrides else None
    run_cases.remote(cases, tag, cpu, memory, timeout, int(cpu), crop_um, res_nm, ov)
    sync(tag)
    gates = summarize(tag)
    print(f"SYNC DONE {tag}")
    if any(not g["gate_ok"] for g in gates):
        return 4
    return 1 if any(d["meta"].get("error") for d in _load_results(tag)) else 0


@app.local_entrypoint()
def main(mode: str, k: str = "", orientation: str = "both", sites: str = "", crop_um: float = 0.0,
         res_nm: float = 0.0, tag: str = "", cpu: float = 4.0, memory: int = 16384, timeout: int = 7200,
         solver_overrides: str = ""):
    if mode == "probe":
        code = _mode_probe()
    elif mode == "unit":
        code = _mode_unit(k)
    elif mode == "window":
        code = _mode_window(orientation, sites, crop_um, res_nm, tag, cpu, memory, timeout, solver_overrides)
    elif mode == "diag":
        runs = json.loads(solver_overrides)
        t0 = time.time()
        for r in diag.remote(runs):
            print("DIAG", json.dumps(r))
        w = time.time() - t0
        append_ledger([dict(utc=_utc(), mode="diag", tag="diag", attempt=1, cpu=4.0, memory_mb=16384,
                            timeout_s=3000, wall_s=round(w, 1), cost_usd=round(cost_usd(w, 4.0, 16384), 4),
                            status="ok")])
        code = 0
    elif mode == "sync":
        sync(tag)
        summarize(tag)
        print(f"SYNC DONE {tag}")
        code = 0
    else:
        raise SystemExit(f"mode {mode!r} not implemented in this build step")
    if code:
        raise SystemExit(code)
