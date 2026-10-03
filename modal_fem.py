"""FEM runner on Modal (D16, P20-P22, P26). Run: ``modal run modal_fem.py --mode probe|unit``.

All dolfinx code runs inside the pinned official dolfinx image; the local side only orchestrates,
keeps the cost ledger and enforces the budget cap.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
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


@app.local_entrypoint()
def main(mode: str, k: str = ""):
    if mode == "probe":
        code = _mode_probe()
    elif mode == "unit":
        code = _mode_unit(k)
    else:
        raise SystemExit(f"mode {mode!r} not implemented in this build step")
    if code:
        raise SystemExit(code)
