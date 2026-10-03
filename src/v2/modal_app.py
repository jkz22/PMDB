"""Phase 4: one detached Modal job on one H100 running many configs concurrently.

    modal run src/v2/modal_app.py --mode calibrate          # short calibration + cost estimate
    modal run --detach src/v2/modal_app.py --mode stage_ab --parallel N
    modal run --detach src/v2/modal_app.py --mode stage_c --parallel N

Crops cache lives in Volume ``pmdb-v2`` (/vol/cache/half) and is loaded into RAM by each
worker. Runs are config-hash keyed under /vol/runs, checkpoint every 500 steps and resume.
Live status: /vol/status/<mode>.csv (queued/running/done/failed + error). Failed runs retry
once. No new run is launched once the estimated spend reaches the budget (default $60).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root/pmdb")
KPI_WT = REPO.parent / "kpi_wt" / "d23a116"
GPU = os.environ.get("PMDB_GPU", "H100")
CPU, MEM_GB = 24.0, 96
# Modal list prices, $/s (https://modal.com/pricing, 2026-10-03)
RATE = {"H100": 0.001097, "A100-80GB": 0.000694}
CPU_RATE, MEM_RATE = 0.0000131, 0.00000222
USD_PER_S = RATE[GPU] + CPU * CPU_RATE + MEM_GB * MEM_RATE

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "transformers==4.57.6", "timm==1.0.30",
                 "segmentation-models-pytorch==0.5.0", "scikit-learn", "scikit-image", "scipy", "pandas",
                 "tifffile", "imagecodecs", "umap-learn", "matplotlib", "numpy<2")
    .run_commands(
        "python -c \"from transformers import Dinov2Model, ViTMAEForPreTraining; "
        "Dinov2Model.from_pretrained('facebook/dinov2-small'); ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')\"",
        "python -c \"import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/jstuckner/"
        "microscopy-efficientnet-b4-imagenet-micronet/resolve/main/efficientnet-b4_imagenet-micronet_weights.pth')\"")
    .env({"PMDB_CACHE": "/vol/cache", "PMDB_RUNS": "/vol/runs", "PMDB_KPI_ROOT": "/kpi/d23a116",
          "PMDB_KPI_COMMIT": "d23a116", "PYTHONPATH": "/root/pmdb", "OMP_NUM_THREADS": "2"})
    .add_local_dir(REPO / "src", "/root/pmdb/src", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "pmdb", "/root/pmdb/pmdb", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "outputs" / "v2" / "kpis", "/root/pmdb/outputs/v2/kpis")
    .add_local_dir(REPO / "outputs" / "v2" / "imaging_stats", "/root/pmdb/outputs/v2/imaging_stats")
    .add_local_dir(KPI_WT, "/kpi/d23a116", ignore=["cache/**", "data/**", "outputs/**", ".git"])
    .add_local_file(KPI_WT / ".kpi_commit", "/kpi/d23a116/.kpi_commit")
)
vol = modal.Volume.from_name("pmdb-v2")
app = modal.App("pmdb-v2", image=image)

LONG_FIRST = ("mae_scratch", "mae_adapted", "dino_ft", "vae_b", "vae_c", "vae_a")


def _gpu_sample():
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    u, m, t = (float(v) for v in q.split(","))
    return u, m, t


def _hash(spec):
    from src.v2.rungrid import train_cfg
    from src.v2.train import cfg_hash
    s = train_cfg(spec)
    return cfg_hash({**s, "steps": 0} if spec["family"].startswith("ots_") else s)


@app.function(gpu=GPU, cpu=CPU, memory=MEM_GB * 1024, volumes={"/vol": vol}, timeout=24 * 3600)
def schedule(specs: list[dict], parallel: int, budget_usd: float, tag: str, max_steps: int | None = None):
    """Run specs concurrently on this GPU. Returns the final status table."""
    os.chdir("/root/pmdb")
    status_dir = Path("/vol/status"); status_dir.mkdir(parents=True, exist_ok=True)
    if max_steps is not None:
        specs = [{**sp, "steps": max_steps} for sp in specs]
    order = sorted(range(len(specs)), key=lambda i: LONG_FIRST.index(specs[i]["family"]) if specs[i]["family"] in LONG_FIRST else 99)
    rows = {i: dict(idx=i, family=specs[i]["family"], factor=specs[i].get("factor", ""), hash=_hash(specs[i]),
                    state="queued", tries=0, start=None, sec=None, error="") for i in order}
    queue, procs, t0, util, mem = list(order), {}, time.time(), [], []
    spent = lambda: (time.time() - t0) * USD_PER_S  # noqa: E731

    def write_status():
        import pandas as pd
        df = pd.DataFrame(rows.values())
        df.to_csv(status_dir / f"{tag}.csv", index=False)
        cnt = df.state.value_counts().to_dict()
        print(f"[{(time.time() - t0) / 60:6.1f} min ${spent():6.2f}] {cnt} util_mean={sum(util[-20:]) / max(1, len(util[-20:])):.0f}%", flush=True)
        vol.commit()

    last = 0.0
    while queue or procs:
        while queue and len(procs) < parallel and spent() < budget_usd:
            i = queue.pop(0)
            spec = dict(specs[i])
            log = open(f"/tmp/run_{i}_{rows[i]['tries']}.log", "w")
            procs[i] = (subprocess.Popen(["python", "-m", "src.v2.worker", json.dumps(spec)], stdout=log,
                                         stderr=subprocess.STDOUT, env={**os.environ, "PMDB_WORKERS": "3"}), log)
            rows[i].update(state="running", start=time.time() - t0, tries=rows[i]["tries"] + 1)
        if queue and spent() >= budget_usd and not procs:
            for i in queue:
                rows[i]["state"] = "skipped_budget"
            queue = []
        for i, (p, log) in list(procs.items()):
            if p.poll() is None:
                continue
            log.close(); del procs[i]
            rows[i]["sec"] = time.time() - t0 - rows[i]["start"]
            if p.returncode == 0:
                rows[i]["state"] = "done"
            else:
                tail = open(log.name).read()[-1500:]
                rows[i]["error"] = tail.replace("\n", " | ")[-600:]
                if rows[i]["tries"] < 2:
                    rows[i]["state"] = "queued_retry"; queue.insert(0, i)
                else:
                    rows[i]["state"] = "failed"
        try:
            u, m, t = _gpu_sample(); util.append(u); rows_mem = m / t; mem.append(m / t)
        except Exception:
            rows_mem = None
        if time.time() - last > 30:
            write_status(); last = time.time()
            if rows_mem is not None:
                print(f"    gpu_mem={100 * rows_mem:.0f}%", flush=True)
        time.sleep(2)
    write_status()
    return {"rows": list(rows.values()), "usd": spent(), "util_mean": sum(util) / max(1, len(util)),
            "util_samples": util[-2000:],
            "util_busy_mean": sum(util[len(util) // 4: 3 * len(util) // 4]) / max(1, len(util) // 2),
            "mem_frac_max": max(mem) if mem else None}


@app.function(volumes={"/vol": vol}, timeout=600)
def fetch(tag: str):
    import glob
    out = {"status": open(f"/vol/status/{tag}.csv").read() if os.path.exists(f"/vol/status/{tag}.csv") else ""}
    out["metrics"] = [json.load(open(p)) for p in glob.glob("/vol/runs/*/metrics.json")]
    return out


@app.local_entrypoint()
def main(mode: str = "calibrate", parallel: int = 6, budget: float = 60.0, steps: int = 150):
    import sys
    sys.path.insert(0, str(REPO))
    from src.v2.models import OFF_THE_SHELF, TRAINABLE
    from src.v2.rungrid import stage_a, stage_b
    print(f"GPU {GPU}; all-in rate ${USD_PER_S * 3600:.2f}/h (GPU+{CPU:.0f} cores+{MEM_GB} GiB)")
    if mode == "calibrate":
        specs = [dict(family=f, stage="cal", factor=f"cal{steps}", seed=1000 + k)
                 for k in range(-(-parallel // len(TRAINABLE))) for f in TRAINABLE][:parallel]
        r = schedule.remote(specs, parallel=parallel, budget_usd=5.0, tag=f"calibrate_p{parallel}", max_steps=steps)
        print(json.dumps({k: v for k, v in r.items() if k != "util_samples"}, indent=1, default=str))
        return
    if mode == "stage_ab":
        specs = [dict(family=f, stage="OTS") for f in OFF_THE_SHELF] + stage_a() + stage_b()
    else:
        specs = json.load(open(REPO / "outputs" / "v2" / "stage_c_specs.json"))
    print(f"{len(specs)} runs; launching detached-safe scheduler (parallel={parallel}, budget=${budget})")
    r = schedule.remote(specs, parallel=parallel, budget_usd=budget, tag=mode)
    print(json.dumps({"usd": r["usd"], "util_mean": r["util_mean"]}, indent=1))
