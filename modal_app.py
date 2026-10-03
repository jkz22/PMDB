"""Run PMDB per-site compute on Modal. See README 'Running on Modal'."""
from __future__ import annotations

import time
from pathlib import Path

import modal

APP_NAME = "pmdb"
VOLUME_NAME = "pmdb-data"
CACHE_MOUNT = "/data"  # volume holds /half/*.npz + /half/manifest.csv

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.4", "scipy==1.14.1", "scikit-image==0.25.2", "pandas==2.1.4",
        "Pillow==12.3.0", "tifffile==2025.5.10", "imagecodecs==2025.3.30",
    )
    .env({"PMDB_CACHE": CACHE_MOUNT})
    .add_local_python_source("pmdb")
    .add_local_file("docs/kpis/kpi_catalogue.csv", "/root/docs/kpis/kpi_catalogue.csv")
)
volume = modal.Volume.from_name(VOLUME_NAME).read_only()
app = modal.App(APP_NAME, image=image)

FN_KW = dict(volumes={CACHE_MOUNT: volume}, cpu=1.0, memory=4096, timeout=900)


def _run_site_impl(batch: str, site_id: str, runner: str) -> dict:
    from pmdb.io import load_site, normalise_image
    from pmdb.kpis import compute_site_kpis
    from pmdb.segment import segment

    t0 = time.time()
    row = {"batch": batch, "site": site_id, "se_detector": "", "runner": runner,
           "elapsed_s": 0.0, "error": ""}
    try:
        s = load_site(batch, site_id, resolution="half", normalise="none")
        row["se_detector"] = s.se_detector
        masks = segment(s)
        bse_norm = normalise_image(s.image)[..., 0]
        values, _curves, _ctx = compute_site_kpis(masks, bse_norm, s.nm_per_px, f"{batch}/{site_id}")
        row.update(values)
    except Exception as e:  # D7: record and continue
        row["error"] = f"{type(e).__name__}: {e}"
    row["elapsed_s"] = round(time.time() - t0, 2)
    return row


@app.function(**FN_KW)
def run_site(batch: str, site_id: str) -> dict:
    return _run_site_impl(batch, site_id, "cpu")


@app.local_entrypoint()
def main(name: str = "site_kpis", smoke: bool = False):
    import pandas as pd
    from pmdb.io import get_cache_root

    manifest = pd.read_csv(get_cache_root() / "half" / "manifest.csv")
    if smoke:
        manifest = manifest.head(1)
        name = f"{name}_smoke"
    rows = list(run_site.map(manifest["batch"].tolist(), manifest["site"].tolist()))
    failed = [r for r in rows if r["error"]]
    out_dir = Path(__file__).resolve().parent / "outputs" / "modal"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous complete table with a partial one.
    out_path = out_dir / (f"{name}_failed.csv" if failed else f"{name}.csv")
    tmp_path = out_path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)
    print(f"Wrote {len(rows)} rows to {out_path} ({len(failed)} errored)")
    for r in failed:
        print(f"  {r['batch']}/{r['site']}: {r['error']}")
    if failed:
        raise SystemExit(1)
