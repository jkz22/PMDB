"""Test-day preprocessing of new organiser images on Modal (half cache, affine2 LUTs, clean, KPIs, fingerprint features).

Everything runs on Modal; local work is file copy, upload and writing the pulled-back small files.
Raw TIFFs go in data_test/Batch_test/img_<site>_<BSE|Inlens|ETD|SE>.tif (gitignored).

Run:
    modal run modal_test_prep.py::main      # upload data_test/, preprocess on Modal, pull back small outputs
    modal run modal_test_prep.py::check --sites fn0mhxef,3e122cbj   # rehearsal: compare with held-out artefacts

Then: modal run modal_patch_mil.py --mode test   (or bash scripts/score_test_sites.sh for both). See docs/patch_mil.md.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import modal

APP_NAME = "pmdb-test-prep"
TEST_ROOT = "/data/test"  # volume path /test
RAW_ROOT = "/data/test/raw"  # list_sites data_root; holds Batch_test/
TEST_BATCH = "Batch_test"
ROOT = Path(__file__).resolve().parent

_SCRIPTS = ("build_cache", "build_harmonised", "build_clean", "run_kpis")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4", "scipy==1.14.1", "scikit-image==0.25.2", "pandas==2.1.4", "Pillow==12.3.0",
                 "tifffile==2025.5.10", "imagecodecs==2025.3.30", "matplotlib==3.10.7")
    .env({"PMDB_CACHE": TEST_ROOT, "MPLBACKEND": "Agg"})
    .add_local_python_source("pmdb")
    .add_local_file("cache/harmonised/affine2/reference.json", "/root/ref/harmonised/affine2/reference.json")
    .add_local_file("outputs/clean/targets.json", "/root/ref/clean/targets.json")
    .add_local_file("outputs/clean/summary.csv", "/root/ref/clean/summary.csv")
    .add_local_file("docs/kpis/kpi_catalogue.csv", "/root/docs/kpis/kpi_catalogue.csv")
)
for _s in _SCRIPTS:
    image = image.add_local_file(f"scripts/{_s}.py", f"/root/scripts/{_s}.py")
data_vol = modal.Volume.from_name("pmdb-data")  # writable
app = modal.App(APP_NAME, image=image)


@app.function(volumes={"/data": data_vol}, cpu=8.0, memory=32768, timeout=3600)
def prep_test_sites(expected_sites: list[str]) -> dict:
    """Returns {"files": {<local repo-relative path>: bytes}, "sites": [...], "timings_s": {stage: s}}."""
    import json
    import shutil

    import pandas as pd

    os.environ["PMDB_CACHE"] = TEST_ROOT
    sys.path.insert(0, "/root/scripts")
    from pmdb import fingerprint as fp
    from pmdb import harmonise as H
    from pmdb.io import list_sites

    timings: dict[str, float] = {}
    t = time.time()

    def lap(name: str) -> None:
        nonlocal t
        timings[name] = round(time.time() - t, 1)
        t = time.time()
        print(f"[stage] {name}: {timings[name]}s", flush=True)

    root = Path(TEST_ROOT)
    man = list_sites(RAW_ROOT)
    got = sorted(man.site.astype(str))
    if set(man.batch) != {TEST_BATCH} or got != sorted(expected_sites):
        raise RuntimeError(f"volume /test/raw has sites {got}, expected {sorted(expected_sites)}; "
                           "run: modal volume rm -r pmdb-data /test/raw")
    n = min(len(expected_sites), 6)
    lap("list")

    import build_cache
    build_cache.build_cache(data_root=Path(RAW_ROOT), cache_root=root, outputs_dir=root, force=True)
    lap("cache")

    import build_harmonised
    ref = H.load_reference(Path("/root/ref"), "affine2")
    anchors, hists = build_harmonised._collect_anchors(root)
    (root / "harmonised").mkdir(parents=True, exist_ok=True)
    anchors.to_csv(root / "harmonised" / "anchors.csv", index=False)
    build_harmonised._fit_all(root, anchors, hists, ref, ["affine2"])
    lap("luts")

    import build_clean
    sys.argv = ["build_clean.py", "--data-root", RAW_ROOT, "--out", f"{TEST_ROOT}/clean",
                "--targets", "/root/ref/clean/targets.json", "--labelled-summary", "/root/ref/clean/summary.csv",
                "--workers", str(n)]
    build_clean.main()
    lap("clean")

    import run_kpis  # after PMDB_CACHE is set: run_kpis.MANIFEST is bound at import
    rc = run_kpis.main(["--out-dir", f"{TEST_ROOT}/kpis", "--no-overlays", "--jobs", str(n)])
    if rc != 0:
        att = root / "kpis" / "run_attempt.json"
        raise RuntimeError(f"run_kpis failed rc={rc}: {att.read_text() if att.exists() else ''}")
    lap("kpis")

    X = fp.read_feature_inputs(f"{TEST_ROOT}/kpis/curves.csv", f"{TEST_ROOT}/kpis/tile_kpis.csv")
    cm = pd.read_csv(root / "half" / "manifest.csv", dtype={"site": str})
    assert not X.isna().any().any(), "NaN in features"
    assert set(X.index) == set(zip(cm.batch, cm.site)), "feature index != manifest"
    X.reset_index().to_csv(root / "features.csv", index=False)
    lap("features")

    data_vol.commit()
    files: dict[str, bytes] = {}

    def grab(src: Path, dst: str) -> None:
        files[dst] = src.read_bytes()

    grab(root / "half" / "manifest.csv", "cache_test/half/manifest.csv")
    grab(root / "harmonised" / "anchors.csv", "cache_test/harmonised/anchors.csv")
    for f in ("luts.npz", "params.csv", "reference.json"):
        grab(root / "harmonised" / "affine2" / f, f"cache_test/harmonised/affine2/{f}")
    grab(root / "raw_intensity_stats.csv", "outputs/test/raw_intensity_stats.csv")
    for f in ("summary.csv", "material_thresholds.json"):
        grab(root / "clean" / f, f"outputs/clean_test/{f}")
    for s in expected_sites:
        grab(root / "clean" / TEST_BATCH / s / "params.json", f"outputs/clean_test/{TEST_BATCH}/{s}/params.json")
    for f in ("site_kpis.csv", "tile_kpis.csv", "curves.csv", "sensitivity.csv", "run_log.json"):
        grab(root / "kpis" / f, f"outputs/test/kpis/{f}")
    grab(root / "features.csv", "outputs/test/features.csv")
    _ = (json, shutil)
    return {"files": files, "sites": sorted(expected_sites), "timings_s": timings}


@app.function(volumes={"/data": data_vol}, cpu=1.0, memory=4096, timeout=600)
def rehearsal_check(test_site: str, heldout_site: str) -> dict:
    import numpy as np

    a = np.load(f"/data/test/half/Batch_test__{test_site}.npz")["image"]
    b = np.load(f"/data/heldout/half/Batch_heldout__{heldout_site}.npz")["image"]
    out = {"half_shape_equal": a.shape == b.shape}
    if a.shape == b.shape:
        d = np.abs(a.astype(int) - b.astype(int))
        out.update(half_equal=bool((d == 0).all()), half_n_diff=int((d > 0).sum()), half_size=int(d.size),
                   half_max_abs_diff=int(d.max()))
    la = np.load("/data/test/harmonised/affine2/luts.npz")[f"Batch_test__{test_site}"]
    lb = np.load("/data/heldout/harmonised/affine2/luts.npz")[f"Batch_heldout__{heldout_site}"]
    out["lut_equal"] = bool(la.shape == lb.shape and np.array_equal(la, lb))
    return out


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


@app.local_entrypoint()
def main():
    import io

    import pandas as pd

    from pmdb.io import list_sites
    from pmdb.parents import parent_id

    t0 = time.time()
    man = list_sites(ROOT / "data_test")
    if set(man.batch) != {TEST_BATCH} or man.empty:
        raise SystemExit(f"data_test must contain only {TEST_BATCH}/img_<site>_<detector>.tif files")
    sites = sorted(man.site.astype(str))
    with data_vol.batch_upload(force=True) as up:
        for r in man.itertuples():
            for col in ("path_bse", "path_inlens", "path_se_type"):
                p = Path(getattr(r, col))
                up.put_file(str(p), f"/test/raw/{TEST_BATCH}/{p.name}")
    print(f"uploaded {len(sites)} sites in {time.time() - t0:.0f}s: {sites}")
    res = prep_test_sites.remote(sites)
    for key, data in res["files"].items():
        if key == "cache_test/half/manifest.csv":
            m = pd.read_csv(io.BytesIO(data), dtype={"site": str})
            m["path"] = [f"cache_test/half/{b}__{s}.npz" for b, s in zip(m["batch"], m["site"])]
            data = m.to_csv(index=False).encode()
        _atomic_write(ROOT / key, data)
    print("sites:", res["sites"], "timings_s:", res["timings_s"], f"total {time.time() - t0:.0f}s")
    summ = pd.read_csv(ROOT / "outputs/clean_test/summary.csv", dtype={"site": str})
    cm = pd.read_csv(ROOT / "cache_test/half/manifest.csv", dtype={"site": str})
    j = cm.drop(columns=["height"]).merge(summ[["batch", "site", "height", "BSE_grey_step"]], on=["batch", "site"])
    for r in j.itertuples():
        print(f"  {r.site}: height {r.height}, BSE_grey_step {r.BSE_grey_step}, "
              f"parent {parent_id(r.height, r.se_detector, r.BSE_grey_step)}")


@app.local_entrypoint()
def check(sites: str = "fn0mhxef"):
    """Rehearsal: test-prep outputs for copies of held-out sites vs the held-out artefacts."""
    import numpy as np
    import pandas as pd

    from pmdb.parents import parent_groups

    pg = parent_groups().set_index(["batch", "site"])
    F = pd.read_csv(ROOT / "outputs/fingerprint/features.csv", dtype={"site": str})
    sd = F.drop(columns=["batch", "site"]).std(ddof=1)
    Hf = pd.read_csv(ROOT / "outputs/fingerprint/heldout_features.csv", dtype={"site": str}).set_index("site")
    Tf = pd.read_csv(ROOT / "outputs/test/features.csv", dtype={"site": str}).set_index("site")
    bad = False
    for s in [x for x in sites.split(",") if x]:
        r = rehearsal_check.remote(s, s)
        print(f"== {s}")
        if r["half_equal"]:
            print("PASS half cache: exact")
        elif r["half_max_abs_diff"] <= 1 and r["half_n_diff"] / r["half_size"] < 1e-3:
            print(f"PASS (warning) half cache: max diff {r['half_max_abs_diff']}, {r['half_n_diff']}/{r['half_size']}")
        else:
            print(f"FAIL half cache: {r}")
            bad = True
        print(("PASS" if r["lut_equal"] else "FAIL") + " affine2 LUT")
        bad |= not r["lut_equal"]
        a, b = pg.loc[("Batch_heldout", s)], pg.loc[("Batch_test", s)]
        ok = all(a[c] == b[c] for c in ("height", "se_detector", "bse_grey_step", "parent_id"))
        print(("PASS" if ok else "FAIL") + f" parent key {a['parent_id']} vs {b['parent_id']}")
        bad |= not ok
        d = (Tf.loc[s, sd.index] - Hf.loc[s, sd.index]).abs() / sd
        print(f"INFO features: max |diff|/SD = {d.max():.4f} on {d.idxmax()} (>0.05: {int((d > 0.05).sum())} features)")
        print("  per-feature |diff|/SD:", {k: round(float(v), 4) for k, v in d.items()})
        _ = np
    if bad:
        raise SystemExit(1)
