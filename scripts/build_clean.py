#!/usr/bin/env python3
"""Run the physical clean/normalise/harmonise pipeline (pmdb.clean) on every site.

Pass 1 (``norm``): patch + normalise each site from the full-resolution TIFFs and write
``<out>/<batch>/<site>/{BSE,Inlens,SE_type}_{norm,mask}.tif`` + ``params.json``.
Pass 2 (``harm``): derive resolution/noise targets from the accepted reference sites (or load
``--targets``), blur/noise every site down to them and write ``*_harm.tif``.
Then ``summary.csv`` (one row per site with every parameter) and ``qc_report.html``.

Examples
--------
    python scripts/build_clean.py --workers 4
    python scripts/build_clean.py --data-root data_heldout --out outputs/clean_heldout \\
        --targets outputs/clean/targets.json --workers 3
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pmdb.clean as C  # noqa: E402
from pmdb.io import REPO_ROOT, list_sites  # noqa: E402

# Accepted reference sites (brief §4.5): the unflagged Batch 3 sites.  Flagged sites are never
# used to fit harmonisation targets.
REFERENCE_SITES = ("ptg8lmto", "utfgcjfa", "vc2whyaq", "xgj4xftb")
CRACK_SPAN_FRAC = 0.75  # pore network spanning >= 75 % of the image height is a crack

FLAGGED_SITES = {
    "71vgq3fw": "strong black-level/gain offset (imaging session)",
    "kbdh4tri": "strong black-level/gain offset (imaging session)",
    "tuy3zymq": "strong black-level/gain offset (imaging session)",
    "x7u69zsw": "strong black-level/gain offset (imaging session)",
    "9luzk4jm": "mild black-level offset",
    "hzumfsms": "mild black-level offset; large crack",
    "ufdvpb81": "mild black-level offset; colour marker columns",
}


def _paths(row) -> dict[str, Path]:
    return {"BSE": Path(row.path_bse), "Inlens": Path(row.path_inlens), "SE_type": Path(row.path_se_type)}


def _portable(p: str | Path) -> str:
    """Repo-relative POSIX path when the file lives inside the repository, else absolute (committed
    ``params.json`` must be readable from any checkout)."""
    q = Path(p).resolve()
    try:
        return q.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(q)


def run_norm(batch: str, site: str, paths: dict[str, str], out_dir: str, nm_per_px: float) -> dict:
    t0 = time.time()
    raw, markers = {}, None
    for d, p in paths.items():
        img, cols = C.read_rgb_tiff(p)
        raw[d] = img
        markers = cols if markers is None else (markers | cols)
    res = C.clean_site(raw, marker_cols=markers, nm_per_px=nm_per_px)
    res.params.update({"batch": batch, "site": site, "paths": {d: _portable(p) for d, p in paths.items()},
                       "seconds_norm": round(time.time() - t0, 1)})
    C.write_site(out_dir, res)
    _thumbs(Path(out_dir), raw, res)
    return res.params


def run_harm(out_dir: str, targets: dict, nm_per_px: float) -> dict:
    out = Path(out_dir)
    params = json.loads((out / "params.json").read_text())
    norm = {d: C.read_site(out, d, "norm")[0] for d in C.DETECTORS}
    mask = {d: C.read_site(out, d, "norm")[1] for d in C.DETECTORS}
    res = C.CleanResult(norm, mask, params)
    t0 = time.time()
    h = C.harmonise_site(res, C.Targets.from_dict(targets), nm_per_px=nm_per_px)
    h.params["seconds_harm"] = round(time.time() - t0, 1)
    C.write_site(out, h)
    return h.params


def _thumbs(out: Path, raw: dict[str, np.ndarray], res: C.CleanResult, step: int = 8) -> None:
    """Small PNG previews (raw / norm / mask) for the QC report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 3 * 1.6), dpi=72)
    for i, d in enumerate(C.DETECTORS):
        axes[i, 0].imshow(raw[d][::step, ::step], cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_ylabel(d)
        axes[i, 1].imshow(res.norm[d][::step, ::step], cmap="gray", vmin=-0.1, vmax=2.2)
        m = res.mask[d][::step, ::step]
        axes[i, 2].imshow(np.log2(m.astype(np.float32) + 1), cmap="tab20", vmin=0, vmax=12, interpolation="nearest")
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])
    axes[0, 0].set_title("raw")
    axes[0, 1].set_title("norm z")
    axes[0, 2].set_title("mask (log2 bits)")
    fig.tight_layout()
    fig.savefig(out / "preview.png")
    plt.close(fig)


def flatten(p: dict) -> dict:
    """One flat row per site for summary.csv."""
    row = {"batch": p.get("batch"), "site": p.get("site"), "height": p["shape"][0], "width": p["shape"][1],
           "n_marker_cols": len(p.get("marker_cols", [])), "code_version": p["code_version"]}
    for d in C.DETECTORS:
        a = p["anchors"][d]
        row.update({f"{d}_D": a["D"], f"{d}_D_se": a["D_se"], f"{d}_D_method": a["D_method"], f"{d}_D_frac_zero": a["D_frac_zero"],
                    f"{d}_G": a["G_level"], f"{d}_G_ptp_rel": a["G_ptp_rel"], f"{d}_si_graphite": a["si_graphite_ratio"],
                    f"{d}_grey_levels": a["grey_levels_used"], f"{d}_grey_step": a.get("grey_step", 1)})
        f = p["fingerprint"][d]
        row.update({f"{d}_sigma_e": f["sigma_e_px"], f"{d}_edge_nm": f["edge_10_90_nm"], f"{d}_noise_alpha": f["noise_alpha"],
                    f"{d}_noise_beta": f["noise_beta"], f"{d}_noise_sigma_g": f["noise_sigma_graphite"]})
        for k, v in p["mask_fractions"][d].items():
            row[f"{d}_mask_{k}"] = v
        row[f"{d}_masked_beyond_border"] = p["masked_beyond_border"][d]
        if "harmonisation" in p:
            h = p["harmonisation"][d]
            row.update({f"{d}_sigma_k": h["sigma_k_px"], f"{d}_sigma_e_after": h["sigma_e_after_px"],
                        f"{d}_resolution_outlier": h["resolution_outlier"], f"{d}_noise_added": h["added"],
                        f"{d}_noise_sigma_g_after": h["noise_sigma_graphite_after"], f"{d}_noise_within_10pct": h["noise_within_10pct"]})
    row.update({"collector_found": p["fov"].get("collector", {}).get("found", False),
                "collector_depth_um": p["fov"].get("collector", {}).get("depth_um"),
                "free_surface_found": p["fov"].get("free_surface", {}).get("found", False),
                "free_surface_depth_um": p["fov"].get("free_surface", {}).get("depth_um"),
                "n_band_events": len(p["bands"]["events"]), "n_bands_corrected": len(p["bands"]["corrected"]),
                "n_bands_masked": len(p["bands"]["masked"]), "n_charging_local": len(p["charging"]["local"]),
                "charging_broad": p["charging"]["broad"].get("found", False),
                "porosity": p["material"]["porosity"], "si_fraction": p["material"]["si_fraction"],
                "porosity_top8": p["material"]["eighth_porosity"]["top"], "porosity_bottom8": p["material"]["eighth_porosity"]["bottom"],
                "largest_pore_um2": p["material"]["pore_networks"][0]["area_um2"] if p["material"]["pore_networks"] else 0.0,
                "largest_pore_span_um": p["material"]["pore_networks"][0]["span_um"] if p["material"]["pore_networks"] else 0.0,
                "seconds_norm": p.get("seconds_norm"), "seconds_harm": p.get("seconds_harm")})
    return row


def material_flags(df: pd.DataFrame, labelled: pd.DataFrame | None = None) -> pd.DataFrame:
    """Crack / pore-rich-band flags from robust across-site outlier thresholds (3.6)."""
    ref = labelled if labelled is not None else df
    thr_crack = C.robust_outlier_threshold(ref["largest_pore_um2"].to_numpy())
    thr_span = C.robust_outlier_threshold(ref["largest_pore_span_um"].to_numpy())
    rel_top = ref["porosity_top8"] / ref["porosity"]
    rel_bot = ref["porosity_bottom8"] / ref["porosity"]
    thr_band = C.robust_outlier_threshold(np.concatenate([rel_top, rel_bot]))
    df = df.copy()
    # a crack is a pore network whose area is an outlier on the labelled distribution AND that is long:
    # longer than the labelled outlier span, or at least most of the coating thickness (hzumfsms: 90 % of
    # the image height). Length alone is not enough: on the short xgj4xftb field a 26 µm wide shallow pore
    # cluster near the free surface already spans 85 % of the height
    height_um = df["height"] * C.FULL_NM_PER_PX / 1000.0
    df["flag_crack"] = (df["largest_pore_um2"] > thr_crack) & (
        (df["largest_pore_span_um"] > thr_span) | (df["largest_pore_span_um"] >= CRACK_SPAN_FRAC * height_um))
    df["flag_pore_band_top"] = df["porosity_top8"] / df["porosity"] > thr_band
    df["flag_pore_band_bottom"] = df["porosity_bottom8"] / df["porosity"] > thr_band
    df["flag_imaging"] = df["site"].map(lambda s: FLAGGED_SITES.get(s, ""))
    df.attrs["thresholds"] = {"crack_area_um2": thr_crack, "crack_span_um": thr_span, "pore_band_rel": thr_band}
    return df


def qc_report(out: Path, df: pd.DataFrame, targets: dict | None) -> None:
    rows = []
    for r in df.itertuples():
        png = out / r.batch / r.site / "preview.png"
        img = base64.b64encode(png.read_bytes()).decode() if png.exists() else ""
        flags = [k for k in ("flag_crack", "flag_pore_band_top", "flag_pore_band_bottom") if getattr(r, k, False)]
        if r.flag_imaging:
            flags.append(f"imaging: {r.flag_imaging}")
        txt = (f"<b>{r.batch} / {r.site}</b> {r.height}x{r.width}px &nbsp; "
               f"D(BSE)={r.BSE_D:.1f}±{r.BSE_D_se:.2f} [{r.BSE_D_method}] G(BSE)={r.BSE_G:.1f} "
               f"Si/G={r.BSE_si_graphite:.3f} σe={r.BSE_sigma_e:.2f}px noise σg={r.BSE_noise_sigma_g:.4f}<br>"
               f"collector={r.collector_found} ({r.collector_depth_um}) free_surface={r.free_surface_found} ({r.free_surface_depth_um}) "
               f"bands corrected/masked={r.n_bands_corrected}/{r.n_bands_masked} charging local/broad={r.n_charging_local}/{r.charging_broad} "
               f"masked beyond border BSE={r.BSE_masked_beyond_border:.3%} Inlens={r.Inlens_masked_beyond_border:.3%}<br>"
               f"porosity={r.porosity:.3f} (top/bottom eighth {r.porosity_top8:.3f}/{r.porosity_bottom8:.3f}) Si={r.si_fraction:.3f} "
               f"largest pore={r.largest_pore_um2:.1f}µm² span={r.largest_pore_span_um:.1f}µm "
               f"<span style='color:#b00'>{' | '.join(flags)}</span>")
        rows.append(f"<div style='margin:12px 0;border-top:1px solid #ccc'><p>{txt}</p><img src='data:image/png;base64,{img}' style='max-width:100%'></div>")
    head = f"<h1>PMDB physical clean QC — {C.CODE_VERSION}</h1><p>{len(df)} sites. Targets: <code>{json.dumps(targets) if targets else 'none (norm only)'}</code></p>"
    (out / "qc_report.html").write_text("<html><body style='font-family:sans-serif;font-size:13px'>" + head + "".join(rows) + "</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "clean"))
    ap.add_argument("--sites", default=None, help="comma-separated subset of site ids")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true", help="reuse params.json/norm from a previous run")
    ap.add_argument("--redo-harm", action="store_true", help="with --skip-existing: recompute the harmonisation stage")
    ap.add_argument("--targets", default=None, help="targets.json from a labelled run (held-out data)")
    ap.add_argument("--no-harm", action="store_true")
    ap.add_argument("--labelled-summary", default=None, help="summary.csv of the labelled run (material-flag thresholds for held-out)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sites = list_sites(args.data_root)
    if args.sites:
        keep = set(args.sites.split(","))
        sites = sites[sites.site.isin(keep)]
    print(f"{len(sites)} sites -> {out}", flush=True)

    params_by_site: dict[str, dict] = {}
    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in sites.itertuples():
            site_dir = out / r.batch / r.site
            if args.skip_existing and (site_dir / "params.json").exists() and (site_dir / "SE_type_norm.tif").exists():
                params_by_site[r.site] = json.loads((site_dir / "params.json").read_text())
                continue
            jobs.append((ex.submit(run_norm, r.batch, r.site, {k: str(v) for k, v in _paths(r).items()}, str(site_dir), float(r.nm_per_px)), r.site))
        for fut, site in jobs:
            pass
        for fut in as_completed([j[0] for j in jobs]):
            p = fut.result()
            params_by_site[p["site"]] = p
            a = p["anchors"]["BSE"]
            print(f"  norm {p['batch']}/{p['site']}: D={a['D']:.1f} [{a['D_method']}] G={a['G_level']:.1f} "
                  f"Si/G={a['si_graphite_ratio']:.3f} bands={len(p['bands']['corrected'])}/{len(p['bands']['masked'])} "
                  f"fov={p['fov'].get('collector', {}).get('found')}/{p['fov'].get('free_surface', {}).get('found')} "
                  f"charging={len(p['charging']['local'])}/{p['charging']['broad'].get('found')} {p['seconds_norm']}s", flush=True)

    targets = None
    if not args.no_harm:
        if args.targets:
            targets = json.loads(Path(args.targets).read_text())
        else:
            targets = C.build_targets(params_by_site, REFERENCE_SITES).to_dict()
            (out / "targets.json").write_text(json.dumps(targets, indent=1))
        print("targets:", json.dumps(targets), flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_harm, str(out / r.batch / r.site), targets, float(r.nm_per_px)): r.site for r in sites.itertuples()
                    if not (args.skip_existing and not args.redo_harm and "harmonisation" in params_by_site.get(r.site, {})
                            and (out / r.batch / r.site / "SE_type_harm.tif").exists())}
            for fut in as_completed(futs):
                p = fut.result()
                params_by_site[p["site"]] = p
                h = p["harmonisation"]["BSE"]
                print(f"  harm {p['batch']}/{p['site']}: σe={h['sigma_e_px']:.2f} -> σk={h['sigma_k_px']:.2f} (after {h['sigma_e_after_px']:.2f}) "
                      f"noise added={h['added']} within10%={h['noise_within_10pct']} {p['seconds_harm']}s", flush=True)

    df = pd.DataFrame([flatten(p) for p in params_by_site.values()]).sort_values(["batch", "site"]).reset_index(drop=True)
    labelled = pd.read_csv(args.labelled_summary) if args.labelled_summary else None
    df = material_flags(df, labelled)
    df.to_csv(out / "summary.csv", index=False)
    (out / "material_thresholds.json").write_text(json.dumps(df.attrs["thresholds"], indent=1))
    qc_report(out, df, targets)
    print(f"wrote {out / 'summary.csv'} and {out / 'qc_report.html'}")


if __name__ == "__main__":
    main()
