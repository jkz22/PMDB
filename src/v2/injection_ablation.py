"""Does each harmonisation route actually remove the known microscope effects?

Inject the documented SEM artefacts into the *raw full-resolution* TIFFs of clean (unflagged)
sites, push the corrupted images through every route exactly as the real data goes, and measure
how much of the injected effect survives in the route's output:

    residual = |stat(route(injected)) - stat(route(original))| / |stat(injected raw) - stat(original raw)|

0 = the route removed the effect completely, 1 = it passed straight through, > 1 = amplified.
Statistics are the imaging fingerprints a classifier can read (black level p1, noise sigma,
Laplacian sharpness) plus the fixed-threshold BSE phase fractions (what a KPI would see) and the
pixel-wise RMS change on jointly valid pixels.

Routes: raw (none) | PR #16 LUT `affine2` and `hybrid` (refitted on the injected image, as the
builder would) | physical clean `norm` and `harm` (full pmdb.clean pipeline, same targets).
Injections (full-res grey levels): strong session (+22 offset, 0.69x gain), mild offset (+6),
Gaussian noise sigma 6, blur sigma 1 px, 3-DN requantisation, and all of them together.

python -m src.v2.injection_ablation [--sites 4ih2ggld 3806gxp0 iv6g2oq0] [--workers 4]
-> outputs/v2/injection_ablation/{ablation_long.csv, residual.csv, residual.png}
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, gaussian_filter, laplace

from pmdb import clean as C
from pmdb import harmonise as H
from pmdb.io import REPO_ROOT, downsample_clean, downsample_to_half, list_sites

from src.v2.common import OUT
from src.v2.materialise_clean import DETS, Z_SCALE

OUT_DIR = OUT / "injection_ablation"
IM_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
PORE_T, SI_T = 0.5, 1.4  # fixed physical thresholds on z (pore ~0, graphite 1, Si > 1.4)


def inject(raw: dict[str, np.ndarray], name: str, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for d, I in raw.items():
        x = I.astype(np.float32)
        if name in ("blur1.0", "all_strong"):
            x = gaussian_filter(x, 1.0)
        if name == "session_strong" or name == "all_strong":
            x = 0.69 * x + 22.0
        if name == "offset6":
            x = x + 6.0
        if name in ("noise6", "all_strong"):
            x = x + rng.normal(0, 6.0, x.shape).astype(np.float32)
        if name in ("requant3", "all_strong"):
            x = np.round(x / 3.0) * 3.0
        out[d] = np.clip(np.round(x), 0, 255).astype(np.uint8)
    return out


INJECTIONS = ("none", "session_strong", "offset6", "noise6", "blur1.0", "requant3", "all_strong")


def fingerprint(a: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    """Imaging statistics on a half-res uint8 channel (same definitions as src.v2.imaging_stats)."""
    a = a.astype(np.float32)
    if valid is not None and not valid.all():
        a = np.where(valid, a, np.nan)
        fill = float(np.nanmedian(a))
        a = np.where(np.isnan(a), fill, a).astype(np.float32)
    p1, p99 = np.nanpercentile(a, [1, 99])
    lap = laplace(a)[1:-1, 1:-1]
    h, w = a.shape
    noise = float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(convolve(a, IM_K)[1:-1, 1:-1]).sum())
    # noise-robust focus measure: Laplacian variance after a sigma=1.5 px smoothing (pixel noise suppressed,
    # structure-scale blur retained), normalised by the dynamic range
    lap_s = laplace(gaussian_filter(a, 1.5))[3:-3, 3:-3]
    return dict(p1=float(p1), p99=float(p99), noise_sigma=noise, sharpness=float(lap.var()) / max(float(p99 - p1), 1.0) ** 2,
                sharp_robust=float(lap_s.var()) / max(float(p99 - p1), 1.0) ** 2)


def restat(batch: str, site: str, inj: str, ref: H.Reference) -> list[dict]:
    """Recompute all statistics from a saved condition (no re-processing)."""
    z = np.load(OUT_DIR / f"{site}__{inj}.npz")
    stats = {}
    for r in ("raw", "affine2", "hybrid"):
        stats[r] = {d: fingerprint(z[r][..., i]) for i, d in enumerate(DETS)}
        stats[r]["BSE"].update(phase_fracs_lut(z[r][..., 0], ref))
    for kind in ("clean_norm", "clean_harm"):
        zh, va = z[f"{kind}_z"], z[f"{kind}_valid"]
        ims = [np.round(np.clip(zh[..., i], 0, 255.0 / Z_SCALE[d]) * Z_SCALE[d]).astype(np.uint8) for i, d in enumerate(DETS)]
        stats[kind] = {d: fingerprint(ims[i], va[..., i]) for i, d in enumerate(DETS)}
        stats[kind]["BSE"].update(phase_fracs_z(zh[..., 0], va[..., 0]))
        stats[kind]["BSE"]["valid_frac"] = float(va[..., 0].mean())
    return [dict(batch=batch, site=site, injection=inj, route=r, detector=d, **s) for r, per in stats.items() for d, s in per.items()]


def phase_fracs_z(z: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    v = z[valid]
    return dict(frac_pore=float((v < PORE_T).mean()), frac_si=float((v > SI_T).mean()),
                frac_graphite=float(((v >= PORE_T) & (v <= SI_T)).mean()))


def phase_fracs_lut(bse_u8: np.ndarray, ref: H.Reference) -> dict[str, float]:
    """Fixed thresholds in the LUT's reference grey scale (midpoints between reference anchors)."""
    a = ref.anchors["BSE"]
    t_p, t_s = (a["pore"] + a["graphite"]) / 2, (a["graphite"] + a["si"]) / 2
    v = bse_u8.astype(np.float32)
    return dict(frac_pore=float((v < t_p).mean()), frac_si=float((v > t_s).mean()),
                frac_graphite=float(((v >= t_p) & (v <= t_s)).mean()))


def _raw_half(raw: dict[str, np.ndarray]) -> np.ndarray:
    W = raw["BSE"].shape[1]
    stack = np.stack([raw[d][:, 4:W - 4] for d in DETS], -1)
    return downsample_to_half(stack)


def one_condition(batch: str, site: str, paths: dict[str, str], nm_per_px: float, inj: str, targets: dict,
                  lut_methods=("affine2", "hybrid")) -> tuple[dict, dict]:
    """Return (stats[route][detector], arrays[route]) for one site x injection."""
    raw, markers = {}, None
    for d, p in paths.items():
        img, cols = C.read_rgb_tiff(p)
        raw[d] = img
        markers = cols if markers is None else (markers | cols)
    raw = inject(raw, inj)
    stats, arrays = {}, {}
    # raw + LUT routes at half res
    half = _raw_half(raw)
    stats["raw"] = {d: fingerprint(half[..., i]) for i, d in enumerate(DETS)}
    stats["raw"]["BSE"].update(phase_fracs_lut(half[..., 0], H.load_reference(REPO_ROOT / "cache", "affine2")))
    arrays["raw"] = half
    anchors, hist = H.estimate_anchors(half, 50.0)
    for m in lut_methods:
        ref = H.load_reference(REPO_ROOT / "cache", m)
        luts, _ = H.fit_lut(m, anchors, hist, ref)
        out = H.apply_lut(half, luts)
        stats[m] = {d: fingerprint(out[..., i]) for i, d in enumerate(DETS)}
        stats[m]["BSE"].update(phase_fracs_lut(out[..., 0], ref))
        arrays[m] = out
    # physical clean route (full pipeline), then materialised like cache/clean
    res = C.clean_site(raw, marker_cols=markers, nm_per_px=nm_per_px)
    harm = C.harmonise_site(res, C.Targets.from_dict(targets), nm_per_px=nm_per_px)
    for kind, zs in (("clean_norm", res.norm), ("clean_harm", harm.harm)):
        ims, vals, zh = [], [], []
        for d in DETS:
            z, m = zs[d], res.mask[d]
            W = z.shape[1]
            z2, m2 = downsample_clean(z[:, 4:W - 4], m[:, 4:W - 4])
            v = C.valid_for_kpis(m2)
            sc = Z_SCALE[d]
            ims.append(np.round(np.clip(z2, 0, 255.0 / sc) * sc).astype(np.uint8)); vals.append(v); zh.append(z2)
        stats[kind] = {d: fingerprint(ims[i], vals[i]) for i, d in enumerate(DETS)}
        stats[kind]["BSE"].update(phase_fracs_z(zh[0], vals[0]))
        stats[kind]["BSE"]["valid_frac"] = float(vals[0].mean())
        arrays[kind] = (np.stack(zh, -1), np.stack(vals, -1))
    return stats, arrays


def _job(args):
    batch, site, paths, nm, inj, targets = args
    stats, arrays = one_condition(batch, site, paths, nm, inj, targets)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{site}__{inj}.npz", raw=arrays["raw"], affine2=arrays["affine2"], hybrid=arrays["hybrid"],
                        clean_norm_z=arrays["clean_norm"][0], clean_norm_valid=arrays["clean_norm"][1],
                        clean_harm_z=arrays["clean_harm"][0], clean_harm_valid=arrays["clean_harm"][1])
    rows = [dict(batch=batch, site=site, injection=inj, route=r, detector=d, **s)
            for r, per in stats.items() for d, s in per.items()]
    print(site, inj, "done", flush=True)
    return rows


def rms_change(out_dir: Path, sites, injections) -> pd.DataFrame:
    """Pixel-wise RMS change of each route's output, injected vs original, on jointly valid pixels,
    in units of the route's own scale (grey levels for raw/LUT, z for clean)."""
    rows = []
    for s in sites:
        base = np.load(out_dir / f"{s}__none.npz")
        for inj in injections:
            if inj == "none":
                continue
            f = out_dir / f"{s}__{inj}.npz"
            if not f.exists():
                continue
            cur = np.load(f)
            for r in ("raw", "affine2", "hybrid"):
                for i, d in enumerate(DETS):
                    a, b = base[r][..., i].astype(np.float32), cur[r][..., i].astype(np.float32)
                    rows.append(dict(site=s, injection=inj, route=r, detector=d, rms=float(np.sqrt(((a - b) ** 2).mean())),
                                     unit="grey"))
            for r in ("clean_norm", "clean_harm"):
                v = base[f"{r}_valid"] & cur[f"{r}_valid"]
                for i, d in enumerate(DETS):
                    a, b = base[f"{r}_z"][..., i], cur[f"{r}_z"][..., i]
                    rows.append(dict(site=s, injection=inj, route=r, detector=d, rms=float(np.sqrt(((a - b)[v[..., i]] ** 2).mean())),
                                     unit="z"))
    return pd.DataFrame(rows)


def site_spread() -> pd.DataFrame:
    """Between-site SD of every statistic per route over the 31 un-injected labelled sites (from the
    caches each route is trained from), so a residual effect can be read in 'site SDs': an injected
    artefact that moves a statistic by < 0.5 site-SD after the route is smaller than the natural
    field-to-field variation and cannot be a batch shortcut."""
    from src.v2.common import CACHE, load_half_clean, manifest
    rows = []
    ref = H.load_reference(REPO_ROOT / "cache", "affine2")
    for r in manifest().itertuples():
        raw = np.load(CACHE / f"{r.batch}__{r.site}.npz")["image"]
        for route in ("raw", "affine2", "hybrid"):
            im = raw if route == "raw" else np.load(REPO_ROOT / "cache" / "harmonised" / route / "half" / f"{r.batch}__{r.site}.npz")["image"]
            for i, d in enumerate(DETS):
                st = fingerprint(im[..., i])
                if d == "BSE":
                    st.update(phase_fracs_lut(im[..., 0], ref))
                rows.append(dict(site=r.site, route=route, detector=d, **st))
        for route in ("clean_norm", "clean_harm"):
            im, va = load_half_clean(r.batch, r.site, route)
            for i, d in enumerate(DETS):
                st = fingerprint(im[..., i], va[..., i])
                if d == "BSE":
                    st.update(phase_fracs_z(im[..., 0].astype(np.float32) / Z_SCALE["BSE"], va[..., 0]))
                rows.append(dict(site=r.site, route=route, detector=d, **st))
    return pd.DataFrame(rows).groupby(["route", "detector"]).std(numeric_only=True)


def residuals(long: pd.DataFrame, spread: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per site/detector/injection/stat/route: the change the route's output shows (delta_route), the
    change in the raw image (delta_raw), their ratio (residual; only meaningful when the injection
    moved the raw statistic) and delta_route in between-site SDs of that route (delta_route_sd)."""
    stats = ["p1", "noise_sigma", "sharpness", "sharp_robust", "frac_pore", "frac_graphite", "frac_si"]
    rows = []
    for (site, det), g in long.groupby(["site", "detector"]):
        base = g[g.injection == "none"].set_index("route")
        for inj in g.injection.unique():
            if inj == "none":
                continue
            cur = g[g.injection == inj].set_index("route")
            for st in stats:
                if st not in cur or pd.isna(cur.loc["raw", st]):
                    continue
                d_raw = cur.loc["raw", st] - base.loc["raw", st]
                for r in cur.index:
                    d_r = cur.loc[r, st] - base.loc[r, st]
                    sd = float(spread.loc[(r, det), st]) if spread is not None else np.nan
                    sd_raw = float(spread.loc[("raw", det), st]) if spread is not None else np.nan
                    rows.append(dict(site=site, detector=det, injection=inj, stat=st, route=r,
                                     delta_raw=d_raw, delta_route=d_r, delta_raw_sd=d_raw / sd_raw if sd_raw else np.nan,
                                     delta_route_sd=d_r / sd if sd else np.nan,
                                     residual=abs(d_r) / abs(d_raw) if abs(d_raw) > 1e-9 else np.nan))
    return pd.DataFrame(rows)


def plot(res: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    routes = ["raw", "affine2", "hybrid", "clean_norm", "clean_harm"]
    stats = ["p1", "noise_sigma", "sharpness", "sharp_robust", "frac_pore", "frac_si"]
    injs = [i for i in INJECTIONS if i != "none"]
    fig, axes = plt.subplots(1, len(stats), figsize=(4.2 * len(stats), 4.5), sharey=True)
    for ax, st in zip(axes, stats):
        g = res[(res.stat == st) & (res.detector == "BSE")].assign(a=lambda d: d.delta_route_sd.abs()).groupby(["injection", "route"]).a.median().unstack("route")
        g = g.reindex(index=injs, columns=routes)
        im = ax.imshow(np.clip(g.to_numpy(float), 0, 3), vmin=0, vmax=3, cmap="RdYlGn_r")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.iloc[i, j]
                ax.text(j, i, "-" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(routes))); ax.set_xticklabels(routes, rotation=45, ha="right")
        ax.set_yticks(range(len(injs))); ax.set_yticklabels(injs)
        ax.set_title(f"BSE {st}")
    fig.colorbar(im, ax=axes, fraction=0.02, label="|change after route| in between-site SDs (0 = removed, >1 = larger than natural field spread)")
    fig.suptitle("Injection ablation: how much of each injected SEM artefact survives each harmonisation route (BSE, median over sites)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=["4ih2ggld", "3806gxp0", "iv6g2oq0"])
    ap.add_argument("--injections", nargs="+", default=list(INJECTIONS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--restat", action="store_true", help="recompute statistics from saved arrays only")
    a = ap.parse_args()
    man = list_sites().set_index("site")
    targets = json.loads((REPO_ROOT / "outputs" / "clean" / "targets.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in a.sites:
        r = man.loc[s]
        paths = {"BSE": str(r.path_bse), "Inlens": str(r.path_inlens), "SE_type": str(r.path_se_type)}
        for inj in a.injections:
            if a.skip_existing and (OUT_DIR / f"{s}__{inj}.npz").exists():
                continue
            jobs.append((r.batch, s, paths, float(r.nm_per_px), inj, targets))
    rows = []
    if a.restat:
        ref = H.load_reference(REPO_ROOT / "cache", "affine2")
        for s_ in a.sites:
            for inj in a.injections:
                rows += restat(man.loc[s_].batch, s_, inj, ref)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            for rr in ex.map(_job, jobs):
                rows += rr
    long = pd.DataFrame(rows)
    f = OUT_DIR / "ablation_long.csv"
    if a.skip_existing and not a.restat and f.exists():
        old = pd.read_csv(f)
        long = pd.concat([old[~old.set_index(["site", "injection"]).index.isin(long.set_index(["site", "injection"]).index)], long])
    long.to_csv(f, index=False)
    spread = site_spread()
    spread.to_csv(OUT_DIR / "site_spread.csv")
    res = residuals(long, spread)
    res.to_csv(OUT_DIR / "residual.csv", index=False)
    rms = rms_change(OUT_DIR, a.sites, a.injections)
    rms.to_csv(OUT_DIR / "rms_change.csv", index=False)
    plot(res, OUT_DIR / "residual.png")
    b = res[res.detector == "BSE"]
    print("median |delta after route| in between-site SDs (BSE):")
    print(b.assign(a=b.delta_route_sd.abs()).groupby(["stat", "injection", "route"]).a.median().unstack("route").round(2).to_string())
    print("median residual fraction of the raw change (BSE):")
    print(b.groupby(["stat", "injection", "route"]).residual.median().unstack("route").round(2).to_string())


if __name__ == "__main__":
    main()
