"""Phase 0.4 KPI gates on the teammate code (segmenter + registry), end to end.

G1 phantom: synthetic BSE with known Si/graphite/pore masks and disc diameters.
    Pass: each mask fraction within +-1 pp of truth, K03 D50 within +-5 % of truth.
G2 upscale: each baseline field bicubic-upscaled 1.2x at fixed nm/px.
    Expect D50 x1.20 (+-5 % relative) and mask fractions unchanged (+-1 pp).
G3 resolution: native 25 nm/px vs cache 50 nm/px for every field.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage

from src.v2 import kpi_adapter as K
from src.v2.common import NM_FULL, NM_HALF, OUT, load_full_raw, load_half_raw, manifest

D = OUT / "kpi_gates"
FRAC_TOL = 0.01
D50_TOL = 0.05


def _discs(shape, centres, radii):
    m = np.zeros(shape, bool)
    lab_area = []
    for (r, c), rad in zip(centres, radii):
        R = int(np.ceil(rad))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        d = yy ** 2 + xx ** 2 <= rad ** 2
        sl = (slice(r - R, r + R + 1), slice(c - R, c + R + 1))
        m[sl] |= d
        lab_area.append(d.sum())
    return m, np.asarray(lab_area)


def phantom(seed=0, size=2000, noise=0.03, psf_sigma=0.8):
    """Graphite matrix (mid grey) with bright Si discs (r 5-10 px) and dark pores (r 12 px)."""
    rng = np.random.default_rng(seed)
    pts, rads = [], []
    target = 1400
    while len(pts) < target:
        p = rng.integers(20, size - 20, 2)
        rad = 12.0 if len(pts) < 300 else float(rng.uniform(5, 10))
        if all(np.hypot(*(p - q)) > rad + r2 + 6 for q, r2 in zip(pts, rads)):
            pts.append(p)
            rads.append(rad)
    pts, rads = np.asarray(pts), np.asarray(rads)
    pore, _ = _discs((size, size), pts[:300], rads[:300])
    si, si_area = _discs((size, size), pts[300:], rads[300:])
    img = np.full((size, size), 0.45)
    img[si] = 0.85
    img[pore] = 0.08
    img = ndimage.gaussian_filter(img, psf_sigma) + noise * rng.standard_normal(img.shape)
    raw = np.clip(img * 255, 0, 255).astype(np.uint8)
    truth = {"frac_si": si.mean(), "frac_pore": pore.mean(), "frac_graphite": 1 - si.mean() - pore.mean(),
             "K03_ecd_d50_um": float(np.percentile(2 * np.sqrt(si_area / np.pi) * NM_HALF / 1000, 50)),
             "K01_si_frac_adm": si.mean()}  # d23a116+: denominator = ~artefact (no artefact here)
    return raw, truth


def g1():
    rows = []
    for noise in (0.02, 0.05):
        raw, truth = phantom(noise=noise)
        got = K.kpis_from_masks(K.segment(raw, NM_HALF), NM_HALF, "phantom")
        for k, t in truth.items():
            g = got[k]
            tol_ok = abs(g - t) / t <= D50_TOL if k == "K03_ecd_d50_um" else abs(g - t) <= FRAC_TOL
            rows.append(dict(noise=noise, metric=k, truth=t, measured=g, passed=bool(tol_ok)))
    return pd.DataFrame(rows)


def _g2_site(args):
    b, s = args
    bse = load_half_raw(b, s)[..., 0]
    up = ndimage.zoom(bse.astype(np.float64), 1.2, order=3)
    a = K.kpis_from_masks(K.segment(bse, NM_HALF), NM_HALF, f"{b}/{s}")
    u = K.kpis_from_masks(K.segment(up, NM_HALF), NM_HALF, f"{b}/{s}/up")
    return dict(batch=b, site=s, **{f"base_{k}": v for k, v in a.items()}, **{f"up_{k}": v for k, v in u.items()})


def _g3_site(args):
    b, s = args
    full = load_full_raw(b, s)[..., 0]
    half = load_half_raw(b, s)[..., 0]
    f = K.kpis_from_masks(K.segment(full, NM_FULL), NM_FULL, f"{b}/{s}")
    h = K.kpis_from_masks(K.segment(half, NM_HALF), NM_HALF, f"{b}/{s}")
    return dict(batch=b, site=s, **{f"full_{k}": v for k, v in f.items()}, **{f"half_{k}": v for k, v in h.items()})


def main(which=("g1", "g2", "g3")):
    D.mkdir(parents=True, exist_ok=True)
    m = manifest()
    summary = {"kpi_commit": K.kpi_commit_hash()}
    if "g1" in which:
        r = g1(); r.to_csv(D / "g1_phantom.csv", index=False)
        summary["g1_pass"] = bool(r.passed.all()); print(r.round(4).to_string())
    with ProcessPoolExecutor(4) as ex:
        if "g2" in which:
            base = m[m.batch == "Batch_1"][["batch", "site"]].itertuples(index=False, name=None)
            r = pd.DataFrame(list(ex.map(_g2_site, base)))
            r["d50_ratio"] = r.up_K03_ecd_d50_um / r.base_K03_ecd_d50_um
            for c in ("frac_si", "frac_graphite", "frac_pore"):
                r[f"d_{c}_pp"] = 100 * (r[f"up_{c}"] - r[f"base_{c}"])
            r.to_csv(D / "g2_upscale.csv", index=False)
            ok = (r.d50_ratio.sub(1.2).abs() / 1.2 <= D50_TOL) & (r.filter(like="_pp").abs() <= 100 * FRAC_TOL).all(axis=1)
            summary["g2_pass"] = bool(ok.all()); summary["g2_d50_ratio_mean"] = float(r.d50_ratio.mean())
            print(r.filter(regex="site|ratio|_pp").round(3).to_string())
        if "g3" in which:
            r = pd.DataFrame(list(ex.map(_g3_site, m[["batch", "site"]].itertuples(index=False, name=None))))
            r.to_csv(D / "g3_resolution.csv", index=False)
            agg = []
            for c in K.CROP_COLS:
                d = r[f"half_{c}"] - r[f"full_{c}"]
                agg.append(dict(kpi=c, mean_full=r[f"full_{c}"].mean(), mean_half=r[f"half_{c}"].mean(),
                                mean_diff=d.mean(), rel_diff=(d / r[f"full_{c}"]).mean(),
                                spearman=r[f"half_{c}"].corr(r[f"full_{c}"], method="spearman")))
            agg = pd.DataFrame(agg); agg.to_csv(D / "g3_resolution_summary.csv", index=False)
            print(agg.round(4).to_string())
    (D / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("g1", "g2", "g3"))
