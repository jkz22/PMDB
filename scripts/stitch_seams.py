"""Seam test: do crops cut from one parent electrode image abut?

Runs on Modal (full-res TIFFs in the `pmdb-fullres` volume, uploaded from data/ and
data_heldout/Batch_heldout/ and test-day data_test/Batch_test/):

    modal run scripts/stitch_seams.py            # all 34 sites
    modal run scripts/stitch_seams.py --smoke    # one same-parent pair

Method (see outputs/stitching/REPORT.md):
  * Each crop edge is summarised by an edge strip (DEPTH px deep) taken after skipping the
    artefact border (SKIP_LR columns on left/right, SKIP_TB rows on top/bottom).
  * Profile = mean over the M px nearest the edge, per channel (BSE, Inlens, ETD|SE).
    hp = band-pass (minus Gaussian sigma 25, plus Gaussian sigma 1): particle-scale texture.
    mp = minus Gaussian sigma 25, plus Gaussian sigma 4: coarser particle-scale band.
    lp = Gaussian sigma 25 low-pass: shading / layer-scale continuity.
  * Score = channel-mean Pearson r between the two edge profiles, maximised over the
    along-edge shift (|s| <= 64 px for left/right edges, any shift with >= 2000 px overlap
    for top/bottom edges).
  * Positive control: each crop split internally at known columns/rows with a known gap,
    scored with the exact same function and search.
  * Null: different-parent pairs and same-parent wrong placements (computed locally from
    the returned table by scripts/stitch_report.py).
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
VOL_MOUNT = "/vol"
SKIP_LR, SKIP_TB, DEPTH, M = 3, 1, 8, 4
LR_SHIFT, TB_MIN_OVERLAP = 64, 2000
CTRL_GAPS = (0, 6, 12, 24, 48, 96)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4", "scipy==1.14.1", "pandas==2.1.4",
                 "tifffile==2025.5.10", "imagecodecs==2025.3.30")
)
volume = modal.Volume.from_name("pmdb-fullres")
app = modal.App("pmdb-stitch", image=image)

GROUPS = {
    "G1612": ["ptg8lmto", "xgj4xftb", "0eryguqq", "fhwrjtet"],
    "G1904": ["0grcilhi", "hawkfj64", "mgxahqnk"],
    "G2048": ["avn74qx1", "3806gxp0", "fn0mhxef"],
    "G2060": ["71vgq3fw", "tuy3zymq", "x7u69zsw", "kbdh4tri"],
    "G2068SE": ["rxax5ozo", "x77cy643", "utfgcjfa", "vc2whyaq"],
    "G2080": ["ffwubibz", "r17byphk", "cfe5vt7s"],
    "G2088": ["ufdvpb81", "hzumfsms", "9luzk4jm", "xrv9xvzb"],
    "G2148": ["f1vzngrs", "epqdaau9", "4hq27w4c", "fspqbkxl"],
    "G2156": ["fzrt2k6r", "b3esycq1", "soo2ax3r"],
    "G2272": ["pl8uabbv", "i9jiqjwl"],
    "G2316": ["4ih2ggld", "5n1q8atc", "3e122cbj"],
    "G1780": ["iv6g2oq0"],
    "G1880": ["uhdslk0o", "y59rxmxl"],
}
PARENT = {s: g for g, ss in GROUPS.items() for s in ss}
VERTICAL_EDGES = ("L", "R")  # profile runs along y
UNFLIPPED = {("R", "L"): "A|B", ("L", "R"): "B|A", ("B", "T"): "A/B", ("T", "B"): "B/A"}


# ----------------------------------------------------------------- scoring helpers
def _profiles(strip):
    """strip (len, depth, 3) -> hp, lp profiles (len, 3)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    p = strip[:, :M, :].mean(1).astype(np.float64)
    lp = gaussian_filter1d(p, 25, axis=0)
    hp = gaussian_filter1d(p - lp, 1.0, axis=0)
    mp = gaussian_filter1d(p - lp, 4.0, axis=0)
    return hp, mp, lp


def _ncc_shifts(a, b):
    """Pearson r for every shift s (b[j] <-> a[j+s]), s in [-(m-1), n-1]. a (n,), b (m,)."""
    import numpy as np
    from scipy.signal import correlate
    n, m = len(a), len(b)
    shifts = np.arange(-(m - 1), n)
    lo_a = np.maximum(shifts, 0)
    hi_a = np.minimum(n, m + shifts)
    lo_b = np.maximum(0, -shifts)
    hi_b = np.minimum(m, n - shifts)
    N = (hi_a - lo_a).astype(np.float64)
    ca, caa = np.concatenate([[0], np.cumsum(a)]), np.concatenate([[0], np.cumsum(a * a)])
    cb, cbb = np.concatenate([[0], np.cumsum(b)]), np.concatenate([[0], np.cumsum(b * b)])
    Sa, Saa = ca[hi_a] - ca[lo_a], caa[hi_a] - caa[lo_a]
    Sb, Sbb = cb[hi_b] - cb[lo_b], cbb[hi_b] - cbb[lo_b]
    # correlate(a, b, 'full')[k] = sum_j a[j + k - (m-1)] b[j]  -> k = s + m - 1
    Sab = correlate(a, b, mode="full", method="fft")
    num = Sab - Sa * Sb / N
    den = np.sqrt(np.clip((Saa - Sa ** 2 / N) * (Sbb - Sb ** 2 / N), 1e-12, None))
    return shifts, num / den, N


def score_edges(sa, sb, vertical):
    """Channel-mean r between two edge strips; best over the allowed shift window."""
    import numpy as np
    ha, ma, la = _profiles(sa)
    hb, mb, lb = _profiles(sb)
    rh, rm, rl = [], [], []
    for c in range(3):
        shifts, r, N = _ncc_shifts(ha[:, c], hb[:, c])
        rh.append(r)
        rm.append(_ncc_shifts(ma[:, c], mb[:, c])[1])
        rl.append(_ncc_shifts(la[:, c], lb[:, c])[1])
    rh, rm, rl = np.mean(rh, 0), np.mean(rm, 0), np.mean(rl, 0)
    ok = (np.abs(shifts) <= LR_SHIFT) if vertical else (N >= TB_MIN_OVERLAP)
    idx = np.where(ok)[0]
    k = idx[np.argmax(rh[idx])]
    km = idx[np.argmax(rm[idx])]
    z0 = np.where(shifts == 0)[0][0]
    return {"hp": float(rh[k]), "shift": int(shifts[k]), "hp0": float(rh[z0]),
            "mp": float(rm[km]), "mp_shift": int(shifts[km]), "mp0": float(rm[z0]),
            "lp_at": float(rl[k]), "lp0": float(rl[z0])}


# ----------------------------------------------------------------- Modal functions
def _load(batch, site):
    import glob

    import numpy as np
    import tifffile
    chans = []
    for ch in ("BSE", "Inlens", "ETD|SE"):
        f = [p for c in ch.split("|")
             for p in glob.glob(f"{VOL_MOUNT}/data/{batch}/img_{site}_{c}.tif")]
        chans.append(tifffile.imread(f[0])[..., 1])
    return np.stack(chans, -1).astype(np.float32)  # (H, W, 3)


@app.function(volumes={VOL_MOUNT: volume}, cpu=1.0, memory=4096, timeout=900)
def extract(batch: str, site: str) -> dict:
    """Write edge strips to the volume; return artefact info + positive-control scores."""
    import numpy as np
    img = _load(batch, site)
    H, W, _ = img.shape
    # artefact border columns: deviation of column mean from the next 20 columns (any channel)
    colm = img.mean(0)
    art_l = [int(i) for i in range(6)
             if np.abs(colm[i] - colm[8:28].mean(0)).max() > 10]
    art_r = [int(i) for i in range(6)
             if np.abs(colm[W - 1 - i] - colm[W - 28:W - 8].mean(0)).max() > 10]
    strips = {
        "L": img[:, SKIP_LR:SKIP_LR + DEPTH],
        "R": img[:, W - SKIP_LR - DEPTH:W - SKIP_LR][:, ::-1],
        "T": img[SKIP_TB:SKIP_TB + DEPTH].transpose(1, 0, 2),
        "B": img[H - SKIP_TB - DEPTH:H - SKIP_TB][::-1].transpose(1, 0, 2),
    }
    Path(f"{VOL_MOUNT}/strips").mkdir(exist_ok=True)
    np.savez(f"{VOL_MOUNT}/strips/{site}.npz", **{k: np.ascontiguousarray(v) for k, v in strips.items()})
    volume.commit()
    ctrl = []
    for frac in (0.25, 0.5, 0.75):
        for g in CTRL_GAPS:
            x = int(W * frac)
            a = img[:, x - DEPTH:x][:, ::-1]
            b = img[:, x + g:x + g + DEPTH]
            ctrl.append({"site": site, "orient": "LR", "frac": frac, "gap": g,
                         **score_edges(a, b, True)})
            y = int(H * frac)
            a = img[y - DEPTH:y][::-1].transpose(1, 0, 2)
            b = img[y + g:y + g + DEPTH].transpose(1, 0, 2)
            ctrl.append({"site": site, "orient": "TB", "frac": frac, "gap": g,
                         **score_edges(a, b, False)})
    return {"site": site, "batch": batch, "H": H, "W": W, "art_left": art_l,
            "art_right": art_r, "ctrl": ctrl}


@app.function(volumes={VOL_MOUNT: volume}, cpu=1.0, memory=4096, timeout=1800)
def score_site(site_a: str, others: list[str]) -> list[dict]:
    """All 16 edge combinations (4 unflipped placements + flips) of site_a vs each other site."""
    import numpy as np
    volume.reload()
    load = lambda s: dict(np.load(f"{VOL_MOUNT}/strips/{s}.npz"))  # noqa: E731
    A = load(site_a)
    rows = []
    for site_b in others:
        B = load(site_b)
        for ea in "LRTB":
            vert = ea in VERTICAL_EDGES
            for eb in (VERTICAL_EDGES if vert else ("T", "B")):
                for rev in (False, True):
                    sb = B[eb][::-1] if rev else B[eb]
                    place = UNFLIPPED.get((ea, eb)) if not rev else None
                    rows.append({"a": site_a, "b": site_b, "ea": ea, "eb": eb, "rev": rev,
                                 "placement": place or "flip",
                                 "same_parent": PARENT[site_a] == PARENT[site_b],
                                 **score_edges(A[ea], sb, vert)})
    return rows


@app.local_entrypoint()
def main(smoke: bool = False):
    import pandas as pd
    sites = []
    for m in [ROOT / "cache/half/manifest.csv", ROOT / "cache_heldout/half/manifest.csv",
              ROOT / "cache_test/half/manifest.csv"]:
        df = pd.read_csv(m)
        sites += list(zip(df["batch"], df["site"]))
    if smoke:
        sites = [s for s in sites if s[1] in ("4ih2ggld", "5n1q8atc")]
    out = ROOT / "outputs/stitching"
    out.mkdir(parents=True, exist_ok=True)
    info = list(extract.starmap(sites))
    ctrl = pd.DataFrame([c for i in info for c in i["ctrl"]])
    meta = pd.DataFrame([{k: v for k, v in i.items() if k != "ctrl"} for i in info])
    meta["parent"] = meta["site"].map(PARENT)
    names = [s for _, s in sites]
    args = [(names[i], names[i + 1:]) for i in range(len(names) - 1)]
    pairs = pd.DataFrame([r for rows in score_site.starmap(args) for r in rows])
    sfx = "_smoke" if smoke else ""
    ctrl.to_csv(out / f"controls{sfx}.csv", index=False)
    meta.to_csv(out / f"sites{sfx}.csv", index=False)
    pairs.to_csv(out / f"pair_scores{sfx}.csv", index=False)
    print(meta.to_string())
    print(ctrl.groupby(["orient", "gap"])[["hp", "hp0", "mp", "mp0", "lp0"]].median().round(3).to_string())
    print(pairs.groupby(["same_parent", "placement"])["hp"].describe().round(3).to_string())
