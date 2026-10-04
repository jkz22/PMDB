"""Is a held-back site a re-image of a labelled field? Peak normalised cross-correlation of BSE at 200 nm/px
(phase correlation for the shift, then NCC on the overlap) against all 31 labelled sites."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import ndimage as ndi
from pmdb.io import list_sites, load_site

ROOT = sys.argv[1] if len(sys.argv) > 1 else ".."


def small(b, s, **kw):
    im = load_site(b, s, resolution="half", normalise="none", **kw).image[..., 0].astype(np.float32)
    im = ndi.uniform_filter(im, 4)[2::4, 2::4]
    return (im - ndi.gaussian_filter(im, 25))  # remove slow shading


def ncc(a, b):
    H, W = max(a.shape[0], b.shape[0]) * 2, max(a.shape[1], b.shape[1]) * 2
    A = np.fft.rfft2((a - a.mean()) / a.std(), (H, W)); B = np.fft.rfft2((b - b.mean()) / b.std(), (H, W))
    R = A * np.conj(B); r = np.fft.irfft2(R / (np.abs(R) + 1e-9), (H, W))
    best = 0.0
    for idx in np.argsort(r.ravel())[-5:]:
        dy, dx = np.unravel_index(idx, r.shape); dy = dy - H if dy > H // 2 else dy; dx = dx - W if dx > W // 2 else dx
        ya, yb = max(dy, 0), max(-dy, 0); xa, xb = max(dx, 0), max(-dx, 0)
        h = min(a.shape[0] - ya, b.shape[0] - yb); w = min(a.shape[1] - xa, b.shape[1] - xb)
        if h < 50 or w < 50: continue
        pa, pb = a[ya:ya+h, xa:xa+w].ravel(), b[yb:yb+h, xb:xb+w].ravel()
        best = max(best, float(np.corrcoef(pa, pb)[0, 1]))
    return best


def job(args):
    hs, b, s = args
    return hs, b, s, ncc(H[hs], small(b, s))


H = {s: small("Batch_heldout", s, data_root=f"{ROOT}/data_heldout", cache_root=f"{ROOT}/cache_heldout")
     for s in pd.read_csv(f"{ROOT}/cache_heldout/half/manifest.csv").site}

if __name__ == "__main__":
    L = list_sites()
    with Pool(8) as p:
        r = p.map(job, [(h, b, s) for h in H for b, s in zip(L.batch, L.site)])
    d = pd.DataFrame(r, columns=["heldout", "batch", "site", "ncc"]).sort_values(["heldout", "ncc"], ascending=[True, False])
    d.to_csv("heldout_overlap.csv", index=False)
    print(d.groupby("heldout").head(3).to_string(index=False))
