"""Where in the image are the outliers? For flagged sites: tiles whose KPI exceeds the 99th percentile of
all tiles from the other sites (red boxes), cracked/irregular Si particles (magenta), anomalous (cyan), and crops."""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import measure, segmentation
from pmdb.io import load_site
from seg import segment, overlay
from kpis import TILE, TILE_KPIS, MIN_D_UM
from compare import hotspot_sites, KPI_INFO

DEFAULT_SITES = [("Batch_1", "4ih2ggld", "si_frac"), ("Batch_1", "5n1q8atc", "si_frac"),
         ("Batch_3", "hzumfsms", "porosity"), ("Batch_2", "epqdaau9", "si_frac")]
SITES = hotspot_sites(json.load(open("compare.json")), DEFAULT_SITES, [k for k in TILE_KPIS if k in KPI_INFO])
NAME = {"si_frac": "Si area fraction", "porosity": "porosity"}
d = pd.read_csv("site_kpis.csv"); T = {(r.batch, r.site): np.load(f"tiles/{r.batch}__{r.site}.npz") for r in d.itertuples()}
summary = {}
for b, s, k in SITES:
    j = TILE_KPIS.index(k); z = T[(b, s)]
    ref = np.concatenate([v["Y"][:, j] for key, v in T.items() if key != (b, s)])
    thr = np.percentile(ref, 99); y = z["Y"][:, j]; hot = y > thr
    site = load_site(b, s, resolution="half", normalise="none"); um = site.nm_per_px / 1000
    bse = site.image[..., 0]; lab, _ = segment(bse); ov = overlay(bse, lab)
    L = measure.label(lab == 2)
    props = [p for p in measure.regionprops(L) if p.equivalent_diameter_area * um >= MIN_D_UM]
    for key, color in (("p_cracked", [1, 0, 1]), ("p_anom", [0, 1, 1])):
        if key not in z: continue   # sparse sites have no particle arrays
        ids = [p.label for p, f in zip(props, z[key]) if f]
        ov[segmentation.find_boundaries(np.isin(L, ids), mode="outer")] = color
    rc = np.rint(z["X"] / um - TILE / 2).astype(int)   # tile top-left corner (row, col) in px
    H, W = bse.shape; ext = [0, W * um, H * um, 0]
    top = np.argsort(-np.where(hot, y, -np.inf))[:4]; top = [t for t in top if hot[t]]
    fig = plt.figure(figsize=(20, 2 * 20 * H / W + 3.4))
    gs = fig.add_gridspec(3, 8, height_ratios=[H / W * 8, H / W * 8, 1.1])
    for row, img, title in ((0, bse, "BSE"), (1, ov, "segmentation (blue pore, orange Si, magenta cracked/irregular, cyan anomalous)")):
        a = fig.add_subplot(gs[row, :]); a.imshow(img, cmap="gray", vmin=0, vmax=255, extent=ext)
        for t in np.where(hot)[0]:
            a.add_patch(Rectangle((rc[t, 1] * um, rc[t, 0] * um), TILE * um, TILE * um, fill=False, ec="red", lw=1.4))
        for n, t in enumerate(top):
            a.text(rc[t, 1] * um, rc[t, 0] * um - .4, str(n + 1), color="red", fontsize=10, weight="bold")
        a.set_title(f"{b} {s}: {title}" + (f" — red = {NAME.get(k, KPI_INFO[k][0])} tile above 99th pct of other sites ({thr:.3f})" if row == 0 else ""), fontsize=10)
        a.set_xlabel("µm"); a.set_ylabel("µm")
    for n, t in enumerate(top):
        r0, c0 = max(rc[t, 0] - TILE // 2, 0), max(rc[t, 1] - TILE // 2, 0); sl = (slice(r0, r0 + 2 * TILE), slice(c0, c0 + 2 * TILE))
        for m, img in enumerate((bse, ov)):
            a = fig.add_subplot(gs[2, 2 * n + m]); a.imshow(img[sl], cmap="gray", vmin=0, vmax=255); a.set_xticks([]); a.set_yticks([])
            a.add_patch(Rectangle((rc[t, 1] - c0, rc[t, 0] - r0), TILE, TILE, fill=False, ec="red", lw=1.2))
            if m == 0: a.set_title(f"#{n + 1}: {NAME.get(k, KPI_INFO[k][0])} {y[t]:.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(f"fig_spots_{s}.jpg", dpi=60, pil_kwargs=dict(quality=82)); plt.close(fig)
    summary[s] = dict(batch=b, kpi=k, threshold=float(thr), n_tiles=int(len(y)), n_hot=int(hot.sum()),
                      pct_hot=float(100 * hot.mean()), expected_pct=1.0, top_values=[float(y[t]) for t in top],
                      n_cracked=int(z["p_cracked"].sum()) if "p_cracked" in z else 0,
                      n_anomalous=int(z["p_anom"].sum()) if "p_anom" in z else 0,
                      hot_share_of_excess=float(np.clip(y[hot] - np.median(ref), 0, None).sum() / max(np.clip(y - np.median(ref), 0, None).sum(), 1e-9)))
    print(s, summary[s], flush=True)
json.dump(summary, open("spots.json", "w"), indent=1)
