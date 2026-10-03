"""Per-site KPIs + per-KPI spatial GP for every PMDB site. Writes site_kpis.csv."""
import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from pmdb.io import list_sites, load_site
from seg import segment

TILE = 64  # px at 50 nm/px = 3.2 µm
TILE_KPIS = ["porosity", "si_frac", "graphite_frac", "interface_um_per_um2", "inlens_texture"]
MIN_D_UM = 1.0        # ignore sub-µm Si specks (noise / partial volume)
CLUSTER_GAP_UM = 1.0  # particles closer than this belong to one cluster
CRACK_HOLE_FRAC = 0.02; CRACK_SOLIDITY = 0.80   # cracked/irregular particle rule
ANOM_D_UM = 8.0; ANOM_AR = 3.0                 # anomalous = oversized or elongated


def tile_table(lab, um, hp):
    edges = ndi.morphological_gradient(lab, size=3) > 0
    H, W = lab.shape; rows = []
    for i in range(0, H - TILE + 1, TILE):
        for j in range(0, W - TILE + 1, TILE):
            t = lab[i:i+TILE, j:j+TILE]; e = edges[i:i+TILE, j:j+TILE]; h = hp[i:i+TILE, j:j+TILE]
            # edge pixels are ~2 px wide -> boundary length ≈ count/2 px
            rows.append([(i+TILE/2)*um, (j+TILE/2)*um, (t == 0).mean(), (t == 2).mean(), (t == 1).mean(),
                         e.sum() / 2 * um / (TILE*um) ** 2, h.std()])
    r = np.array(rows); return r[:, :2], r[:, 2:]


def gp_mean(X, y):
    mu0, sd0 = y.mean(), y.std() + 1e-12
    z = (y - mu0) / sd0
    k = C(1.0, (1e-3, 1e2)) * RBF(5.0, (0.5, 200.0)) + WhiteKernel(0.5, (1e-4, 2.0))
    gp = GaussianProcessRegressor(k, n_restarts_optimizer=1, random_state=0).fit(X, z)
    K = gp.kernel_(X); Ki1 = np.linalg.solve(K, np.ones(len(z))); v = 1 / Ki1.sum()
    p = gp.kernel_.get_params()
    return dict(mean=mu0 + sd0 * v * (Ki1 @ z), err=sd0 * np.sqrt(v), naive_err=y.std() / np.sqrt(len(y)),
                ls_um=p["k1__k2__length_scale"],
                spatial_frac=p["k1__k1__constant_value"] / (p["k1__k1__constant_value"] + p["k2__noise_level"]))


def inlens_maps(inl):
    """Per-image relative Inlens texture: high-pass and gradient, divided by the image's own 5-95% range."""
    f = inl.astype(np.float32); rng = np.percentile(f, 95) - np.percentile(f, 5) + 1e-6
    return (f - ndi.gaussian_filter(f, 2)) / rng, ndi.gaussian_gradient_magnitude(f, 1) / rng


def particles(lab, um):
    m = lab == 2; L = measure.label(m); area_um2 = lab.size * um * um
    props = [p for p in measure.regionprops(L) if p.equivalent_diameter_area * um >= MIN_D_UM]
    nan = float("nan")
    d = np.asarray([p.equivalent_diameter_area for p in props], float) * um
    sol = np.array([p.solidity for p in props])
    ar = np.array([p.major_axis_length / max(p.minor_axis_length, 1e-6) for p in props])
    circ = np.clip([4 * np.pi * p.area / max(p.perimeter, 1e-6) ** 2 for p in props], 0, 1)
    holes = np.array([(p.area_filled - p.area) / p.area_filled for p in props])
    cracked = (holes >= CRACK_HOLE_FRAC) | (sol < CRACK_SOLIDITY)
    anom = (d > ANOM_D_UM) | (ar > ANOM_AR)
    if len(props) < 3:
        return dict.fromkeys(["si_d10_um", "si_d50_um", "si_d90_um", "si_psd_span", "si_aspect_ratio", "si_circularity",
                              "si_solidity", "si_nn_um", "si_clark_evans", "si_cluster_size", "si_cracked_frac",
                              "si_anomalous_per_1000um2"], nan) | dict(si_count_per_1000um2=len(props) / area_um2 * 1000,
                                                                   si_n_cracked=int(cracked.sum()), si_n_anomalous=int(anom.sum())), {}
    c = np.array([p.centroid for p in props]) * um; n = len(c); lam = n / area_um2
    nn = cKDTree(c).query(c, k=2)[0][:, 1]
    keep = np.isin(L, [p.label for p in props])
    CL = measure.label(ndi.distance_transform_edt(~keep) <= CLUSTER_GAP_UM / 2 / um)
    cid = np.array([CL[tuple(p.coords[0])] for p in props]); size = np.bincount(cid)[cid]
    d10, d50, d90 = np.percentile(d, [10, 50, 90])
    kp = dict(si_d10_um=d10, si_d50_um=d50, si_d90_um=d90, si_psd_span=(d90 - d10) / d50,
              si_count_per_1000um2=n / area_um2 * 1000, si_aspect_ratio=float(np.median(ar)),
              si_circularity=float(np.median(circ)), si_solidity=float(np.median(sol)),
              si_nn_um=float(nn.mean()), si_clark_evans=float(nn.mean() / (0.5 / np.sqrt(lam))),
              si_cluster_size=float(size.mean()), si_n_cracked=int(cracked.sum()), si_cracked_frac=float(cracked.mean()),
              si_n_anomalous=int(anom.sum()), si_anomalous_per_1000um2=float(anom.sum() / area_um2 * 1000))
    arrs = dict(p_d=d, p_ar=ar, p_circ=circ, p_cracked=cracked, p_anom=anom,
                p_bbox=np.array([p.bbox for p in props]))
    return kp, arrs


def run(row):
    s = load_site(row["batch"], row["site"], resolution="half", normalise="none")
    um = s.nm_per_px / 1000
    lab, info = segment(s.image[..., 0])
    hp, grad = inlens_maps(s.image[..., 1])
    X, Y = tile_table(lab, um, hp)
    out = dict(batch=row["batch"], site=row["site"], se_detector=row["se_detector"], n_tiles=len(Y),
               bse_black=info["black"], bse_graphite=info["graphite"], si_thr=info["si_thr"], si_peak=info["si_peak"])
    for n, k in enumerate(TILE_KPIS):
        for a, v in gp_mean(X, Y[:, n]).items():
            out[f"{k}" if a == "mean" else f"{k}__{a}"] = v
    kp, arrs = particles(lab, um); out.update(kp)
    out["inlens_edge_binder"] = float(grad[lab == 1].mean())
    np.savez_compressed(f"tiles/{row['batch']}__{row['site']}.npz", X=X, Y=Y, lab=lab[::2, ::2], **arrs)
    print(row["batch"], row["site"], "done", flush=True)
    return out


if __name__ == "__main__":
    import os; os.makedirs("tiles", exist_ok=True)
    m = list_sites().to_dict("records")
    with Pool(8) as p:
        res = p.map(run, m)
    pd.DataFrame(res).to_csv("site_kpis.csv", index=False)
