"""Per-site KPIs + per-KPI spatial GP for every PMDB site. Writes site_kpis.csv."""
import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage import measure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from pmdb.io import list_sites, load_site
from seg import segment

TILE = 64  # px at 50 nm/px = 3.2 µm
TILE_KPIS = ["porosity", "si_frac", "graphite_frac", "interface_um_per_um2"]


def tile_table(lab, um):
    edges = ndi.morphological_gradient(lab, size=3) > 0
    H, W = lab.shape; rows = []
    for i in range(0, H - TILE + 1, TILE):
        for j in range(0, W - TILE + 1, TILE):
            t = lab[i:i+TILE, j:j+TILE]; e = edges[i:i+TILE, j:j+TILE]
            # edge pixels are ~2 px wide -> boundary length ≈ count/2 px
            rows.append([(i+TILE/2)*um, (j+TILE/2)*um, (t == 0).mean(), (t == 2).mean(), (t == 1).mean(),
                         e.sum() / 2 * um / (TILE*um) ** 2])
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


def particles(lab, um):
    props = measure.regionprops(measure.label(lab == 2))
    d = np.array([p.equivalent_diameter_area * um for p in props])
    sol = np.array([p.solidity for p in props])
    area_um2 = lab.size * um * um
    big = d >= 1.0  # ignore sub-µm specks (noise / partial-volume)
    return dict(si_d50_um=float(np.median(d[big])) if big.any() else np.nan,
                si_d90_um=float(np.percentile(d[big], 90)) if big.any() else np.nan,
                si_count_per_1000um2=float(big.sum() / area_um2 * 1000),
                si_solidity=float(np.median(sol[big])) if big.any() else np.nan)


def run(row):
    s = load_site(row["batch"], row["site"], resolution="half", normalise="none")
    um = s.nm_per_px / 1000
    lab, info = segment(s.image[..., 0])
    X, Y = tile_table(lab, um)
    out = dict(batch=row["batch"], site=row["site"], se_detector=row["se_detector"], n_tiles=len(Y),
               bse_black=info["black"], bse_graphite=info["graphite"], si_thr=info["si_thr"], si_peak=info["si_peak"])
    for n, k in enumerate(TILE_KPIS):
        for a, v in gp_mean(X, Y[:, n]).items():
            out[f"{k}" if a == "mean" else f"{k}__{a}"] = v
    out.update(particles(lab, um))
    np.savez_compressed(f"tiles/{row['batch']}__{row['site']}.npz", X=X, Y=Y, lab=lab[::2, ::2])
    print(row["batch"], row["site"], "done", flush=True)
    return out


if __name__ == "__main__":
    import os; os.makedirs("tiles", exist_ok=True)
    m = list_sites().to_dict("records")
    with Pool(8) as p:
        res = p.map(run, m)
    pd.DataFrame(res).to_csv("site_kpis.csv", index=False)
