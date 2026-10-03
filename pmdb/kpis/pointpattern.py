"""Point-pattern Si KPIs on object centroids: K05-K09 (D-013, D-014, D-016)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay, QhullError, cKDTree

from pmdb.kpis.common import NAN, KpiContext, KpiOutput, safe_cv

K06_CLUSTER_CUT = 0.5
K06_VOID_CUT = 2.0
PCF_R_UM = np.round(np.arange(1, 41) * 0.25, 4)  # 0.25 .. 10 um (D-016)
PCF_SECTOR_DEG = 30.0
PCF_STOYAN = 0.15  # Epanechnikov half-width h = PCF_STOYAN / sqrt(lambda) (Stoyan & Stoyan 1994; spatstat default)
ENV_LO_PCT, ENV_HI_PCT = 2.5, 97.5


# ---------------------------------------------------------------- Voronoi (D-014)

def _border_labels(owner: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([owner[0, :], owner[-1, :], owner[:, 0], owner[:, -1]]))


def voronoi_cell_areas(points_rc: np.ndarray, admissible: np.ndarray) -> np.ndarray:
    """Discrete Voronoi of point seeds: admissible-pixel area of every non-border cell."""
    shape = admissible.shape
    r = np.clip(np.rint(points_rc[:, 0]).astype(np.int64), 0, shape[0] - 1)
    c = np.clip(np.rint(points_rc[:, 1]).astype(np.int64), 0, shape[1] - 1)
    seeds = np.zeros(shape, dtype=np.int32)
    seeds[r, c] = np.arange(1, len(r) + 1, dtype=np.int32)
    n = len(r)
    ir, ic = ndimage.distance_transform_edt(seeds == 0, return_distances=False, return_indices=True)
    owner = seeds[ir, ic]
    areas = np.bincount(owner[admissible], minlength=n + 1).astype(np.float64)
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    keep[_border_labels(owner)] = False
    keep &= areas > 0
    return areas[keep]


def voronoi_sigma(areas: np.ndarray) -> float:
    if areas.size < 2:
        return NAN
    return float((areas / areas.mean()).std())


def skiz_local_area_fraction(si_labels: np.ndarray, n: int, admissible: np.ndarray) -> np.ndarray:
    """Per-object Si area / SKIZ cell admissible area, border cells dropped (K05, D-014)."""
    si = si_labels > 0
    ir, ic = ndimage.distance_transform_edt(~si, return_distances=False, return_indices=True)
    owner = si_labels[ir, ic]
    cell = np.bincount(owner[admissible], minlength=n + 1).astype(np.float64)
    obj = np.bincount(si_labels[si], minlength=n + 1).astype(np.float64)
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    keep[_border_labels(owner)] = False
    keep &= cell > 0
    return obj[keep] / cell[keep]


def k05_voronoi(ctx: KpiContext, with_null: bool = True) -> KpiOutput:
    ctx.require_objects()
    areas = voronoi_cell_areas(ctx.centroids_rc, ctx.masks.admissible)
    sigma = voronoi_sigma(areas)
    out = {"K05_voronoi_sigma": sigma}
    if with_null:
        null = np.array([voronoi_sigma(voronoi_cell_areas(p, ctx.masks.admissible)) for p in ctx.null_points_rc])
        mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
        out["K05_voronoi_sigma_null_mean"] = mu
        out["K05_voronoi_sigma_z"] = float((sigma - mu) / sd) if sd > 0 else NAN
        lab, n = ctx.si_labels
        out["K05_local_af_cv"] = safe_cv(skiz_local_area_fraction(lab, n, ctx.masks.admissible))
    return KpiOutput(out)


def k05_voronoi_tile(ctx: KpiContext) -> KpiOutput:
    return k05_voronoi(ctx, with_null=False)


def k06_voronoi_regions(ctx: KpiContext) -> KpiOutput:
    """Area fractions of *all* admissible space in non-border Voronoi cells with normalised
    area < K06_CLUSTER_CUT (cluster) / > K06_VOID_CUT (void); border cells count in the
    denominator only (catalogue K06, D-014)."""
    ctx.require_objects()
    areas = voronoi_cell_areas(ctx.centroids_rc, ctx.masks.admissible)
    norm = areas / areas.mean()
    tot = float(ctx.masks.admissible.sum())
    return KpiOutput({
        "K06_cluster_region_frac": float(areas[norm < K06_CLUSTER_CUT].sum() / tot),
        "K06_void_region_frac": float(areas[norm > K06_VOID_CUT].sum() / tot),
    })


# ---------------------------------------------------------------- nearest neighbour (K07)

def mean_nn_distance(points: np.ndarray) -> float:
    d, _ = cKDTree(points).query(points, k=2)
    return float(d[:, 1].mean())


def k07_nearest_neighbour(ctx: KpiContext) -> KpiOutput:
    ctx.require_objects()
    px = ctx.px_um
    obs = mean_nn_distance(ctx.centroids_rc * px)
    null = np.mean([mean_nn_distance(p * px) for p in ctx.null_points_rc])
    n = ctx.n_objects
    expected_csr = 0.5 * np.sqrt(ctx.image_area_um2 / n)
    return KpiOutput({"K07_R_rl": float(obs / null), "K07_R_csr": float(obs / expected_csr)})


# ---------------------------------------------------------------- pair correlation (K08)

def directional_pair_counts(points_um: np.ndarray, r_grid: np.ndarray = PCF_R_UM,
                            sector_deg: float = PCF_SECTOR_DEG,
                            bandwidth_um: float = 0.5) -> dict[str, np.ndarray]:
    """Kernel-smoothed counts of pairs at separation r within +-sector of x and of z.

    ``points_um`` columns are (z, x) = (row, col) in um.
    """
    tree = cKDTree(points_um)
    pairs = tree.query_pairs(r_grid.max() + bandwidth_um, output_type="ndarray")
    out = {}
    if len(pairs) == 0:
        return {"x": np.zeros_like(r_grid), "z": np.zeros_like(r_grid)}
    dv = points_um[pairs[:, 1]] - points_um[pairs[:, 0]]
    dz, dx = np.abs(dv[:, 0]), np.abs(dv[:, 1])
    dist = np.hypot(dz, dx)
    ang = np.degrees(np.arctan2(dz, dx))
    for name, sel in (("x", ang <= sector_deg), ("z", ang >= 90.0 - sector_deg)):
        d = dist[sel]
        u = (r_grid[:, None] - d[None, :]) / bandwidth_um
        k = np.where(np.abs(u) < 1.0, 0.75 * (1.0 - u * u), 0.0) / bandwidth_um
        out[name] = k.sum(axis=1)
    return out


def pcf_bandwidth_um(n: int, admissible_area_um2: float) -> float:
    """Stoyan's rule of thumb with intensity taken over admissible space."""
    return PCF_STOYAN / np.sqrt(n / admissible_area_um2)


def k08_pair_correlation(ctx: KpiContext) -> KpiOutput:
    """g relative to random labelling: g(r) = C_obs(r) / mean_sims C_sim(r) (D-013, D-016)."""
    ctx.require_objects()
    px = ctx.px_um
    h = pcf_bandwidth_um(ctx.n_objects, ctx.admissible_area_um2)
    obs = directional_pair_counts(ctx.centroids_rc * px, bandwidth_um=h)
    sims = [directional_pair_counts(p * px, bandwidth_um=h) for p in ctx.null_points_rc]
    values: dict[str, float] = {}
    curves: list[tuple[str, float, float]] = []
    for direction in ("x", "z"):
        sim_c = np.array([s[direction] for s in sims])
        denom = sim_c.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g_obs = np.where(denom > 0, obs[direction] / denom, np.nan)
            g_sim = np.where(denom > 0, sim_c / denom, np.nan)
        env_lo = np.percentile(g_sim, ENV_LO_PCT, axis=0)
        env_hi = np.percentile(g_sim, ENV_HI_PCT, axis=0)
        diff = g_obs - env_hi
        if np.all(np.isnan(diff)):
            raise ValueError(f"K08: no pairs within {PCF_R_UM.max()} um in direction {direction}")
        i = int(np.nanargmax(diff))
        excess = max(0.0, float(diff[i]))
        values[f"K08_pcf_excess_max_{direction}"] = excess
        values[f"K08_pcf_rpeak_{direction}_um"] = float(PCF_R_UM[i]) if excess > 0 else -1.0
        for name, arr in (("g_obs", g_obs), ("env_lo", env_lo), ("env_hi", env_hi)):
            curves.extend((f"{name}_{direction}", float(r), float(v)) for r, v in zip(PCF_R_UM, arr))
    return KpiOutput(values, curves)


# ---------------------------------------------------------------- MST (K09)

def mst_edge_lengths(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n < 2:
        return np.zeros(0)
    try:
        tri = Delaunay(points)
        s = tri.simplices
        e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    except QhullError:
        ii, jj = np.triu_indices(n, k=1)
        e = np.column_stack([ii, jj])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.linalg.norm(points[e[:, 0]] - points[e[:, 1]], axis=1)
    g = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return minimum_spanning_tree(g).data


def k09_mst(ctx: KpiContext) -> KpiOutput:
    ctx.require_objects()
    edges = mst_edge_lengths(ctx.centroids_rc * ctx.px_um)
    scale = np.sqrt(ctx.admissible_area_um2 / ctx.n_objects)
    return KpiOutput({
        "K09_mst_m_norm": float(edges.mean() / scale),
        "K09_mst_sigma_norm": float(edges.std() / scale),
    })
