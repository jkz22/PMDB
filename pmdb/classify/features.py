"""Feature specs, FEM tile grid, FEM table loader and curated tile features (classifier plan C2-C5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pmdb.kpis import catalogue_columns

BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
HELDOUT_BATCH = "Batch_heldout"
TILE_ID = ["batch", "site", "tile"]
N_FEM_TILES = 6
FEM_TILE_COL = "tile"
FEM_ORIENTATION = "sym"
FEM_FRAMES = 11
FEM_REQUIRED_METRICS = (
    "swelling", "surface_rough", "porosity_rel_change", "pore_closed_frac", "vm_si_p95_MPa",
    "si_yield_frac", "p_si_mean_MPa", "vm_binder_p95_MPa", "vm_gr_p95_MPa", "sxx_mean_MPa",
    "J_si_mean", "band_vm_maxdev",
)
CLOSURE_NEVER = 1.1
GRID_TOL_UM = 0.1

FEM_FEATURES: dict[str, tuple[str, str]] = {
    "fem_swell_50": ("swelling @ frame 5", "electrode thickness strain at 50% SOC"),
    "fem_swell_100": ("swelling @ frame 10", "electrode thickness strain at full charge"),
    "fem_swell_slope_early": ("OLS slope of swelling vs s, frames 0-2",
                              "swelling rate in the Si-dominated stage (s < 0.25)"),
    "fem_swell_slope_late": ("OLS slope of swelling vs s, frames 3-10",
                             "swelling rate once graphite lithiation dominates (s > 0.25)"),
    "fem_surface_rough_100": ("surface_rough @ frame 10",
                              "non-uniformity of the free-surface rise across the tile"),
    "fem_pore_left_100": ("1 + porosity_rel_change @ frame 10",
                          "fraction of the initial pore area still open at full charge"),
    "fem_pore_closed_frac_100": ("pore_closed_frac @ frame 10",
                                 "fraction of pore cells collapsed (J < 0.1) at full charge"),
    "fem_first_closure_s": ("first s with tile pore_closed_frac > 0 (1.1 if never)",
                            "SOC at which the first pore is crushed"),
    "fem_vm_si_p95_100": ("vm_si_p95_MPa @ frame 10", "peak Si von Mises stress (particle fracture driver)"),
    "fem_si_yield_frac_100": ("si_yield_frac @ frame 10", "fraction of Si above its yield stress"),
    "fem_p_si_mean_100": ("p_si_mean_MPa @ frame 10", "mean hydrostatic compression of Si"),
    "fem_vm_binder_p95_100": ("vm_binder_p95_MPa @ frame 10",
                              "peak binder stress, proxy for Si/matrix interface load transfer"),
    "fem_vm_gr_p95_100": ("vm_gr_p95_MPa @ frame 10", "peak graphite stress imposed by swelling Si neighbours"),
    "fem_sxx_mean_100": ("sxx_mean_MPa @ frame 10", "mean in-plane stress (lateral constraint)"),
    "fem_J_si_mean_100": ("J_si_mean @ frame 10", "realised Si volume expansion"),
    "fem_band_vm_maxdev_100": ("band_vm_maxdev @ frame 10", "through-thickness heterogeneity of stress"),
}

KPI_MEANINGS: dict[str, str] = {
    "K01_si_frac_adm": "Si area / non-artefact area (Si loading)",
    "K02_si_density_per_1000um2": "Si objects per 1000 um2",
    "K03_ecd_d50_um": "Si particle equivalent-circle diameter, median",
    "K03_ecd_d90_um": "Si particle equivalent-circle diameter, 90th percentile",
    "K03_ecd_max_um": "Si particle equivalent-circle diameter, maximum",
    "K04_agglom_frac": "share of Si area in agglomerates (clusters with ECD > 5 um)",
    "K04_n_clusters_per_1000um2": "Si cluster count density",
    "K05_voronoi_sigma": "spread of Voronoi cell areas of Si centroids (dispersion non-uniformity)",
    "K07_R_rl": "nearest-neighbour index vs random labelling (< 1 clustered)",
    "K07_R_csr": "nearest-neighbour index vs CSR (< 1 clustered)",
    "K09_mst_m_norm": "normalised mean minimum-spanning-tree edge length over Si",
    "K09_mst_sigma_norm": "normalised SD of minimum-spanning-tree edge length over Si",
    "K14_empty_p50_um": "median distance to the nearest Si (Si-free pockets)",
    "K14_empty_p95_um": "95th percentile distance to the nearest Si (Si-free pockets)",
    "K15_si_graphite_contact_frac": "share of Si boundary touching graphite",
    "K16_si_graphite_dist_median_um": "median Si-to-graphite distance",
}


def fem_grid_slices(width_half: int) -> list[slice]:
    """Half-resolution column slices of the 6 FEM tiles (reproduces fem-build P15), C2."""
    wc = width_half // 2
    e = np.linspace(200, wc - 200, N_FEM_TILES + 1).round().astype(int)
    return [slice(int(2 * a), int(2 * b)) for a, b in zip(e[:-1], e[1:])]


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    m = s.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if m.isna().any():
        raise ValueError("table: 'heldout' column is not boolean")
    return m.astype(bool)


def load_fem_tile_curves(path: str | Path) -> pd.DataFrame:
    """The only code that knows the FEM tile_curves.csv schema (C3). Returns long sym rows."""
    df = pd.read_csv(path)
    need = ["batch", "site", "heldout", "orientation", FEM_TILE_COL, "frame", "s", "tile_x0_um",
            "tile_x1_um", *FEM_REQUIRED_METRICS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"FEM table {path} missing columns: {missing}")
    df = df[df["orientation"] == FEM_ORIENTATION].copy()
    df["heldout"] = _as_bool(df["heldout"])
    df["tile"] = df["tile"].astype(int)
    df["frame"] = df["frame"].astype(int)
    if (df.groupby("site")["batch"].nunique() != 1).any():
        raise ValueError("FEM table: a site maps to more than one batch")
    if (df["heldout"] != (df["batch"] == HELDOUT_BATCH)).any():
        raise ValueError("FEM table: heldout flag disagrees with batch == Batch_heldout")
    if (np.abs(df["s"] - df["frame"] / 10.0) >= 1e-6).any():
        raise ValueError("FEM table: s != frame/10")
    want = {(t, f) for t in range(N_FEM_TILES) for f in range(FEM_FRAMES)}
    for site, g in df.groupby("site"):
        keys = list(zip(g["tile"], g["frame"]))
        if len(keys) != len(want) or set(keys) != want:
            raise ValueError(f"FEM table: site {site} does not have exactly tiles 0..5 x frames 0..10")
    keep = ["batch", "site", "heldout", "tile", "frame", "s", "tile_x0_um", "tile_x1_um",
            *FEM_REQUIRED_METRICS]
    return df[keep].sort_values(["batch", "site", "tile", "frame"]).reset_index(drop=True)


def _ols_slope(s: np.ndarray, y: np.ndarray) -> float:
    return np.nan if np.isnan(y).any() else float(np.polyfit(s, y, 1)[0])


def fem_tile_features(curves: pd.DataFrame) -> pd.DataFrame:
    """One row per (batch, site, tile): 16 curated fem_* features (C4)."""
    rows = []
    for (batch, site, tile), g in curves.groupby(TILE_ID, sort=True):
        g = g.sort_values("frame")
        s = g["s"].to_numpy(float)
        m = {k: g[k].to_numpy(float) for k in FEM_REQUIRED_METRICS}
        sw, pcf = m["swelling"], m["pore_closed_frac"]
        closed = np.flatnonzero(np.nan_to_num(pcf, nan=0.0) > 0)
        rows.append({
            "batch": batch, "site": site, "tile": int(tile), "heldout": bool(g["heldout"].iloc[0]),
            "tile_x0_um": float(g["tile_x0_um"].iloc[0]), "tile_x1_um": float(g["tile_x1_um"].iloc[0]),
            "fem_swell_50": sw[5], "fem_swell_100": sw[10],
            "fem_swell_slope_early": _ols_slope(s[0:3], sw[0:3]),
            "fem_swell_slope_late": _ols_slope(s[3:11], sw[3:11]),
            "fem_surface_rough_100": m["surface_rough"][10],
            "fem_pore_left_100": 1.0 + m["porosity_rel_change"][10],
            "fem_pore_closed_frac_100": pcf[10],
            "fem_first_closure_s": (float(s[closed[0]]) if closed.size
                                    else (CLOSURE_NEVER if np.isfinite(pcf[10]) else np.nan)),
            "fem_vm_si_p95_100": m["vm_si_p95_MPa"][10],
            "fem_si_yield_frac_100": m["si_yield_frac"][10],
            "fem_p_si_mean_100": m["p_si_mean_MPa"][10],
            "fem_vm_binder_p95_100": m["vm_binder_p95_MPa"][10],
            "fem_vm_gr_p95_100": m["vm_gr_p95_MPa"][10],
            "fem_sxx_mean_100": m["sxx_mean_MPa"][10],
            "fem_J_si_mean_100": m["J_si_mean"][10],
            "fem_band_vm_maxdev_100": m["band_vm_maxdev"][10],
        })
    return pd.DataFrame(rows).sort_values(TILE_ID).reset_index(drop=True)


def kpi_tile_columns() -> list[str]:
    _, tile_cols = catalogue_columns()
    return [c for group in tile_cols.values() for c in group]


def kpi_feature_columns(kpi_tiles: pd.DataFrame) -> list[str]:
    """C5: catalogue tile columns minus those with zero nan-std over labelled rows."""
    lab = kpi_tiles[~kpi_tiles["heldout"]]
    return [c for c in kpi_tile_columns() if np.nanstd(lab[c].to_numpy(float)) > 0]


def load_kpi_tiles6(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["heldout"] = _as_bool(df["heldout"])
    counts = df.groupby("site").size()
    if not (counts == N_FEM_TILES).all():
        raise ValueError(f"expected {N_FEM_TILES} tiles per site, got {counts[counts != N_FEM_TILES].to_dict()}")
    return df.sort_values(TILE_ID).reset_index(drop=True)


def build_arm_table(arm: str, kpi: pd.DataFrame | None, fem: pd.DataFrame | None
                    ) -> tuple[pd.DataFrame, list[str]]:
    """Tile table (TILE_ID, heldout, x-range, features) and feature list for one arm (C15)."""
    if arm not in ("KPI", "FEM", "KPI+FEM"):
        raise ValueError(f"unknown arm {arm}")
    if arm in ("KPI", "KPI+FEM") and kpi is None:
        raise ValueError("KPI table required")
    if arm in ("FEM", "KPI+FEM") and fem is None:
        raise ValueError("FEM table required")
    fem_cols = list(FEM_FEATURES)
    base = ["batch", "site", "tile", "heldout", "tile_x0_um", "tile_x1_um"]
    if arm == "KPI":
        feats = kpi_feature_columns(kpi)
        return kpi[base + feats].sort_values(TILE_ID).reset_index(drop=True), feats
    if arm == "FEM":
        return fem[base + fem_cols].sort_values(TILE_ID).reset_index(drop=True), fem_cols
    feats = kpi_feature_columns(kpi)
    j = kpi[base + feats].merge(fem[TILE_ID + ["tile_x0_um", "tile_x1_um"] + fem_cols], on=TILE_ID,
                                suffixes=("", "_fem"), how="inner")
    n_sites = kpi["site"].nunique()
    if j["site"].nunique() != n_sites or len(j) != n_sites * N_FEM_TILES:
        raise ValueError("KPI+FEM join lost sites or tiles: FEM and KPI site sets differ")
    for c in ("tile_x0_um", "tile_x1_um"):
        bad = (j[c] - j[c + "_fem"]).abs() > GRID_TOL_UM
        if bad.any():
            r = j[bad].iloc[0]
            raise ValueError(f"tile grid mismatch {c}: {r['batch']}/{r['site']} tile {r['tile']}: "
                             f"KPI {r[c]} vs FEM {r[c + '_fem']}")
    j = j.drop(columns=["tile_x0_um_fem", "tile_x1_um_fem"])
    return j.sort_values(TILE_ID).reset_index(drop=True), feats + fem_cols


def feature_catalogue(features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        if f in FEM_FEATURES:
            rows.append({"feature": f, "arm_source": "FEM", "source": FEM_FEATURES[f][0],
                         "meaning": FEM_FEATURES[f][1]})
        else:
            rows.append({"feature": f, "arm_source": "KPI", "source": "outputs/classifier/kpi_tiles6.csv",
                         "meaning": KPI_MEANINGS.get(f, "")})
    return pd.DataFrame(rows)
