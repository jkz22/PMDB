"""Tile and site curve features from FEM fields (P15-P17, D8 tiers 1, 3, 4). Pure numpy/pandas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmdb.fem.materials import BINDER, GRAPHITE, PORE, SI, lithiation_state, si_yield_MPa
from pmdb.fem.result import SimResult
from pmdb.kpis.fields import band_profile, band_summary

TIER1 = ("swelling", "surface_rough", "sxx_mean_MPa", "syy_mean_MPa", "porosity", "porosity_change",
         "porosity_rel_change", "J_si_mean", "J_gr_mean", "J_binder_mean", "vm_si_p50_MPa",
         "vm_si_p95_MPa", "vm_gr_p95_MPa", "vm_binder_p95_MPa", "p_si_mean_MPa", "si_yield_frac",
         "pore_closed_frac")
QUANTILES = (5, 25, 50, 75, 95, 99)
_PHASES = (("si", SI), ("gr", GRAPHITE), ("binder", BINDER))
TIER3 = tuple(f"q{q}_{f}_{ph}" for f in ("vm", "p", "J") for ph, _ in _PHASES for q in QUANTILES) + tuple(
    f"q{q}_J_pore" for q in QUANTILES)
TIER4 = ("band_vm_maxdev", "band_vm_absslope", "band_J_maxdev", "band_J_absslope")
METRIC_NAMES = TIER1 + TIER3 + TIER4
META_COLUMNS = ("batch", "site", "heldout", "orientation", "tile", "tile_x0_um", "tile_x1_um", "frame", "s",
                "converged", "failed_at_s", "first_pore_closure_s")


def fem_tile_slices(width_px: int, px_um: float, n_tiles: int = 6, edge_um: float = 20.0) -> list[slice]:
    edge_px = int(round(edge_um / px_um))
    edges = np.linspace(edge_px, width_px - edge_px, n_tiles + 1).round().astype(int)
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def _nan_metrics() -> dict[str, float]:
    return {k: float("nan") for k in METRIC_NAMES}


def _porosity(J: np.ndarray, pore: np.ndarray) -> float:
    tot = float(J.sum(dtype=np.float64))
    return float(J[pore].sum(dtype=np.float64)) / tot if tot > 0 else float("nan")


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x, dtype=np.float64)) if x.size else float("nan")


def _quant(x: np.ndarray, qs) -> list[float]:
    if x.size == 0:
        return [float("nan")] * len(qs)
    return [float(v) for v in np.percentile(x, qs)]


def region_metrics(r: SimResult, frame: int, cols: slice, orientation: str, p: dict) -> dict[str, float]:
    if not bool(r.converged[frame]):
        return _nan_metrics()
    H, W = r.labels.shape
    h = r.px_um
    H_um = H * h
    lab = r.labels[:, cols]
    F = {k: v[frame][:, cols] for k, v in r.fields.items()}
    out = _nan_metrics()

    # Tier 1: free-edge surface
    n_free = 1.0 if orientation == "bottom" else -1.0
    row = 0 if orientation == "bottom" else H
    uz = n_free * r.u_nodes[frame, row, cols.start:cols.stop + 1, 1].astype(np.float64)
    out["swelling"] = float(uz.mean() / H_um)
    out["surface_rough"] = float(uz.std() / H_um)
    out["sxx_mean_MPa"] = _mean(F["sxx"])
    out["syy_mean_MPa"] = _mean(F["syy"])

    pore = lab == PORE
    por = _porosity(F["J"], pore)
    por0 = _porosity(r.fields["J"][0][:, cols], pore)
    out["porosity"] = por
    out["porosity_change"] = por - por0
    out["porosity_rel_change"] = por / por0 - 1.0 if por0 and np.isfinite(por0) else float("nan")

    pres = -(F["sxx"] + F["szz"] + F["syy"]) / 3.0
    fields = {"vm": F["vm"], "p": pres, "J": F["J"]}
    masks = {ph: lab == code for ph, code in _PHASES}

    out["J_si_mean"] = _mean(F["J"][masks["si"]])
    out["J_gr_mean"] = _mean(F["J"][masks["gr"]])
    out["J_binder_mean"] = _mean(F["J"][masks["binder"]])
    vm_si, vm_gr, vm_b = F["vm"][masks["si"]], F["vm"][masks["gr"]], F["vm"][masks["binder"]]
    out["vm_si_p50_MPa"] = _quant(vm_si, [50])[0]
    out["vm_si_p95_MPa"] = _quant(vm_si, [95])[0]
    out["vm_gr_p95_MPa"] = _quant(vm_gr, [95])[0]
    out["vm_binder_p95_MPa"] = _quant(vm_b, [95])[0]
    out["p_si_mean_MPa"] = _mean(pres[masks["si"]])
    if vm_si.size:
        u, _ = lithiation_state(float(r.s[frame]), p)
        out["si_yield_frac"] = float(np.mean(vm_si > si_yield_MPa(u, p)))
    if pore.any():
        out["pore_closed_frac"] = float(np.mean(F["J"][pore] < p["pore"]["closure_J"]))

    # Tier 3
    for f, arr in fields.items():
        for ph, _ in _PHASES:
            vals = _quant(arr[masks[ph]], list(QUANTILES))
            for q, v in zip(QUANTILES, vals):
                out[f"q{q}_{f}_{ph}"] = v
    for q, v in zip(QUANTILES, _quant(F["J"][pore], list(QUANTILES))):
        out[f"q{q}_J_pore"] = v

    # Tier 4 (magnitude only; solid cells)
    solid = np.isin(lab, (SI, GRAPHITE, BINDER))
    if solid.any():
        nb = int(p["features"]["n_depth_bands"])
        solid_f = solid.astype(np.float64)
        for name, arr in (("vm", F["vm"]), ("J", F["J"])):
            _, maxdev, absslope = band_summary(band_profile(arr.astype(np.float64) * solid_f, solid_f, nb))
            out[f"band_{name}_maxdev"] = maxdev
            out[f"band_{name}_absslope"] = absslope
    return out


def _first_pore_closure_s(r: SimResult, p: dict) -> float:
    pore = r.labels == PORE
    if not pore.any():
        return float("nan")
    for i, s in enumerate(r.s):
        if bool(r.converged[i]) and np.any(r.fields["J"][i][pore] < p["pore"]["closure_J"]):
            return float(s)
    return float("nan")


def run_curves(r: SimResult, orientation: str, p: dict, window: bool) -> tuple[list[dict], list[dict]]:
    H, W = r.labels.shape
    feat = p["features"]
    fpc = _first_pore_closure_s(r, p)
    if window:
        tiles: list[slice] = []
        site_cols = slice(0, W)
    else:
        tiles = fem_tile_slices(W, r.px_um, feat["n_tiles"], feat["edge_um"])
        site_cols = slice(tiles[0].start, tiles[-1].stop)

    def head(i: int) -> dict:
        return {"frame": i, "s": float(r.s[i]), "converged": bool(r.converged[i]),
                "failed_at_s": float(r.failed_at_s), "first_pore_closure_s": fpc}

    site_rows, tile_rows = [], []
    for i in range(len(r.s)):
        site_rows.append({**head(i), **region_metrics(r, i, site_cols, orientation, p)})
        for t, sl in enumerate(tiles):
            tile_rows.append({**head(i), "tile": t, "tile_x0_um": sl.start * r.px_um,
                              "tile_x1_um": sl.stop * r.px_um, **region_metrics(r, i, sl, orientation, p)})
    return site_rows, tile_rows


def symmetrise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Append orientation="sym" rows: per-metric mean of bottom and top (P17)."""
    metric_cols = [c for c in df.columns if c not in META_COLUMNS]
    b = df[df["orientation"] == "bottom"].set_index(keys)
    t = df[df["orientation"] == "top"].set_index(keys)
    t = t.reindex(b.index)
    sym = b.copy()
    sym["orientation"] = "sym"
    for c in metric_cols:
        sym[c] = (b[c].astype(float) + t[c].astype(float)) / 2.0  # NaN propagates
    if "converged" in b:
        sym["converged"] = b["converged"].astype(bool) & t["converged"].fillna(False).astype(bool)
    for c in ("failed_at_s", "first_pore_closure_s"):
        if c in b:
            sym[c] = np.fmin(b[c].astype(float), t[c].astype(float))
    sym = sym.reset_index()[list(df.columns)]
    return pd.concat([df, sym], ignore_index=True)


def swelling_gate(swelling_sym: float, p: dict) -> dict:
    g = p["gates"]
    if swelling_sym is None or not np.isfinite(swelling_sym):
        return {"swelling_sym": float("nan"), "gate_ok": False, "lit_band_ok": False}
    lo, hi = g["swelling_stop"]
    blo, bhi = g["swelling_lit_band"]
    return {"swelling_sym": float(swelling_sym), "gate_ok": bool(lo <= swelling_sym <= hi),
            "lit_band_ok": bool(blo <= swelling_sym <= bhi)}
