"""Case runner: one (site, orientation) simulation -> result dict (P23, Step 7). Needs dolfinx (runs in the image)."""

from __future__ import annotations

import resource
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from pmdb.fem.config import case_params_hash, load_params
from pmdb.fem.features import run_curves
from pmdb.fem.geometry import central_cols, coarsen_image, coarsen_labels, labels_from_masks
from pmdb.fem.gif import render_site_gif
from pmdb.fem.materials import phase_properties
from pmdb.fem.result import save_npz
from pmdb.fem.solver import BCSpec, simulate
from pmdb.io import load_site
from pmdb.segment import segment


def _versions() -> dict:
    import dolfinx
    import petsc4py
    import scipy
    import skimage

    return {"python": sys.version.split()[0], "dolfinx": dolfinx.__version__, "petsc4py": petsc4py.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__}


def run_case(batch: str, site: str, orientation: Literal["bottom", "top"], *,
             cache_root: str | Path | None = None, res_nm: float | None = None, crop_um: float | None = None,
             fields_path: Path | None = None, render_gif: bool = False,
             solver_overrides: dict | None = None, log=print) -> dict:
    p = load_params()
    if solver_overrides:
        p["solver"].update(solver_overrides)
    lateral = p["solver"].pop("lateral", "both")  # "left" = right edge free: crops of a strip may expand in x
    s_obj = load_site(batch, site, resolution="half", normalise="none", cache_root=cache_root)
    masks = segment(s_obj)
    labels = labels_from_masks(masks)
    bse = np.asarray(s_obj.image[..., 0], dtype=np.float32)
    if crop_um:
        cols = central_cols(labels.shape[1], crop_um, s_obj.nm_per_px)
        labels, bse = labels[:, cols], bse[:, cols]
    res = float(res_nm or p["mesh"]["res_nm"])
    factor = int(round(res / s_obj.nm_per_px))
    labels = coarsen_labels(labels, factor)
    bse = coarsen_image(bse, factor)
    px_um = factor * s_obj.nm_per_px / 1000.0
    H, W = labels.shape

    r = simulate(labels, px_um, lambda s: phase_properties(s, p), BCSpec(orientation, lateral), p["solver"],
                 np.linspace(0.0, 1.0, p["soc"]["frames"]), extra_targets=(p["soc"]["s_star"],),
                 log=lambda rec: log(f"  substep {rec}"),
                 compaction=(p["pore"]["compaction_Jc"], p["pore"]["compaction_kappa_MPa"]),
                 mechanics=p.get("mechanics", "finite"))
    if fields_path:
        save_npz(r, Path(fields_path))
    site_rows, tile_rows = run_curves(r, orientation, p, window=crop_um is not None)

    gif, gif_error = None, None
    if render_gif:
        try:
            gif, _ = render_site_gif(bse, r, f"{batch}/{site} {orientation}",
                                     [row["swelling"] for row in site_rows], p["gif"])
        except Exception as e:  # GifTooLarge, or a render failure on NaN frames
            gif_error = f"{type(e).__name__}: {e}"

    heldout = batch == "Batch_heldout"
    key = {"batch": batch, "site": site, "heldout": heldout, "orientation": orientation}
    for row in site_rows + tile_rows:
        row.update(key)
    fpc = site_rows[0]["first_pore_closure_s"] if site_rows else float("nan")
    meta = {**key, "H": int(H), "W": int(W), "px_um": px_um, "n_cells": int(H * W),
            "crop_um": crop_um, "res_nm": res, "lateral": lateral, "failed_at_s": float(r.failed_at_s),
            "first_pore_closure_s": fpc, "n_substeps": len(r.substeps),
            "newton_its_total": int(sum(x["its"] for x in r.substeps)), "substeps": r.substeps,
            "wall_s": float(r.wall_s), "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "versions": _versions(), "params": p, "params_hash": case_params_hash(p, lateral), "gif_error": gif_error}
    return {"meta": meta, "site_rows": site_rows, "tile_rows": tile_rows, "gif": gif}
