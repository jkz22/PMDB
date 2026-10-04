#!/usr/bin/env python3
"""2D -> 3D sectioning noise: how much field-to-field KPI spread is just where the knife fell?

Synthetic 3D electrode volumes (graphite flakes as in-plane oblate ellipsoids, Si as lognormal spheres,
small spherical pores) with *identical* generating parameters are sliced on random planes; the v1 KPIs
and the SOC-1 swelling test are computed on every slice exactly as on real fields. The slice-to-slice SD
is the sectioning floor; compared with the observed between-field SD inside Batch 3 (the largest batch)
it says what fraction of the real spread a one-slice-per-field design cannot distinguish from sampling.

    python scripts/run_sectioning_sim.py [--packings 3] [--slices 30] -> outputs/sectioning/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmdb import functional  # noqa: E402
from pmdb.kpis import REGISTRY  # noqa: E402
from pmdb.kpis.common import KpiContext, TooFewObjects  # noqa: E402
from pmdb.segment import Masks  # noqa: E402

OUT = ROOT / "outputs" / "sectioning"
VOX_UM = 0.1
BOX_UM = (50.0, 16.0, 175.0)  # z (through-thickness), y (slice normal), x (in-plane): the real half-res FOV is ~52 x 175 um
SI_FRAC, GR_FRAC, PORE_FRAC = 0.06, 0.65, 0.04
SI_D_MEDIAN_UM, SI_D_SIGMA, SI_D_MAX_UM = 0.6, 0.7, 8.0
GR_AXES_UM = (1.2, 4.0, 4.0)  # oblate flake, short axis along z
PORE_D_UM = (0.4, 1.0)
KPI_IDS = ("K01", "K02", "K03", "K15")
COLS = ["K01_si_frac_adm", "K02_si_density_per_1000um2", "K03_ecd_d50_um", "K03_ecd_d90_um",
        "K15_si_graphite_contact_frac", "F02_pore_loss", "F02_into_graphite"]


def _paint_ellipsoid(vol: np.ndarray, c: np.ndarray, axes_vox: np.ndarray) -> None:
    lo = np.maximum(np.floor(c - axes_vox).astype(int), 0)
    hi = np.minimum(np.ceil(c + axes_vox).astype(int) + 1, vol.shape)
    if np.any(hi <= lo):
        return
    g = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    d = sum(((gi - ci) / ai) ** 2 for gi, ci, ai in zip(g, c, axes_vox))
    vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= d <= 1.0


def _fill(vol: np.ndarray, target: float, rng: np.random.Generator, draw_axes, forbid: np.ndarray | None) -> None:
    shape = np.array(vol.shape)
    n_target, n = target * vol.size, 0
    while n < n_target:
        for _ in range(25):
            c = rng.uniform(0, shape - 1)
            if forbid is not None and forbid[tuple(c.astype(int))]:
                continue
            _paint_ellipsoid(vol, c, draw_axes() / VOX_UM)
        n = np.count_nonzero(vol)


def build_volume(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = tuple(int(round(b / VOX_UM)) for b in BOX_UM)
    gr = np.zeros(shape, bool)
    _fill(gr, GR_FRAC, rng, lambda: np.array(GR_AXES_UM) * rng.uniform(0.7, 1.3, 3), None)
    si = np.zeros(shape, bool)

    def si_axes():
        d = min(SI_D_MEDIAN_UM * np.exp(SI_D_SIGMA * rng.standard_normal()), SI_D_MAX_UM)
        return np.full(3, d / 2)

    _fill(si, SI_FRAC, rng, si_axes, None)  # Si may sit on graphite, as the real fields do (docs/functional.md 2.6)
    pore = np.zeros(shape, bool)
    _fill(pore, PORE_FRAC, rng, lambda: np.full(3, rng.uniform(*PORE_D_UM) / 2), gr | si)
    gr &= ~si
    pore &= ~(si | gr)
    return si, gr, pore


def slice_kpis(si: np.ndarray, gr: np.ndarray, pore: np.ndarray, y: int, key: str) -> dict[str, float]:
    s, g, p = si[:, y, :], gr[:, y, :], pore[:, y, :]
    art = np.zeros_like(s)
    m = Masks(si=s, graphite=g, pore=p, artefact=art, admissible=~g, version="synthetic", params={})
    ctx = KpiContext(masks=m, nm_per_px=VOX_UM * 1000, seed_key=key)
    out = {}
    for k in KPI_IDS:
        try:
            out.update(REGISTRY[k].site_fn(ctx).values)
        except TooFewObjects:
            pass
    sw = functional.swelling_test(m, 1.0)
    out.update({f"F02_{k}": v for k, v in sw.items()})
    return {c: float(out.get(c, np.nan)) for c in COLS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packings", type=int, default=4)
    ap.add_argument("--slices", type=int, default=8)
    a = ap.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    for pk in range(a.packings):
        si, gr, pore = build_volume(pk)
        ny = si.shape[1]
        ys = np.linspace(8, ny - 9, a.slices).astype(int)
        for y in ys:
            r = slice_kpis(si, gr, pore, int(y), f"synthetic/{pk}/{y}")
            rows.append({"packing": pk, "y_um": y * VOX_UM, **r})
        print(f"packing {pk}: vol fractions si {si.mean():.3f} gr {gr.mean():.3f} pore {pore.mean():.3f}; "
              f"slice K01 {np.mean([x['K01_si_frac_adm'] for x in rows if x['packing'] == pk]):.3f}")
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "slices.csv", index=False)

    K = pd.read_csv(ROOT / "outputs" / "kpis" / "site_kpis.csv")
    F = pd.read_csv(ROOT / "outputs" / "functional" / "site_functional.csv")
    R = K.merge(F[["batch", "site", "F02_soc100_pore_loss", "F02_soc100_into_graphite"]], on=["batch", "site"])
    R = R.rename(columns={"F02_soc100_pore_loss": "F02_pore_loss", "F02_soc100_into_graphite": "F02_into_graphite"})
    b3 = R[R.batch == "Batch_3"]
    summ = []
    for c in COLS:
        within = S.groupby("packing")[c].std().mean()
        between_pk = S.groupby("packing")[c].mean().std()
        obs = b3[c].std()
        obs_all = R[c].std()
        summ.append({"kpi": c, "synthetic_mean": S[c].mean(), "sectioning_sd": within, "packing_sd": between_pk,
                     "observed_sd_batch3": obs, "observed_sd_all31": obs_all,
                     "sectioning_share_of_batch3_var": min((within / obs) ** 2, 1.0) if obs > 0 else np.nan,
                     "sectioning_cv": within / S[c].mean() if S[c].mean() else np.nan,
                     "observed_cv_batch3": obs / b3[c].mean() if b3[c].mean() else np.nan})
    T = pd.DataFrame(summ)
    T.to_csv(OUT / "summary.csv", index=False)
    print(T.round(3).to_string(index=False))
    (OUT / "params.json").write_text(json.dumps({"vox_um": VOX_UM, "box_um": BOX_UM, "si_frac": SI_FRAC, "gr_frac": GR_FRAC,
                                                 "pore_frac": PORE_FRAC, "si_d_median_um": SI_D_MEDIAN_UM, "si_d_sigma": SI_D_SIGMA,
                                                 "gr_axes_um": GR_AXES_UM, "pore_d_um": PORE_D_UM, "packings": a.packings,
                                                 "slices": a.slices}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(T))
    ax.bar(x - 0.2, T.sectioning_cv, 0.4, label="sectioning CV (same volume, different plane)")
    ax.bar(x + 0.2, T.observed_cv_batch3, 0.4, label="observed CV across Batch 3 fields")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_per_1000um2", "").replace("_adm", "") for c in T.kpi], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("coefficient of variation")
    ax.legend(fontsize=8)
    ax.set_title("How much of the between-field spread is the slice?", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "sectioning.png", dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
