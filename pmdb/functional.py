"""Functional morphology: what the Si / graphite / pore arrangement implies for access and swelling.

The KPI families in :mod:`pmdb.kpis` describe *where* the Si sits (composition, clustering,
localisation). This module asks the follow-on question a cell engineer would ask of the same
masks: *does that arrangement still work as an electrode, and what happens to it on lithiation?*
Everything is computed on the ``Masks`` from :func:`pmdb.segment.segment`, so it inherits the
segmenter's affine (offset / gain) invariance and needs no harmonised input.

F01 - pore access and connectivity
    In a 2D section a 3-5 % pore phase cannot percolate (2D site-percolation threshold ~ 0.59), so
    network transport (tortuosity) is not measurable from one section; :func:`tortuosity_factor`
    is provided and tested (TauFactor construction, Cooper et al. 2016) and reports ``spans`` so
    this is recorded rather than assumed. What *is* measurable is access: how far each Si pixel is
    from the nearest pore (electrolyte access length), how the pore chords are oriented (x vs z),
    how fragmented the pore phase is, and whether any Si is cut off from the solid network.

F02 - swelling stress test
    Si expands ~ 280 % by volume on full lithiation (Obrovac & Christensen 2004; Beaulieu et al.
    2001). Grow every Si object isotropically to its lithiated cross-section area and record where
    the growth lands: on pore (free swelling, pore loss), on binder / carbon black (soft), or on
    graphite (constrained swelling, a stress proxy), and how many Si objects merge into contact.

These are **2D section proxies**: a section under-reports connectivity (Cooper et al. 2014) and
the swelling model moves no material, so the numbers compare fields and batches under one rule;
they are not absolute predictions. Directions follow spec 002: columns are x (in-plane), rows
are z (through-thickness).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

from pmdb.segment import Masks

FUNCTIONAL_VERSION = "f1"
NAN = float("nan")

SI_VOL_EXPANSION = 2.8  # relative volume increase of Si at full lithiation (Li15Si4 ~ 280 %)
DEFAULT_SOCS = (0.25, 0.5, 1.0)
ACCESS_UM = 1.0  # "within reach of a pore" radius used for F01_si_pore_access_frac

_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_EIGHT = np.ones((3, 3), dtype=bool)


# ---------------------------------------------------------------------------
# percolation and tortuosity
# ---------------------------------------------------------------------------

@dataclass
class Percolation:
    spans: bool
    spanning_frac: float  # share of phase pixels in clusters touching both ends
    largest_frac: float  # share of phase pixels in the largest cluster
    n_clusters: int


def percolation(mask: np.ndarray, axis: int, connectivity: int = 1) -> Percolation:
    """Cluster statistics of ``mask`` for transport along ``axis`` (0 = z, 1 = x).

    ``connectivity=1`` (4-neighbour) is the graph the finite-volume solve uses, so a phase that
    does not percolate here has ``tau = inf`` there. ``connectivity=2`` is 8-neighbour.
    """
    mask = np.asarray(mask, dtype=bool)
    n_phase = int(mask.sum())
    if n_phase == 0:
        return Percolation(False, 0.0, 0.0, 0)
    lab, n = ndimage.label(mask, structure=_FOUR if connectivity == 1 else _EIGHT)
    first = np.take(lab, 0, axis=axis)
    last = np.take(lab, -1, axis=axis)
    spanning = np.intersect1d(first[first > 0], last[last > 0])
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    spanning_px = int(sizes[spanning - 1].sum()) if spanning.size else 0
    return Percolation(bool(spanning.size), spanning_px / n_phase, float(sizes.max() / n_phase), int(n))


@dataclass
class Transport:
    phase_frac: float  # eps = phase pixels / domain pixels
    d_eff: float  # D_eff / D_0 (0 if not percolating)
    tau: float  # eps / d_eff (inf if not percolating)
    spans: bool
    n_iter: int


def _solve(a: sparse.csr_matrix, b: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    """Solve the SPD system: algebraic multigrid if pyamg is installed, Jacobi-CG otherwise."""
    try:
        import pyamg
    except ImportError:  # pragma: no cover - depends on environment
        pyamg = None
    if pyamg is not None and a.shape[0] > 500:
        ml = pyamg.smoothed_aggregation_solver(a.tocsr(), max_coarse=200)
        res: list[float] = []
        x = ml.solve(b, tol=tol, residuals=res, accel="cg", maxiter=400)
        return x, len(res)
    it = {"n": 0}

    def _count(_):
        it["n"] += 1

    x, info = cg(a, b, M=sparse.diags(1.0 / a.diagonal()), rtol=tol, maxiter=50000, callback=_count)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")
    return x, it["n"]


def tortuosity_factor(mask: np.ndarray, axis: int, domain: np.ndarray | None = None,
                      tol: float = 1e-8) -> Transport:
    """TauFactor-style tortuosity factor of ``mask`` for diffusion along ``axis``.

    Unit concentration on a ghost plane before the first slice, zero on one after the last, no
    flux through the other faces or into non-phase pixels, unit conductance per 4-neighbour link.
    With ghost planes the domain is ``n + 1`` cells long, so ``D_eff / D_0 = Q (n + 1) / A`` with
    ``Q`` the inlet flux and ``A`` the number of cross-section pixels, and ``tau = eps / D_eff``.
    ``eps`` is taken over ``domain`` (default: the whole array) so artefact pixels can be left out
    of the fraction while still blocking flow.
    """
    mask = np.asarray(mask, dtype=bool)
    if axis == 1:
        mask = mask.T
        domain = None if domain is None else np.asarray(domain, dtype=bool).T
    n_dom = mask.size if domain is None else int(np.count_nonzero(domain))
    eps = float(mask.sum() / n_dom) if n_dom else 0.0
    lab, _ = ndimage.label(mask, structure=_FOUR)
    keep = np.intersect1d(lab[0][lab[0] > 0], lab[-1][lab[-1] > 0])
    if keep.size == 0:
        return Transport(eps, 0.0, float("inf"), False, 0)

    active = np.isin(lab, keep)  # dead-end clusters carry no flux and would make A singular
    n_rows, n_cols = active.shape
    idx = -np.ones(active.shape, dtype=np.int64)
    n_unk = int(active.sum())
    idx[active] = np.arange(n_unk)

    rows, cols, vals = [], [], []
    diag = np.zeros(n_unk)
    for dr, dc in ((0, 1), (1, 0)):
        both = active[: n_rows - dr, : n_cols - dc] & active[dr:, dc:]
        i = idx[: n_rows - dr, : n_cols - dc][both]
        j = idx[dr:, dc:][both]
        rows += [i, j]
        cols += [j, i]
        vals += [-np.ones(i.size), -np.ones(i.size)]
        np.add.at(diag, i, 1.0)
        np.add.at(diag, j, 1.0)
    inlet = idx[0][active[0]]
    outlet = idx[-1][active[-1]]
    diag[inlet] += 1.0
    diag[outlet] += 1.0
    rhs = np.zeros(n_unk)
    rhs[inlet] = 1.0
    ar = np.arange(n_unk)
    a_mat = sparse.coo_matrix(
        (np.concatenate(vals + [diag]), (np.concatenate(rows + [ar]), np.concatenate(cols + [ar]))),
        shape=(n_unk, n_unk),
    ).tocsr()
    c, n_iter = _solve(a_mat, rhs, tol)
    q = 0.5 * (float(np.sum(1.0 - c[inlet])) + float(np.sum(c[outlet])))
    d_eff = q * (n_rows + 1) / n_cols
    return Transport(eps, float(d_eff), float(eps / d_eff) if d_eff > 0 else float("inf"), True, int(n_iter))


# ---------------------------------------------------------------------------
# F01 pore access
# ---------------------------------------------------------------------------

def chord_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Lengths (px) of all maximal runs of ``mask`` along ``axis`` (1 = x rows, 0 = z columns)."""
    m = np.asarray(mask, dtype=bool)
    if axis == 0:
        m = m.T
    d = np.diff(np.pad(m.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return (ends - starts).astype(np.float64)


def pore_access(masks: Masks, nm_per_px: float) -> dict[str, float]:
    px_um = nm_per_px / 1000.0
    domain = masks.fraction_space
    area_um2 = float(domain.sum()) * px_um ** 2
    pore = masks.pore & domain
    solid = domain & ~masks.pore
    out: dict[str, float] = {"F01_pore_frac": float(pore.sum() / domain.sum())}

    pz = percolation(pore, axis=0)
    px_ = percolation(pore, axis=1)
    out["F01_pore_spans_z"] = float(pz.spans)
    out["F01_pore_spans_x"] = float(px_.spans)
    out["F01_pore_largest_frac"] = pz.largest_frac
    out["F01_pore_clusters_per_1000um2"] = pz.n_clusters / area_um2 * 1000.0 if area_um2 else NAN

    cx, cz = chord_lengths(pore, 1) * px_um, chord_lengths(pore, 0) * px_um
    out["F01_pore_chord_x_p50_um"] = float(np.median(cx)) if cx.size else NAN
    out["F01_pore_chord_z_p50_um"] = float(np.median(cz)) if cz.size else NAN
    out["F01_pore_chord_x_p90_um"] = float(np.percentile(cx, 90)) if cx.size else NAN
    out["F01_pore_chord_z_p90_um"] = float(np.percentile(cz, 90)) if cz.size else NAN
    out["F01_pore_chord_anisotropy"] = (out["F01_pore_chord_x_p50_um"] / out["F01_pore_chord_z_p50_um"]
                                        if cx.size and cz.size and out["F01_pore_chord_z_p50_um"] > 0 else NAN)

    si = masks.si & domain
    n_si = int(si.sum())
    if n_si and pore.any():
        d = ndimage.distance_transform_edt(~pore)[si] * px_um
        out["F01_si_pore_dist_p50_um"] = float(np.median(d))
        out["F01_si_pore_dist_p90_um"] = float(np.percentile(d, 90))
        out["F01_si_pore_access_frac"] = float(np.mean(d <= ACCESS_UM))
    else:
        out["F01_si_pore_dist_p50_um"] = out["F01_si_pore_dist_p90_um"] = out["F01_si_pore_access_frac"] = NAN

    lab, n = ndimage.label(solid, structure=_FOUR)
    if n and n_si:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        sizes[0] = 0
        giant = lab == int(np.argmax(sizes))
        out["F01_si_isolated_frac"] = float(1.0 - (si & giant).sum() / n_si)
    else:
        out["F01_si_isolated_frac"] = NAN
    return out


# ---------------------------------------------------------------------------
# F02 swelling stress test
# ---------------------------------------------------------------------------

def area_factor(soc: float, vol_expansion: float = SI_VOL_EXPANSION) -> float:
    """Cross-section area factor of Si at state of charge ``soc`` (isotropic, linear in volume)."""
    return float((1.0 + vol_expansion * soc) ** (2.0 / 3.0))


def swell_si(si: np.ndarray, soc: float, vol_expansion: float = SI_VOL_EXPANSION) -> np.ndarray:
    """Grow each 8-connected Si object isotropically so its area scales by :func:`area_factor`.

    An object of area ``A`` has equivalent radius ``R = sqrt(A / pi)`` and is dilated by
    ``R (sqrt(f) - 1)``. The per-object radii are applied through the Euclidean distance transform
    to the nearest Si pixel, so each object grows by its own increment. Overlaps are allowed;
    they are what the stress proxy and the merging ratio count.
    """
    si = np.asarray(si, dtype=bool)
    if soc <= 0 or not si.any():
        return si.copy()
    f = area_factor(soc, vol_expansion)
    lab, n = ndimage.label(si, structure=_EIGHT)
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    grow = np.concatenate([[0.0], np.sqrt(areas / np.pi) * (np.sqrt(f) - 1.0)])
    dist, (ri, ci) = ndimage.distance_transform_edt(~si, return_indices=True)
    return si | (dist <= grow[lab[ri, ci]])


def swelling_test(masks: Masks, soc: float, vol_expansion: float = SI_VOL_EXPANSION) -> dict[str, float]:
    """Where does the lithiation growth of Si land? Keys are suffixes (caller adds the SOC prefix)."""
    domain = masks.fraction_space
    n_dom = int(domain.sum())
    si0 = masks.si & domain
    pore0 = masks.pore & domain
    binder = domain & ~masks.si & ~masks.graphite & ~masks.pore
    swollen = swell_si(masks.si, soc, vol_expansion) & domain
    growth = swollen & ~si0
    n_growth = int(growth.sum())
    n_obj0 = int(ndimage.label(si0, structure=_EIGHT)[1])
    n_obj1 = int(ndimage.label(swollen, structure=_EIGHT)[1])
    p0 = int(pore0.sum())
    return {
        "si_frac": float(swollen.sum() / n_dom),
        "pore_frac": float((pore0 & ~swollen).sum() / n_dom),
        "pore_loss": float((pore0 & swollen).sum() / p0) if p0 else NAN,
        "into_graphite": float((growth & masks.graphite).sum() / n_growth) if n_growth else NAN,
        "into_pore": float((growth & pore0).sum() / n_growth) if n_growth else NAN,
        "into_binder": float((growth & binder).sum() / n_growth) if n_growth else NAN,
        "si_objects_ratio": float(n_obj1 / n_obj0) if n_obj0 else NAN,
    }


# ---------------------------------------------------------------------------
# per-field summary
# ---------------------------------------------------------------------------

def compute_functional(masks: Masks, nm_per_px: float, socs: tuple[float, ...] = DEFAULT_SOCS,
                       vol_expansion: float = SI_VOL_EXPANSION) -> dict[str, float]:
    """All functional-morphology columns for one field or tile: ``F01_*`` and ``F02_soc{ppp}_*``."""
    out = pore_access(masks, nm_per_px)
    for soc in socs:
        prefix = f"F02_soc{int(round(100 * soc)):03d}"
        out.update({f"{prefix}_{k}": v for k, v in swelling_test(masks, soc, vol_expansion).items()})
    return out
