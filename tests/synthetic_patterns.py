"""Synthetic point patterns and masks with known answers (spec 002, Acceptance 1)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pmdb.segment import Masks

SIZE = 2000
RADIUS = 4
NM_PER_PX = 50.0  # production working resolution (D-009): 1 um = 20 px
HARD_CORE_PX = 2 * RADIUS + 2  # discs never touch, so one disc = one Si object


def _accept_hard_core(cands: np.ndarray, n_target: int, min_dist: float) -> np.ndarray:
    kept: list[np.ndarray] = []
    tree = None
    for i, p in enumerate(cands):
        if tree is not None and tree.query_ball_point(p, min_dist - 1e-9):
            continue
        if kept and np.min(np.hypot(*(np.asarray(kept[-64:]) - p).T)) < min_dist:
            continue
        kept.append(p)
        if len(kept) % 64 == 0:
            tree = cKDTree(np.asarray(kept))
        if len(kept) == n_target:
            break
    return np.asarray(kept)


def poisson_points(rng, n: int, rows: tuple[int, int] = (0, SIZE), size: int = SIZE,
                   margin: int = RADIUS + 1) -> np.ndarray:
    lo, hi = max(rows[0], margin), min(rows[1], size - margin)
    cands = np.column_stack([rng.uniform(lo, hi, 20 * n), rng.uniform(margin, size - margin, 20 * n)])
    pts = _accept_hard_core(cands, n, HARD_CORE_PX)
    assert len(pts) == n
    return pts


def stripe_points(rng, n: int, stripes: list[tuple[int, int]]) -> np.ndarray:
    """Uniform points restricted to horizontal stripes (disc fully inside its stripe)."""
    heights = np.array([b - a - 2 * (RADIUS + 1) for a, b in stripes], dtype=float)
    k = rng.choice(len(stripes), size=20 * n, p=heights / heights.sum())
    lo = np.array([a + RADIUS + 1 for a, _ in stripes])[k]
    r = lo + rng.uniform(0, 1, 20 * n) * heights[k]
    c = rng.uniform(RADIUS + 1, SIZE - RADIUS - 1, 20 * n)
    pts = _accept_hard_core(np.column_stack([r, c]), n, HARD_CORE_PX)
    assert len(pts) == n
    return pts


def hex_points(spacing: float = 100.0, size: int = SIZE) -> np.ndarray:
    dy = spacing * np.sqrt(3) / 2
    pts = []
    for i, r in enumerate(np.arange(spacing / 2, size - spacing / 4, dy)):
        off = spacing / 2 if i % 2 else 0.0
        for c in np.arange(spacing / 2 + off, size - spacing / 4, spacing):
            pts.append((r, c))
    return np.asarray(pts)


def thomas_points(rng, n_parents: int = 20, n_offspring: int = 20, sigma_px: float = 20.0) -> np.ndarray:
    margin = RADIUS + 1
    parents = rng.uniform(4 * sigma_px, SIZE - 4 * sigma_px, size=(n_parents, 2))
    cands = []
    for _ in range(30):
        for p in parents:
            cands.append(p + rng.normal(0, sigma_px, size=(n_offspring, 2)))
    cands = np.concatenate(cands)
    cands = cands[np.all((cands >= margin) & (cands < SIZE - margin), axis=1)]
    # keep exactly n_offspring per parent, respecting the hard core
    owner = np.argmin(np.linalg.norm(cands[:, None, :] - parents[None], axis=2), axis=1)
    out = []
    accepted = np.zeros((0, 2))
    count = np.zeros(n_parents, dtype=int)
    for p, o in zip(cands, owner):
        if count[o] >= n_offspring:
            continue
        if len(accepted) and np.min(np.hypot(*(accepted - p).T)) < HARD_CORE_PX:
            continue
        accepted = np.vstack([accepted, p])
        out.append(p)
        count[o] += 1
    assert count.min() == n_offspring
    return np.asarray(out)


def discs_mask(points: np.ndarray, radius: int = RADIUS, size: int | tuple[int, int] = SIZE) -> np.ndarray:
    shape = (size, size) if np.isscalar(size) else size
    m = np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disc = yy ** 2 + xx ** 2 <= radius ** 2
    for r, c in np.rint(points).astype(int):
        r0, c0 = r - radius, c - radius
        rs = slice(max(r0, 0), min(r0 + 2 * radius + 1, shape[0]))
        cs = slice(max(c0, 0), min(c0 + 2 * radius + 1, shape[1]))
        m[rs, cs] |= disc[rs.start - r0: rs.stop - r0, cs.start - c0: cs.stop - c0]
    return m


def make_masks(si: np.ndarray, graphite: np.ndarray | None = None, pore: np.ndarray | None = None) -> Masks:
    graphite = np.zeros_like(si) if graphite is None else graphite
    pore = np.zeros_like(si) if pore is None else pore
    artefact = np.zeros_like(si)
    graphite = graphite & ~si
    return Masks(si=si, graphite=graphite, pore=pore & ~si, artefact=artefact,
                 admissible=~graphite & ~artefact, version="synthetic")


def synthetic_bse(rng, size: int = SIZE, n_si: int = 400, n_pores: int = 60, pore_radius: int = 12,
                  matrix: float = 0.5, si_level: float = 0.85, si_grain: float = 0.05,
                  pore_level: float = 0.1, noise: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mid-grey matrix, bright grainy Si discs, dark holes, additive Gaussian noise.

    Returns (image, si_truth, pore_truth). Si discs and holes never overlap.
    """
    pts = poisson_points(rng, n_si + n_pores, size=size, margin=pore_radius + 2)
    # make room for the larger holes: drop Si centres too close to a hole
    pores_c = pts[:n_pores]
    si_c = pts[n_pores:]
    d = np.min(np.linalg.norm(si_c[:, None] - pores_c[None], axis=2), axis=1)
    si_c = si_c[d > pore_radius + RADIUS + 3]
    si = discs_mask(si_c, RADIUS, size)
    pore = discs_mask(pores_c, pore_radius, size)
    img = np.full((size, size), matrix, dtype=np.float64)
    img[si] = si_level + si_grain * rng.standard_normal(int(si.sum()))
    img[pore] = pore_level
    img += noise * rng.standard_normal(img.shape)
    return img, si, pore
