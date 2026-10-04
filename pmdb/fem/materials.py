"""Per-phase material properties and the SOC driver (P7-P9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BINDER, SI, GRAPHITE, PORE, ARTEFACT = 0, 1, 2, 3, 4
PHASE_NAMES = {BINDER: "binder", SI: "si", GRAPHITE: "gr", PORE: "pore", ARTEFACT: "artefact"}
SOLID = (SI, GRAPHITE, BINDER)


@dataclass(frozen=True)
class PhaseProps:
    E: float  # MPa
    nu: float
    stretch: tuple[float, float, float]  # (lambda_x, lambda_z, lambda_y)


def _frac(s: float, ratio, s_star: float) -> float:
    a, b = ratio
    norm = a * s_star + b * (1.0 - s_star)
    return (a * min(s, s_star) + b * max(s - s_star, 0.0)) / norm


def soc_fractions(s: float, p: dict) -> tuple[float, float]:
    """(f_si, f_gr), the Yao two-region split (P9)."""
    soc = p["soc"]
    return (
        _frac(s, soc["si_ratio"], soc["s_star"]),
        _frac(s, soc["gr_ratio"], soc["s_star"]),
    )


def lithiation_state(s: float, p: dict) -> tuple[float, float]:
    f_si, f_gr = soc_fractions(s, p)
    return p["soc"]["u_max"] * f_si, p["soc"]["y_max"] * f_gr


def graphite_strains(y: float, p: dict) -> tuple[float, float]:
    """(eps_a, eps_c) at graphite lithiation y."""
    g = p["graphite"]
    pts = np.asarray(g["eps_c_points"], dtype=float)
    eps_c = float(np.interp(y, pts[:, 0], pts[:, 1]))
    r0, r1 = g["eps_a_over_c"]
    return r0 * eps_c / r1, eps_c


def lame(E: float, nu: float) -> tuple[float, float]:
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def si_yield_MPa(u: float, p: dict) -> float:
    si = p["si"]
    x = si["x_per_u"] * u
    a, b = si["yield_MPa"]
    return a - b * x / (1.0 + x)


def phase_properties(s: float, p: dict) -> dict[int, PhaseProps]:
    u, y = lithiation_state(s, p)
    si, gr = p["si"], p["graphite"]
    lam_si = (1.0 + si["beta"] * u) ** (1.0 / 3.0)
    eps_a, eps_c = graphite_strains(y, p)
    e_b, nu_b = p["binder"]["E_MPa"], p["binder"]["nu"]
    e_p = p["pore"]["E_rel_binder"] * e_b
    nu_p = p["pore"]["nu"]
    unit = (1.0, 1.0, 1.0)
    return {
        SI: PhaseProps(si["E_MPa"][0] + si["E_MPa"][1] * u, si["nu"][0] + si["nu"][1] * u,
                       (lam_si, lam_si, lam_si)),
        GRAPHITE: PhaseProps(gr["E_MPa"][0] + gr["E_MPa"][1] * y, gr["nu"][0] + gr["nu"][1] * y,
                             (1.0 + eps_a, 1.0 + eps_c, 1.0 + eps_a)),
        BINDER: PhaseProps(e_b, nu_b, unit),
        PORE: PhaseProps(e_p, nu_p, unit),
        ARTEFACT: PhaseProps(e_p, nu_p, unit),
    }
