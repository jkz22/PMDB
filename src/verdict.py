"""Verdict rules. Simple and printable:

reject      multivariate p < 0.01 AND at least one |effect size| > 2
investigate multivariate p < 0.05 OR  any |effect size| > 1.5
accept      otherwise

Thresholds are placeholders; calibrate against the leave-one-out
false-alarm rate (src/stats.py), never against the known batch labels.
"""

from __future__ import annotations

import pandas as pd

# physical-meaning sentences, quoted verbatim in reports and Q&A
MEANINGS = {
    "mat_porosity": "The number every QC engineer already measures from cross-section SEM; shifts with calendering pressure and slurry solids content.",
    "mat_graphite_d10": "Feedstock particle size distribution; first thing that moves when a supplier changes milling or grade.",
    "mat_graphite_d50": "Feedstock particle size distribution; first thing that moves when a supplier changes milling or grade.",
    "mat_graphite_d90": "Feedstock particle size distribution; first thing that moves when a supplier changes milling or grade.",
    "mat_bright_fraction": "Phase loading of the higher-Z component (e.g. silicon or SiOx content); formulation signature.",
    "mat_orientation_anisotropy": "Degree of flake alignment produced by calendering; over-alignment raises through-plane tortuosity and sheet resistance.",
    "mat_crack_fraction": "Intra-particle damage from over-calendering or weak feedstock.",
    "mat_active_fraction": "Solid active loading; complements porosity and catches binder/rim changes.",
    "mat_rim_coverage": "Binder and carbon-black distribution.",
    "phys_tortuosity_y": "Through-plane ion transport resistance; directly linked to rate capability and lithium plating risk.",
    "phys_tortuosity_x": "In-plane transport; the ratio to the through-plane value is the transport anisotropy caused by calendering.",
    "phys_deff_pore_y": "Normalised electrolyte diffusivity through the coating.",
    "phys_pore_connectivity": "Fraction of porosity that actually carries ions; dead pores count toward porosity but not transport.",
    "phys_graphite_percolation_y": "Electronic percolation of the active phase through the coating.",
    "phys_tpc_decay_x": "Characteristic pore scale per direction.",
    "phys_tpc_decay_y": "Characteristic pore scale per direction.",
}

TIER_ORDER = {"mat": 0, "phys": 1, "geo": 2}


def decide(effects: pd.DataFrame, p_multivariate: float) -> tuple[str, pd.DataFrame]:
    """Return (verdict, driver table) for one incoming batch.

    effects: rows for this batch from stats.effect_sizes.
    """
    max_abs = effects.effect_size.abs().max()
    if p_multivariate < 0.01 and max_abs > 2:
        verdict = "reject"
    elif p_multivariate < 0.05 or max_abs > 1.5:
        verdict = "investigate"
    else:
        verdict = "accept"
    return verdict, drivers(effects)


def drivers(effects: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Top drivers by |effect size|, ordered materials -> physics ->
    geometry. Never geometry-only: if the top n are all geo_*, the
    strongest mat_/phys_ driver is appended."""
    df = effects.copy()
    df["abs_effect"] = df.effect_size.abs()
    df["tier_rank"] = df.kpi.str.split("_").str[0].map(TIER_ORDER)
    top = df.nlargest(n, "abs_effect")
    if (top.tier_rank == TIER_ORDER["geo"]).all():
        non_geo = df[df.tier_rank < TIER_ORDER["geo"]]
        if len(non_geo):
            top = pd.concat([top, non_geo.nlargest(1, "abs_effect")])
    top = top.sort_values(["tier_rank", "abs_effect"], ascending=[True, False])
    top["meaning"] = top.kpi.map(MEANINGS).fillna("")
    return top.drop(columns=["abs_effect", "tier_rank"])
