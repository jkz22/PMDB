"""Run design for the PR #33 clean rebuild (``clean2_norm`` / ``clean2_harm``) and the KPI-phase-masked VAE.

Writes outputs/v2/clean2_specs.json (``--mode clean2``) and outputs/v2/phase_specs.json (``--mode phase``).

* clean2: the same classification and transductive-representation design as plan_clean (PR #27 arrays)
  on the arrays rebuilt after PR #33 (collector connected to the true image edge, half-res mask word from
  contributing parents only, slightly shifted harm targets), so the two builds can be compared like for like.
* phase: VAE-C (all labelled fields, aug1) whose reconstruction is driven by the teammate KPI segmentation of
  the crop: ``phase_mask='inpaint'`` hides one minority phase (Si or pore) in the input and reconstructs it from
  the matrix; ``phase_mask='weight'`` keeps the input and gives Si / graphite / pore a third of the loss each.
  Both on PR #16 hybrid and on clean2_harm input.
"""
from __future__ import annotations

import json

from src.v2.common import OUT
from src.v2.plan_clean import cls_specs as _cls, rep_specs as _rep
from src.v2.rungrid import STEPS

CLEAN2 = {"clean_norm": "clean2_norm", "clean_harm": "clean2_harm"}


def _retag(specs: list[dict]) -> list[dict]:
    out = []
    for s in specs:
        if s["harmonise"] in CLEAN2:
            out.append({**s, "harmonise": CLEAN2[s["harmonise"]], "stage": "CLEAN2"})
    return out


def phase_specs(harms=("hybrid", "clean2_harm")) -> list[dict]:
    return [dict(family="vae_c", steps=STEPS["vae_c"], train_set="all_fields", harmonise=h, phase_mask=pm, stage="PHASE")
            for h in harms for pm in ("inpaint", "weight")]


def verify_specs() -> list[dict]:
    """The PR #33 rebuild is byte-identical to the PR #28 arrays on every labelled and held-out site (norm) and
    within 0.3 grey on 3 sites (harm), so the full fan-out would only reproduce the clean round. This subset
    checks that: VAE-C x {clean2_norm, clean2_harm} and ResNet-18 stack clean2_harm (5 folds)."""
    c2 = _retag(_rep()) + _retag(_cls())
    return [s for s in c2 if ("family" in s and s["family"] == "vae_c" and s.get("view", "stack") == "stack" and s.get("aug", "aug1") == "aug1")
            or (s.get("task") == "cls" and s["arch"] == "resnet18_imnet" and s["view"] == "stack" and s["harmonise"] == "clean2_harm" and s.get("aug", "aug1") == "aug1")]


if __name__ == "__main__":
    import sys
    c2 = verify_specs() if "--verify" in sys.argv else _retag(_rep()) + _retag(_cls())
    (OUT / "clean2_specs.json").write_text(json.dumps(c2 + phase_specs(("clean2_harm",)), indent=1))
    (OUT / "phase_specs.json").write_text(json.dumps(phase_specs(("hybrid",)), indent=1))
    print(len(c2), "clean2 runs (+2 phase on clean2_harm);", 2, "phase runs on hybrid")
