"""Run design for the imported harmonisation pipelines of PR #37 (``pmdb.harmonise_ext``):
``nyul`` = N4ITK bias field + Nyul-Udupa histogram standardisation, ``basic`` = BaSiC flat/dark field + baseline.
Same tracks as the hybrid / clean rounds so the rows are directly comparable: 3-class and binary 'off'
classifiers (ResNet-18, DINOv2-S; stack; stratified grouped 5-fold, saved + attributed), VAE-C (+ phase-inpaint)
and MAE-adapted representations over all labelled fields, and a Batch_3-only VAE-C for novelty scoring.
Writes outputs/v2/ext_{rep,cls_a,cls_b}_specs.json."""
from __future__ import annotations

import json

from src.v2.common import EXT_METHODS, OUT
from src.v2.rungrid import STEPS


def cls_specs(arch: str, n_folds: int = 5) -> list[dict]:
    return [dict(task="cls", arch=arch, view="stack", input="raw", harmonise=m, labels=lab, fold=k, n_folds=n_folds,
                 stage="EXT", save=True, attrib=True)
            for m in EXT_METHODS for lab in ("batch", "off") for k in range(n_folds)]


def rep_specs() -> list[dict]:
    runs = []
    for m in EXT_METHODS:
        base = dict(input="raw", harmonise=m, stage="EXT")
        runs += [dict(family=fam, steps=STEPS[fam], train_set="all_fields", **base) for fam in ("vae_c", "mae_adapted")]
        runs.append(dict(family="vae_c", steps=STEPS["vae_c"], train_set="all_fields", phase_mask="inpaint", **base))
        runs.append(dict(family="vae_c", steps=STEPS["vae_c"], train_set="baseline", **base))
    return runs


if __name__ == "__main__":
    groups = {"ext_rep": rep_specs(), "ext_cls_a": cls_specs("resnet18_imnet"), "ext_cls_b": cls_specs("dinov2_ft")}
    for name, specs in groups.items():
        (OUT / f"{name}_specs.json").write_text(json.dumps(specs, indent=1))
        print(name, len(specs), "runs")
