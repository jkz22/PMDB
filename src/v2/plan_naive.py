"""Run design for the naive-harmonisation baseline (``input='naive'``, raw grey levels).

Naive route (src.v2.data.FieldStore / naive_transform): KPI phase segmentation keeps only pore / Si /
graphite pixels (everything else -> 0), each 256-px crop is rescaled to its own p1-p99 (~min-max) and
N(0, 0.1) noise (~25 grey levels) is added at train and test time. Same families / probes as the clean
round so the rows are comparable: VAE-C (+ phase-inpaint), MAE-adapted, DINO-FT on all labelled fields,
and ResNet-18 / DINOv2-S stratified 5-fold classifiers on the stack and on BSE alone.
Writes outputs/v2/naive_{rep,cls_a,cls_b}_specs.json for three parallel ``modal run`` schedulers.
"""
from __future__ import annotations

import json

from src.v2.common import OUT
from src.v2.rungrid import STEPS


def rep_specs() -> list[dict]:
    base = dict(input="naive", harmonise="none", train_set="all_fields", stage="NAIVE")
    runs = [dict(family=fam, steps=STEPS[fam], **base) for fam in ("vae_c", "mae_adapted", "dino_ft")]
    runs.append(dict(family="vae_c", steps=STEPS["vae_c"], phase_mask="inpaint", **base))
    runs.append(dict(family="vae_c", steps=STEPS["vae_c"], view="BSE", **base))
    return runs


def cls_specs(arch: str, n_folds: int = 5) -> list[dict]:
    return [dict(task="cls", arch=arch, view=view, input="naive", harmonise="none", fold=k, n_folds=n_folds,
                 stage="NAIVE", save=view == "stack", attrib=view == "stack")
            for view in ("stack", "BSE") for k in range(n_folds)]


if __name__ == "__main__":
    groups = {"naive_rep": rep_specs(), "naive_cls_a": cls_specs("resnet18_imnet"), "naive_cls_b": cls_specs("dinov2_ft")}
    for name, specs in groups.items():
        (OUT / f"{name}_specs.json").write_text(json.dumps(specs, indent=1))
        print(name, len(specs), "runs")
