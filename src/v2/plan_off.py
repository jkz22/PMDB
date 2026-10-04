"""Run design for the 'is this image off?' track: Batch_3 (supplier baseline) vs Batch_1+2.

Two harmonisation strengths next to the raw control: *normal* = PR #16 ``hybrid`` LUT (black level / gain
only) and *extreme* = ``input='extreme'`` (KPI phase-only pixels, per-crop p1-p99 rescale, 1 px Gaussian
blur, N(0, 0.1) noise: removes grey level, gain, sharpness, noise texture, clipping and grey-step cues);
``naive`` (same without the blur) sits between them.
* Classifiers: ResNet-18 / DINOv2-S, stack, stratified grouped 5-fold, ``labels='off'`` (binary), saved
  and attributed for Grad-CAM / occlusion / calibration.
* Representations: VAE-C (+ phase-inpaint) and MAE-adapted on ``extreme`` over all labelled fields, and
  VAE-C trained on Batch_3 only (``train_set='baseline'``) on hybrid / extreme for reconstruction- and
  embedding-novelty scoring of Batch_1+2 and the held-out sites.
Writes outputs/v2/off_{rep,cls_a,cls_b}_specs.json for three parallel ``modal run`` schedulers.
"""
from __future__ import annotations

import json

from src.v2.common import OUT
from src.v2.rungrid import STEPS

ROUTES = (dict(input="raw", harmonise="none"), dict(input="raw", harmonise="hybrid"),
          dict(input="naive", harmonise="none"), dict(input="extreme", harmonise="none"))


def cls_specs(arch: str, n_folds: int = 5) -> list[dict]:
    return [dict(task="cls", arch=arch, view="stack", labels="off", fold=k, n_folds=n_folds, stage="OFF", save=True,
                 attrib=True, **r) for r in ROUTES for k in range(n_folds)]


def rep_specs() -> list[dict]:
    ext = dict(input="extreme", harmonise="none", stage="OFF")
    runs = [dict(family=fam, steps=STEPS[fam], train_set="all_fields", **ext) for fam in ("vae_c", "mae_adapted")]
    runs.append(dict(family="vae_c", steps=STEPS["vae_c"], train_set="all_fields", phase_mask="inpaint", **ext))
    for r in (dict(input="raw", harmonise="hybrid"), dict(input="extreme", harmonise="none")):
        runs.append(dict(family="vae_c", steps=STEPS["vae_c"], train_set="baseline", stage="OFF", **r))
    return runs


if __name__ == "__main__":
    groups = {"off_rep": rep_specs(), "off_cls_a": cls_specs("resnet18_imnet"), "off_cls_b": cls_specs("dinov2_ft")}
    for name, specs in groups.items():
        (OUT / f"{name}_specs.json").write_text(json.dumps(specs, indent=1))
        print(name, len(specs), "runs")
