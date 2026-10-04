"""Run design for the physical-clean round (PR #27 ``clean_norm`` / ``clean_harm`` inputs).

Writes outputs/v2/clean_specs.json for ``modal run --detach src/v2/modal_app.py --mode clean``.

* Classification (stratified grouped 5-fold by field, 3 classes): ResNet-18, DINOv2-S, MicroNet
  EfficientNet-B4 x {stack, BSE} x {clean_norm, clean_harm}; ResNet-18 stack also with aug2 (random
  offset/gain/blur/noise, the imaging-invariance augmentation) on clean_harm and, for comparison, on
  hybrid. Stack runs are saved (``save``) and attributed (occlusion + Grad-CAM) for the SAE /
  calibration / attribution analyses.
* Representation (transductive SSL, ``train_set='all_fields'``: reconstruction / self-distillation on
  every labelled field, nothing held out; batch labels never enter training): VAE-C, MAE-adapted,
  DINO-FT x {clean_norm, clean_harm}, plus VAE-C on none / hybrid for a like-for-like comparison and
  VAE-C clean_harm + aug2. Embeddings are then probed leave-one-field-out on CPU (cls_probe).
"""
from __future__ import annotations

import json

from src.v2.common import OUT
from src.v2.rungrid import STEPS

CLEAN = ("clean_norm", "clean_harm")


def cls_specs(n_folds: int = 5) -> list[dict]:
    runs = []
    for arch in ("resnet18_imnet", "dinov2_ft", "effb4_micronet"):
        for view in ("stack", "BSE"):
            for harm in CLEAN:
                for k in range(n_folds):
                    runs.append(dict(task="cls", arch=arch, view=view, harmonise=harm, fold=k, n_folds=n_folds,
                                     stage="CLEAN", save=view == "stack", attrib=view == "stack"))
    for harm in ("clean_harm", "hybrid"):
        for k in range(n_folds):
            runs.append(dict(task="cls", arch="resnet18_imnet", view="stack", harmonise=harm, aug="aug2", fold=k,
                             n_folds=n_folds, stage="CLEAN", save=True, attrib=True))
    return runs


def rep_specs() -> list[dict]:
    runs = []
    for fam in ("vae_c", "mae_adapted", "dino_ft"):
        for harm in CLEAN:
            runs.append(dict(family=fam, steps=STEPS[fam], train_set="all_fields", harmonise=harm, stage="CLEAN"))
    for harm in ("none", "hybrid"):
        runs.append(dict(family="vae_c", steps=STEPS["vae_c"], train_set="all_fields", harmonise=harm, stage="CLEAN"))
    runs.append(dict(family="vae_c", steps=STEPS["vae_c"], train_set="all_fields", harmonise="clean_harm", aug="aug2", stage="CLEAN"))
    runs.append(dict(family="vae_c", steps=STEPS["vae_c"], train_set="all_fields", harmonise="clean_harm", view="BSE", stage="CLEAN"))
    return runs


if __name__ == "__main__":
    specs = rep_specs() + cls_specs()
    (OUT / "clean_specs.json").write_text(json.dumps(specs, indent=1))
    print(len(specs), "runs:", sum(s.get("task") == "cls" for s in specs), "cls,", sum("family" in s for s in specs), "rep")
