"""Data tests for pmdb.fem.geometry (coarsening fractions, P24 harmonisation invariance)."""

from __future__ import annotations

import numpy as np
import pytest

from pmdb.fem.geometry import coarsen_labels, labels_from_masks
from pmdb.fem.materials import BINDER, GRAPHITE, PORE, SI
from pmdb.io import load_site
from pmdb.segment import segment

pytestmark = pytest.mark.data


def test_coarsening_fractions():
    site = load_site("Batch_3", "vc2whyaq", resolution="half", normalise="none")
    fine = labels_from_masks(segment(site))
    coarse = coarsen_labels(fine, 2)
    ratios = {}
    for name, lab in (("si", SI), ("gr", GRAPHITE), ("pore", PORE), ("binder", BINDER)):
        a_f = (fine == lab).mean()
        a_c = (coarse == lab).mean()
        ratios[name] = a_c / a_f - 1.0
    print("coarse/fine area ratio - 1:", ratios)
    assert abs(ratios["si"]) <= 0.20
    assert abs(ratios["gr"]) <= 0.05


def test_segmentation_harmonisation_invariant():
    for site_id in ("71vgq3fw", "kbdh4tri", "tuy3zymq", "x7u69zsw", "9luzk4jm"):
        labs = {}
        for h in ("none", "hybrid"):
            s = load_site("Batch_3", site_id, resolution="half", normalise="none", harmonise=h)
            labs[h] = labels_from_masks(segment(s))
        diff = float((labs["none"] != labs["hybrid"]).mean())
        fr = [abs(float((labs["none"] == k).mean()) - float((labs["hybrid"] == k).mean()))
              for k in range(5)]
        print(site_id, "differing fraction", diff, "max area-fraction diff", max(fr))
        assert diff <= 0.005, site_id
        assert max(fr) <= 0.002, site_id
