"""Phase 0 G1: teammate segmenter + K03 on a phantom with known Si discs.

Failed at KPI commit 9d83e3a (Si over-segmented by ~1-3 px radius); passes at d23a116
"""
import pytest

from src.v2.common import NM_HALF
from src.v2.kpi_gates import D50_TOL, FRAC_TOL, phantom

K = pytest.importorskip("src.v2.kpi_adapter")


@pytest.mark.parametrize("psf_sigma", [0.0, 0.8])
def test_phantom_recovery(psf_sigma):
    raw, truth = phantom(noise=0.05, psf_sigma=psf_sigma)
    got = K.kpis_from_masks(K.segment(raw, NM_HALF), NM_HALF, "phantom")
    assert abs(got["frac_si"] - truth["frac_si"]) <= FRAC_TOL
    assert abs(got["frac_pore"] - truth["frac_pore"]) <= FRAC_TOL
    assert abs(got["K03_ecd_d50_um"] / truth["K03_ecd_d50_um"] - 1) <= D50_TOL
