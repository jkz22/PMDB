"""BSE phase segmentation with per-image anchors (no baseline needed).
a = (BSE - pore_black) / (graphite_level - pore_black): pores ~0, graphite = 1, Si ~ 1.6-2.5.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage import morphology

PORE_T = 0.5  # halfway between pore black and graphite


def anchor(bse_uint8):
    b = ndi.gaussian_filter(bse_uint8.astype(np.float32), 1.5)
    h, _ = np.histogram(b, bins=256, range=(0, 256))
    g = float(np.argmax(ndi.gaussian_filter1d(h.astype(float), 2)))  # graphite = dominant phase
    k = float(np.percentile(b, 0.5))
    return (b - k) / (g - k), dict(black=k, graphite=g)


def si_threshold(a):
    """Si = tallest histogram peak above the graphite shoulder; threshold = valley before it."""
    h, e = np.histogram(a, bins=300, range=(1.0, 4.0))
    c = e[:-1] + 0.005
    hs = ndi.gaussian_filter1d(h.astype(float), 3)
    pk, pr = find_peaks(np.log1p(hs), prominence=0.05)
    keep = c[pk] >= 1.4
    if not keep.any():
        return 1.6, float("nan")  # no Si peak found: fixed fallback, flagged downstream
    peak = float(c[pk[keep][np.argmax(hs[pk[keep]])]])  # tallest peak = the main bright phase
    win = (c >= 1.15) & (c <= peak)
    return float(c[win][np.argmin(hs[win])]), peak


def segment(bse_uint8):
    a, anc = anchor(bse_uint8)
    st, peak = si_threshold(a)
    pore = morphology.remove_small_objects(a < PORE_T, 16)
    si = morphology.binary_opening(a > st, morphology.disk(2))
    si = morphology.remove_small_objects(si, 40)
    lab = np.ones(a.shape, np.uint8)  # 1 graphite/binder
    lab[pore] = 0; lab[si] = 2
    return lab, dict(anc, si_thr=st, si_peak=peak)


def overlay(bse_uint8, lab):
    g = bse_uint8.astype(np.float32) / 255.0
    rgb = np.stack([g, g, g], -1) * 0.55
    rgb[lab == 0] = [0.1, 0.3, 1.0]   # pores blue
    rgb[lab == 2] = rgb[lab == 2] * 0.3 + np.array([1.0, 0.55, 0.0]) * 0.7  # Si orange
    return np.clip(rgb, 0, 1)
