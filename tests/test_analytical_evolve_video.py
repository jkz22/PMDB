import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analytical_benchmarks"))
import evolve_video as E  # noqa: E402


def _setup(ny=10, nx=12):
    E.G.update(Hm=ny, Wm=nx, mb=2, px=0.1)
    return np.random.default_rng(0).uniform(0, 255, (ny * 2 * 4, nx * 2 * 4)).astype(np.float32)


def test_zero_displacement_keeps_real_texture():
    T = _setup(); u = np.zeros(2 * 11 * 13); vol = np.zeros(120)
    g, st = E.warp(T, 4, 0, 0, u, vol, 0, 0, *T.shape)
    assert np.allclose(g, T, atol=1e-3) and np.allclose(st, 0)


def test_uniform_vertical_stretch_lifts_top_surface():
    T = _setup(); ny, nx = 10, 12
    yy = np.repeat(np.arange(ny + 1)[:, None], nx + 1, 1)          # node row 0 = top
    u = np.zeros((ny + 1, nx + 1, 2)); u[..., 1] = 0.1 * (ny - yy) * 2 * 0.1   # uy = 10% strain, up positive
    g, _ = E.warp(T, 4, 0, 0, u.ravel(), np.full(120, 0.1), -4, 0, T.shape[0] + 16, T.shape[1])
    rows = np.flatnonzero(~np.isnan(g[:, 5]))
    assert abs(rows.min() - (16 - 8)) <= 1 and np.isfinite(g[rows]).all()      # top moved up by 10% of 2 µm = 0.2 µm = 8 px
