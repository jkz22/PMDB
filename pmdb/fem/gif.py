"""Per-site deformation GIF (P19, D6, D15). Matplotlib Agg + Pillow."""

from __future__ import annotations

import io
import math
from typing import Sequence

import numpy as np

from pmdb.fem.materials import BINDER, GRAPHITE, SI
from pmdb.fem.result import SimResult


class GifTooLarge(RuntimeError):
    """No rung of the size ladder produced a GIF within the byte limit."""


def _corner_idx(n: int, f: int) -> np.ndarray:
    idx = list(range(0, n + 1, f))
    if idx[-1] != n:
        idx.append(n)
    return np.asarray(idx)


def _block_sum(a: np.ndarray, ri: np.ndarray, ci: np.ndarray) -> np.ndarray:
    return np.add.reduceat(np.add.reduceat(a, ri[:-1], axis=0), ci[:-1], axis=1)


def _block_mean(a: np.ndarray, ri: np.ndarray, ci: np.ndarray) -> np.ndarray:
    ones = np.ones_like(a, dtype=np.float64)
    return _block_sum(a.astype(np.float64), ri, ci) / _block_sum(ones, ri, ci)


def _render_frames(bse: np.ndarray, r: SimResult, title_prefix: str, swelling: Sequence[float],
                   cfg: dict, width_px: int) -> list[np.ndarray]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    H, W = r.labels.shape
    h = r.px_um
    f = max(1, math.ceil(W / width_px))
    ri, ci = _corner_idx(H, f), _corner_idx(W, f)
    solid = np.isin(r.labels, (SI, GRAPHITE, BINDER)).astype(np.float64)
    solid_blk = _block_sum(solid, ri, ci)
    lo, hi = np.percentile(bse, [1, 99])
    bse_blk = _block_mean(bse, ri, ci)
    x0 = (ci * h)[None, :] * np.ones((len(ri), 1))
    z0 = ((H - ri) * h)[:, None] * np.ones((1, len(ci)))
    vmin, vmax = cfg["vm_range_MPa"]
    norm = LogNorm(vmin=vmin, vmax=vmax)

    xlim = (-1.0, W * h + 1.0)
    zlim = (-0.05 * H * h, 1.45 * H * h)
    ax_w = 0.9 * width_px
    ax_h = ax_w * (zlim[1] - zlim[0]) / (xlim[1] - xlim[0])
    fig_h = ax_h + 140.0
    dpi = 100

    conv = np.asarray(r.converged, dtype=bool)
    last_ok = 0
    frames = []
    for i in range(len(r.s)):
        if conv[i]:
            last_ok = i
        k = last_ok
        failed = not conv[i]
        fig = plt.figure(figsize=(width_px / dpi, fig_h / dpi), dpi=dpi)
        ax = fig.add_axes([0.05, 70.0 / fig_h, 0.9, ax_h / fig_h])
        cax = fig.add_axes([0.25, 25.0 / fig_h, 0.5, 14.0 / fig_h])
        un = r.u_nodes[k][np.ix_(ri, ci)].astype(np.float64)
        X, Z = x0 + un[..., 0], z0 + un[..., 1]
        ax.pcolormesh(X, Z, bse_blk, cmap="gray", vmin=lo, vmax=hi, shading="flat", rasterized=True)
        vm = r.fields["vm"][k].astype(np.float64)
        vm_blk = _block_sum(np.nan_to_num(vm) * solid, ri, ci) / np.maximum(solid_blk, 1.0)
        vm_blk = np.ma.masked_where((solid_blk <= 0) | (vm_blk <= 0), vm_blk)
        mesh = ax.pcolormesh(X, Z, vm_blk, cmap="magma", norm=norm, alpha=cfg["alpha"], shading="flat",
                             rasterized=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (um)", fontsize=8)
        ax.set_ylabel("z (um)", fontsize=8)
        ax.tick_params(labelsize=7)
        sw = swelling[k] if k < len(swelling) else float("nan")
        sw_txt = f"{sw * 100:.1f}%" if np.isfinite(sw) else "n/a"
        fig.text(0.5, 1.0 - 14.0 / fig_h, f"{title_prefix} | SOC {r.s[i] * 100:.0f}% | swelling {sw_txt}",
                 ha="center", va="center", fontsize=10)
        if failed:
            ax.text(0.5, 0.92, f"solver failed at SOC {r.failed_at_s * 100:.0f}%", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"))
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.set_label("von Mises (MPa)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return frames


def render_site_gif(bse_coarse: np.ndarray, r: SimResult, title_prefix: str, swelling: Sequence[float],
                    cfg: dict) -> tuple[bytes, dict]:
    from PIL import Image

    cache: dict[int, list[np.ndarray]] = {}
    last = None
    for width_px, colors in cfg["ladder"]:
        if width_px not in cache:
            cache[width_px] = _render_frames(bse_coarse, r, title_prefix, swelling, cfg, width_px)
        pil = [Image.fromarray(a).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
               for a in cache[width_px]]
        buf = io.BytesIO()
        pil[0].save(buf, format="GIF", save_all=True, append_images=pil[1:], duration=cfg["frame_ms"],
                    loop=0, optimize=True)
        data = buf.getvalue()
        last = (width_px, colors, len(data))
        if len(data) <= cfg["max_bytes"]:
            return data, {"width_px": int(width_px), "colors": int(colors), "bytes": len(data)}
    raise GifTooLarge(f"no ladder rung fits {cfg['max_bytes']} bytes; last rung (width, colors, bytes) = {last}")
