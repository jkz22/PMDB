"""Segmentation review overlays (D-019)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pmdb.segment import Masks

COLOURS = {  # D-019
    "si": (255, 0, 0),
    "graphite": (0, 80, 255),
    "pore": (255, 230, 0),
    "artefact": (255, 0, 255),
}
ALPHA = 0.45
CROP_UM = 40.0
THUMB_WIDTH = 1400
LABEL_H = 22


def _grey(bse_norm: np.ndarray) -> np.ndarray:
    g = np.clip(np.asarray(bse_norm, dtype=np.float64), 0.0, 1.0)
    return np.repeat((g * 255.0)[..., None], 3, axis=2)


def colourise(bse_norm: np.ndarray, masks: Masks, alpha: float = ALPHA) -> np.ndarray:
    rgb = _grey(bse_norm)
    for name, col in COLOURS.items():
        m = getattr(masks, name)
        rgb[m] = (1.0 - alpha) * rgb[m] + alpha * np.asarray(col, dtype=np.float64)
    return rgb.astype(np.uint8)


def crop_starts(width: int, crop_px: int, n: int = 2) -> list[int]:
    """Left edges of n crops centred at evenly spaced positions across the slice."""
    centres = [(i + 1) * width / (n + 1) for i in range(n)]
    return [int(np.clip(round(c - crop_px / 2), 0, max(width - crop_px, 0))) for c in centres]


def _label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + LABEL_H), (255, 255, 255))
    out.paste(img, (0, LABEL_H))
    ImageDraw.Draw(out).text((4, 4), text, fill=(0, 0, 0))
    return out


def render_overlay(bse_norm: np.ndarray, masks: Masks, nm_per_px: float, title: str,
                   path: str | Path, n_crops: int = 1) -> Path:
    """Full-slice thumbnail plus full-height 40 um crops (raw | overlay), saved as a palette PNG."""
    h, w = bse_norm.shape
    crop_px = int(round(CROP_UM * 1000.0 / nm_per_px))
    over = colourise(bse_norm, masks)
    grey = _grey(bse_norm).astype(np.uint8)

    thumb = Image.fromarray(over)
    scale = min(1.0, THUMB_WIDTH / w)
    thumb = thumb.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    draw = ImageDraw.Draw(thumb)
    starts = crop_starts(w, crop_px, n_crops)
    for x0 in starts:
        draw.rectangle([x0 * scale, 0, (x0 + crop_px) * scale - 1, thumb.height - 1], outline=(0, 255, 0), width=2)
    legend = ("Si red | graphite blue | pore yellow | artefact magenta | green boxes = 40 um crops"
              f" | segmenter {masks.version}")
    panels = [_label(thumb, f"{title}  full slice ({w}x{h} px @ {nm_per_px:g} nm/px)  {legend}")]
    for i, x0 in enumerate(starts):
        cs = slice(x0, x0 + crop_px)
        pair = np.concatenate([grey[:, cs], np.full((h, 8, 3), 255, np.uint8), over[:, cs]], axis=1)
        panels.append(_label(Image.fromarray(pair), f"crop {i + 1}: x = {x0 * nm_per_px / 1000:.1f}-"
                             f"{(x0 + crop_px) * nm_per_px / 1000:.1f} um, full height, 1:1 pixels (raw | overlay)"))
    width = max(p.width for p in panels)
    canvas = Image.new("RGB", (width, sum(p.height for p in panels) + 6 * len(panels)), (255, 255, 255))
    y = 0
    for p in panels:
        canvas.paste(p, (0, y))
        y += p.height + 6
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.quantize(colors=24, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    return path
