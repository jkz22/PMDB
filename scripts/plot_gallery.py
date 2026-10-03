"""Browsable gallery of the 31 labelled sites: processed half-res channels plus the
segmentation review overlays from outputs/overlays/.

Writes web-sized JPEGs and a self-contained index.html to outputs/gallery/. Held-back
sites (data_heldout/) are not included.

    python scripts/plot_gallery.py
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb.io import list_sites, load_site  # noqa: E402

WIDTH = 1400  # px; the half-res strips are ~3500 px wide
QUALITY = 82
KPIS = [  # (column, label, format)
    ("K01_si_frac_adm", "Si area frac", "{:.3f}"),
    ("K02_si_density_per_1000um2", "Si / 1000 µm²", "{:.0f}"),
    ("K03_ecd_d50_um", "Si ECD d50 µm", "{:.2f}"),
    ("K04_agglom_frac", "Agglom frac", "{:.2f}"),
    ("D01_graphite_frac_mean", "Graphite frac", "{:.3f}"),
]


def _save(arr: np.ndarray, path: Path) -> None:
    im = Image.fromarray(arr)
    h = int(round(im.height * WIDTH / im.width))
    im.resize((WIDTH, h), Image.LANCZOS).save(path, quality=QUALITY, optimize=True)


def _u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def build(out_dir: Path) -> list[dict]:
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    kpis = pd.read_csv(ROOT / "outputs" / "kpis" / "site_kpis.csv").set_index(["batch", "site"])
    sites = []
    for row in list_sites().itertuples():
        key = f"{row.batch}__{row.site}"
        site = load_site(row.batch, row.site, resolution="half")
        img = site.image
        names = [c if c != "SE_type" else site.se_detector for c in site.channels]
        files = {}
        for i, name in enumerate(names):
            files[name] = f"img/{key}_{i}.jpg"
            _save(_u8(img[..., i]), out_dir / files[name])
        files["Composite"] = f"img/{key}_rgb.jpg"
        _save(_u8(img), out_dir / files["Composite"])
        overlay = Image.open(ROOT / "outputs" / "overlays" / f"{key}.png").convert("RGB")
        files["Overlay"] = f"img/{key}_overlay.jpg"
        overlay.save(out_dir / files["Overlay"], quality=88, optimize=True)
        k = kpis.loc[(row.batch, row.site)]
        sites.append({
            "batch": row.batch,
            "site": row.site,
            "detector": site.se_detector,
            "channels": names,
            "w_um": round(img.shape[1] * site.nm_per_px / 1000, 1),
            "h_um": round(img.shape[0] * site.nm_per_px / 1000, 1),
            "files": files,
            "kpis": [[label, fmt.format(k[col])] for col, label, fmt in KPIS],
        })
        print(key)
    return sites


def write_html(sites: list[dict], out_dir: Path) -> None:
    template = (Path(__file__).with_name("gallery_template.html")).read_text(encoding="utf-8")
    page = template.replace("/*SITES*/[]", json.dumps(sites)).replace(
        "{{N_SITES}}", html.escape(str(len(sites))))
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "gallery"))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    write_html(build(out_dir), out_dir)
    print(f"wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
