"""Phase 0.1: dump TIFF metadata, assign detector/field, verify co-registration."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile
from skimage.registration import phase_cross_correlation

from src.v2.common import OUT, REPO, load_full_raw, manifest

MD = OUT / "metadata"


def dump_tags():
    rows, full = [], {}
    for f in sorted((REPO / "data").glob("Batch_*/*.tif")):
        with tifffile.TiffFile(f) as t:
            p = t.pages[0]
            tags = {tg.name: str(tg.value) for tg in p.tags.values()
                    if tg.name not in ("StripOffsets", "StripByteCounts")}
            arr = p.asarray()
            npages = len(t.pages)
        stem = f.stem.split("_")
        xr = p.tags["XResolution"].value
        nm = 2.54e7 / (xr[0] / xr[1])
        rows.append(dict(file=str(f.relative_to(REPO)), batch=f.parent.name, site=stem[1], detector_file=stem[2],
                         height=p.shape[0], width=p.shape[1], nm_per_px=nm, n_pages=npages,
                         rgb_identical=bool((arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all()),
                         rgb_identical_excl_border=bool((arr[:, 4:-4, 0] == arr[:, 4:-4, 1]).all() and (arr[:, 4:-4, 1] == arr[:, 4:-4, 2]).all()),
                         software=tags.get("Software"), description=tags.get("ImageDescription"),
                         has_microscope_metadata=any(k not in ("ImageWidth", "ImageLength", "BitsPerSample", "Compression",
                             "PhotometricInterpretation", "ImageDescription", "SamplesPerPixel", "RowsPerStrip",
                             "XResolution", "YResolution", "PlanarConfiguration", "ResolutionUnit", "Software", "Predictor")
                             for k in tags)))
        full[str(f.relative_to(REPO))] = tags
    df = pd.DataFrame(rows)
    df.to_csv(MD / "tiff_tags.csv", index=False)
    (MD / "tiff_tags_full.json").write_text(json.dumps(full, indent=1))
    return df


WINDOWS = {"left": 0.0, "centre": 0.5, "right": 1.0}


def coreg_site(args):
    batch, site, se = args
    img = load_full_raw(batch, site).astype(np.float32)
    h, w, _ = img.shape
    ww = 2048
    names = ["BSE", "Inlens", se]
    out = []
    for wname, pos in WINDOWS.items():
        x0 = int(round(pos * (w - ww)))
        win = img[:, x0:x0 + ww]
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            a, b = win[..., i], win[..., j]
            shift, err, _ = phase_cross_correlation(a - a.mean(), b - b.mean(), upsample_factor=20)
            ga = np.hypot(*np.gradient(a)); gb = np.hypot(*np.gradient(b))
            gshift, gerr, _ = phase_cross_correlation(ga, gb, upsample_factor=20)
            out.append(dict(batch=batch, site=site, window=wname, x0=x0, pair=f"{names[i]}-{names[j]}",
                            dy=shift[0], dx=shift[1], err=err, dy_grad=gshift[0], dx_grad=gshift[1], err_grad=gerr))
    return out


def main():
    MD.mkdir(parents=True, exist_ok=True)
    tags = dump_tags()
    m = manifest()
    assign = tags.merge(m[["batch", "site", "se_detector", "group_id"]], on=["batch", "site"], how="left")
    assign["channel"] = assign["detector_file"].map({"BSE": 0, "Inlens": 1, "ETD": 2, "SE": 2})
    assign.to_csv(MD / "detector_field_assignment.csv", index=False)
    with ProcessPoolExecutor(4) as ex:
        res = [r for rr in ex.map(coreg_site, m[["batch", "site", "se_detector"]].itertuples(index=False, name=None)) for r in rr]
    co = pd.DataFrame(res)
    co["mag"] = np.hypot(co.dy, co.dx)
    co["mag_grad"] = np.hypot(co.dy_grad, co.dx_grad)
    co.to_csv(MD / "coregistration_offsets.csv", index=False)
    summ = co.groupby(["pair", "window"])[["mag", "mag_grad"]].agg(["median", "max"])
    summ.to_csv(MD / "coregistration_summary.csv")
    print(tags.groupby(["batch", "detector_file"]).size().unstack(), "\n", summ)
    print("microscope metadata present in any file:", tags.has_microscope_metadata.any())


if __name__ == "__main__":
    main()
