"""Per-crop gated KPIs (teammate segmenter run once per field, masks cropped).

Writes outputs/v2/kpis/crop_kpis_{eval,train}.csv keyed by group_id, y, x (50 nm/px,
256 px crops; eval = non-overlapping grid, train = stride 128).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.v2 import kpi_adapter as K
from src.v2.common import CROP, NM_HALF, OUT, STRIDE_TRAIN, grid, load_half_raw, manifest

D = OUT / "kpis"


def _site(args):
    b, s = args
    bse = load_half_raw(b, s)[..., 0]
    m = K.segment(bse, NM_HALF)
    out = {}
    for name, stride in (("eval", CROP), ("train", STRIDE_TRAIN)):
        rows = K.crop_kpis(m, NM_HALF, f"{b}/{s}", grid(*bse.shape, CROP, stride), CROP)
        out[name] = [{"batch": b, "site": s, "group_id": f"{b}/{s}", **{k: r[k] for k in ("y", "x", *K.GATED_COLS)}}
                     for r in rows]
    return out


def main():
    D.mkdir(parents=True, exist_ok=True)
    sites = manifest()[["batch", "site"]].itertuples(index=False, name=None)
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_site, sites))
    for name in ("eval", "train"):
        df = pd.DataFrame([r for x in res for r in x[name]])
        df.to_csv(D / f"crop_kpis_{name}.csv", index=False)
        print(name, df.shape, df[list(K.GATED_COLS)].isna().mean().round(4).to_dict())
    (D / "config.json").write_text(json.dumps({"kpi_commit": K.kpi_commit_hash(), "cols": K.GATED_COLS,
                                               "crop": CROP, "stride_train": STRIDE_TRAIN, "nm_per_px": NM_HALF}, indent=2))


if __name__ == "__main__":
    main()
