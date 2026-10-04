"""Materials (mat_) KPIs for the held-back sites, same pipeline as src/run.py.

Writes outputs/heldout/kpis/materials_site_kpis.csv (batch, site, mat_*).

    python scripts/heldout_materials_kpis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import load_config
from src.run import process_image


def main() -> None:
    cfg = load_config()
    rows = []
    for path in sorted((ROOT / "data_heldout" / "Batch_heldout").glob("img_*_BSE.tif")):
        res = process_image(path, "Batch_heldout", cfg)
        if res["failures"]:
            raise RuntimeError(f"{path.name}: {res['failures']}")
        rows += res["rows"]
        print(f"done {path.stem}")
    long = pd.DataFrame(rows)
    long["site"] = long["image"].str.extract(r"^img_([A-Za-z0-9]+)_BSE$")[0]
    wide = long.pivot(index=["batch", "site"], columns="kpi", values="value").reset_index()
    wide.columns.name = None
    out = ROOT / "outputs" / "heldout" / "kpis" / "materials_site_kpis.csv"
    wide.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
