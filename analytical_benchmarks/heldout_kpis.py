"""Same KPIs + GPs (kpis.run) for the unlabelled held-back sites. Writes heldout_kpis.csv; never touches site_kpis.csv."""
import os, sys
from multiprocessing import Pool
import pandas as pd
from kpis import run

ROOT = os.environ.get("PMDB_HELDOUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA, CACHE = f"{ROOT}/data_heldout", f"{ROOT}/cache_heldout"

if __name__ == "__main__":
    os.makedirs("tiles_heldout", exist_ok=True)
    m = pd.read_csv(f"{CACHE}/half/manifest.csv")[["batch", "site", "se_detector"]].to_dict("records")
    for r in m: r.update(data_root=DATA, cache_root=CACHE, tiles_dir="tiles_heldout")
    with Pool(min(8, len(m))) as p:
        res = p.map(run, m)
    pd.DataFrame(res).to_csv(sys.argv[1] if len(sys.argv) > 1 else "heldout_kpis.csv", index=False)
