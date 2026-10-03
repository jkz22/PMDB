"""Post-hoc probes of every saved embedding against all of Kevin's crop KPIs, including the
ones that fail gates G2/G3 (K02, K03, K04 cluster density). Same held-out fields and grouped
ridge probe as ``evaluate.kpi_r2``; adds ``kpi_r2_all`` and ``kpi_r2_all__<col>`` to metrics.json.

    python -m src.v2.kpi_all [--runs DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.v2.common import OUT
from src.v2.data import heldout_split
from src.v2.evaluate import probe_r2
from src.v2.kpi_adapter import ALL_COLS, UNGATED_COLS, crop_kpi_frame
from src.v2.latent_audit import load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(OUT / "runs"))
    a = ap.parse_args(argv)
    held = set(heldout_split().query("heldout").group_id)
    kp = crop_kpi_frame("eval", ALL_COLS)
    n = 0
    for mp in sorted(Path(a.runs).glob("*/metrics.json")):
        m = json.loads(mp.read_text())
        if "kpi_r2_all" in m or not (mp.parent / "embeddings_eval.npz").exists():
            continue
        E, meta = load(mp.parent)
        sel = meta.group_id.isin(held).to_numpy()
        Y = meta[sel].merge(kp, on=["group_id", "y", "x"], how="left")[list(ALL_COLS)]
        r = probe_r2(E[sel], Y, meta.group_id.to_numpy()[sel])
        m["kpi_r2_all"] = float(np.mean(list(r.values())))
        m["kpi_r2_ungated"] = float(np.mean([r[c] for c in UNGATED_COLS if c in r]))
        m.update({f"kpi_r2_all__{k}": v for k, v in r.items()})
        mp.write_text(json.dumps(m, indent=2, default=float))
        n += 1
    print(f"{n} runs updated")


if __name__ == "__main__":
    main()
