#!/usr/bin/env python3
"""QC sheet: one swelling card (BSE crop + where SOC-1 Si growth lands) per labelled field, one PNG per batch.

    python scripts/swelling_cards.py  -> outputs/functional/figures/swelling_cards_<batch>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from explain_heldout_functional import OUT, swelling_card  # noqa: E402


def main() -> int:
    F = pd.read_csv(OUT / "site_functional.csv")
    for batch, grp in F.groupby("batch"):
        grp = grp.sort_values("F02_soc100_into_graphite")
        fig, axes = plt.subplots(len(grp), 2, figsize=(11, 2.9 * len(grp)), squeeze=False)
        for (_, r), (a1, a2) in zip(grp.iterrows(), axes):
            swelling_card(a1, a2, batch, r["site"], f"{batch}\n{r['site']}", False)
            a2.set_title(f"SOC 1: on graphite {r['F02_soc100_into_graphite']:.2f}, pore lost "
                         f"{r['F02_soc100_pore_loss']:.2f}, Si -> {r['F02_soc100_si_frac']:.3f}", fontsize=9)
        axes[0, 0].set_title("BSE (half res)", fontsize=9)
        fig.suptitle(f"{batch}, sorted by constrained share. Si (yellow); growth on graphite (red), "
                     "pore (blue), binder/CB (green)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(OUT / "figures" / f"swelling_cards_{batch}.png", dpi=110)
        plt.close(fig)
        print(batch, len(grp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
