#!/usr/bin/env python3
"""How do the held-out sites differ from the Batch 3 baseline in functional (F) and stretch (S) terms?

Writes outputs/functional/heldout_explain.csv (robust z vs each batch, Batch 3 percentile) and
outputs/functional/figures/heldout_swelling_cards.png (BSE crop + where SOC-1 Si growth lands,
held-out sites next to a typical Batch 3 field). Read-only use of the held-out data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pmdb.segment as segment_mod  # noqa: E402
from pmdb.functional import swell_si  # noqa: E402
from pmdb.io import load_site  # noqa: E402

OUT = ROOT / "outputs" / "functional"
BATCHES = ("Batch_1", "Batch_2", "Batch_3")
SHOW = ["F02_soc100_into_graphite", "F02_soc100_pore_loss", "F02_soc100_si_objects_ratio",
        "F01_si_pore_dist_p50_um", "F01_si_pore_access_frac", "F01_pore_frac",
        "S01_c2_anisotropy", "S02_euler_merge_radius_um", "S03_h0_life_p50_um", "S04_si_beta"]


def _key(df: pd.DataFrame) -> pd.Index:
    return pd.Index(df["batch"] + "/" + df["site"])


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    F = pd.read_csv(OUT / "site_functional.csv")
    S = pd.read_csv(ROOT / "outputs" / "stretch" / "site_stretch.csv")
    L = F.merge(S.drop(columns=["se_detector", "segmenter_version"]), on=["batch", "site"])
    HF = pd.read_csv(OUT / "heldout_site_functional.csv")
    HS = pd.read_csv(ROOT / "outputs" / "stretch" / "heldout_site_stretch.csv")
    H = HF.merge(HS.drop(columns=["se_detector", "segmenter_version"]), on=["batch", "site"])
    return L, H


def robust_z(x: float, ref: np.ndarray) -> float:
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    return float((x - med) / mad) if mad > 0 else float("nan")


def explain(L: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, h in H.iterrows():
        for c in SHOW:
            b3 = L.loc[L.batch == "Batch_3", c].to_numpy()
            row = {"site": h["site"], "column": c, "value": h[c], "B3_median": float(np.median(b3)),
                   "B3_percentile": float((b3 < h[c]).mean() * 100)}
            for b in BATCHES:
                row[f"z_vs_{b[-1]}"] = robust_z(h[c], L.loc[L.batch == b, c].to_numpy())
            rows.append(row)
    return pd.DataFrame(rows)


def swelling_card(ax_bse, ax_sw, batch: str, site: str, title: str, heldout: bool, soc: float = 1.0,
                  crop: tuple[int, int] = (360, 720)) -> None:
    kw = {"data_root": ROOT / "data_heldout", "cache_root": ROOT / "cache_heldout"} if heldout else {}
    raw = load_site(batch, site, resolution="half", normalise="none", **kw)
    bse = np.asarray(raw.image[..., 0], dtype=np.float64)
    masks = segment_mod.segment_bse(bse, raw.nm_per_px)
    h, w = bse.shape
    r0, c0 = h // 2 - crop[0] // 2, w // 2 - crop[1] // 2
    sl = (slice(r0, r0 + crop[0]), slice(c0, c0 + crop[1]))
    si, gr, po = masks.si[sl], masks.graphite[sl], masks.pore[sl]
    grown = swell_si(si, soc) & ~si
    rgb = np.repeat((bse[sl] / 255.0)[..., None], 3, axis=2) * 0.6
    rgb[si] = (1.0, 0.85, 0.1)                       # Si today
    rgb[grown & gr] = (0.85, 0.1, 0.1)                # growth on graphite (constrained)
    rgb[grown & po] = (0.1, 0.5, 1.0)                 # growth into pore
    rgb[grown & ~gr & ~po] = (0.3, 0.8, 0.3)          # growth into binder / CB
    ax_bse.imshow(bse[sl], cmap="gray", vmin=0, vmax=255)
    ax_sw.imshow(rgb)
    for ax in (ax_bse, ax_sw):
        ax.set_xticks([])
        ax.set_yticks([])
    ax_bse.set_ylabel(title, fontsize=9)
    um = 5.0 / (raw.nm_per_px / 1000.0)
    ax_bse.plot([20, 20 + um], [crop[0] - 20, crop[0] - 20], "w-", lw=3)
    ax_bse.text(20, crop[0] - 30, "5 um", color="w", fontsize=8)


def main() -> int:
    L, H = load_tables()
    ex = explain(L, H)
    ex.to_csv(OUT / "heldout_explain.csv", index=False)
    piv = ex.pivot(index="column", columns="site", values="z_vs_3").loc[SHOW].round(2)
    piv["B3_median"] = ex.groupby("column")["B3_median"].first().loc[SHOW].round(3)
    print("robust z vs Batch 3 (MAD units):")
    print(piv.to_string())
    print(ex.pivot(index="column", columns="site", values="B3_percentile").loc[SHOW].round(0).to_string())

    b3 = L[L.batch == "Batch_3"]
    ref = b3.iloc[(b3["F02_soc100_into_graphite"] - b3["F02_soc100_into_graphite"].median()).abs().argsort().iloc[0]]
    cards = [("Batch_3", ref["site"], f"Batch 3 reference\n{ref['site']}", False)] + \
            [("Batch_heldout", s, f"held-out\n{s}", True) for s in H["site"]]
    fig, axes = plt.subplots(len(cards), 2, figsize=(11, 2.9 * len(cards)))
    for (batch, site, title, held), (a1, a2) in zip(cards, axes):
        swelling_card(a1, a2, batch, site, title, held)
        row = (H if held else L)
        r = row[row.site == site].iloc[0]
        a2.set_title(f"SOC 1: on graphite {r['F02_soc100_into_graphite']:.2f}, pore lost {r['F02_soc100_pore_loss']:.2f}, "
                     f"Si {r['F02_soc100_si_frac'] / (1 + 2.8) ** (2 / 3):.3f} -> {r['F02_soc100_si_frac']:.3f}", fontsize=9)
    axes[0, 0].set_title("BSE (half res)", fontsize=9)
    fig.suptitle("Si today (yellow); SOC-1 growth landing on graphite (red), pore (blue), binder/CB (green)", fontsize=10)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "heldout_swelling_cards.png", dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
