"""Calibrate seam scores from scripts/stitch_seams.py and reassemble parent images.

Reads outputs/stitching/{pair_scores,controls,sites}.csv (produced on Modal), writes
outputs/stitching/{same_parent_verdicts.csv,chains.csv,null_summary.csv} and stitched
preview PNGs (from the half-res cache). Run locally: python scripts/stitch_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pmdb.io import load_site  # noqa: E402

OUT = ROOT / "outputs/stitching"
Z_MIN, MAX_SHIFT = 4.0, 8  # verdict: z (vs different-parent null) and |dy| near 0
STAT = "mp"  # particle-scale band, best over |shift| <= 64 (LR) / any shift (TB)
LR = {"A|B", "B|A"}


def main() -> None:
    pairs = pd.read_csv(OUT / "pair_scores.csv")
    ctrl = pd.read_csv(OUT / "controls.csv")
    sites = pd.read_csv(OUT / "sites.csv").set_index("site")
    pairs["orient"] = np.where(pairs["ea"].isin(["L", "R"]), "LR", "TB")
    unfl = pairs[pairs["placement"] != "flip"].copy()

    # null: different-parent pairs, unflipped placements, per orientation
    null = unfl[~unfl["same_parent"]].groupby("orient")[STAT].agg(["mean", "std", "max", "count"])
    null["p99"] = unfl[~unfl["same_parent"]].groupby("orient")[STAT].quantile(0.99)
    unfl["z"] = (unfl[STAT] - unfl["orient"].map(null["mean"])) / unfl["orient"].map(null["std"])
    nulls = {o: np.sort(unfl[(~unfl["same_parent"]) & (unfl["orient"] == o)][STAT].values)
             for o in ("LR", "TB")}
    unfl["p_emp"] = [(np.sum(nulls[o] >= v) + 1) / (len(nulls[o]) + 1)
                     for o, v in zip(unfl["orient"], unfl[STAT])]

    # positive control summary (same statistic, same search)
    pos = ctrl.groupby(["orient", "gap"])[[STAT, f"{STAT}0", "lp0"]].agg(["median", "min"]).round(3)
    pos.to_csv(OUT / "positive_control_summary.csv")

    sp = unfl[unfl["same_parent"]].copy()
    sp["pair"] = sp["a"] + "-" + sp["b"]
    best = sp.sort_values("z", ascending=False).groupby("pair").head(1).copy()
    # same-parent wrong placements: the other 3 unflipped placements of every same-parent pair
    wrong = sp.drop(best.index)
    best["second_z"] = best["pair"].map(wrong.groupby("pair")["z"].max())
    flips = pairs[(pairs["placement"] == "flip") & pairs["same_parent"]].copy()
    flips["z"] = (flips[STAT] - flips["orient"].map(null["mean"])) / flips["orient"].map(null["std"])
    flips["pair"] = flips["a"] + "-" + flips["b"]
    best["best_flip_z"] = best["pair"].map(flips.groupby("pair")["z"].max())
    shift_col = f"{STAT}_shift"
    best["verdict"] = np.where(
        (best["z"] > Z_MIN) & (best[shift_col].abs() <= MAX_SHIFT) & best["placement"].isin(LR),
        "ABUT", "no seam")
    # directed edge: left crop | right crop
    best["left"] = np.where(best["placement"] == "A|B", best["a"], best["b"])
    best["right"] = np.where(best["placement"] == "A|B", best["b"], best["a"])
    best["parent"] = best["a"].map(sites["parent"])
    best["batch_a"] = best["a"].map(sites["batch"])
    best["batch_b"] = best["b"].map(sites["batch"])
    cols = ["parent", "a", "b", "batch_a", "batch_b", "placement", STAT, shift_col, f"{STAT}0",
            "lp0", "z", "p_emp", "second_z", "best_flip_z", "verdict", "left", "right"]
    best = best.sort_values(["parent", "z"], ascending=[True, False])[cols]
    best.round(3).to_csv(OUT / "same_parent_verdicts.csv", index=False)

    wrong_null = wrong[STAT].describe()
    summ = null.round(3).copy()
    summ.loc["same_parent_wrong_placements"] = [wrong_null["mean"], wrong_null["std"],
                                                 wrong_null["max"], wrong_null["count"],
                                                 wrong[STAT].quantile(0.99)]
    summ.round(3).to_csv(OUT / "null_summary.csv")

    # chains
    ab = best[best["verdict"] == "ABUT"]
    right_of = dict(zip(ab["left"], ab["right"]))
    assert len(right_of) == len(ab), "a crop has two right neighbours"
    assert len(set(ab["right"])) == len(ab), "a crop has two left neighbours"
    chains = []
    for parent, g in sites.groupby("parent"):
        members = list(g.index)
        heads = [s for s in members if s not in set(ab["right"])]
        for h in heads:
            chain = [h]
            while chain[-1] in right_of:
                chain.append(right_of[chain[-1]])
            chains.append({"parent": parent, "n_members": len(members),
                           "chain": " | ".join(chain),
                           "batches": " | ".join(sites.loc[c, "batch"] for c in chain),
                           "art_left_of_first": sites.loc[chain[0], "art_left"],
                           "art_right_of_last": sites.loc[chain[-1], "art_right"]})
    chains = pd.DataFrame(chains)
    chains.to_csv(OUT / "chains.csv", index=False)

    print("null:\n", summ.round(3).to_string())
    print("positive control:\n", pos.to_string())
    print(best.drop(columns=["left", "right"]).round(3).to_string(index=False))
    print(chains.to_string(index=False))
    previews(chains, sites)


def _load_half(batch: str, site: str) -> np.ndarray:
    kw = {}
    if batch == "Batch_heldout":
        kw = dict(data_root=ROOT / "data_heldout", cache_root=ROOT / "cache_heldout")
    elif batch == "Batch_test":
        kw = dict(data_root=ROOT / "data_test", cache_root=ROOT / "cache_test")
    s = load_site(batch, site, resolution="half", normalise="none", **kw)
    return s.image[..., 0].astype(np.float32)  # BSE


def previews(chains: pd.DataFrame, sites: pd.DataFrame) -> None:
    rows, zooms = [], []
    for _, c in chains.iterrows():
        names = c["chain"].split(" | ")
        if len(names) < 2:
            continue
        imgs = [_load_half(sites.loc[n, "batch"], n) for n in names]
        # per-crop percentile stretch so batch grey-level offsets do not hide the seam
        imgs = [np.clip((i - np.percentile(i, 1)) / (np.percentile(i, 99) - np.percentile(i, 1)), 0, 1)
                for i in imgs]
        strip = np.concatenate(imgs, axis=1)
        im = Image.fromarray((strip * 255).astype(np.uint8))
        im = im.resize((strip.shape[1] // 4, strip.shape[0] // 4), Image.LANCZOS)
        rows.append(np.asarray(im))
        # 1:1 zoom (half-res) on the first seam: 400 px either side, middle 400 rows
        w0 = imgs[0].shape[1]
        h = strip.shape[0] // 2
        zooms.append(strip[h - 200:h + 200, w0 - 400:w0 + 400])
    width = max(r.shape[1] for r in rows)
    pad = [np.pad(r, ((0, 12), (0, width - r.shape[1])), constant_values=255) for r in rows]
    Image.fromarray(np.concatenate(pad, 0)).save(OUT / "stitched_all_parents.png")
    zpad = [np.pad((z * 255).astype(np.uint8), ((0, 0), (0, 12)), constant_values=255) for z in zooms]
    Image.fromarray(np.concatenate(zpad, 1)).save(OUT / "seam_zooms.png")


if __name__ == "__main__":
    main()
