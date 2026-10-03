"""Evaluate every harmonisation method: how much artefact it removes vs how much material signal it keeps.

Writes to outputs/harmonisation/:
    site_metrics.csv     per method x site: intensity stats, re-estimated anchors, fixed-threshold phase fractions
    summary.csv          one row per method (see docs/harmonisation.md for the column definitions)
    groups.csv           which Batch_3 sites are in the strong / mild / clean black-level groups
    heldout_metrics.csv  the same site metrics for the held-back sites
    *.png                figures
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pmdb import harmonise as H
from pmdb.io import normalise_image
from pmdb.segment import V0_PARAMS, segment_bse
from pmdb.stats import raw_intensity_stats

NM = 50.0
EVAL_METHODS = list(H.METHODS) + ["percentile"]
BATCH_COLOURS = {"Batch_1": "tab:blue", "Batch_2": "tab:green", "Batch_3": "tab:red", "Batch_heldout": "k"}
STAT_KEYS = ("p0_5", "p1", "p50", "p99", "mean", "std", "frac_255")


def _harmonised(arr: np.ndarray, method: str, cache_root: Path, batch: str, site: str) -> np.ndarray:
    if method == "percentile":
        return np.round(normalise_image(arr) * 255).astype(np.uint8)
    return H.apply_lut(arr, H.load_lut(cache_root, method, batch, site))


def _fixed_thresholds(ref: H.Reference) -> tuple[float, float]:
    a = ref.anchors["BSE"]
    t_pore = a["black"] + V0_PARAMS["pore_anchor_frac"] * (a["graphite"] - a["black"])
    t_si = a["graphite"] + V0_PARAMS["si_anchor_frac"] * (a["si"] - a["graphite"])
    return float(t_pore), float(t_si)


def _site_metrics(arr: np.ndarray, raw: np.ndarray, t_pore: float, t_si: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for c, ch in enumerate(H.CHANNELS):
        st = raw_intensity_stats(arr[..., c])
        for k in STAT_KEYS:
            out[f"{ch}_{k}"] = st[k]
    anchors, _ = H.estimate_anchors(arr, NM)
    for ch in H.CHANNELS:
        for a in H.ANCHOR_NAMES:
            out[f"{ch}_anchor_{a}"] = anchors[ch][a]
        b, g, s = anchors[ch]["black"], anchors[ch]["graphite"], anchors[ch]["si"]
        out[f"{ch}_contrast_ratio"] = (s - b) / (g - b) if g > b else float("nan")
    g = ndimage.gaussian_filter(arr[..., 0].astype(np.float64), V0_PARAMS["gauss_sigma_px"])
    out["fixed_f_pore"] = float((g < t_pore).mean())
    out["fixed_f_si"] = float((g > t_si).mean())
    masks = segment_bse(arr[..., 0].astype(np.float64), NM)
    out["seg_f_pore"] = float(masks.pore.mean())
    out["seg_f_si"] = float(masks.si.mean())
    out["mean_abs_change"] = float(np.mean(np.abs(arr.astype(np.int16) - raw.astype(np.int16))))
    return out


def _collect(cache_root: Path, ref_root: Path, methods: list[str]) -> pd.DataFrame:
    manifest = pd.read_csv(cache_root / "half" / "manifest.csv")
    ref = H.load_reference(ref_root, "affine2")
    t_pore, t_si = _fixed_thresholds(ref)
    rows = []
    for _, r in manifest.iterrows():
        raw = np.load(cache_root / "half" / f"{r.batch}__{r.site}.npz")["image"]
        for m in methods:
            arr = _harmonised(raw, m, cache_root, r.batch, r.site)
            row = {"method": m, "batch": r.batch, "site": r.site}
            row.update(_site_metrics(arr, raw, t_pore, t_si))
            rows.append(row)
        print(f"  {r.batch}/{r.site} done", flush=True)
    return pd.DataFrame(rows)


def _groups(anchors: pd.DataFrame) -> pd.DataFrame:
    g = anchors[["batch", "site", "BSE_black"]].copy()
    g["group"] = np.where(g.BSE_black >= 15, "strong", np.where(g.BSE_black >= 2, "mild", "clean"))
    return g


def _shortcut_accuracy(df: pd.DataFrame, feature_cols: list[str], y: np.ndarray) -> float:
    X = df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(clf, X, y, cv=LeaveOneOut())
    return float((pred == y).mean())


def _summarise(site_df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    stat_cols = [f"{ch}_{k}" for ch in H.CHANNELS for k in STAT_KEYS]
    raw = site_df[site_df.method == "none"].set_index("site")
    rows = []
    for m, d in site_df.groupby("method", sort=False):
        d = d.merge(groups[["site", "group"]], on="site").set_index("site")
        b3 = d[d.batch == "Batch_3"]
        strong, rest = b3[b3.group == "strong"], b3[b3.group != "strong"]
        clean = d[d.batch.isin(["Batch_1", "Batch_2"])]
        row: dict[str, float | str] = {"method": m}
        for ch in H.CHANNELS:
            row[f"black_sd_{ch}"] = float(d[f"{ch}_p0_5"].std())
            row[f"black_gap_strong_{ch}"] = float(strong[f"{ch}_p0_5"].mean() - rest[f"{ch}_p0_5"].mean())
            row[f"graphite_sd_{ch}"] = float(d[f"{ch}_anchor_graphite"].std())
        row["fixed_fsi_gap_strong"] = float(strong.fixed_f_si.mean() - rest.fixed_f_si.mean())
        row["fixed_fpore_gap_strong"] = float(strong.fixed_f_pore.mean() - rest.fixed_f_pore.mean())
        row["fixed_fsi_sd_batch3"] = float(b3.fixed_f_si.std())
        row["fixed_vs_seg_fsi_r"] = float(np.corrcoef(d.fixed_f_si, d.seg_f_si)[0, 1])
        row["fixed_vs_seg_fpore_r"] = float(np.corrcoef(d.fixed_f_pore, d.seg_f_pore)[0, 1])
        row["fixed_fsi_gap_b1_b3"] = float(d[d.batch == "Batch_1"].fixed_f_si.mean() - b3.fixed_f_si.mean())
        cr_now = d.loc[clean.index, "BSE_contrast_ratio"]
        cr_raw = raw.loc[clean.index, "BSE_contrast_ratio"]
        row["contrast_rel_change_clean"] = float(np.nanmean(np.abs(cr_now / cr_raw - 1)))
        row["contrast_ratio_sd_all"] = float(d["BSE_contrast_ratio"].std())
        for ch in H.CHANNELS:
            row[f"clip255_strong_{ch}"] = float(strong[f"{ch}_frac_255"].mean())
            row[f"clip255_clean_{ch}"] = float(clean[f"{ch}_frac_255"].mean())
        row["mean_abs_change_clean"] = float(clean.mean_abs_change.mean())
        row["mean_abs_change_strong"] = float(strong.mean_abs_change.mean())
        row["shortcut_batch_acc"] = _shortcut_accuracy(d, stat_cols, d.batch.to_numpy())
        row["shortcut_batch3_recall"] = _shortcut_accuracy(d, stat_cols, (d.batch == "Batch_3").to_numpy())
        row["shortcut_strong_vs_rest_b3"] = _shortcut_accuracy(b3, stat_cols, (b3.group == "strong").to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def _fig_p05(site_df: pd.DataFrame, groups: pd.DataFrame, out: Path) -> None:
    methods = list(dict.fromkeys(site_df.method))
    fig, axes = plt.subplots(3, len(methods), figsize=(3.2 * len(methods), 8), sharey="row")
    d = site_df.merge(groups[["site", "group"]], on="site")
    for j, m in enumerate(methods):
        dm = d[d.method == m]
        for i, ch in enumerate(H.CHANNELS):
            ax = axes[i, j]
            for k, b in enumerate(["Batch_1", "Batch_2", "Batch_3"]):
                db = dm[dm.batch == b]
                x = np.full(len(db), k) + np.linspace(-0.25, 0.25, len(db))
                mk = np.where(db.group == "strong", "s", np.where(db.group == "mild", "^", "o"))
                for xi, yi, mki in zip(x, db[f"{ch}_p0_5"], mk):
                    ax.scatter(xi, yi, c=BATCH_COLOURS[b], marker=mki, s=28, edgecolor="k", linewidth=0.3)
            ax.set_xticks([0, 1, 2], ["B1", "B2", "B3"])
            if i == 0:
                ax.set_title(m)
            if j == 0:
                ax.set_ylabel(f"{ch} p0.5 (black level)")
            ax.grid(alpha=0.3)
    fig.suptitle("Black level (0.5th percentile) per site after each method. Squares = strong group, triangles = mild")
    fig.tight_layout()
    fig.savefig(out / "black_level_by_method.png", dpi=130)
    plt.close(fig)


def _fig_hist(cache_root: Path, groups: pd.DataFrame, methods: list[str], out: Path) -> None:
    manifest = pd.read_csv(cache_root / "half" / "manifest.csv")
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 3.6), sharey=True)
    grp = groups.set_index("site").group
    for _, r in manifest.iterrows():
        raw = np.load(cache_root / "half" / f"{r.batch}__{r.site}.npz")["image"]
        for ax, m in zip(axes, methods):
            arr = _harmonised(raw, m, cache_root, r.batch, r.site)
            h = np.bincount(arr[..., 0].ravel(), minlength=256) / arr[..., 0].size
            strong = grp[r.site] == "strong"
            ax.plot(h, color=BATCH_COLOURS[r.batch], lw=1.8 if strong else 0.6, ls="--" if strong else "-",
                    alpha=1.0 if strong else 0.6)
    for ax, m in zip(axes, methods):
        ax.set_title(m)
        ax.set_xlim(0, 180)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 0.1)
        ax.set_xlabel("BSE grey level")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of pixels")
    fig.suptitle("BSE histograms, all 31 sites (blue B1, green B2, red B3; dashed = strong black-level group)")
    fig.tight_layout()
    fig.savefig(out / "bse_hist_by_method.png", dpi=130)
    plt.close(fig)


def _fig_visual(cache_root: Path, methods: list[str], out: Path) -> None:
    picks = [("Batch_3", "71vgq3fw", "B3 strong"), ("Batch_3", "9luzk4jm", "B3 mild"),
             ("Batch_3", "0grcilhi", "B3 clean"), ("Batch_1", "4ih2ggld", "B1")]
    fig, axes = plt.subplots(len(picks), len(methods), figsize=(3.0 * len(methods), 2.2 * len(picks)))
    for i, (b, s, label) in enumerate(picks):
        raw = np.load(cache_root / "half" / f"{b}__{s}.npz")["image"]
        for j, m in enumerate(methods):
            arr = _harmonised(raw, m, cache_root, b, s)
            crop = arr[200:500, 1000:1700, 0]
            ax = axes[i, j]
            ax.imshow(crop, cmap="gray", vmin=0, vmax=160, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(m)
            if j == 0:
                ax.set_ylabel(f"{label}\n{s}")
    fig.suptitle("BSE crops on one fixed grey scale (0-160)")
    fig.tight_layout()
    fig.savefig(out / "visual_by_method.png", dpi=110)
    plt.close(fig)


def _fig_fractions(site_df: pd.DataFrame, groups: pd.DataFrame, out: Path) -> None:
    methods = list(dict.fromkeys(site_df.method))
    d = site_df.merge(groups[["site", "group"]], on="site")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ["fixed_f_si", "fixed_f_pore"],
                              ["Si fraction, fixed global threshold", "Pore fraction, fixed global threshold"]):
        for j, m in enumerate(methods):
            dm = d[d.method == m]
            for b in ["Batch_1", "Batch_2", "Batch_3"]:
                db = dm[dm.batch == b]
                x = np.full(len(db), j) + {"Batch_1": -0.25, "Batch_2": 0.0, "Batch_3": 0.25}[b]
                mk = np.where(db.group == "strong", "s", np.where(db.group == "mild", "^", "o"))
                for xi, yi, mki in zip(x, db[col], mk):
                    ax.scatter(xi, yi, c=BATCH_COLOURS[b], marker=mki, s=26, edgecolor="k", linewidth=0.3)
        seg = d[d.method == "none"]
        ax.axhline(seg[col.replace("fixed", "seg")].mean(), color="grey", ls=":", lw=1,
                   label="mean of per-image segmenter")
        ax.set_xticks(range(len(methods)), methods, rotation=20)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Phase fractions with ONE global threshold: strong-group squares should sit with the other red points")
    fig.tight_layout()
    fig.savefig(out / "fixed_threshold_fractions.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", default=str(REPO_ROOT / "cache"))
    ap.add_argument("--heldout-cache-root", default=str(REPO_ROOT / "cache_heldout"))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "harmonisation"))
    ap.add_argument("--methods", nargs="+", default=EVAL_METHODS)
    args = ap.parse_args()
    cache_root, out = Path(args.cache_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    anchors = pd.read_csv(cache_root / H.HARMONISED_DIRNAME / "anchors.csv")
    groups = _groups(anchors)
    groups.to_csv(out / "groups.csv", index=False)
    print("Batch_3 groups:", groups[groups.batch == "Batch_3"].groupby("group").site.apply(list).to_dict())

    print("Collecting per-site metrics ...")
    methods = list(dict.fromkeys(["none", *args.methods]))  # "none" is the baseline for the summary/figures
    site_df = _collect(cache_root, cache_root, methods)
    site_df.to_csv(out / "site_metrics.csv", index=False)
    summary = _summarise(site_df, groups)
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print(summary.set_index("method").T)

    heldout = Path(args.heldout_cache_root)
    if (heldout / "half" / "manifest.csv").exists():
        h = _collect(heldout, cache_root, methods)
        h.to_csv(out / "heldout_metrics.csv", index=False)

    print("Figures ...")
    _fig_p05(site_df, groups, out)
    _fig_hist(cache_root, groups, methods, out)
    _fig_visual(cache_root, methods, out)
    _fig_fractions(site_df, groups, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
