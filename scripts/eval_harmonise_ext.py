"""Evaluate the imported harmonisation pipelines (``cache/harmonised_ext``) and draw the visual report.

Compares ``none`` (raw grey at the same half-res coordinates and masks) with the imported intensity routes
``nyul`` / ``basic`` or the shift routes ``spectrum`` / ``fda`` (``--methods``) –
and ``hybrid`` (LUT route) where its cache exists – on: residual Batch-3 black-level gap, imaging-
statistics shortcut classifier (leave-one-out logistic regression on grey statistics), fixed-threshold
phase fractions vs the in-house segmenter, Si/graphite contrast preservation and mean absolute change.
Anchors/segmentation are used as *measurements* only, never inside the imported pipelines.

Figures (outputs/harmonisation_ext/):
    effects_<det>.png     every site, raw | method strips with the exclusion mask overlaid
    effects_examples.png  compact version: strong / mild / clean / Batch-1 / held-out site, BSE + Inlens
    crops_gallery.png     every site whose mask flags more than the border: what is cut out and why
    crops_table.csv       per site/detector masked fractions by reason
    hist_by_method.png, black_level_by_method.png, fractions_by_method.png
    site_metrics.csv, summary.csv, heldout_metrics.csv, REPORT.md
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import LeaveOneOut, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pmdb import clean as C  # noqa: E402
from pmdb import harmonise as H  # noqa: E402
from pmdb import harmonise_ext as X  # noqa: E402
from pmdb import harmonise_shift as S  # noqa: E402
from pmdb.io import get_cache_root, list_clean_sites  # noqa: E402
from pmdb.segment import V0_PARAMS, segment_bse  # noqa: E402

NM = 50.0
STATS = ("p1", "p10", "p50", "p90", "p99", "mean", "std", "frac_0", "frac_255")
STRONG = ("71vgq3fw", "kbdh4tri", "tuy3zymq", "x7u69zsw")
MILD = ("9luzk4jm", "hzumfsms", "ufdvpb81", "ptg8lmto", "xgj4xftb")
REASONS = [  # (bit, label, colour) – what is cut out of an image and why
    (C.BIT_BORDER, "border / colour-marker columns", "#808080"),
    (C.BIT_COLLECTOR, "Cu collector", "#ff7f0e"),
    (C.BIT_FREE_SURFACE, "free surface / beyond coating", "#9467bd"),
    (C.BIT_BAND_BAD, "scan band (uncorrectable)", "#d62728"),
    (C.BIT_CHARGE_LOCAL, "local charging", "#e377c2"),
    (C.BIT_CHARGE_BROAD, "broad charging", "#bcbd22"),
    (C.BIT_CLIP_HIGH, "saturated (255)", "#17becf"),
    (C.BIT_CLIP_LOW, "clipped at 0", "#1f77b4"),
    (C.BIT_CRACK, "crack / delamination (flag)", "#2ca02c"),
]


def _stats(x: np.ndarray) -> dict[str, float]:
    p = np.percentile(x, [1, 10, 50, 90, 99])
    return {"p1": p[0], "p10": p[1], "p50": p[2], "p90": p[3], "p99": p[4], "mean": float(x.mean()),
            "std": float(x.std()), "frac_0": float((x <= 0).mean()), "frac_255": float((x >= 255).mean())}


def _group(site: str) -> str:
    return "strong" if site in STRONG else "mild" if site in MILD else "clean"


def _load(method: str, batch: str, site: str, raw: dict, hybrid_root: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """(image uint8 (H, W, 3), mask uint16 (H, W, 3)) for a method; ``none`` = raw grey."""
    if method == "none":
        return raw["image"], raw["mask"]
    if method == "hybrid":
        # LUT route: apply the per-site LUT to the same raw grey (so masks/coordinates coincide)
        lut = H.load_lut(hybrid_root, "hybrid", batch, site)
        return H.apply_lut(raw["image"], lut), raw["mask"]
    if method in S.METHODS:
        return S.load_shift(batch, site, method)
    img, msk = X.load_ext(batch, site, method)
    return img, msk


CROP_GALLERY_STATS_FRAC = 0.05  # a field enters the crop gallery when > 5 % of a detector's interior is excluded
SMOOTH_BINS = np.arange(0, 255.5, 0.25)


def _fixed_fractions(site_df: pd.DataFrame) -> pd.DataFrame:
    """Fixed-threshold pore/Si fractions with thresholds on *each method's own grey scale*.

    Thresholds are the segmenter's anchor fractions (``V0_PARAMS``) between the median black/graphite/Si
    anchors of the Batch 1/2 sites under that method, so a method that re-scales grey (Nyúl) is judged on
    its own scale; this mirrors the fixed-threshold evaluation of the LUT route.
    """
    out = site_df.copy()
    out["fixed_f_pore"], out["fixed_f_si"], out["t_pore"], out["t_si"] = np.nan, np.nan, np.nan, np.nan
    centres = 0.5 * (SMOOTH_BINS[:-1] + SMOOTH_BINS[1:])
    for m, d in site_df.groupby("method", sort=False):
        ref = d[d.batch.isin(["Batch_1", "Batch_2"])]
        b, g, si = (float(ref[f"BSE_anchor_{a}"].median()) for a in ("black", "graphite", "si"))
        t_pore = b + V0_PARAMS["pore_anchor_frac"] * (g - b)
        t_si = g + V0_PARAMS["si_anchor_frac"] * (si - g)
        for i in d.index:
            h = np.asarray(site_df.at[i, "_hist_smooth"])
            out.at[i, "fixed_f_pore"], out.at[i, "fixed_f_si"] = float(h[centres < t_pore].sum()), float(h[centres > t_si].sum())
            out.at[i, "t_pore"], out.at[i, "t_si"] = t_pore, t_si
    return out.drop(columns=["_hist_smooth"])


def _fill_invalid(ch: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace excluded pixels by the median of the valid ones so they cannot move percentiles or thresholds."""
    f = ch.astype(np.float64)
    if not valid.all():
        f[~valid] = np.median(f[valid]) if valid.any() else 0.0
    return f


def _site_metrics(img: np.ndarray, msk: np.ndarray, raw_img: np.ndarray) -> dict:
    """All statistics, anchors and segmentation thresholds are derived from the stats-valid pixels of each detector;
    excluded pixels (border, bad bands, charging, clipping) are filled with the valid median before any filtering or
    percentile, and phase fractions are counted over KPI-valid pixels only."""
    out: dict = {}
    valid = [C.valid_for_stats(msk[..., c]) for c in range(3)]
    for c, d in enumerate(X.DETECTORS):
        for k, val in _stats(img[..., c][valid[c]].astype(np.float32)).items():
            out[f"{d}_{k}"] = float(val)
    bse = _fill_invalid(img[..., 0], valid[0])
    masks = segment_bse(bse, NM)  # percentile thresholds from the filled BSE = valid pixels only
    for c, d in enumerate(X.DETECTORS):
        ch, v = img[..., c], valid[c]
        anchors = {"black": float(np.percentile(ch[v], 0.5))}
        for a in ("pore", "graphite", "si"):
            sel = getattr(masks, a) & v
            anchors[a] = float(np.median(ch[sel])) if sel.any() else float("nan")
        for a in H.ANCHOR_NAMES:
            out[f"{d}_anchor_{a}"] = anchors[a]
        b, g, s = anchors["black"], anchors["graphite"], anchors["si"]
        out[f"{d}_contrast_ratio"] = (s - b) / (g - b) if g > b else float("nan")
    v0 = C.valid_for_kpis(msk[..., 0])
    g = ndimage.gaussian_filter(bse, V0_PARAMS["gauss_sigma_px"])
    out["_hist_smooth"] = np.histogram(g[v0], bins=SMOOTH_BINS)[0] / v0.sum()  # fixed-threshold fractions later
    out["seg_f_pore"] = float(masks.pore[v0].mean())
    out["seg_f_si"] = float(masks.si[v0].mean())
    out["mean_abs_change"] = float(np.mean(np.abs(img[..., 0][v0].astype(np.int16) - raw_img[..., 0][v0].astype(np.int16))))
    for c, d in enumerate(X.DETECTORS):
        out.update({f"{d}_{k}": v for k, v in _texture(img[..., c], valid[c]).items()})
    return out


TEXTURE = ("hf_ratio", "noise_sigma", "grad_p90", "edge_sigma_px")
METHOD_LABEL = {"none": "raw", "nyul": "N4 + Nyúl–Udupa", "basic": "BaSiC", "hybrid": "hybrid LUT",
                "spectrum": "spectrum (NPS filter)", "fda": "FDA"}
METHOD_BLURB = {
    "nyul": "**nyul** = N4ITK bias-field correction → Nyúl–Udupa piecewise-linear histogram standardisation (`pmdb/harmonise_ext.py`, `docs/harmonisation_ext.md`)",
    "basic": "**basic** = BaSiC flat-field/dark-field + per-image baseline (`pmdb/harmonise_ext.py`)",
    "spectrum": "**spectrum** = radial amplitude-spectrum (MTF/NPS) matching filter to the labelled median, DC kept (`pmdb/harmonise_shift.py`, `docs/harmonisation_shift.md`)",
    "fda": "**fda** = Fourier Domain Adaptation, low-frequency amplitude window (β = 0.01) from the labelled reference (`pmdb/harmonise_shift.py`)",
    "hybrid": "**hybrid** = in-house LUT route (`pmdb/harmonise.py`), for reference",
}


def _texture(ch: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Non-intensity (shift) statistics of one detector, on the invalid-filled image:
    hf_ratio = radial amplitude 0.35–0.5 c/px over 0.05–0.15 c/px (blur/noise texture),
    noise_sigma = robust sigma of the Laplacian (pixel noise), grad_p90 = 90th percentile of the
    gradient magnitude (sharpness), edge_sigma_px = fitted Gaussian blur width from the amplitude
    roll-off between 0.1 and 0.4 c/px (log-amplitude slope vs f^2)."""
    f = S.fill_invalid(ch, valid)
    a = S.radial_amplitude(f, valid)
    fc = S.bin_centres()
    lo, hi = a[(fc >= 0.05) & (fc < 0.15)].mean(), a[(fc >= 0.35) & (fc <= 0.5)].mean()
    sel = (fc >= 0.1) & (fc <= 0.4)
    slope = np.polyfit(fc[sel] ** 2, np.log(np.maximum(a[sel], 1e-9)), 1)[0]  # log A ≈ -2 pi^2 sigma^2 f^2
    sig = float(np.sqrt(max(-slope, 0.0) / (2 * np.pi ** 2)))
    lap = ndimage.laplace(f)
    gy, gx = np.gradient(f)
    g = np.hypot(gx, gy)
    return {"hf_ratio": float(hi / lo), "noise_sigma": float(1.4826 * np.median(np.abs(lap[valid] - np.median(lap[valid]))) / np.sqrt(20)),
            "grad_p90": float(np.percentile(g[valid], 90)), "edge_sigma_px": sig}


def _shortcut_pred(df: pd.DataFrame, cols: list[str], y: np.ndarray) -> np.ndarray | None:
    if len(np.unique(y)) < 2:
        return None
    Xm = np.nan_to_num(df[cols].to_numpy(dtype=float))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_predict(clf, Xm, y, cv=LeaveOneOut())


def _shortcut(df: pd.DataFrame, cols: list[str], y: np.ndarray) -> float:
    """Leave-one-out accuracy of a logistic regression on grey statistics only."""
    pred = _shortcut_pred(df, cols, y)
    return float("nan") if pred is None else float((pred == y).mean())


def _shortcut_recall(df: pd.DataFrame, cols: list[str], y: np.ndarray) -> float:
    """Leave-one-out recall of the positive class (fraction of positives predicted positive)."""
    pred = _shortcut_pred(df, cols, y)
    return float("nan") if pred is None or not y.any() else float(pred[y].mean())


def _summarise(site_df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"{d}_{k}" for d in X.DETECTORS for k in STATS]
    raw = site_df[site_df.method == "none"].set_index("site")
    rows = []
    for m, d in site_df.groupby("method", sort=False):
        d = d.set_index("site")
        b3 = d[d.batch == "Batch_3"]
        strong, rest = b3[b3.group == "strong"], b3[b3.group != "strong"]
        clean = d[d.batch.isin(["Batch_1", "Batch_2"])]
        row: dict = {"method": m}
        for det in X.DETECTORS:
            row[f"black_gap_strong_{det}"] = float(strong[f"{det}_p1"].mean() - rest[f"{det}_p1"].mean())
            row[f"black_sd_{det}"] = float(d[f"{det}_p1"].std())
            row[f"graphite_sd_{det}"] = float(d[f"{det}_anchor_graphite"].std())
            row[f"iqr_gap_strong_{det}"] = float((strong[f"{det}_p90"] - strong[f"{det}_p10"]).mean()
                                                 - (rest[f"{det}_p90"] - rest[f"{det}_p10"]).mean())
        row["fixed_fsi_gap_strong"] = float(strong.fixed_f_si.mean() - rest.fixed_f_si.mean())
        row["fixed_fpore_gap_strong"] = float(strong.fixed_f_pore.mean() - rest.fixed_f_pore.mean())
        row["fixed_vs_seg_fsi_r"] = float(np.corrcoef(d.fixed_f_si, d.seg_f_si)[0, 1])
        row["fixed_vs_seg_fpore_r"] = float(np.corrcoef(d.fixed_f_pore, d.seg_f_pore)[0, 1])
        row["seg_fsi_sd_all"] = float(d.seg_f_si.std())
        row["seg_fsi_gap_b1_b3"] = float(d[d.batch == "Batch_1"].seg_f_si.mean() - b3.seg_f_si.mean())
        cr_now, cr_raw = d.loc[clean.index, "BSE_contrast_ratio"], raw.loc[clean.index, "BSE_contrast_ratio"]
        row["contrast_rel_change_clean"] = float(np.nanmean(np.abs(cr_now / cr_raw - 1)))
        row["contrast_ratio_sd_all"] = float(d["BSE_contrast_ratio"].std())
        row["clip255_BSE"] = float(d["BSE_frac_255"].mean())
        row["clip0_BSE"] = float(d["BSE_frac_0"].mean())
        row["mean_abs_change_clean"] = float(clean.mean_abs_change.mean())
        row["mean_abs_change_strong"] = float(strong.mean_abs_change.mean())
        row["shortcut_batch_acc"] = _shortcut(d, cols, d.batch.to_numpy())
        row["shortcut_batch3_recall"] = _shortcut_recall(d, cols, (d.batch == "Batch_3").to_numpy())
        row["shortcut_strong_vs_rest_b3"] = _shortcut(b3, cols, (b3.group == "strong").to_numpy())
        tcols = [f"{d}_{k}" for d in X.DETECTORS for k in TEXTURE]
        for det in X.DETECTORS:
            row[f"hf_ratio_gap_strong_{det}"] = float(strong[f"{det}_hf_ratio"].mean() / rest[f"{det}_hf_ratio"].mean() - 1)
            row[f"hf_ratio_cv_{det}"] = float(d[f"{det}_hf_ratio"].std() / d[f"{det}_hf_ratio"].mean())
            row[f"noise_sigma_cv_{det}"] = float(d[f"{det}_noise_sigma"].std() / d[f"{det}_noise_sigma"].mean())
            row[f"edge_sigma_sd_{det}"] = float(d[f"{det}_edge_sigma_px"].std())
        row["texture_shortcut_batch_acc"] = _shortcut(d, tcols, d.batch.to_numpy())
        row["texture_shortcut_strong_vs_rest_all"] = _shortcut(d, tcols, (d.group == "strong").to_numpy())
        row["texture_shortcut_batch3_recall"] = _shortcut_recall(d, tcols, (d.batch == "Batch_3").to_numpy())
        row["all_shortcut_batch_acc"] = _shortcut(d, cols + tcols, d.batch.to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def _overlay(ax, img: np.ndarray, mask: np.ndarray, vmin=0, vmax=255, step: int = 2) -> None:
    ax.imshow(img[::step, ::step], cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    m = mask[::step, ::step]
    rgba = np.zeros(m.shape + (4,), dtype=float)
    for bit, _label, colour in REASONS:
        sel = (m & bit) > 0
        if sel.any():
            rgba[sel] = matplotlib.colors.to_rgba(colour, alpha=0.55)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])


def _legend(fig) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.7) for _b, _l, c in REASONS]
    fig.legend(handles, [lbl for _b, lbl, _c in REASONS], loc="lower center", ncol=5, fontsize=8, frameon=False)


def _fig_effects(sites: list[tuple[str, str]], methods: list[str], loader, det: int, out: Path) -> None:
    n = len(sites)
    fig, axes = plt.subplots(n, len(methods), figsize=(5.2 * len(methods), 1.35 * n), squeeze=False)
    for i, (b, s) in enumerate(sites):
        for j, m in enumerate(methods):
            img, msk = loader(m, b, s)
            _overlay(axes[i, j], img[..., det], msk[..., det], step=3)
            if j == 0:
                axes[i, j].set_ylabel(f"{b[-1] if b.startswith('Batch_') else 'H'}/{s}\n{_group(s)}", fontsize=7)
            if i == 0:
                axes[i, j].set_title(m, fontsize=10)
    _legend(fig)
    fig.suptitle(f"{X.DETECTORS[det]}: grey 0–255, same display range for every site and method; "
                 "coloured = excluded by the mask", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    fig.savefig(out, dpi=80)
    plt.close(fig)


def _fig_examples(loader, out: Path, hs, methods=("none", "nyul", "basic")) -> None:
    """Compact before/after: one strong, one mild, one clean Batch-3 site, one Batch-1 site, one held-out; BSE and Inlens."""
    picks = [("Batch_3", "71vgq3fw"), ("Batch_3", "9luzk4jm"), ("Batch_3", "vc2whyaq"), ("Batch_1", "4ih2ggld")]
    if hs is not None and len(hs):
        picks.append((hs.batch.iloc[-1], hs.site.iloc[-1]))
    methods = [m for m in methods if m != "hybrid"] or ["none"]
    fig, axes = plt.subplots(2 * len(picks), len(methods), figsize=(5.4 * len(methods), 1.5 * 2 * len(picks)), squeeze=False)
    for i, (b, s) in enumerate(picks):
        for k, det in enumerate((0, 1)):
            for j, m in enumerate(methods):
                img, msk = loader(m, b, s)
                ax = axes[2 * i + k, j]
                _overlay(ax, img[..., det], msk[..., det], step=3)
                if j == 0:
                    ax.set_ylabel(f"{s} ({_group(s)})\n{X.DETECTORS[det]}", fontsize=8)
                if i == 0 and k == 0:
                    ax.set_title(m, fontsize=11)
    _legend(fig)
    fig.suptitle("Harmonisation effects – " + " | ".join(METHOD_LABEL.get(m, m) for m in methods) + ", same 0–255 display range; coloured = masked", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out, dpi=90)
    plt.close(fig)


def _fig_crops(sites: list[tuple[str, str]], loader, out: Path) -> pd.DataFrame:
    rows = []
    for b, s in sites:
        _img, msk = loader("none", b, s)
        for c, d in enumerate(X.DETECTORS):
            m = msk[..., c]
            inner = (m & C.BIT_BORDER) == 0
            row = {"batch": b, "site": s, "detector": d, "frac_border": float(1 - inner.mean())}
            for bit, label, _c in REASONS[1:]:
                row[f"frac_{C.BIT_NAMES[bit]}"] = float(((m & bit) > 0)[inner].mean())
            row["frac_invalid_kpi_inner"] = float((~C.valid_for_kpis(m))[inner].mean())
            row["frac_invalid_stats_inner"] = float((~C.valid_for_stats(m))[inner].mean())
            rows.append(row)
    table = pd.DataFrame(rows)
    # poorly imaged fields, same rule on every detector: > CROP_GALLERY_STATS_FRAC of the interior excluded from
    # statistics (clipping included), or any KPI-invalid interior pixel (bad band / charging), or a crack flag
    bad = table[(table.frac_invalid_stats_inner > CROP_GALLERY_STATS_FRAC) | (table.frac_invalid_kpi_inner > 0.001)
                | (table.frac_crack > 0)]
    keys = sorted({(r.batch, r.site) for r in bad.itertuples()})
    if not keys:
        return table
    fig, axes = plt.subplots(len(keys), 3, figsize=(16, 1.6 * len(keys)), squeeze=False)
    for i, (b, s) in enumerate(keys):
        img, msk = loader("none", b, s)
        for c, d in enumerate(X.DETECTORS):
            _overlay(axes[i, c], img[..., c], msk[..., c], step=3)
            t = table[(table.batch == b) & (table.site == s) & (table.detector == d)].iloc[0]
            why = ", ".join(f"{lbl} {100 * t[f'frac_{C.BIT_NAMES[bit]}']:.1f} %" for bit, lbl, _c in REASONS[1:]
                            if t[f"frac_{C.BIT_NAMES[bit]}"] > 0.0005)
            axes[i, c].set_title(f"{b}/{s} {d}: {100 * t.frac_invalid_stats_inner:.1f} % of interior excluded"
                                 + (f" – {why}" if why else " – nothing beyond the border"), fontsize=7)
    _legend(fig)
    fig.suptitle("What the mask cuts out of the poorly imaged fields and why\n"
                 f"(fields with > {100 * CROP_GALLERY_STATS_FRAC:.0f} % of a detector's interior excluded, any charging / bad band, or a "
                 "crack; no Cu collector / free surface in any field – border + colour-marker columns are the only fixed crop)", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return table


def _fig_hist(sites, methods, loader, out: Path) -> None:
    fig, axes = plt.subplots(len(X.DETECTORS), len(methods), figsize=(4.2 * len(methods), 8), squeeze=False)
    colours = {"strong": "#d62728", "mild": "#ff7f0e", "clean": "#1f77b4"}
    for j, m in enumerate(methods):
        for b, s in sites:
            img, msk = loader(m, b, s)
            for c in range(3):
                v = C.valid_for_stats(msk[..., c])
                h, _ = np.histogram(img[..., c][v], bins=np.arange(257), density=True)
                axes[c, j].plot(h, color=colours[_group(s)], alpha=0.5, lw=0.8)
                axes[c, j].set_yticks([])
                if c == 0:
                    axes[c, j].set_title(m)
                if j == 0:
                    axes[c, j].set_ylabel(X.DETECTORS[c])
    handles = [plt.Line2D([], [], color=c, label=g) for g, c in colours.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Grey-level histograms of valid pixels per site (Batch-3 strong / mild groups vs clean sites)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out, dpi=90)
    plt.close(fig)


def _fig_black(site_df: pd.DataFrame, out: Path) -> None:
    methods = list(dict.fromkeys(site_df.method))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colours = {"strong": "#d62728", "mild": "#ff7f0e", "clean": "#1f77b4"}
    for c, d in enumerate(X.DETECTORS):
        for g, col in colours.items():
            sub = site_df[site_df.group == g]
            for m_i, m in enumerate(methods):
                vals = sub[sub.method == m][f"{d}_p1"]
                axes[c].scatter(np.full(len(vals), m_i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals)),
                                vals, color=col, s=14, alpha=0.8, label=g if (c == 0 and m_i == 0) else None)
        axes[c].set_xticks(range(len(methods)), methods)
        axes[c].set_title(f"{d}: 1st percentile of valid pixels (black level)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def _fig_fractions(site_df: pd.DataFrame, out: Path) -> None:
    methods = list(dict.fromkeys(site_df.method))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colours = {"strong": "#d62728", "mild": "#ff7f0e", "clean": "#1f77b4"}
    for ax, key, title in zip(axes, ["fixed_f_si", "fixed_f_pore"], ["fixed-threshold Si fraction", "fixed-threshold pore fraction"]):
        for g, col in colours.items():
            sub = site_df[site_df.group == g]
            for m_i, m in enumerate(methods):
                vals = sub[sub.method == m][key]
                ax.scatter(np.full(len(vals), m_i) + np.random.default_rng(1).uniform(-0.15, 0.15, len(vals)),
                           vals, color=col, s=14, alpha=0.8, label=g if m_i == 0 else None)
        ax.set_xticks(range(len(methods)), methods)
        ax.set_title(title)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "harmonisation_ext"))
    ap.add_argument("--methods", nargs="+", default=None,
                    help="default: none nyul basic hybrid; with --figures-only, the methods saved in summary.csv")
    ap.add_argument("--no-heldout", action="store_true")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures/report from the saved site_metrics.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.methods = resolve_methods(args.methods, args.figures_only, out)

    hybrid_root = get_cache_root()  # H.load_lut / H.load_reference append 'harmonised/<method>' themselves
    methods = [m for m in args.methods if m != "hybrid" or (hybrid_root / "harmonised" / "hybrid" / "luts.npz").exists()]
    sites = [(b, s) for b, s in zip(*[list_clean_sites()[k] for k in ("batch", "site")])]
    raw_cache: dict = {}

    def raw_of(b, s):
        if (b, s) not in raw_cache:
            imgs, msks = [], []
            for d in X.DETECTORS:
                g, m = X.load_half_raw(b, s, d)
                imgs.append(np.clip(np.round(g), 0, 255).astype(np.uint8))
                msks.append(m)
            raw_cache[(b, s)] = {"image": np.stack(imgs, -1), "mask": np.stack(msks, -1)}
        return raw_cache[(b, s)]

    def loader(m, b, s):
        return _load(m, b, s, raw_of(b, s), hybrid_root)

    if args.figures_only:
        site_df = pd.read_csv(out / "site_metrics.csv")
        summary = pd.read_csv(out / "summary.csv")
        hrows = pd.read_csv(out / "heldout_metrics.csv").to_dict("records") if (out / "heldout_metrics.csv").exists() else []
        hs = list_clean_sites(heldout=True) if hrows else None
        _figures(out, methods, sites, hs, hrows, loader, site_df, summary)
        return

    rows = []
    for b, s in sites:
        raw = raw_of(b, s)
        for m in methods:
            img, msk = loader(m, b, s)
            row = {"method": m, "batch": b, "site": s, "group": _group(s)}
            row.update(_site_metrics(img, msk, raw["image"]))
            rows.append(row)
        print(f"  metrics {b}/{s}", flush=True)
    site_df = _fixed_fractions(pd.DataFrame(rows))
    site_df.to_csv(out / "site_metrics.csv", index=False)
    summary = _summarise(site_df)
    summary.to_csv(out / "summary.csv", index=False)

    # held-out: apply-only metrics (no fitting), same loader
    hrows = []
    if not args.no_heldout:
        try:
            hs = list_clean_sites(heldout=True)
            for b, s in zip(hs.batch, hs.site):
                raw = raw_of(b, s)
                for m in methods:
                    if m == "hybrid":
                        continue
                    img, msk = loader(m, b, s)
                    row = {"method": m, "batch": b, "site": s, "group": _group(s)}
                    row.update(_site_metrics(img, msk, raw["image"]))
                    hrows.append(row)
            # held-out fixed thresholds: those of the labelled sites under the same method
            held_df = pd.DataFrame(hrows)
            thr = site_df.groupby("method")[["t_pore", "t_si"]].first()
            centres = 0.5 * (SMOOTH_BINS[:-1] + SMOOTH_BINS[1:])
            held_df["fixed_f_pore"] = [float(np.asarray(h)[centres < thr.loc[m, "t_pore"]].sum()) for h, m in zip(held_df._hist_smooth, held_df.method)]
            held_df["fixed_f_si"] = [float(np.asarray(h)[centres > thr.loc[m, "t_si"]].sum()) for h, m in zip(held_df._hist_smooth, held_df.method)]
            hrows = held_df.drop(columns=["_hist_smooth"]).to_dict("records")
            pd.DataFrame(hrows).to_csv(out / "heldout_metrics.csv", index=False)
        except FileNotFoundError as e:
            print(f"  held-out skipped: {e}")

    _figures(out, methods, sites, hs if hrows else None, hrows, loader, site_df, summary)


def resolve_methods(methods, figures_only: bool, out: Path) -> list[str]:
    """Explicit --methods wins; --figures-only takes the saved summary.csv methods (or fails); else the intensity default."""
    if methods is not None:
        return list(methods)
    if figures_only:
        summ = out / "summary.csv"
        if not summ.exists():
            raise SystemExit(f"--figures-only: cannot determine methods, {summ} is missing; pass --methods")
        return list(pd.read_csv(summ)["method"])
    return ["none", "nyul", "basic", "hybrid"]


def _figures(out: Path, methods, sites, hs, hrows, loader, site_df, summary) -> None:
    print("  figures", flush=True)
    ext_methods = [m for m in methods if m in ("none", "nyul", "basic") or m in S.METHODS]
    for det in range(3):
        _fig_effects(sites, ext_methods, loader, det, out / f"effects_{X.DETECTORS[det]}.png")
    _fig_examples(loader, out / "effects_examples.png", hs, ext_methods)
    all_sites = sites + ([(b, s) for b, s in zip(hs.batch, hs.site)] if hs is not None else [])
    table = _fig_crops(all_sites, loader, out / "crops_gallery.png")
    table.to_csv(out / "crops_table.csv", index=False)
    _fig_hist(sites, methods, loader, out / "hist_by_method.png")
    _fig_black(site_df, out / "black_level_by_method.png")
    _fig_fractions(site_df, out / "fractions_by_method.png")
    _write_report(out, summary, site_df, pd.DataFrame(hrows), table)
    print(f"wrote {out}")


def _write_report(out: Path, summary: pd.DataFrame, site_df: pd.DataFrame, held: pd.DataFrame, crops: pd.DataFrame) -> None:
    s = summary.set_index("method")
    shift_run = any(m in S.METHODS for m in s.index)
    L = ["# " + ("Shift (non-intensity) harmonisation – results" if shift_run else "Imported harmonisation pipelines – results"), "",
         "Methods compared on the `pmdb.clean`-masked half-res grey (`none` is the raw grey at the same coordinates):", ""]
    L += [f"* {METHOD_BLURB[m]}" for m in s.index if m in METHOD_BLURB]
    L += ["", "## Summary (labelled sites)", ""]
    cols = ["black_gap_strong_BSE", "black_gap_strong_Inlens", "black_gap_strong_SE_type", "iqr_gap_strong_BSE",
            "black_sd_BSE", "graphite_sd_BSE", "contrast_rel_change_clean", "contrast_ratio_sd_all",
            "fixed_fsi_gap_strong", "fixed_fpore_gap_strong", "fixed_vs_seg_fsi_r", "seg_fsi_sd_all",
            "clip0_BSE", "clip255_BSE", "mean_abs_change_clean", "mean_abs_change_strong",
            "shortcut_batch_acc", "shortcut_batch3_recall", "shortcut_strong_vs_rest_b3"]
    cols += [c for c in ("hf_ratio_gap_strong_BSE", "hf_ratio_cv_BSE", "noise_sigma_cv_BSE", "edge_sigma_sd_BSE",
                         "hf_ratio_gap_strong_Inlens", "hf_ratio_cv_Inlens", "noise_sigma_cv_Inlens",
                         "texture_shortcut_batch_acc", "texture_shortcut_strong_vs_rest_all", "texture_shortcut_batch3_recall",
                         "all_shortcut_batch_acc") if c in s.columns]
    L.append("| metric | " + " | ".join(s.index) + " |")
    L.append("|---|" + "---|" * len(s.index))
    for c in cols:
        L.append(f"| {c} | " + " | ".join(f"{s.loc[m, c]:.3f}" for m in s.index) + " |")
    L += ["", "* `black_gap_strong_*`: mean 1st percentile of the 4 strong Batch-3 sites minus the other Batch-3 sites (grey levels; 0 = artefact removed).",
          "* `iqr_gap_strong_BSE`: p90−p10 width difference strong vs rest (gain artefact; BaSiC cannot correct gain).",
          "* `contrast_rel_change_clean`: relative change of the Si/graphite contrast ratio on Batch 1/2 sites (material preservation; 0 = untouched).",
          "* `fixed_fsi_gap_strong`, `seg_fsi_sd_all`: Si fraction at a fixed threshold (set on each method's own grey scale from the Batch 1/2 anchors) /",
          "  with the segmenter. Nyúl matches 11 landmarks per site, which by construction forces equal percentile positions and so pulls phase",
          "  fractions towards a common value (the known limitation of histogram standardisation).",
          "* `mean_abs_change_*`: |method − raw| per pixel in stored uint8 units; for `nyul` (standard scale) and `basic` (raw − bᵢ + 64) this includes the scale change itself.",
          "* `shortcut_*`: leave-one-out logistic regression on grey statistics only; `_acc`/`_b3` = accuracy (chance 0.45 batch, 0.76 strong-vs-rest), `batch3_recall` = fraction of Batch-3 sites predicted Batch 3.",
          "* `hf_ratio_gap_strong_*`: relative high-frequency (0.35–0.5 c/px) / low-frequency amplitude ratio of the strong sites vs the rest (0 = texture shift removed);",
          "  `hf_ratio_cv_*`, `noise_sigma_cv_*`, `edge_sigma_sd_*`: across-site spread of texture, noise and blur width (lower = more homogeneous).",
          "* `texture_shortcut_*`: the same leave-one-out classifier on the four texture statistics only (is the site still identifiable from blur/noise?); `all_shortcut_batch_acc` uses grey + texture.", ""]
    L += ["## What is cut out ('crops')", "",
          "No field contains a Cu collector or the coating free surface (`collector_found`/`free_surface_found` are False on all 34 sites), so",
          "nothing is cropped for those reasons; `crops_gallery.png` shows the fields where the mask excludes more than 0.2 % of the interior and why", ""]
    worst = crops.sort_values("frac_invalid_stats_inner", ascending=False).head(12)
    L.append("| batch | site | detector | interior excluded | border |")
    L.append("|---|---|---|---|---|")
    for r in worst.itertuples():
        L.append(f"| {r.batch} | {r.site} | {r.detector} | {100 * r.frac_invalid_stats_inner:.2f} % | {100 * r.frac_border:.1f} % |")
    if len(held):
        L += ["", "## Held-out (models fitted on labelled sites only)", ""]
        for m, d in held.groupby("method", sort=False):
            L.append(f"* {m}: BSE p1 = " + ", ".join(f"{r.site} {r.BSE_p1:.1f}" for r in d.itertuples())
                     + "; BSE graphite anchor = " + ", ".join(f"{r.BSE_anchor_graphite:.0f}" for r in d.itertuples()))
    L += ["", "## Figures", "", "`effects_BSE.png`, `effects_Inlens.png`, `effects_SE_type.png` (" + " | ".join(m for m in s.index if m != "hybrid") + " for every site, mask overlaid),",
          "`crops_gallery.png`, `hist_by_method.png`, `black_level_by_method.png`, `fractions_by_method.png`."]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
