"""D10 pre-checks 1-2 on per-tile KPIs: is the batch signal in a few witness tiles?

Reads outputs/kpis/tile_kpis.csv only. Writes to outputs/pooling_checks/:
  - tile_predictions.csv:      LOSO tile-classifier probabilities (multiclass, B1_vs_B3, B2_vs_B3)
  - site_tile_spread.csv:      per (variant, site) spread of tile predictions, category, pooling
  - batch_summary.csv:         per (variant, batch) category counts, false-alarm rate, verdict
  - pooling_side_note.csv:     per variant accuracy of mean vs max pooling (side note)
  - variance_by_feature.csv:   ICC, nested variance fractions, single-tile flags per feature
  - site_feature_spread.csv:   per (site, feature) within-site spread
  - fig_tile_probs.png, fig_variance.png
  - results.md:                script-generated report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb.kpis import catalogue_columns  # noqa: E402

BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
VARIANTS = {
    "multiclass": ["Batch_1", "Batch_2", "Batch_3"],
    "B1_vs_B3": ["Batch_1", "Batch_3"],
    "B2_vs_B3": ["Batch_2", "Batch_3"],
}
P_COLS = ["p_Batch_1", "p_Batch_2", "p_Batch_3"]
WITNESS_P = 0.7
SEED = 0


def batch_colours(batches: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {b: cmap(i) for i, b in enumerate(sorted(batches))}


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(float(v), floatfmt)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- loading
def load_tiles(path) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(path)
    _, tile_cols = catalogue_columns()
    cols = [c for group in tile_cols.values() for c in group]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"tile_kpis missing catalogue columns: {missing}")
    counts = df.groupby("site").size()
    if not (counts == 4).all():
        raise ValueError(f"expected exactly 4 rows per site, got {counts[counts != 4].to_dict()}")
    if (df.groupby("site")["batch"].nunique() != 1).any():
        raise ValueError("a site maps to more than one batch")
    if set(df["batch"]) != set(BATCHES):
        raise ValueError(f"batch set is {sorted(set(df['batch']))}, expected {BATCHES}")
    reason = df["nan_reason"].fillna("").astype(str)
    bad = df[cols].isna().any(axis=1) & (reason == "")
    if bad.any():
        raise ValueError(f"NaN in feature columns without nan_reason on {int(bad.sum())} rows")
    dropped = [c for c in cols if np.nanstd(df[c].to_numpy(float)) == 0]
    features = [c for c in cols if c not in dropped]
    return df, features, dropped


# ---------------------------------------------------------------- check 1
def make_pipeline() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED)),
    ])


def loso_tile_probs(df: pd.DataFrame, features: list[str], variant: str) -> pd.DataFrame:
    sub = df[df.batch.isin(VARIANTS[variant])].reset_index(drop=True)
    X, y, g = sub[features].to_numpy(float), sub.batch.to_numpy(), sub.site.to_numpy()
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        model = make_pipeline()
        model.fit(X[tr], y[tr])
        P = model.predict_proba(X[te])
        classes = list(model.classes_)
        for k, i in enumerate(te):
            rec = {"variant": variant, "batch": y[i], "site": g[i], "tile": int(sub.tile[i])}
            for c in BATCHES:
                rec["p_" + c] = P[k, classes.index(c)] if c in classes else np.nan
            rec["pred"] = classes[int(np.argmax(P[k]))]
            rec["p_true"] = rec["p_" + y[i]]
            rec["correct"] = bool(rec["pred"] == y[i])
            rows.append(rec)
    out = pd.DataFrame(rows)
    out = out[["variant", "batch", "site", "tile", "pred", "p_true", "correct"] + P_COLS]
    return out.sort_values(["variant", "batch", "site", "tile"]).reset_index(drop=True)


def categorise(n_correct: int, p_true_max: float) -> str:
    if n_correct >= 3:
        return "diffuse_correct"
    if n_correct == 0:
        return "miss"
    return "witness" if p_true_max >= WITNESS_P else "mixed"


def site_summary(tile_preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, batch, site), t in tile_preds.groupby(["variant", "batch", "site"], sort=True):
        classes = VARIANTS[variant]
        pc = ["p_" + c for c in classes]
        n_correct = int(t.correct.sum())
        pt = t.p_true.to_numpy(float)
        binary = variant != "multiclass"
        nonbase = [c for c in classes if c != "Batch_3"][0] if binary else None
        mean_pred = classes[int(np.argmax(t[pc].mean().to_numpy()))]
        if binary:
            max_pred = nonbase if t["p_" + nonbase].max() >= 0.5 else "Batch_3"
        else:
            max_pred = classes[int(np.argmax(t[pc].max().to_numpy()))]
        rows.append({
            "variant": variant, "batch": batch, "site": site, "n_tiles": len(t),
            "n_correct": n_correct, "frac_correct": n_correct / len(t),
            "p_true_min": pt.min(), "p_true_median": float(np.median(pt)), "p_true_max": pt.max(),
            "p_true_range": pt.max() - pt.min(),
            "category": categorise(n_correct, pt.max()),
            "max_p_nonbase": t["p_" + nonbase].max() if binary and batch == "Batch_3" else np.nan,
            "mean_pool_pred": mean_pred, "max_pool_pred": max_pred,
            "mean_pool_correct": mean_pred == batch, "max_pool_correct": max_pred == batch,
        })
    return pd.DataFrame(rows)


def batch_summary(site_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        sv = site_df[site_df.variant == variant]
        binary = variant != "multiclass"
        fa = np.nan
        if binary:
            b3 = sv[sv.batch == "Batch_3"]
            fa = float((b3.max_p_nonbase >= WITNESS_P).mean())
        for batch in VARIANTS[variant]:
            s = sv[sv.batch == batch]
            rec = {"variant": variant, "batch": batch, "n_sites": len(s)}
            for k in range(5):
                rec[f"n_correct_{k}"] = int((s.n_correct == k).sum())
            rec["median_frac_correct"] = float(s.frac_correct.median())
            for c in ["diffuse_correct", "witness", "mixed", "miss"]:
                rec[c] = int((s.category == c).sum())
            is_a = binary and batch != "Batch_3"
            rec["fa_rate"] = fa if is_a else np.nan
            verdict = ""
            if is_a:
                n_a, W, D = len(s), rec["witness"], rec["diffuse_correct"]
                if W >= 2 and W / n_a > fa:
                    verdict = "localised"
                elif D >= 4:
                    verdict = "diffuse"
                elif D + W <= 2:
                    verdict = "weak"
                else:
                    verdict = "inconclusive"
            rec["verdict"] = verdict
            rows.append(rec)
    return pd.DataFrame(rows)


def pooling_side_note(tile_preds: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        t = tile_preds[tile_preds.variant == variant]
        s = site_df[site_df.variant == variant]
        rows.append({
            "variant": variant, "n_sites": len(s),
            "tile_balanced_acc": balanced_accuracy_score(t.batch, t.pred),
            "site_acc_mean_pool": float(s.mean_pool_correct.mean()),
            "site_acc_max_pool": float(s.max_pool_correct.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- check 2
def icc1(values, groups) -> tuple[float, float]:
    v = np.asarray(values, float)
    gr = np.asarray(groups)
    ok = ~np.isnan(v)
    v, gr = v[ok], gr[ok]
    ids, counts = np.unique(gr, return_counts=True)
    keep = set(ids[counts >= 2])
    m_ok = np.array([x in keep for x in gr])
    v, gr = v[m_ok], gr[m_ok]
    ids = sorted(keep)
    N, g = len(v), len(ids)
    if g < 2 or N <= g:
        return np.nan, np.nan
    grand = v.mean()
    n_i = np.array([(gr == i).sum() for i in ids], float)
    m_i = np.array([v[gr == i].mean() for i in ids])
    msb = float((n_i * (m_i - grand) ** 2).sum() / (g - 1))
    msw = float(sum(((v[gr == i] - m) ** 2).sum() for i, m in zip(ids, m_i)) / (N - g))
    k0 = (N - (n_i ** 2).sum() / N) / (g - 1)
    denom = msb + (k0 - 1) * msw
    icc = (msb - msw) / denom if denom != 0 else np.nan
    return float(icc), msw


def nested_fractions(sub: pd.DataFrame, f: str) -> tuple[float, float, float]:
    d = sub[["batch", "site", f]].dropna()
    x = d[f].to_numpy(float)
    grand = x.mean()
    total = ((x - grand) ** 2).sum()
    if total == 0:
        return np.nan, np.nan, np.nan
    mb = d.groupby("batch")[f].transform("mean").to_numpy()
    ms = d.groupby("site")[f].transform("mean").to_numpy()
    ss_b = ((mb - grand) ** 2).sum()
    ss_s = ((ms - mb) ** 2).sum()
    ss_t = ((x - ms) ** 2).sum()
    return ss_b / total, ss_s / total, ss_t / total


def site_feature_spread(df: pd.DataFrame, features: list[str], sw_all: dict[str, float]) -> pd.DataFrame:
    rows = []
    for (batch, site), t in df.groupby(["batch", "site"], sort=True):
        for f in features:
            x = t[f].to_numpy(float)
            med, mn, mx = np.nanmedian(x), np.nanmin(x), np.nanmax(x)
            rows.append({
                "batch": batch, "site": site, "feature": f, "median": med, "min": mn, "max": mx,
                "range": mx - mn, "max_abs_dev": np.nanmax(np.abs(x - med)),
                "range_sw": (mx - mn) / sw_all[f],
            })
    return pd.DataFrame(rows)


def flag_contrast(spread_f: pd.DataFrame, a: str, sw: float) -> tuple[str, float, float]:
    A = spread_f[spread_f.batch == a]
    B = spread_f[spread_f.batch == "Batch_3"]
    dmed = (A["median"].median() - B["median"].median()) / sw
    dmax = (A["max"].median() - B["max"].median()) / sw
    dmin = (A["min"].median() - B["min"].median()) / sw
    dext = dmax if abs(dmax) >= abs(dmin) else dmin
    if abs(dext) >= 1.0 and abs(dext) >= 2 * abs(dmed):
        flag = "tail_driven"
    elif abs(dmed) >= 1.0:
        flag = "diffuse_shift"
    else:
        flag = "none"
    return flag, dmed, dext


def variance_table(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sw_all = {}
    for f in features:
        _, msw = icc1(df[f], df["site"])
        if not msw > 0:
            raise ValueError(f"sw_all is zero or undefined for {f}")
        sw_all[f] = float(np.sqrt(msw))
    spread = site_feature_spread(df, features, sw_all)
    rows = []
    for f in features:
        rec = {"feature": f}
        rec["icc_all"], _ = icc1(df[f], df["site"])
        sw_b = {}
        for b in BATCHES:
            sb = df[df.batch == b]
            icc_b, msw_b = icc1(sb[f], sb["site"])
            rec["icc_" + b.replace("atch_", "")] = icc_b
            sw_b[b] = np.sqrt(msw_b)
        rec["ss_batch"], rec["ss_site"], rec["ss_tile"] = nested_fractions(df, f)
        rec["sw_all"] = sw_all[f]
        sf = spread[spread.feature == f]
        for b in ["Batch_1", "Batch_2"]:
            k = b.replace("atch_", "")
            rec["sw_ratio_" + k] = sw_b[b] / sw_b["Batch_3"] if sw_b["Batch_3"] > 0 else np.nan
        for b in ["Batch_1", "Batch_2"]:
            k = b.replace("atch_", "")
            rec["flag_" + k], rec["dmed_" + k], rec["dext_" + k] = flag_contrast(sf, b, sw_all[f])
        for b in BATCHES:
            rec["range_sw_med_" + b.replace("atch_", "")] = float(sf[sf.batch == b].range_sw.median())
        rows.append(rec)
    cols = ["feature", "icc_all", "icc_B1", "icc_B2", "icc_B3", "ss_batch", "ss_site", "ss_tile",
            "sw_all", "sw_ratio_B1", "sw_ratio_B2", "flag_B1", "flag_B2",
            "dmed_B1", "dext_B1", "dmed_B2", "dext_B2",
            "range_sw_med_B1", "range_sw_med_B2", "range_sw_med_B3"]
    return pd.DataFrame(rows)[cols], spread


def overall_reading(bsum: pd.DataFrame, var: pd.DataFrame) -> dict[str, str]:
    out = {}
    for k, variant, batch in [("B1", "B1_vs_B3", "Batch_1"), ("B2", "B2_vs_B3", "Batch_2")]:
        verdict = bsum[(bsum.variant == variant) & (bsum.batch == batch)].verdict.iloc[0]
        n_tail = int((var["flag_" + k] == "tail_driven").sum())
        n_shift = int((var["flag_" + k] == "diffuse_shift").sum())
        if verdict == "localised" or n_tail >= 3:
            r = "localised"
        elif verdict == "diffuse" and n_shift > n_tail:
            r = "diffuse"
        else:
            r = "mixed/inconclusive"
        out[k] = r
    return out


# ---------------------------------------------------------------- figures
def fig_tile_probs(tile_preds, bsum, out: Path) -> None:
    colours = batch_colours(BATCHES)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), squeeze=False)
    for ax, variant in zip(axes[:, 0], VARIANTS):
        t = tile_preds[tile_preds.variant == variant]
        mean_pt = t.groupby(["batch", "site"]).p_true.mean().reset_index()
        order = []
        for b in VARIANTS[variant]:
            order += list(mean_pt[mean_pt.batch == b].sort_values(["p_true", "site"]).site)
        pos = {s: i for i, s in enumerate(order)}
        for b in VARIANTS[variant]:
            tb = t[t.batch == b]
            x = np.array([pos[s] for s in tb.site]) + rng.uniform(-0.15, 0.15, len(tb))
            ax.scatter(x, tb.p_true, s=14, color=colours[b], label=b)
        ax.axhline(0.5, ls="--", c="grey", lw=0.8)
        ax.axhline(0.7, ls="--", c="k", lw=0.8)
        if variant == "multiclass":
            ax.axhline(1 / 3, ls="--", c="grey", lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("p_true")
        verd = bsum[(bsum.variant == variant) & (bsum.verdict != "")]
        vtxt = ", ".join(f"{r.batch}: {r.verdict}" for r in verd.itertuples()) or "no verdict"
        ax.set_title(f"{variant} ({vtxt})", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_variance(var: pd.DataFrame, out: Path) -> None:
    n = len(var)
    fig, ax = plt.subplots(figsize=(10, 0.45 * n + 2))
    y = np.arange(n)
    left = np.zeros(n)
    for col, lab, colr in [("ss_batch", "batch", "tab:red"), ("ss_site", "site", "tab:blue"),
                           ("ss_tile", "tile", "tab:grey")]:
        ax.barh(y, var[col], left=left, label=lab, color=colr)
        left += var[col].to_numpy()
    labels = []
    for r in var.itertuples():
        tags = []
        for k in ["B1", "B2"]:
            fl = getattr(r, "flag_" + k)
            if fl != "none":
                tags.append(f"{k}:{'tail' if fl == 'tail_driven' else 'shift'}")
        labels.append(r.feature + (f" [{', '.join(tags)}]" if tags else ""))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of total variance")
    ax.legend(fontsize=7, loc="lower right")
    ax2 = ax.twiny()
    ax2.scatter(var.icc_all, y, c="k", s=14, zorder=5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("icc_all (black dots)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- results.md
def write_results(path_in, out: Path, df, features, dropped, bsum, site_df, side, var, reading) -> None:
    nsites = df.site.nunique()
    L = []
    L.append("# Tile-signal checks 1-2 (D10 pre-checks)\n")
    L.append(f"- Input: `{path_in}`")
    L.append(f"- Rows: {len(df)}; sites: {nsites} (4 tiles each); batches: "
             + ", ".join(f"{b}={df[df.batch == b].site.nunique()}" for b in BATCHES))
    L.append(f"- Features used ({len(features)}): " + ", ".join(features))
    L.append(f"- Features dropped (zero variance): {dropped}")
    L.append("- Seeds: LogisticRegression random_state=0; figure jitter np.random.default_rng(0). "
             "Outputs are deterministic.\n")
    L.append("## Definitions\n")
    L.append("- Classifier: median impute, standard scale, LogisticRegression(C=1.0, class_weight=balanced, "
             "max_iter=5000), refitted per fold; leave-one-site-out (groups = site).")
    L.append("- Site category (DD7): `diffuse_correct` n_correct >= 3; `witness` n_correct in {1,2} and max tile "
             "p_true >= 0.7; `mixed` n_correct in {1,2} and max tile p_true < 0.7; `miss` n_correct == 0.")
    L.append("- False-alarm rate FA (DD8, binary variants): fraction of Batch_3 sites with any tile "
             "p(non-B3 class) >= 0.7.")
    L.append("- Batch verdict (DD9, binary, n_A = 7; W = #witness, D = #diffuse_correct), first match wins: "
             "`localised` if W >= 2 and W/n_A > FA; `diffuse` if D >= 4; `weak` if D + W <= 2; else `inconclusive`.")
    L.append("- ICC (DD11): one-way random-effects ICC(1), unbalanced-safe k0; bands (Koo & Li 2016): "
             ">= 0.5 site-level feature, < 0.5 tile-dominated. ss_batch/ss_site/ss_tile are nested sum-of-squares "
             "fractions summing to 1. sw_all = sqrt(MSW) pooled within-site SD; sw_ratio = sw_batch / sw_B3.")
    L.append("- Single-tile flag (DD12), per contrast A vs Batch_3: dmed, dmax, dmin = difference over sites of "
             "the median of site median / max / min, divided by sw_all; dext = larger-|.| of dmax, dmin. "
             "`tail_driven` if |dext| >= 1.0 and |dext| >= 2|dmed|; else `diffuse_shift` if |dmed| >= 1.0; "
             "else `none`.")
    L.append("- Overall reading (DD14): Localised = any binary verdict is `localised` OR >= 3 features flagged "
             "`tail_driven` for that contrast. Diffuse = verdict is `diffuse` and `diffuse_shift` flags outnumber "
             "`tail_driven` flags. Otherwise `mixed/inconclusive`.\n")
    L.append("## Check 1: LOSO tile classifier\n")
    L.append("### Batch summary\n")
    L.append(md_table(bsum) + "\n")
    L.append("### Per-site spread\n")
    cols = ["variant", "batch", "site", "n_correct", "p_true_min", "p_true_median", "p_true_max", "category"]
    L.append(md_table(site_df[cols]) + "\n")
    L.append("### Pooling side note (side note; see check 3)\n")
    L.append(md_table(side) + "\n")
    L.append("## Check 2: variance decomposition and single-tile flags\n")
    L.append(md_table(var.round(3)) + "\n")
    L.append("## Overall reading\n")
    L.append(f"B1 vs B3: {reading['B1']}")
    L.append("")
    L.append(f"B2 vs B3: {reading['B2']}\n")
    L.append("## Caveats\n")
    L.append("- n = 7 sites per non-B3 batch, so the counts are descriptive and not significance tests.")
    L.append("- SE-detector confound: 1 Batch_2 site and 3 Batch_3 sites use SE rather than ETD; not adjusted for.")
    L.append("- The KPIs come from the provisional v0 segmenter (`segmenter_version` v0r1).")
    (out / "results.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile-kpis", default=str(ROOT / "outputs/kpis/tile_kpis.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs/pooling_checks"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, features, dropped = load_tiles(args.tile_kpis)
    tile_preds = pd.concat([loso_tile_probs(df, features, v) for v in VARIANTS], ignore_index=True)
    tile_preds.to_csv(out / "tile_predictions.csv", index=False)
    site_df = site_summary(tile_preds)
    site_df.to_csv(out / "site_tile_spread.csv", index=False)
    bsum = batch_summary(site_df)
    bsum.to_csv(out / "batch_summary.csv", index=False)
    side = pooling_side_note(tile_preds, site_df)
    side.to_csv(out / "pooling_side_note.csv", index=False)
    var, spread = variance_table(df, features)
    var.to_csv(out / "variance_by_feature.csv", index=False)
    spread.to_csv(out / "site_feature_spread.csv", index=False)
    fig_tile_probs(tile_preds, bsum, out / "fig_tile_probs.png")
    fig_variance(var, out / "fig_variance.png")
    reading = overall_reading(bsum, var)
    write_results(args.tile_kpis, out, df, features, dropped, bsum, site_df, side, var, reading)
    print(f"B1 vs B3: {reading['B1']}")
    print(f"B2 vs B3: {reading['B2']}")


if __name__ == "__main__":
    main()
