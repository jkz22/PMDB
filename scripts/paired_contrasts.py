"""Within-parent paired contrasts: candidate feature table, contrast table, figure (local, light).

    python scripts/paired_contrasts.py

Writes outputs/paired_contrasts/{features.csv, heldout_features.csv, contrasts.csv,
contrasts_top.md, contrasts.png}. The CV / permutation evaluation is scripts/modal_paired.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import paired  # noqa: E402
from pmdb.classify.features import fem_tile_features, load_fem_tile_curves  # noqa: E402

OUT = ROOT / "outputs/paired_contrasts"
KEY = ["batch", "site"]
PAIR_NAMES = {(0, 1): "B1vB2", (0, 2): "B1vB3", (1, 2): "B2vB3"}


def _read(p):
    return pd.read_csv(p, dtype={"batch": str, "site": str})


def _kpi_block(site_csv, tile_csv, curves_csv) -> pd.DataFrame:
    s = _read(site_csv).set_index(KEY)
    s = s.drop(columns=["se_detector", "segmenter_version"]).astype(float).add_prefix("kpi:")
    t = _read(tile_csv)
    tcols = [c for c in t.columns if c[:1] in "KD" and c[1:3].isdigit()]
    tstd = t.groupby(KEY)[tcols].std().add_prefix("tilestd:")
    c = _read(curves_csv)
    feats = {}
    for curve in ("band_graphite_frac", "band_pore_frac"):
        band = c[c["curve"] == curve].pivot_table(index=KEY, columns="x", values="value")
        rel = band.div(band.mean(axis=1), axis=0)
        for b in range(5):
            feats[f"curve:{curve}_rel{b}"] = rel[float(b)]
        feats[f"curve:{curve}_slope"] = rel[4.0] - rel[0.0]
    col = c[c["curve"] == "column_si_frac"].groupby(KEY)["value"]
    feats["curve:column_si_frac_cv"] = col.std() / col.mean()
    cv = c[c["curve"] == "cv"].pivot_table(index=KEY, columns="x", values="value")
    for w in cv.columns:
        feats[f"curve:cv_w{int(w)}"] = cv[w]
    return s.join(tstd).join(pd.DataFrame(feats))


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(labelled features, held-out features, labelled-only extras without held-out values)."""
    X = _kpi_block(ROOT / "outputs/kpis/site_kpis.csv", ROOT / "outputs/kpis/tile_kpis.csv",
                   ROOT / "outputs/kpis/curves.csv")
    H = _kpi_block(ROOT / "outputs/heldout/kpis/site_kpis.csv",
                   ROOT / "outputs/heldout/kpis/tile_kpis.csv",
                   ROOT / "outputs/heldout/kpis/curves.csv")
    fpx = _read(ROOT / "outputs/fingerprint/features.csv").set_index(KEY).add_prefix("fp:")
    fph = _read(ROOT / "outputs/fingerprint/heldout_features.csv").set_index(KEY).add_prefix("fp:")
    t = fem_tile_features(load_fem_tile_curves(ROOT / "outputs/fem/tile_curves.csv"))
    fem = t.groupby(KEY)[[c for c in t.columns if c.startswith("fem_")]].median()
    fem.columns = ["fem:" + c[4:] for c in fem.columns]
    X = fpx.join(X).join(fem)  # fp first so duplicates keep the fingerprint name
    H = fph.join(H).join(fem)
    # Drop features missing anywhere (labelled or held-out) or constant.
    ok = X.notna().all() & H.reindex(columns=X.columns).notna().all() & (X.std() > 0)
    X = X.loc[:, ok]
    X = X.loc[:, ~X.T.round(12).duplicated()]  # identical columns (e.g. K15 tile std twice)
    H = H.loc[:, X.columns]

    mat = _read(ROOT / "outputs/kpis/materials_site_kpis.csv").set_index(KEY).add_prefix("mat:")
    anna = _read(ROOT / "outputs/kpis/anna_site_kpis.csv").set_index(KEY)
    anna = anna.drop(columns=["se_detector"]).select_dtypes("number").add_prefix("anna:")
    extra = mat.join(anna).reindex(X.index)
    extra = extra.loc[:, extra.notna().all() & (extra.std() > 0)]
    return X, H, extra


def contrast_table(F: pd.DataFrame, codes, parents, batches, has_heldout: bool) -> pd.DataFrame:
    x = F.to_numpy(float)
    k = len(batches)
    diffs = paired.within_parent_diffs(x, codes, parents, k)
    s = paired.pooled_within_sd(x, codes, k)
    rows = {"feature": F.columns, "has_heldout": has_heldout}
    # batch-pure parents: difference of pure-parent means, same standardisation
    pure = {p for p in set(parents) if len(set(codes[parents == p])) == 1}
    pmean = {p: x[parents == p].mean(axis=0) for p in pure}
    pbatch = {p: int(codes[parents == p][0]) for p in pure}
    for pr, nm in PAIR_NAMES.items():
        names, d = diffs[pr]
        mean, cons, score = paired.pair_scores(d)
        rows[f"{nm}_n"] = len(names)
        rows[f"{nm}_mean_d"] = mean
        rows[f"{nm}_consistency"] = cons
        rows[f"{nm}_score"] = score
        for j, p in enumerate(names):
            rows[f"{nm}_d_{p}"] = d[j]
        a, b = pr
        ma = [pmean[p] for p in pure if pbatch[p] == a]
        mb = [pmean[p] for p in pure if pbatch[p] == b]
        rows[f"{nm}_pure_d"] = (np.mean(mb, axis=0) - np.mean(ma, axis=0)) / np.where(s > 0, s, np.nan)
        rows[f"{nm}_pure_agrees"] = np.sign(rows[f"{nm}_pure_d"]) == np.sign(mean)
    out = pd.DataFrame(rows)
    out["max_score"] = out[[f"{nm}_score" for nm in PAIR_NAMES.values()]].max(axis=1)
    return out


def _figure(tab: pd.DataFrame, chosen: list[str], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = []
    for nm in PAIR_NAMES.values():
        top += list(tab.sort_values(f"{nm}_score", ascending=False)["feature"].head(5))
    top = list(dict.fromkeys(top))
    t = tab.set_index("feature").loc[top]
    dcols = [c for c in tab.columns if "_d_G" in c]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(dcols) + 0.6 * 3, 0.36 * len(top) + 1.8))
    cols = dcols + [f"{nm}_pure_d" for nm in PAIR_NAMES.values()]
    m = t[cols].to_numpy(float)
    im = ax.imshow(m, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(m[i, j]) > 1.6 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_d_", "\n").replace("_pure_d", "\npure") for c in cols],
                       fontsize=7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([("* " if f in chosen else "") + f for f in top], fontsize=7)
    ax.axvline(len(dcols) - 0.5, color="k", lw=1)
    fig.colorbar(im, ax=ax, label="standardised difference (later batch minus earlier)", shrink=0.6)
    ax.set_title("Within-parent batch contrasts (left) vs batch-pure parents (right)\n"
                 "top-5 features per pair; * = selected on all 31 crops", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, H, extra = build_tables()
    sites = X.index.get_level_values("site")
    parents = np.array(paired.parent_of(sites, require=set(sites)))
    assert set(paired.parent_of(H.index.get_level_values("site"))) <= set(paired.PARENT_SITES)
    batches = sorted(X.index.get_level_values("batch").unique())
    codes = np.array([batches.index(b) for b in X.index.get_level_values("batch")])

    Xo = X.copy()
    Xo.insert(0, "parent", parents)
    Xo.to_csv(OUT / "features.csv")
    Ho = H.copy()
    Ho.insert(0, "parent", paired.parent_of(H.index.get_level_values("site")))
    Ho.to_csv(OUT / "heldout_features.csv")

    tab = pd.concat([contrast_table(X, codes, parents, batches, True),
                     contrast_table(extra, codes, parents, batches, False)], ignore_index=True)
    tab = tab.sort_values("max_score", ascending=False)
    tab.to_csv(OUT / "contrasts.csv", index=False, float_format="%.4f")

    chosen = [X.columns[j] for j in paired.select(X.to_numpy(float), codes, parents, len(batches))]
    lines = [f"Selected on all 31 crops (K={paired.K_SELECT}, round-robin): " + ", ".join(chosen), ""]
    for nm in PAIR_NAMES.values():
        dcols = [c for c in tab.columns if c.startswith(f"{nm}_d_") and tab[c].notna().any()]
        t = tab.sort_values(f"{nm}_score", ascending=False).head(8)
        lines += [f"### {nm} (n parents = {int(t[f'{nm}_n'].iloc[0])}: "
                  + ", ".join(c.split("_d_")[1] for c in dcols) + ")", "",
                  "| feature | held-out | mean d | consistency | per-parent d | pure-parent d |",
                  "|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            per = " ".join(f"{r[c]:+.2f}" for c in dcols)
            lines.append(f"| {r['feature']} | {'yes' if r['has_heldout'] else 'no'} | "
                         f"{r[f'{nm}_mean_d']:+.2f} | {r[f'{nm}_consistency']:.2f} | {per} | "
                         f"{r[f'{nm}_pure_d']:+.2f} |")
        lines.append("")
    (OUT / "contrasts_top.md").write_text("\n".join(lines))
    _figure(tab[tab["has_heldout"]], chosen, OUT / "contrasts.png")
    print("\n".join(lines))
    print(f"{X.shape[1]} model candidates, {extra.shape[1]} labelled-only extras; wrote {OUT}")


if __name__ == "__main__":
    main()
