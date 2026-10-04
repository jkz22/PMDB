"""KPI-language explanations of the supervised patch probe (see docs/patch_mil.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy import ndimage

from pmdb.patch_mil import PATCH
from pmdb.patch_probe import BATCHES

R2_MIN = 0.2
MAX_NAN_FRAC = 0.2
KPI_COLS = ["si_frac", "si_density_per_1000um2", "si_mean_area_um2", "si_graphite_contact_frac",
            "porosity", "depth_frac"]  # graphite_frac is collinear (Si + graphite + pore = 1): cached, not regressed
KPI_NAMES = dict(zip(KPI_COLS, [
    "Si area fraction", "Si particle number density", "Si particle size", "Si-graphite contact",
    "porosity", "depth position in the image"]))


def patch_kpis(masks, coords, nm_per_px, seed_prefix):
    px_um2 = (nm_per_px / 1000.0) ** 2
    rows = []
    for i, (row, col, y0, x0) in enumerate(coords):
        y0, x0 = int(y0), int(x0)
        tm = masks.crop(slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
        nonart = ~tm.artefact
        n = int(nonart.sum())
        nan = float("nan")
        si = tm.si & nonart
        _, n_si = ndimage.label(si)
        edge = si & ~ndimage.binary_erosion(si)
        touch = edge & ndimage.binary_dilation(tm.graphite & nonart)
        k = {"si_frac": si.sum() / n if n else nan,
             "si_density_per_1000um2": n_si / (n * px_um2 / 1000.0) if n else nan,
             "si_mean_area_um2": si.sum() * px_um2 / n_si if n_si else nan,
             "si_graphite_contact_frac": touch.sum() / edge.sum() if edge.any() else nan,
             "graphite_frac": (tm.graphite & nonart).sum() / n if n else nan,
             "porosity": (tm.pore & nonart).sum() / n if n else nan,
             "depth_frac": (y0 + PATCH / 2) / masks.shape[0]}
        rows.append({"i": i, "row": int(row), "col": int(col), "y0": y0, "x0": x0,
                     **{c: float(v) for c, v in k.items()}})
    return rows


def _ols(y, X):
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ beta
    return beta[1:], 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


def pc_kpi_regression(Z, K, kpi_cols):
    nan_frac = K[kpi_cols].isna().mean()
    kept = [k for k in kpi_cols if nan_frac[k] <= MAX_NAN_FRAC]
    ok = K[kept].notna().all(axis=1).to_numpy()
    Kk = K.loc[ok, kept]
    Kz = ((Kk - Kk.mean()) / Kk.std(ddof=0)).to_numpy()
    rows = []
    for j in range(Z.shape[1]):
        y = Z[ok, j]
        B, r2 = _ols(y, Kz)
        sd = float(y.std())
        b = B / sd
        if r2 >= R2_MIN:
            top = np.argsort(-np.abs(b))[:2]
            label = ", ".join(("+" if b[t] > 0 else "-") + KPI_NAMES[kept[t]] for t in top)
        else:
            label = "unexplained texture"
        row = {"pc": f"PC{j + 1}", "sd": sd, "r2": float(r2), "explained": bool(r2 >= R2_MIN), "label": label}
        row.update({f"b_{k}": float(v) for k, v in zip(kept, b)})
        row.update({f"B_{k}": float(v) for k, v in zip(kept, B)})
        rows.append(row)
    kr = []
    for k in kpi_cols:
        r2 = _ols(Kz[:, kept.index(k)], Z[ok])[1] if k in kept else np.nan
        kr.append({"kpi": k, "kpi_name": KPI_NAMES[k], "nan_frac": float(nan_frac[k]), "kept": k in kept,
                   "r2": float(r2), "n_patches": int(ok.sum())})
    return pd.DataFrame(rows), pd.DataFrame(kr)


def _kept(pc):
    return [c[2:] for c in pc.columns if c.startswith("B_")]


def batch_directions(clf, pc_meanings):
    kept = _kept(pc_meanings)
    W = clf.coef_ - clf.coef_.mean(0)
    Bm = pc_meanings[[f"B_{k}" for k in kept]].to_numpy()
    L = W @ Bm
    sd = pc_meanings["sd"].to_numpy()
    ex = pc_meanings["explained"].to_numpy(bool)
    rows = []
    for bi, b in enumerate(BATCHES):
        share = (np.abs(W[bi]) * sd * ex).sum() / (np.abs(W[bi]) * sd).sum()
        order = np.argsort(-np.abs(L[bi]))
        summ = f"{b} <-> " + ", ".join(("more " if L[bi, t] > 0 else "less ") + KPI_NAMES[kept[t]] for t in order[:3])
        for rank, t in enumerate(order, 1):
            rows.append({"batch": b, "kpi": kept[t], "kpi_name": KPI_NAMES[kept[t]],
                         "logit_per_sd": float(L[bi, t]), "rank": rank, "explainable_share": float(share),
                         "summary": summ})
    return pd.DataFrame(rows)


def explain_site(Z_site, clf, pc_meanings):
    kept = _kept(pc_meanings)
    lp = clf.predict_log_proba(Z_site).mean(0)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    c, r = np.argsort(-lp)[:2]
    zbar = Z_site.mean(0)
    contrib = (clf.coef_[c] - clf.coef_[r]) * zbar
    icpt = clf.intercept_[c] - clf.intercept_[r]
    margin = lp[c] - lp[r]
    assert np.isclose(contrib.sum() + icpt, margin, atol=1e-6)
    expl = pc_meanings["explained"].to_numpy(bool)
    share = np.abs(contrib[expl]).sum() / np.abs(contrib).sum()
    top = [j for j in np.argsort(-contrib)[:3] if contrib[j] > 0]
    phrases = []
    for j in top:
        if expl[j]:
            b = pc_meanings.loc[j, [f"b_{k}" for k in kept]].to_numpy(float)
            for t in np.argsort(-np.abs(b))[:2]:
                phrases.append(("higher " if np.sign(zbar[j]) * np.sign(b[t]) > 0 else "lower ") + KPI_NAMES[kept[t]])
        else:
            phrases.append("fine texture not captured by our measurements")
    phrases = list(dict.fromkeys(phrases)) or ["no single dominant feature"]
    ph = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]

    def name(x):
        return x.replace("_", " ")

    sentence = (f"The image's local structure resembles {name(BATCHES[c])} more than {name(BATCHES[r])} mainly through "
                f"{ph}; about {round(100 * share)}% of this judgement maps onto measured microstructure, "
                f"the rest is fine texture not captured by our KPIs.")
    return {"call": BATCHES[c], "runner_up": BATCHES[r], **{f"p_{b}": float(p[i]) for i, b in enumerate(BATCHES)},
            "margin": float(margin), "intercept_term": float(icpt), "explained_share": float(share),
            "top_pcs": ";".join(f"PC{j + 1}:{contrib[j]:+.3f}" for j in np.argsort(-np.abs(contrib))[:5]),
            "sentence": sentence, "contrib": contrib}


def plot_pc_kpi_heatmap(pc_meanings, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept = _kept(pc_meanings)
    M = pc_meanings[[f"b_{k}" for k in kept]].to_numpy()
    lim = np.abs(M).max()
    fig, ax = plt.subplots(figsize=(8, 14))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{r.pc} R2={r.r2:.2f}" for r in pc_meanings.itertuples()], fontsize=6)
    for lbl, ex in zip(ax.get_yticklabels(), pc_meanings["explained"]):
        if ex:
            lbl.set_fontweight("bold")
    ax.set_xticks(range(len(kept)))
    ax.set_xticklabels([KPI_NAMES[k] for k in kept], rotation=60, ha="right")
    fig.colorbar(im, ax=ax, label="standardised coefficient")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
