"""Patch-level embedding + calibrated kNN batch classifier ("patch MIL").

Pure numpy/pandas/scipy logic; the torch feature extractor lives in
``pmdb.patch_embed`` and the Modal orchestration in ``modal_patch_mil.py``.
See docs/patch_mil.md.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

BATCHES: tuple[str, ...] = ("Batch_1", "Batch_2", "Batch_3")
REF_BATCH = "Batch_3"
PATCH = 224
K_NN = 3
TOP_FRAC = 0.10
ANOM_U = 0.95
SOFTMAX_TAU = 0.1
MAD_SCALE = 1.4826
SHORTCUT_RHO = 0.6


def grid_coords(h: int, w: int, patch: int = PATCH) -> np.ndarray:
    """int32 (n, 4) [row, col, y0, x0], row-major, centred non-overlapping grid."""
    nr, nc = h // patch, w // patch
    oy, ox = (h - nr * patch) // 2, (w - nc * patch) // 2
    out = [(r, c, oy + r * patch, ox + c * patch) for r in range(nr) for c in range(nc)]
    return np.asarray(out, dtype=np.int32).reshape(-1, 4)


def combine_channels(f_bse: np.ndarray, f_inlens: np.ndarray) -> np.ndarray:
    """Per-channel L2 normalise, concat, divide by sqrt(2) -> float32 unit rows."""
    def nrm(f):
        f = np.asarray(f, dtype=np.float32)
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    return (np.concatenate([nrm(f_bse), nrm(f_inlens)], axis=1) / np.sqrt(2.0)).astype(np.float32)


def site_distance_matrix(query: np.ndarray, query_site: np.ndarray,
                         banks: list[np.ndarray], k: int = K_NN) -> np.ndarray:
    """(n_query, len(banks)) mean of the min(k, len(bank)) smallest cosine distances."""
    q = np.asarray(query, dtype=np.float32)
    out = np.empty((q.shape[0], len(banks)), dtype=np.float32)
    for s, bank in enumerate(banks):
        d = 1.0 - q @ np.asarray(bank, dtype=np.float32).T
        kk = min(k, d.shape[1])
        if kk < d.shape[1]:
            d = np.partition(d, kk - 1, axis=1)[:, :kk]
        out[:, s] = d.mean(axis=1)
        out[np.asarray(query_site) == s, s] = np.inf
    return out


def reference_distribution(D: np.ndarray, patch_site: np.ndarray, bank_sites: np.ndarray) -> np.ndarray:
    """Sorted 1-D reference distances of the bank's own patches against the other bank sites."""
    bank_sites = np.asarray(bank_sites)
    if len(bank_sites) < 2:
        raise ValueError("calibration needs at least 2 training sites per batch bank")
    rows = np.isin(patch_site, bank_sites)
    return np.sort(D[np.ix_(rows, bank_sites)].min(axis=1))


def calibrate(d: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    n = len(ref_sorted)
    u = np.searchsorted(ref_sorted, d, side="right") / n
    med = np.median(ref_sorted)
    sc = max(MAD_SCALE * np.median(np.abs(ref_sorted - med)), 1e-6)
    return np.where(d > ref_sorted[-1], 1.0 + (d - ref_sorted[-1]) / sc, u)


def loo_bank_scores(D_rows: np.ndarray, bank_sites: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """(m, n) calibrated u for every dropped bank site j (D6)."""
    sub = D_rows[:, bank_sites]
    m = len(bank_sites)
    am = sub.argmin(axis=1)
    part = np.partition(sub, 1, axis=1)
    dj = np.where(am[None, :] == np.arange(m)[:, None], part[:, 1][None, :], part[:, 0][None, :])
    return calibrate(dj, ref_sorted)


def top_mean(u: np.ndarray, frac: float = TOP_FRAC) -> np.ndarray:
    """Mean of the k = max(1, ceil(frac * n)) largest values along the last axis."""
    n = u.shape[-1]
    k = max(1, math.ceil(frac * n))
    return np.sort(u, axis=-1)[..., n - k:].mean(axis=-1)


@dataclass
class SiteScore:
    u: np.ndarray                  # (n_patches, n_batches) mean over dropped bank site
    pooled: dict[str, np.ndarray]  # {"top10": (n_batches,), "mean": (n_batches,)}


def score_site(D_rows, D, patch_site, site_labels, train_mask, n_batches) -> SiteScore:
    n = D_rows.shape[0]
    u = np.empty((n, n_batches))
    top = np.empty(n_batches)
    mean = np.empty(n_batches)
    for b in range(n_batches):
        bank = np.flatnonzero(train_mask & (site_labels == b))
        ref = reference_distribution(D, patch_site, bank)
        uj = loo_bank_scores(D_rows, bank, ref)
        u[:, b] = uj.mean(0)
        top[b] = top_mean(uj).mean()
        mean[b] = uj.mean()
    return SiteScore(u=u, pooled={"top10": top, "mean": mean})


def loo_scores(D, patch_site, site_labels, n_batches) -> list[SiteScore]:
    n_sites = len(site_labels)
    return [score_site(D[patch_site == i], D, patch_site, site_labels,
                       np.arange(n_sites) != i, n_batches) for i in range(n_sites)]


def heldout_scores(Dh, patch_site_h, n_heldout, D, patch_site, site_labels, n_batches) -> list[SiteScore]:
    train = np.ones(len(site_labels), dtype=bool)
    return [score_site(Dh[patch_site_h == h], D, patch_site, site_labels, train, n_batches)
            for h in range(n_heldout)]


def softmax_conf(pooled: np.ndarray, tau: float = SOFTMAX_TAU) -> np.ndarray:
    pooled = np.asarray(pooled, dtype=float)
    z = np.exp(-(pooled - pooled.min(axis=-1, keepdims=True)) / tau)
    return z / z.sum(axis=-1, keepdims=True)


def metrics_from_confusion(conf: dict[str, dict[str, int]]) -> dict:
    names = list(conf)
    n = sum(sum(r.values()) for r in conf.values())
    correct = sum(conf[b].get(b, 0) for b in names)
    recall, f1 = {}, {}
    for b in names:
        tp = conf[b].get(b, 0)
        support = sum(conf[b].values())
        predicted = sum(conf[t].get(b, 0) for t in names)
        rec = tp / support if support else 0.0
        prec = tp / predicted if predicted else 0.0
        recall[b] = float(rec)
        f1[b] = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "n": int(n),
        "accuracy": float(correct / n) if n else 0.0,
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "macro_f1": float(np.mean(list(f1.values()))),
        "recall": recall,
        "f1": f1,
        "confusion": {t: {p: int(v) for p, v in r.items()} for t, r in conf.items()},
    }


def classification_metrics(true: np.ndarray, pred: np.ndarray, names=BATCHES) -> dict:
    conf = {t: {p: 0 for p in names} for t in names}
    for t, p in zip(true, pred):
        conf[names[int(t)]][names[int(p)]] += 1
    return metrics_from_confusion(conf)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)[:, None]
    neg = np.asarray(neg, dtype=float)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def _loo_accuracy(D, patch_site, labels, n_batches) -> float:
    sc = loo_scores(D, patch_site, labels, n_batches)
    pred = np.array([int(np.argmin(s.pooled["top10"])) for s in sc])
    return float((pred == labels).mean())


def permutation_test(D, patch_site, site_labels, n_batches, n_perm=1000, seed=0) -> dict:
    site_labels = np.asarray(site_labels)
    observed = _loo_accuracy(D, patch_site, site_labels, n_batches)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        cp = rng.permutation(site_labels)
        null[k] = _loo_accuracy(D, patch_site, cp, n_batches)
    p = (1.0 + float((null >= observed).sum())) / (n_perm + 1.0)
    return {
        "observed_accuracy": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float(p),
        "n_perm": int(n_perm),
        "seed": int(seed),
    }


def explain_heldout(site: str, assigned: str, confidence: float, pooled: np.ndarray, frac_anom: float,
                    n_patches: int, top_rows: list[int], n_rows: int, ctx: dict) -> str:
    verdict = "inside" if pooled[2] <= ctx["b3_max"] else "above"
    return (
        f"{site}: assigned {assigned} (confidence {confidence:.2f}; top-10% scores B1 {pooled[0]:.2f}, "
        f"B2 {pooled[1]:.2f}, B3 {pooled[2]:.2f}, lower = more typical). Versus Batch 3 (supplier baseline): "
        f"anomaly score {pooled[2]:.2f}, {verdict} the LOO range of Batch 3 sites "
        f"({ctx['b3_min']:.2f}-{ctx['b3_max']:.2f}, median {ctx['b3_median']:.2f}; "
        f"Batch 1/2 sites median {ctx['b12_median']:.2f}); {frac_anom:.0%} of {n_patches} patches exceed "
        f"the Batch 3 95th-percentile self-distance (LOO Batch 3 sites median {ctx['b3_frac_median']:.0%}). "
        f"Most Batch-3-atypical patches lie in image rows {top_rows} of 0-{n_rows - 1}; "
        f"see figures/heldout_{site}.png."
    )


def fingerprint_comparison(fp_eval: dict, ours_eval: dict) -> dict:
    src = "outputs/fingerprint/evaluation.json"

    def pack(m, p):
        return {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"], "recall": m["recall"], "permutation_p": float(p),
                "source": src}

    fpb = pack(metrics_from_confusion(fp_eval["loo"]["confusion"]), fp_eval["permutation"]["p_value"])
    ours = pack(ours_eval["loo"], ours_eval["permutation"]["p_value"])
    return {
        "fingerprint_nb": fpb,
        "patch_mil_top10": ours,
        "delta_accuracy": float(ours["accuracy"] - fpb["accuracy"]),
        "delta_balanced_accuracy": float(ours["balanced_accuracy"] - fpb["balanced_accuracy"]),
    }


def build_outputs(D, patch_site, site_labels, sites: pd.DataFrame, coords: np.ndarray, bse_hf: np.ndarray,
                  Dh, patch_site_h, heldout_ids: list[str], coords_h: np.ndarray, bse_hf_h: np.ndarray,
                  n_perm: int = 1000, seed: int = 0) -> dict:
    from scipy.stats import spearmanr

    nb = len(BATCHES)
    site_labels = np.asarray(site_labels)
    n_sites = len(site_labels)
    loo = loo_scores(D, patch_site, site_labels, nb)

    rows, prow = [], []
    for i, sc in enumerate(loo):
        top, mean = sc.pooled["top10"], sc.pooled["mean"]
        p = softmax_conf(top)
        sel = patch_site == i
        c = coords[sel]
        rows.append({
            "batch": sites["batch"].iloc[i], "site": sites["site"].iloc[i],
            "true": BATCHES[site_labels[i]], "assigned": BATCHES[int(np.argmin(top))],
            "confidence": float(p.max()),
            **{f"p_{b}": float(p[j]) for j, b in enumerate(BATCHES)},
            **{f"score_{b}": float(top[j]) for j, b in enumerate(BATCHES)},
            "assigned_mean_pool": BATCHES[int(np.argmin(mean))],
            **{f"score_mean_{b}": float(mean[j]) for j, b in enumerate(BATCHES)},
            "anomaly_vs_B3": float(top[2]),
            "frac_patches_anomalous": float((sc.u[:, 2] > ANOM_U).mean()),
            "n_patches": int(sel.sum()), "bse_hf": float(bse_hf[i]),
        })
        prow.append(pd.DataFrame({
            "batch": sites["batch"].iloc[i], "site": sites["site"].iloc[i], "split": "loo",
            "row": c[:, 0], "col": c[:, 1], "y0": c[:, 2], "x0": c[:, 3],
            **{f"u_{b}": sc.u[:, j] for j, b in enumerate(BATCHES)},
            "anomalous_vs_B3": sc.u[:, 2] > ANOM_U,
        }))
    cols = ["batch", "site", "true", "assigned", "confidence", "p_Batch_1", "p_Batch_2", "p_Batch_3",
            "score_Batch_1", "score_Batch_2", "score_Batch_3", "assigned_mean_pool",
            "score_mean_Batch_1", "score_mean_Batch_2", "score_mean_Batch_3",
            "anomaly_vs_B3", "frac_patches_anomalous", "n_patches", "bse_hf"]
    loo_df = pd.DataFrame(rows)[cols]

    a3 = loo_df["anomaly_vs_B3"].to_numpy()
    is_b3 = site_labels == 2
    fr = loo_df["frac_patches_anomalous"].to_numpy()
    ctx = {"b3_median": float(np.median(a3[is_b3])), "b3_min": float(a3[is_b3].min()),
           "b3_max": float(a3[is_b3].max()), "b12_median": float(np.median(a3[~is_b3])),
           "b3_frac_median": float(np.median(fr[is_b3]))}

    hsc = heldout_scores(Dh, patch_site_h, len(heldout_ids), D, patch_site, site_labels, nb)
    hrows = []
    for h, sc in enumerate(hsc):
        top = sc.pooled["top10"]
        p = softmax_conf(top)
        sel = patch_site_h == h
        c = coords_h[sel]
        n = int(sel.sum())
        k = max(1, math.ceil(TOP_FRAC * n))
        top_idx = np.argsort(-sc.u[:, 2], kind="stable")[:k]
        top_rows = sorted({int(r) for r in c[top_idx, 0]})
        frac = float((sc.u[:, 2] > ANOM_U).mean())
        assigned = BATCHES[int(np.argmin(top))]
        conf = float(p.max())
        hrows.append({
            "batch": "Batch_heldout", "site": heldout_ids[h], "assigned": assigned, "confidence": conf,
            **{f"p_{b}": float(p[j]) for j, b in enumerate(BATCHES)},
            **{f"score_{b}": float(top[j]) for j, b in enumerate(BATCHES)},
            "anomaly_vs_B3": float(top[2]), "frac_patches_anomalous": frac,
            "n_patches": n, "bse_hf": float(bse_hf_h[h]),
            "explanation": explain_heldout(heldout_ids[h], assigned, conf, top, frac, n, top_rows,
                                           int(c[:, 0].max()) + 1, ctx),
        })
        prow.append(pd.DataFrame({
            "batch": "Batch_heldout", "site": heldout_ids[h], "split": "heldout",
            "row": c[:, 0], "col": c[:, 1], "y0": c[:, 2], "x0": c[:, 3],
            **{f"u_{b}": sc.u[:, j] for j, b in enumerate(BATCHES)},
            "anomalous_vs_B3": sc.u[:, 2] > ANOM_U,
        }))
    hcols = ["batch", "site", "assigned", "confidence", "p_Batch_1", "p_Batch_2", "p_Batch_3",
             "score_Batch_1", "score_Batch_2", "score_Batch_3", "anomaly_vs_B3",
             "frac_patches_anomalous", "n_patches", "bse_hf", "explanation"]
    held_df = pd.DataFrame(hrows)[hcols]
    patch_df = pd.concat(prow, ignore_index=True)

    true = site_labels
    pred_top = np.array([int(np.argmin(s.pooled["top10"])) for s in loo])
    pred_mean = np.array([int(np.argmin(s.pooled["mean"])) for s in loo])
    counts = np.bincount(true, minlength=nb)
    rho, pv = spearmanr(a3, np.asarray(bse_hf))
    evaluation = {
        "n_sites": int(n_sites),
        "n_patches": int(len(patch_site)),
        "patches_per_site": {"min": int(loo_df["n_patches"].min()), "max": int(loo_df["n_patches"].max())},
        "majority_baseline": float(counts.max() / n_sites),
        "loo": classification_metrics(true, pred_top),
        "loo_mean_pooling": classification_metrics(true, pred_mean),
        "permutation": permutation_test(D, patch_site, site_labels, nb, n_perm=n_perm, seed=seed),
        "auroc_vs_B3": {"Batch_1": auroc(a3[true == 0], a3[is_b3]), "Batch_2": auroc(a3[true == 1], a3[is_b3])},
        "loo_anomaly_vs_B3": {BATCHES[b]: {"median": float(np.median(a3[true == b])),
                                           "min": float(a3[true == b].min()),
                                           "max": float(a3[true == b].max())} for b in range(nb)},
        "shortcut_diagnostic": {
            "feature": "std of Laplacian of affine2-harmonised BSE (grey/255)",
            "spearman_rho": float(rho), "p_value": float(pv), "flag": bool(abs(rho) > SHORTCUT_RHO),
        },
    }
    return {"evaluation": evaluation, "loo_predictions": loo_df, "heldout_predictions": held_df,
            "patch_scores": patch_df}


def render_heldout_figure(bse: np.ndarray, inlens: np.ndarray, coords: np.ndarray, u_b3: np.ndarray,
                          title: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    coords = np.asarray(coords)
    u_b3 = np.asarray(u_b3, dtype=float)
    nr, nc = int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1
    grid = np.full((nr, nc), np.nan)
    grid[coords[:, 0], coords[:, 1]] = u_b3
    y0min, x0min = int(coords[:, 2].min()), int(coords[:, 3].min())
    y0max, x0max = int(coords[:, 2].max()), int(coords[:, 3].max())
    n_show = min(8, len(u_b3))
    order = np.argsort(-u_b3, kind="stable")[:n_show]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = GridSpec(3, 8, height_ratios=[2, 1, 1], figure=fig)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(bse, cmap="gray", vmin=0, vmax=1)
    im = ax.imshow(grid, extent=[x0min, x0max + PATCH, y0max + PATCH, y0min], cmap="magma", alpha=0.45,
                   vmin=0, vmax=max(1.0, float(np.nanmax(grid))), interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="calibrated distance to Batch 3 (u)")
    for rank, idx in enumerate(order, 1):
        y0, x0 = int(coords[idx, 2]), int(coords[idx, 3])
        ax.add_patch(Rectangle((x0, y0), PATCH, PATCH, fill=False, edgecolor="red", linewidth=1.5))
        ax.text(x0 + 6, y0 + 6, str(rank), color="red", fontsize=9, va="top", ha="left")
    ax.set_xlim(0, bse.shape[1])
    ax.set_ylim(bse.shape[0], 0)
    ax.set_title(title)
    ax.axis("off")
    for rank, idx in enumerate(order, 1):
        r, c, y0, x0 = (int(v) for v in coords[idx])
        a1 = fig.add_subplot(gs[1, rank - 1])
        a1.imshow(bse[y0:y0 + PATCH, x0:x0 + PATCH], cmap="gray", vmin=0, vmax=1)
        a1.set_title(f"#{rank} r{r}c{c} u={u_b3[idx]:.2f}", fontsize=8)
        a1.axis("off")
        a2 = fig.add_subplot(gs[2, rank - 1])
        a2.imshow(inlens[y0:y0 + PATCH, x0:x0 + PATCH], cmap="gray", vmin=0, vmax=1)
        a2.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
