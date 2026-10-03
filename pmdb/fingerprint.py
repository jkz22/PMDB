"""Batch fingerprint model: conformal batch assignment from spatial-arrangement features.

Given an unlabelled site, the model answers the organiser's question directly:
which batch does it belong to, with what calibrated confidence, and is it like
*any* known batch at all (in / out of distribution)?

Why these features
------------------
The v1 scalar KPIs carry almost no batch signal: the best of 51 site-level
KPIs reaches only Kruskal-Wallis p = 0.065 across batches, and leave-one-out
classification on the screened KPI set stays below the majority-class baseline.
The batches have the same *composition* statistics. What differs is the
*spatial arrangement*, which lives in the full curves that the scalar KPIs
summarise away:

- `band_si_frac` (Si area fraction per depth band): Batch 3, the supplier
  baseline, is flat (uniform Si through the coating). Batch 2 is top-heavy
  with a depleted mid-depth (drying-migration signature; its band-2 dip is the
  single strongest batch discriminator, KW p ~ 0.001). Batch 1 is
  bottom-heavy and much more variable site-to-site (sedimentation signature).
- `g_obs_x` / `g_obs_z` (pair correlation of Si centroids): the lag range
  where g recovers toward 1 differs by batch (medium-range ordering).
- Within-site tile spread of K15 (Si-graphite contact fraction): batch-level
  differences in electrode homogeneity (KW p ~ 0.01).

Why this model
--------------
31 labelled sites (7/7/17) forbid fitted weights. The classifier is a robust
per-batch naive Bayes with Laplace scores: each batch contributes a median and
a robust scale per feature, shrunk toward the pooled scale. The per-batch
scales matter because the batches differ in *dispersion* as much as location
(Batch 1 is the loose one); a nearest-centroid rule is blind to that and
performs measurably worse.

Confidence must mean something, so on top of the likelihood score sits
class-conditional (Mondrian) conformal calibration: per batch, the
leave-one-out scores of its own training sites form the calibration set, and
a test site gets a finite-sample-valid p-value per batch (quantised to
multiples of 1/(n_b + 1); at n_b = 7 that is coarse, and honestly so).
Out-of-distribution falls out for free: if every batch p-value is small, the
site is like no known batch ("reject the shipment").

Terminology (standard conformal usage):

- credibility = p-value of the assigned batch (how typical the site would be
  if it truly belonged there).
- confidence  = 1 - second-highest p-value (how firmly the alternatives are
  excluded).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Consistency constant: 1.4826 * MAD estimates the SD for Gaussian data.
MAD_TO_SD = 1.4826

#: A batch is "rejected" for a site when its p-value is below this, or when the
#: p-value sits at its achievable floor 1/(n_b + 1), i.e. the site is more
#: nonconforming than every calibration site of that batch. The floor rule
#: matters because with n_b = 7 the smallest possible p-value is 0.125, which
#: alpha = 0.1 alone could never flag. A site with every batch rejected is
#: out-of-distribution.
DEFAULT_OOD_ALPHA = 0.1

#: MAD collapses on heavily quantized features (>= half the within-batch
#: residuals identical gives MAD ~ 0 while the true spread is finite). When the
#: MAD-based scale falls below this fraction of the corresponding SD, the SD is
#: used instead.
MIN_MAD_TO_SD_RATIO = 0.1

#: Standardized values are winsorized at this many robust SDs, so a single
#: outlying feature cannot dominate a likelihood score.
MAX_Z = 10.0

#: Per-batch scales are shrunk toward the pooled scale (1 in standardized
#: space): s_used = SHRINK * 1 + (1 - SHRINK) * s_batch. At n_b = 7 a raw
#: per-batch scale is too noisy to trust on its own.
SCALE_SHRINK = 0.5

#: Lag bins (um) over which the observed pair-correlation curves are averaged.
G_OBS_BINS = ((0.5, 2.0), (2.0, 4.0), (4.0, 7.0), (7.0, 10.0))


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(curves: pd.DataFrame, tile_kpis: pd.DataFrame) -> pd.DataFrame:
    """Build the per-site fingerprint feature table.

    Args:
        curves: Long table from run_kpis (`curves.csv`): batch, site, kpi_id,
            curve, x, value.
        tile_kpis: Tile-level KPI table (`tile_kpis.csv`): batch, site, tile,
            KPI columns.

    Returns:
        DataFrame indexed by (batch, site) with the fingerprint features:
        relative Si depth-band fractions (profile shape, composition-free),
        depth slope and mid-depth dip, binned pair-correlation levels in x and
        z, and the tile spread of the Si-graphite contact fraction.
    """
    feats: dict[str, pd.Series] = {}

    band = curves[curves["curve"] == "band_si_frac"].pivot_table(
        index=["batch", "site"], columns="x", values="value")
    rel = band.div(band.mean(axis=1), axis=0)
    for b in range(5):
        feats[f"si_depth_rel_band{b}"] = rel[float(b)]
    feats["si_depth_slope"] = rel[4.0] - rel[0.0]
    feats["si_depth_mid_dip"] = rel[2.0] - (rel[0.0] + rel[4.0]) / 2

    for ax in ("x", "z"):
        g = curves[curves["curve"] == f"g_obs_{ax}"].pivot_table(
            index=["batch", "site"], columns="x", values="value")
        lags = g.columns.to_numpy(float)
        for lo, hi in G_OBS_BINS:
            m = (lags >= lo) & (lags < hi)
            feats[f"g{ax}_{lo}_{hi}"] = g.loc[:, g.columns[m]].mean(axis=1)

    tstd = tile_kpis.groupby(["batch", "site"])["K15_si_graphite_contact_frac"].std()
    feats["k15_contact_tilestd"] = tstd

    return pd.DataFrame(feats)


def read_feature_inputs(curves_csv, tile_kpis_csv) -> pd.DataFrame:
    """Read curves.csv and tile_kpis.csv and build the feature table."""
    curves = pd.read_csv(curves_csv, dtype={"batch": str, "site": str})
    tiles = pd.read_csv(tile_kpis_csv, dtype={"batch": str, "site": str})
    return build_features(curves, tiles)


# ---------------------------------------------------------------------------
# Model (numpy core, pandas at the edges)
# ---------------------------------------------------------------------------

@dataclass
class FingerprintModel:
    """Fitted conformal robust-naive-Bayes model.

    Attributes:
        features: Feature names used, in column order.
        batches: Batch labels, sorted.
        center: Per-feature pooled median (standardization offset).
        scale: Per-feature pooled robust scale (standardization divisor).
        batch_center: (batch x feature) medians in standardized space.
        batch_scale: (batch x feature) shrunk robust scales in standardized space.
        calibration: Per batch, sorted leave-one-out nonconformity scores of
            its training sites (the conformal calibration set).
    """

    features: list[str]
    batches: list[str]
    center: pd.Series
    scale: pd.Series
    batch_center: pd.DataFrame
    batch_scale: pd.DataFrame
    calibration: dict[str, np.ndarray] = field(repr=False)


def _robust_scale(resid: np.ndarray) -> np.ndarray:
    """MAD-based scale with SD fallback for quantized features."""
    mad = MAD_TO_SD * np.nanmedian(np.abs(resid), axis=0)
    sd = np.nanstd(resid, axis=0)
    return np.where(mad >= MIN_MAD_TO_SD_RATIO * sd, mad, sd)


def _batch_params(z: np.ndarray, codes: np.ndarray, n_batches: int):
    """Per-batch medians and shrunk robust scales in standardized space."""
    mu = np.empty((n_batches, z.shape[1]))
    sb = np.empty((n_batches, z.shape[1]))
    for b in range(n_batches):
        zb = z[codes == b]
        mu[b] = np.nanmedian(zb, axis=0)
        s = _robust_scale(zb - mu[b])
        s = SCALE_SHRINK * 1.0 + (1.0 - SCALE_SHRINK) * s
        sb[b] = np.where(s > 0, s, 1.0)
    return mu, sb


def _scores(z: np.ndarray, mu: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """Per-batch nonconformity = mean negative Laplace log-likelihood.

    score(x, b) = mean_k [ |z_k - mu_bk| / s_bk + log s_bk ].  Lower is more
    typical of the batch. The mean (not sum) keeps scores comparable when a
    site has missing features.
    """
    dev = np.abs(z[:, None, :] - mu[None, :, :]) / sb[None, :, :]
    return np.nanmean(dev + np.log(sb)[None, :, :], axis=2)


def _fit_core(x: np.ndarray, codes: np.ndarray, n_batches: int,
              with_calibration: bool = True):
    """Numpy fit: standardization, batch params, LOO conformal calibration.

    The per-batch calibration scores are leave-one-out within the batch (each
    site scored against parameters re-estimated from its batchmates only), so
    a test site's score is exchangeable with them. The pooled standardization
    is computed once on the full training set; re-estimating it per left-out
    site moves third decimals at n = 31 and is not worth the opacity.
    """
    center = np.nanmedian(x, axis=0)
    scale = _robust_scale(x - center)
    ok = scale > 0
    if not ok.any():
        raise ValueError("all features have zero spread")
    z = np.clip((x[:, ok] - center[ok]) / scale[ok], -MAX_Z, MAX_Z)

    mu, sb = _batch_params(z, codes, n_batches)

    calibration = []
    if not with_calibration:
        return center, scale, ok, mu, sb, calibration
    for b in range(n_batches):
        zb = z[codes == b]
        n_b = len(zb)
        scores = np.empty(n_b)
        mask = np.ones(n_b, dtype=bool)
        for i in range(n_b):
            mask[i] = False
            mu_i = np.nanmedian(zb[mask], axis=0)
            s_i = _robust_scale(zb[mask] - mu_i)
            s_i = SCALE_SHRINK * 1.0 + (1.0 - SCALE_SHRINK) * s_i
            s_i = np.where(s_i > 0, s_i, 1.0)
            scores[i] = float(_scores(zb[i:i + 1], mu_i[None], s_i[None])[0, 0])
            mask[i] = True
        calibration.append(np.sort(scores))
    return center, scale, ok, mu, sb, calibration


def fit(X: pd.DataFrame, y: pd.Series, features: list[str] | None = None) -> FingerprintModel:
    """Fit the model on labelled sites.

    Args:
        X: Feature table (sites x features), numeric.
        y: Batch label per site (aligned with X).
        features: Feature subset; defaults to all columns of X.
    """
    features = list(features) if features is not None else list(X.columns)
    y = y.astype(str)
    batches = sorted(y.unique())
    codes = np.asarray([batches.index(b) for b in y])

    missing = [f for f in features if f not in X.columns]
    if missing:
        raise ValueError(f"input is missing feature columns: {missing}")
    x = X[features].to_numpy(dtype=float)
    center, scale, ok, mu, sb, calibration = _fit_core(x, codes, len(batches))
    kept = [f for f, k in zip(features, ok) if k]

    return FingerprintModel(
        features=kept,
        batches=batches,
        center=pd.Series(center[ok], index=kept),
        scale=pd.Series(scale[ok], index=kept),
        batch_center=pd.DataFrame(mu, index=batches, columns=kept),
        batch_scale=pd.DataFrame(sb, index=batches, columns=kept),
        calibration={b: calibration[i] for i, b in enumerate(batches)},
    )


def _standardize(model: FingerprintModel, X: pd.DataFrame) -> np.ndarray:
    missing = [f for f in model.features if f not in X.columns]
    if missing:
        raise ValueError(f"input is missing feature columns: {missing}")
    x = X[model.features].to_numpy(dtype=float)
    z = (x - model.center.to_numpy()) / model.scale.to_numpy()
    return np.clip(z, -MAX_Z, MAX_Z)


def nonconformity(model: FingerprintModel, X: pd.DataFrame) -> pd.DataFrame:
    """Per-batch nonconformity score for each site (sites x batches, lower = closer)."""
    z = _standardize(model, X)
    s = _scores(z, model.batch_center.to_numpy(), model.batch_scale.to_numpy())
    return pd.DataFrame(s, index=X.index, columns=model.batches)


def conformal_p(model: FingerprintModel, X: pd.DataFrame) -> pd.DataFrame:
    """Class-conditional conformal p-value per batch (sites x batches).

    p_b = (1 + #{calibration scores of batch b >= score(x, b)}) / (n_b + 1).
    """
    score = nonconformity(model, X)
    out = {}
    for b in model.batches:
        cal = model.calibration[b]
        ge = len(cal) - np.searchsorted(cal, score[b].to_numpy(), side="left")
        out[b] = (1.0 + ge) / (len(cal) + 1.0)
    return pd.DataFrame(out, index=X.index)


def predict(
    model: FingerprintModel,
    X: pd.DataFrame,
    ood_alpha: float = DEFAULT_OOD_ALPHA,
) -> pd.DataFrame:
    """Assign a batch to each site, with calibrated credibility and confidence.

    Assignment is by best (lowest) likelihood score; the conformal p-values
    quantify how typical the site would be of each batch. Returns a DataFrame
    with columns: `assigned`, `credibility` (p-value of the assigned batch),
    `confidence` (1 - highest p-value among the other batches), `ood` (True if
    every batch is rejected: p below `ood_alpha` or at its achievable floor
    1/(n_b + 1)), one `p_<batch>` and one `score_<batch>` column per batch. An
    assignment is always made.
    """
    score = nonconformity(model, X)
    p = conformal_p(model, X)
    floor = {b: 1.0 / (len(model.calibration[b]) + 1.0) for b in model.batches}

    rows = []
    for idx in X.index:
        sv, pv = score.loc[idx], p.loc[idx]
        best = sv.idxmin()
        others = [b for b in model.batches if b != best]
        rejected = {b: pv[b] < ood_alpha or pv[b] <= floor[b] + 1e-12
                    for b in model.batches}
        rows.append({
            "assigned": best,
            "credibility": float(pv[best]),
            "confidence": float(1.0 - max(pv[b] for b in others)),
            "ood": bool(all(rejected.values())),
            **{f"p_{b}": float(pv[b]) for b in model.batches},
            **{f"score_{b}": float(sv[b]) for b in model.batches},
        })
    return pd.DataFrame(rows, index=X.index)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _loo_assignments(x: np.ndarray, codes: np.ndarray, n_batches: int) -> np.ndarray:
    """Numpy-only LOO: assigned batch code (lowest score) per site."""
    n = len(x)
    assigned = np.empty(n, dtype=int)
    idx = np.arange(n)
    for i in range(n):
        tr = idx != i
        center, scale, ok, mu, sb, _ = _fit_core(x[tr], codes[tr], n_batches,
                                                 with_calibration=False)
        z = np.clip((x[i:i + 1, ok] - center[ok]) / scale[ok], -MAX_Z, MAX_Z)
        assigned[i] = int(np.argmin(_scores(z, mu, sb)[0]))
    return assigned


def loo_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str] | None = None,
    ood_alpha: float = DEFAULT_OOD_ALPHA,
) -> tuple[pd.DataFrame, dict]:
    """Leave-one-site-out evaluation.

    Returns:
        (per-site predictions with a `true` column, metrics dict with
        `accuracy`, `n`, per-batch `recall`, and a `confusion` nested dict
        confusion[true][assigned] = count).
    """
    y = y.astype(str)
    preds = []
    for i in range(len(X)):
        m = fit(X.drop(X.index[i]), y.drop(y.index[i]), features=features)
        preds.append(predict(m, X.iloc[[i]], ood_alpha=ood_alpha))
    pred = pd.concat(preds)
    pred.insert(0, "true", y.values)

    correct = pred["assigned"] == pred["true"]
    batches = sorted(y.unique())
    metrics = {
        "n": int(len(pred)),
        "accuracy": float(correct.mean()),
        "recall": {
            b: float(correct[pred["true"] == b].mean()) for b in batches
        },
        "confusion": {
            bt: {ba: int(((pred["true"] == bt) & (pred["assigned"] == ba)).sum())
                 for ba in batches}
            for bt in batches
        },
    }
    return pred, metrics


def permutation_test(
    X: pd.DataFrame,
    y: pd.Series,
    n_perm: int = 500,
    seed: int = 0,
    features: list[str] | None = None,
) -> dict:
    """Permutation test of the whole LOO pipeline.

    Shuffles batch labels, re-runs leave-one-site-out, and compares the
    observed accuracy with the null distribution. The p-value uses the
    standard (1 + #{null >= observed}) / (n_perm + 1) estimator.
    """
    y = y.astype(str)
    features = list(features) if features is not None else list(X.columns)
    batches = sorted(y.unique())
    codes = np.asarray([batches.index(b) for b in y])
    x = X[features].to_numpy(dtype=float)

    observed = float((_loo_assignments(x, codes, len(batches)) == codes).mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        cp = rng.permutation(codes)
        null[k] = (_loo_assignments(x, cp, len(batches)) == cp).mean()
    p = (1.0 + float((null >= observed).sum())) / (n_perm + 1.0)
    return {
        "observed_accuracy": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float(p),
        "n_perm": int(n_perm),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Explanation (feeds the fingerprint cards)
# ---------------------------------------------------------------------------

def explain(model: FingerprintModel, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature evidence for each site against each batch.

    Returns a long DataFrame with columns `site_index`, `feature`, `z` (the
    site's standardized value), and per batch `center_<batch>`,
    `scale_<batch>` and `dev_<batch>` (the feature's scaled deviation
    |z - center| / scale, its contribution to that batch's score). Sorting a
    site's rows by the gap between `dev_` columns shows which features decided
    the assignment.
    """
    z = pd.DataFrame(_standardize(model, X), index=X.index, columns=model.features)
    rows = []
    for idx in X.index:
        for f in model.features:
            row = {"site_index": idx, "feature": f, "z": float(z.loc[idx, f])}
            for b in model.batches:
                c = float(model.batch_center.loc[b, f])
                s = float(model.batch_scale.loc[b, f])
                row[f"center_{b}"] = c
                row[f"scale_{b}"] = s
                row[f"dev_{b}"] = (abs(row["z"] - c) / s
                                   if np.isfinite(row["z"]) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)
