"""Generic unsupervised KPI screening tool (measurement quality only, no outcome label).

Five sequential pass/fail gates; the first gate a KPI fails decides its fate:

0. degeneracy: too many missing values, too few distinct values, or zero MAD.
1. artefact: tracks an imaging covariate within batches (batch-partial Spearman).
2. reliability: ICC(1) of replicates (batch-residualised) with a low upper CI bound.
3. robustness: site ranking changes across analysis-parameter settings.
4. redundancy: duplicates a better KPI (bootstrap co-clustering).

Usage::

    python -m pmdb.screen --kpis outputs/kpis/site_kpis.csv
    python -m pmdb.screen --kpis a.csv b.csv --replicates tiles.csv --covariates cov.csv
    python -m pmdb.screen --kpis site.csv --sensitivity sweep.csv --sensitivity-params d_um

The submission contract and full gate definitions are in ``docs/kpis/screening.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

KEY_COLS = ("batch", "site")
MISSING_TOKENS = ("", "NaN", "nan")
DEFAULT_IGNORE_COLS = ("se_detector", "segmenter_version", "nan_reason", "runner", "elapsed_s", "error")
KPI_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
MIN_PAIRED_SITES = 5
STREAM_COVARIATE, STREAM_ICC, STREAM_CLUSTER = 1, 2, 3
GATES = ("degeneracy", "artefact", "reliability", "robustness", "redundancy")
SCREEN_VERSION = "1.0"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "kpis" / "screen"
DEFAULT_SITE_MANIFEST = Path(__file__).resolve().parents[1] / "cache" / "half" / "manifest.csv"

TABLE_COLUMNS = [
    "kpi", "decision", "deciding_gate", "untested_gates", "flags", "rank",
    "n_missing_frac", "n_unique", "mad", "batch_eta2",
    "max_abs_rho_covariate", "covariate_of_max", "covariate_rho", "covariate_rho_lo",
    "covariate_rho_hi", "covariate_rho_marginal",
    "icc", "icc_lo", "icc_hi", "n_rep_sites",
    "robustness_rho", "n_settings",
    "cluster_id", "cluster_rep",
]


class ScreenInputError(ValueError):
    """Raised when an input table violates the submission contract."""


@dataclass(frozen=True)
class ScreenConfig:
    """Screening thresholds, bootstrap settings and input conventions.

    Attributes
    ----------
    max_missing_frac : float
        Gate 0. Drop if the fraction of missing sites is greater than this.
    min_unique : int
        Gate 0. Drop if the number of distinct non-missing values is less than this.
    eta2_flag : float
        Gate 1. Flag ``high_batch_eta2`` if rank-based batch eta-squared is greater
        than this (never drops).
    covariate_rho : float
        Gate 1. Drop if |within-batch Spearman rho| with any covariate is >= this
        and its bootstrap CI excludes 0.
    ci_level : float
        Confidence level of all percentile bootstrap CIs.
    n_boot_cov : int
        Gate 1. Number of batch-stratified bootstrap resamples for covariate CIs.
    icc_upper_min : float
        Gate 2. Drop if the ICC(1) CI upper bound is < this.
    n_boot_icc : int
        Gate 2. Number of site-cluster bootstrap resamples for the ICC CI.
    robustness_min : float
        Gate 3. Drop if the minimum pairwise Spearman rho across settings is < this.
    redundancy_rho : float
        Gate 4. Two KPIs are in one cluster when |rho| >= this (average linkage,
        cut at distance 1 - this).
    cocluster_frac : float
        Gate 4. KPIs are linked when co-clustered in >= this fraction of resamples.
    n_boot_cluster : int
        Gate 4. Number of batch-stratified bootstrap resamples for clustering.
    seed : int
        Base seed; each gate uses ``default_rng([seed, stream])``.
    sentinels : tuple of (str, float)
        Declared per-column sentinel values treated as missing.
    replicate_col : str
        Name of the replicate (tile) identifier column in the replicate table.
    sensitivity_params : tuple of str
        Parameter columns defining a setting in the sensitivity table.
    ignore_cols : tuple of str
        Column names never treated as KPIs or covariates.
    """

    max_missing_frac: float = 0.2
    min_unique: int = 5
    eta2_flag: float = 0.5
    covariate_rho: float = 0.5
    ci_level: float = 0.95
    n_boot_cov: int = 1000
    icc_upper_min: float = 0.5
    n_boot_icc: int = 1000
    robustness_min: float = 0.8
    redundancy_rho: float = 0.8
    cocluster_frac: float = 0.8
    n_boot_cluster: int = 500
    seed: int = 0
    sentinels: tuple[tuple[str, float], ...] = ()
    replicate_col: str = "tile"
    sensitivity_params: tuple[str, ...] = ()
    ignore_cols: tuple[str, ...] = DEFAULT_IGNORE_COLS


@dataclass(frozen=True)
class ScreenResult:
    """Output of :func:`screen`.

    Attributes
    ----------
    table : pandas.DataFrame
        Content of ``kpi_screen.csv``.
    filtered : pandas.DataFrame
        Content of ``site_kpis_filtered.csv`` (kept KPIs only, original values).
    """

    table: pd.DataFrame
    filtered: pd.DataFrame


# --------------------------------------------------------------------------- input


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV with string ``batch`` and ``site`` columns.

    Parameters
    ----------
    path : str or Path
        CSV file.

    Only empty, ``NaN`` and ``nan`` cells are read as missing (``inf``/``-inf`` parse
    as floats); any other text (``NA``, ``n/a``, ...) stays text.

    Returns
    -------
    pandas.DataFrame
        The table.

    Raises
    ------
    ScreenInputError
        If a column header is duplicated.
    """
    df = pd.read_csv(path, dtype={"batch": str, "site": str},
                     keep_default_na=False, na_values=list(MISSING_TOKENS))
    header = pd.read_csv(path, header=None, nrows=1, dtype=str,
                         keep_default_na=False).iloc[0].tolist()
    dups = sorted({h for h in header if header.count(h) > 1})
    if "" in dups:
        raise ScreenInputError(f"{path}: {header.count('')} blank column headers (trailing commas?); remove the empty columns")
    if dups:
        raise ScreenInputError(f"{path}: duplicated column header(s) {dups}")
    return df


def read_site_manifest(path: str | Path) -> set[tuple[str, str]]:
    """Canonical (batch, site) set from a manifest CSV.

    Parameters
    ----------
    path : str or Path
        CSV with ``batch`` and ``site`` columns (default ``cache/half/manifest.csv``,
        written by ``scripts/build_cache.py`` from ``pmdb.io.list_sites()``).

    Returns
    -------
    set of tuple of str
        The ``(batch, site)`` keys.

    Raises
    ------
    ScreenInputError
        If the file is missing or lacks valid keys.
    """
    p = Path(path)
    if not p.is_file():
        raise ScreenInputError(f"site manifest not found: {p} (pass --site-manifest)")
    df = read_table(p)
    _check_keys(df, f"site manifest {p}")
    return _keys(df)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["batch"], df["site"]))


def _check_keys(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ScreenInputError(f"{label}: missing required column(s) {missing}")
    dup = df.duplicated(list(KEY_COLS), keep=False)
    if dup.any():
        ex = df.loc[dup, list(KEY_COLS)].drop_duplicates().head(5).values.tolist()
        raise ScreenInputError(f"{label}: duplicated (batch, site) keys, e.g. {ex}")


def _check_same_sites(df: pd.DataFrame, site: pd.DataFrame, label: str) -> None:
    a, b = _keys(df), _keys(site)
    if a != b:
        raise ScreenInputError(
            f"{label}: site set differs from the KPI table "
            f"({len(a)} vs {len(b)} sites; only in {label}: {sorted(a - b)[:5]}; "
            f"only in KPI table: {sorted(b - a)[:5]})"
        )


def kpi_columns(df: pd.DataFrame, ignore_cols: Iterable[str],
                exclude: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    """Split a table's non-key columns into numeric KPI columns and ignored text columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with ``batch`` and ``site`` columns.
    ignore_cols : iterable of str
        Column names silently excluded.
    exclude : iterable of str, optional
        Further columns silently excluded (e.g. replicate or parameter columns).

    Returns
    -------
    tuple of (list of str, list of str)
        Numeric KPI columns in file order, and non-numeric non-key columns that were
        not in ``ignore_cols``/``exclude`` (reported as ignored).

    Raises
    ------
    ScreenInputError
        If a KPI column name does not match ``KPI_NAME_RE``.
    """
    skip = set(KEY_COLS) | set(ignore_cols) | set(exclude)
    kpis, ignored = [], []
    for c in df.columns:
        if c in skip:
            continue
        if _is_numeric(df[c]):
            if not KPI_NAME_RE.match(str(c)):
                raise ScreenInputError(
                    f"invalid KPI name {c!r}: must match {KPI_NAME_RE.pattern} "
                    "(duplicated headers get a '.1' suffix from pandas)")
            kpis.append(c)
        else:
            ignored.append(c)
    return kpis, ignored


def merge_kpi_tables(frames: Sequence[pd.DataFrame], sources: Sequence[str],
                     ignore_cols: Iterable[str] = DEFAULT_IGNORE_COLS,
                     expected_sites: set[tuple[str, str]] | None = None
                     ) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Merge one or more site-level KPI tables on (batch, site).

    Parameters
    ----------
    frames : sequence of pandas.DataFrame
        One site-level table per source; rows are keyed by (batch, site).
    sources : sequence of str
        Source names (file paths) used in messages.
    ignore_cols : iterable of str, optional
        Column names silently excluded from KPIs.
    expected_sites : set of tuple of str, optional
        Canonical (batch, site) set the first frame must match exactly; ``None``
        means no manifest check.

    Returns
    -------
    tuple
        ``(table, kpis, ignored)``: ``batch, site`` plus all KPI columns in the row
        order of ``frames[0]``; the KPI names in file-then-column order; and
        ``{source: ignored text columns}``.

    Raises
    ------
    ScreenInputError
        On missing keys, duplicate keys, bad names, KPI names repeated across files,
        differing site sets, a first file differing from ``expected_sites``, or a file
        with no KPI columns.
    """
    ignore_cols = tuple(ignore_cols)
    kpis: list[str] = []
    owner: dict[str, str] = {}
    ignored: dict[str, list[str]] = {}
    per_frame: list[list[str]] = []
    for df, src in zip(frames, sources):
        _check_keys(df, src)
        cols, ign = kpi_columns(df, ignore_cols)
        if not cols:
            raise ScreenInputError(f"{src}: no numeric KPI columns")
        for c in cols:
            if c in owner:
                raise ScreenInputError(f"KPI {c!r} appears in both {owner[c]} and {src}")
            owner[c] = src
        kpis.extend(cols)
        per_frame.append(cols)
        if ign:
            ignored[src] = ign
    base = frames[0]
    if expected_sites is not None:
        a = _keys(base)
        if a != expected_sites:
            raise ScreenInputError(
                f"{sources[0]}: site set differs from the site manifest "
                f"({len(a)} vs {len(expected_sites)} sites; missing: {sorted(expected_sites - a)[:5]}; "
                f"not in manifest: {sorted(a - expected_sites)[:5]})")
    for df, src in zip(frames[1:], sources[1:]):
        a, b = _keys(base), _keys(df)
        if a != b:
            raise ScreenInputError(
                f"site sets differ between {sources[0]} ({len(a)} sites) and {src} ({len(b)} sites); "
                f"only in {sources[0]}: {sorted(a - b)[:5]}; only in {src}: {sorted(b - a)[:5]}")
    out = base[list(KEY_COLS) + per_frame[0]].reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(out[list(KEY_COLS)])
    for df, cols in zip(frames[1:], per_frame[1:]):
        other = df.set_index(list(KEY_COLS))[cols].reindex(idx).reset_index(drop=True)
        out = pd.concat([out, other], axis=1)
    return out, kpis, ignored


def parse_sentinels(items: Sequence[str]) -> tuple[tuple[str, float], ...]:
    """Parse ``COL=VALUE`` sentinel declarations.

    Parameters
    ----------
    items : sequence of str
        Items such as ``"K08_pcf_rpeak_x_um=-1"``.

    Returns
    -------
    tuple of (str, float)
        Deduplicated, sorted by (column, value).

    Raises
    ------
    ScreenInputError
        If an item has no ``=`` or a non-float value.
    """
    out = set()
    for it in items:
        if "=" not in it:
            raise ScreenInputError(f"malformed --sentinel {it!r}: expected COL=VALUE")
        col, val = it.rsplit("=", 1)
        try:
            out.add((col, float(val)))
        except ValueError:
            raise ScreenInputError(f"malformed --sentinel {it!r}: {val!r} is not a number") from None
    return tuple(sorted(out))


def apply_sentinels(df: pd.DataFrame, cols: Sequence[str],
                    sentinels: Sequence[tuple[str, float]]) -> pd.DataFrame:
    """Return a copy with +-inf and declared sentinels replaced by NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    cols : sequence of str
        Columns in which +-inf becomes NaN.
    sentinels : sequence of (str, float)
        Declared (column, value) sentinels; columns absent from ``df`` are skipped.

    Returns
    -------
    pandas.DataFrame
        Cleaned copy.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(float).replace([np.inf, -np.inf], np.nan)
    for c, v in sentinels:
        if c in out.columns:
            out[c] = out[c].astype(float).mask(out[c] == v)
    return out


def validate_replicates(rep: pd.DataFrame, site: pd.DataFrame, kpis: Sequence[str],
                        cfg: ScreenConfig) -> list[str]:
    """Validate the replicate (tile) table.

    Parameters
    ----------
    rep : pandas.DataFrame
        Table with ``batch, site, <replicate_col>`` and KPI columns.
    site : pandas.DataFrame
        Site-level table.
    kpis : sequence of str
        Site-level KPI names.
    cfg : ScreenConfig
        Supplies ``replicate_col`` and ``ignore_cols``.

    Returns
    -------
    list of str
        Replicate KPI columns in file order.

    Raises
    ------
    ScreenInputError
        On missing columns, duplicate rows, site-set mismatch or unknown KPI names.
    """
    missing = [c for c in (*KEY_COLS, cfg.replicate_col) if c not in rep.columns]
    if missing:
        raise ScreenInputError(f"replicates: missing required column(s) {missing}")
    if rep.duplicated([*KEY_COLS, cfg.replicate_col]).any():
        raise ScreenInputError(f"replicates: duplicated (batch, site, {cfg.replicate_col}) rows")
    _check_same_sites(rep, site, "replicates")
    cols, ign = kpi_columns(rep, cfg.ignore_cols, exclude=[cfg.replicate_col])
    bad = [c for c in ign if c in set(kpis)]
    if bad:
        raise ScreenInputError(f"replicates: column(s) {bad} name site KPIs but are not numeric")
    unknown = [c for c in cols if c not in set(kpis)]
    if unknown:
        raise ScreenInputError(f"replicates: KPI column(s) {unknown} not in the site KPI table (naming typo?)")
    return cols


def validate_sensitivity(sens: pd.DataFrame, site: pd.DataFrame, kpis: Sequence[str],
                         cfg: ScreenConfig) -> list[str]:
    """Validate the sensitivity (parameter sweep) table.

    Parameters
    ----------
    sens : pandas.DataFrame
        Table with ``batch, site, <params>`` and KPI columns.
    site : pandas.DataFrame
        Site-level table.
    kpis : sequence of str
        Site-level KPI names.
    cfg : ScreenConfig
        Supplies ``sensitivity_params`` and ``ignore_cols``.

    Returns
    -------
    list of str
        Sensitivity KPI columns in file order.

    Raises
    ------
    ScreenInputError
        On missing/NaN params, fewer than 2 settings, incomplete settings, duplicates
        or unknown KPI names.
    """
    absent = [c for c in KEY_COLS if c not in sens.columns]
    if absent:
        raise ScreenInputError(f"sensitivity: missing required column(s) {absent}")
    params = list(cfg.sensitivity_params)
    if not params:
        raise ScreenInputError("sensitivity: --sensitivity-params is required")
    missing = [p for p in params if p not in sens.columns]
    if missing:
        raise ScreenInputError(f"sensitivity: missing parameter column(s) {missing}")
    if sens[params].isna().any().any():
        raise ScreenInputError("sensitivity: parameter columns contain NaN")
    if sens.duplicated([*KEY_COLS, *params]).any():
        raise ScreenInputError("sensitivity: duplicated (batch, site, *params) rows")
    if len(sens[params].drop_duplicates()) < 2:
        raise ScreenInputError("sensitivity: need at least 2 distinct settings")
    for _, grp in sens.groupby(params, sort=False):
        if _keys(grp) != _keys(site):
            raise ScreenInputError("sensitivity: a setting does not cover exactly the site set "
                                   f"({len(_keys(grp))} vs {len(_keys(site))} sites)")
    cols, ign = kpi_columns(sens, cfg.ignore_cols, exclude=params)
    bad = [c for c in ign if c in set(kpis)]
    if bad:
        raise ScreenInputError(f"sensitivity: column(s) {bad} name site KPIs but are not numeric")
    unknown = [c for c in cols if c not in set(kpis)]
    if unknown:
        raise ScreenInputError(f"sensitivity: KPI column(s) {unknown} not in the site KPI table (naming typo?)")
    return cols


def validate_covariates(cov: pd.DataFrame, site: pd.DataFrame, cfg: ScreenConfig) -> list[str]:
    """Validate the wide covariate table.

    Parameters
    ----------
    cov : pandas.DataFrame
        Table with ``batch, site`` and numeric covariate columns.
    site : pandas.DataFrame
        Site-level table.
    cfg : ScreenConfig
        Supplies ``ignore_cols``.

    Returns
    -------
    list of str
        Numeric covariate columns in file order.

    Raises
    ------
    ScreenInputError
        On missing keys, duplicates, site-set mismatch, a non-numeric column or no numeric covariate.
    """
    _check_keys(cov, "covariates")
    _check_same_sites(cov, site, "covariates")
    cand = [c for c in cov.columns if c not in KEY_COLS and c not in cfg.ignore_cols]
    bad = [c for c in cand if not _is_numeric(cov[c])]
    if bad:
        raise ScreenInputError(f"covariates: column(s) {bad} not numeric (write missing values as empty/NaN)")
    cols = cand
    if not cols:
        raise ScreenInputError("covariates: no numeric covariate columns")
    return cols


def pivot_raw_intensity_stats(raw: pd.DataFrame,
                              stats: Sequence[tuple[str, str]] = (("BSE", "p1"), ("BSE", "p50"))
                              ) -> pd.DataFrame:
    """Pivot the long raw-intensity table into a wide covariate table.

    Parameters
    ----------
    raw : pandas.DataFrame
        Long table (``outputs/raw_intensity_stats.csv``) with ``channel_slot`` and stat columns.
    stats : sequence of (str, str)
        ``(channel_slot, stat)`` pairs to extract.

    Returns
    -------
    pandas.DataFrame
        ``batch, site`` plus one column ``f"{channel_slot}_{stat}"`` per pair.
    """
    parts = []
    for slot, stat in stats:
        sub = raw.loc[raw["channel_slot"] == slot, [*KEY_COLS, stat]]
        parts.append(sub.set_index(list(KEY_COLS))[stat].rename(f"{slot}_{stat}"))
    return pd.concat(parts, axis=1).reset_index()


# ------------------------------------------------------------------ gate statistics


def degeneracy_stats(x: pd.Series) -> dict[str, float]:
    """Gate 0 statistics for one KPI.

    Parameters
    ----------
    x : pandas.Series
        Sentinel-cleaned values (missing as NaN).

    Returns
    -------
    dict
        ``n_missing_frac``, ``n_unique`` (distinct non-missing) and ``mad``
        (unscaled median absolute deviation, NaN if no values).
    """
    v = x.to_numpy(float)
    v = v[np.isfinite(v)]
    mad = float(np.median(np.abs(v - np.median(v)))) if v.size else float("nan")
    return {"n_missing_frac": float(1 - v.size / max(len(x), 1)), "n_unique": float(len(np.unique(v))), "mad": mad}


def batch_eta2(x: np.ndarray, batches: np.ndarray) -> float:
    """Rank-based batch eta-squared (SS_between / SS_total on average ranks).

    Parameters
    ----------
    x : numpy.ndarray
        Site values (NaN allowed).
    batches : numpy.ndarray
        Batch label per site.

    Returns
    -------
    float
        eta-squared, NaN if no variance or fewer than 2 values.
    """
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return float("nan")
    r = rankdata(x[ok])
    codes = np.unique(batches[ok], return_inverse=True)[1]
    sst = float(((r - r.mean()) ** 2).sum())
    if sst == 0:
        return float("nan")
    n = np.bincount(codes)
    means = np.bincount(codes, r) / n
    return float((n * (means - r.mean()) ** 2).sum() / sst)


def partial_spearman(x: np.ndarray, y: np.ndarray, batches: np.ndarray) -> float:
    """Within-batch (batch-partial) Spearman correlation.

    Parameters
    ----------
    x, y : numpy.ndarray
        Paired site values (NaN allowed; pairwise-complete sites are used).
    batches : numpy.ndarray
        Batch label per site; pass zeros for the marginal Spearman rho.

    Returns
    -------
    float
        Pearson correlation of batch-demeaned ranks; NaN if fewer than
        ``MIN_PAIRED_SITES`` complete sites or zero variance.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_PAIRED_SITES:
        return np.nan
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    codes = np.unique(batches[ok], return_inverse=True)[1]
    n = np.bincount(codes)
    rx = rx - (np.bincount(codes, rx) / n)[codes]
    ry = ry - (np.bincount(codes, ry) / n)[codes]
    den = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / den) if den > 0 else np.nan


def bootstrap_indices(batches: np.ndarray, n_boot: int, seed: int, stream: int) -> np.ndarray:
    """Batch-stratified site bootstrap indices.

    Parameters
    ----------
    batches : numpy.ndarray
        Batch label per site.
    n_boot : int
        Number of resamples.
    seed : int
        Base seed.
    stream : int
        Gate stream id (``STREAM_*``).

    Returns
    -------
    numpy.ndarray
        Shape ``(n_boot, n_sites)``, dtype intp; within each batch the same number of
        sites is drawn with replacement.
    """
    rng = np.random.default_rng([seed, stream])
    out = np.empty((n_boot, len(batches)), dtype=np.intp)
    for b in sorted(set(batches)):
        pos = np.flatnonzero(batches == b)
        out[:, pos] = rng.choice(pos, size=(n_boot, len(pos)), replace=True)
    return out


def percentile_ci(samples: np.ndarray, n_boot: int, level: float) -> tuple[float, float]:
    """Percentile CI over finite bootstrap samples.

    Parameters
    ----------
    samples : numpy.ndarray
        Bootstrap statistic values (NaN allowed).
    n_boot : int
        Planned number of resamples.
    level : float
        Confidence level.

    Returns
    -------
    tuple of float
        ``(lo, hi)``; ``(nan, nan)`` if fewer than ``n_boot / 2`` samples are finite.
    """
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size < n_boot / 2 or s.size == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo, hi = np.quantile(s, [a, 1 - a])
    return float(lo), float(hi)


def icc1(values: np.ndarray, groups: np.ndarray, batches: np.ndarray) -> float:
    """ICC(1) from a one-way random-effects ANOVA on batch-residualised values.

    Parameters
    ----------
    values : numpy.ndarray
        Replicate measurements.
    groups : numpy.ndarray
        Group (site) id per measurement.
    batches : numpy.ndarray
        Batch label per measurement.

    Returns
    -------
    float
        ``(MSB - MSW) / (MSB + (n0 - 1) MSW)`` with MSB on ``a - g`` df; NaN when
        undefined. Negative values are not clipped.
    """
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v, g, b = v[ok], np.asarray(groups)[ok], np.asarray(batches)[ok]
    if v.size == 0:
        return float("nan")
    gc = np.unique(g, return_inverse=True)[1]
    keep = np.bincount(gc)[gc] >= 2
    v, g, b = v[keep], g[keep], b[keep]
    if v.size == 0:
        return float("nan")
    bc = np.unique(b, return_inverse=True)[1]
    v = v - (np.bincount(bc, v) / np.bincount(bc))[bc]
    gc = np.unique(g, return_inverse=True)[1]
    a, nb, n_tot = int(gc.max()) + 1, int(bc.max()) + 1, v.size
    if a - nb < 1 or n_tot - a < 1:
        return float("nan")
    ni = np.bincount(gc)
    gm = np.bincount(gc, v) / ni
    ssb = float((ni * (gm - v.mean()) ** 2).sum())
    ssw = float(((v - gm[gc]) ** 2).sum())
    msb, msw = ssb / (a - nb), ssw / (n_tot - a)
    n0 = (n_tot - (ni ** 2).sum() / n_tot) / (a - 1)
    den = msb + (n0 - 1) * msw
    if not np.isfinite(den) or den == 0:
        return float("nan")
    return float((msb - msw) / den)


def icc1_bootstrap(per_site_values: Sequence[np.ndarray], site_batches: np.ndarray, idx: np.ndarray,
                   level: float) -> tuple[float, float, float, int]:
    """ICC(1) with a batch-stratified site-cluster bootstrap CI.

    Parameters
    ----------
    per_site_values : sequence of numpy.ndarray
        Replicate values per site, in site-table order.
    site_batches : numpy.ndarray
        Batch label per site.
    idx : numpy.ndarray
        Bootstrap indices from ``bootstrap_indices(..., STREAM_ICC)``.
    level : float
        Confidence level.

    Returns
    -------
    tuple
        ``(icc, lo, hi, n_rep_sites)``; ``n_rep_sites`` counts sites with >= 2
        non-missing replicates.
    """
    clean = [np.asarray(v, float)[np.isfinite(v)] for v in per_site_values]
    counts = np.array([len(v) for v in clean])
    n_sites = len(clean)

    def _icc(sel: np.ndarray) -> float:
        c = counts[sel]
        vals = np.concatenate([clean[j] for j in sel]) if c.sum() else np.empty(0)
        return icc1(vals, np.repeat(np.arange(len(sel)), c), np.repeat(site_batches[sel], c))

    point = _icc(np.arange(n_sites))
    boots = np.array([_icc(row) for row in idx])
    lo, hi = percentile_ci(boots, len(idx), level)
    return point, lo, hi, int((counts >= 2).sum())


def robustness_rho(sens: pd.DataFrame, kpi: str, params: Sequence[str]) -> tuple[float, int, bool, bool]:
    """Minimum pairwise Spearman rho across parameter settings.

    Parameters
    ----------
    sens : pandas.DataFrame
        Sentinel-cleaned sensitivity table.
    kpi : str
        KPI column.
    params : sequence of str
        Parameter columns defining a setting.

    Returns
    -------
    tuple
        ``(min rho over pairs sharing >= MIN_PAIRED_SITES finite sites, constant pairs
        counted as 0.0; NaN if no such pair, n_settings, had_constant_pair,
        had_insufficient_pair)``.
    """
    wide = sens.pivot(index=list(KEY_COLS), columns=list(params), values=kpi)
    vals = wide.to_numpy(float)
    k = vals.shape[1]
    zeros = np.zeros(vals.shape[0])
    rhos: list[float] = []
    constant = insufficient = False
    for i in range(k):
        for j in range(i + 1, k):
            ok = np.isfinite(vals[:, i]) & np.isfinite(vals[:, j])
            if ok.sum() < MIN_PAIRED_SITES:
                insufficient = True
                continue
            r = partial_spearman(vals[:, i], vals[:, j], zeros)
            if np.isnan(r):
                constant = True
                r = 0.0
            rhos.append(r)
    return (float(min(rhos)) if rhos else float("nan")), k, constant, insufficient


# ------------------------------------------------------------------- redundancy


def _abs_spearman(frame: pd.DataFrame) -> np.ndarray:
    r = frame.corr(method="spearman").abs().to_numpy()
    r = np.nan_to_num(r, nan=0.0)
    np.fill_diagonal(r, 1.0)
    return r


def _cluster_labels(frame: pd.DataFrame, redundancy_rho: float) -> np.ndarray:
    d = np.clip(1 - _abs_spearman(frame), 0, None)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="average")
    return fcluster(z, t=1 - redundancy_rho, criterion="distance")


def redundancy_clusters(site: pd.DataFrame, kpis: Sequence[str], icc: Mapping[str, float],
                        batches: np.ndarray, cfg: ScreenConfig) -> pd.DataFrame:
    """Gate 4: bootstrap co-clustering of surviving KPIs.

    Parameters
    ----------
    site : pandas.DataFrame
        Sentinel-cleaned site table in site order.
    kpis : sequence of str
        Surviving KPIs in input column order.
    icc : mapping of str to float
        ICC point estimates (NaN or absent when unavailable).
    batches : numpy.ndarray
        Batch label per site.
    cfg : ScreenConfig
        Thresholds and bootstrap settings.

    Returns
    -------
    pandas.DataFrame
        Indexed by KPI with ``cluster_id`` (int, numbered by earliest member) and
        ``cluster_rep`` (str).
    """
    kpis = list(kpis)
    m = len(kpis)
    if m == 0:
        return pd.DataFrame({"cluster_id": [], "cluster_rep": []}, index=pd.Index([], name="kpi"))
    if m == 1:
        return pd.DataFrame({"cluster_id": [1], "cluster_rep": kpis}, index=pd.Index(kpis, name="kpi"))
    frame = site[kpis].reset_index(drop=True)
    idx = bootstrap_indices(batches, cfg.n_boot_cluster, cfg.seed, STREAM_CLUSTER)
    freq = np.zeros((m, m))
    for row in idx:
        lab = _cluster_labels(frame.iloc[row].reset_index(drop=True), cfg.redundancy_rho)
        freq += lab[:, None] == lab[None, :]
    freq /= len(idx)
    _, comp = connected_components(csr_matrix(freq >= cfg.cocluster_frac), directed=False)
    full_r = _abs_spearman(frame)
    order: dict[int, int] = {}
    for c in comp:
        order.setdefault(int(c), len(order) + 1)
    ids = np.array([order[int(c)] for c in comp])
    reps = {}
    for cid in set(ids.tolist()):
        members = [i for i in range(m) if ids[i] == cid]
        finite = [i for i in members if np.isfinite(icc.get(kpis[i], np.nan))]
        if finite:
            best = min(finite, key=lambda i: (-icc[kpis[i]], kpis[i]))
        else:
            def mean_rho(i: int) -> float:
                others = [j for j in members if j != i]
                return float(np.mean([full_r[i, j] for j in others])) if others else 1.0
            best = min(members, key=lambda i: (-mean_rho(i), kpis[i]))
        reps[cid] = kpis[best]
    return pd.DataFrame({"cluster_id": ids, "cluster_rep": [reps[int(c)] for c in ids]},
                        index=pd.Index(kpis, name="kpi"))


# ------------------------------------------------------------------ orchestrator


def screen(site: pd.DataFrame, kpis: Sequence[str], cfg: ScreenConfig = ScreenConfig(),
           replicates: pd.DataFrame | None = None, sensitivity: pd.DataFrame | None = None,
           covariates: pd.DataFrame | None = None) -> ScreenResult:
    """Run the five screening gates on a site-level KPI table.

    Parameters
    ----------
    site : pandas.DataFrame
        ``batch, site`` plus KPI columns, one row per site (original values).
    kpis : sequence of str
        KPI column names, in input order.
    cfg : ScreenConfig, optional
        Thresholds and settings.
    replicates : pandas.DataFrame, optional
        Tile table for the reliability gate.
    sensitivity : pandas.DataFrame, optional
        Parameter sweep table for the robustness gate (needs ``cfg.sensitivity_params``).
    covariates : pandas.DataFrame, optional
        Wide covariate table for the artefact gate.

    Returns
    -------
    ScreenResult
        The per-KPI table and the filtered site table.

    Raises
    ------
    ScreenInputError
        If inputs violate the contract.
    """
    kpis = list(kpis)
    _check_keys(site, "kpis")
    for c, _ in cfg.sentinels:
        if c not in kpis:
            raise ScreenInputError(f"--sentinel column {c!r} is not a site KPI")
    rep_cols = validate_replicates(replicates, site, kpis, cfg) if replicates is not None else []
    if (sensitivity is None) != (not cfg.sensitivity_params):
        raise ScreenInputError("--sensitivity and --sensitivity-params must be given together")
    sens_cols = validate_sensitivity(sensitivity, site, kpis, cfg) if sensitivity is not None else []
    cov_cols = validate_covariates(covariates, site, cfg) if covariates is not None else []

    S = apply_sentinels(site, kpis, cfg.sentinels)
    R = apply_sentinels(replicates, rep_cols, cfg.sentinels) if replicates is not None else None
    N = apply_sentinels(sensitivity, sens_cols, cfg.sentinels) if sensitivity is not None else None
    batches = S["batch"].to_numpy()
    site_keys = pd.MultiIndex.from_frame(S[list(KEY_COLS)])
    covmat = None
    if covariates is not None:
        C = apply_sentinels(covariates, cov_cols, ())
        covmat = C.set_index(list(KEY_COLS))[cov_cols].reindex(site_keys).to_numpy(float)

    idx_cov = bootstrap_indices(batches, cfg.n_boot_cov, cfg.seed, STREAM_COVARIATE)
    idx_icc = bootstrap_indices(batches, cfg.n_boot_icc, cfg.seed, STREAM_ICC)
    rep_by_kpi: dict[str, list[np.ndarray]] = {}
    if R is not None:
        rkeys = pd.MultiIndex.from_frame(R[list(KEY_COLS)])
        pos = pd.Series(np.arange(len(site_keys)), index=site_keys).reindex(rkeys).to_numpy()
        for k in rep_cols:
            buckets: list[list[float]] = [[] for _ in range(len(site_keys))]
            for p, v in zip(pos, R[k].to_numpy(float)):
                buckets[p].append(v)
            rep_by_kpi[k] = [np.array(b, float) for b in buckets]

    rows = []
    for k in kpis:
        x = S[k].to_numpy(float)
        row: dict = {"kpi": k, **degeneracy_stats(S[k])}
        row["batch_eta2"] = batch_eta2(x, batches)
        flags: list[str] = []
        untested: list[str] = []

        # gate 1 statistics
        art_cov = None
        suspect = None
        ci_undef = None
        if covmat is not None:
            covs = []
            for j, cname in enumerate(cov_cols):
                y = covmat[:, j]
                rho = partial_spearman(x, y, batches)
                boots = np.array([partial_spearman(x[i], y[i], batches[i]) for i in idx_cov])
                lo, hi = percentile_ci(boots, len(idx_cov), cfg.ci_level)
                excl = bool(np.isfinite(lo) and (lo > 0 or hi < 0))
                qual = bool(np.isfinite(rho) and abs(rho) >= cfg.covariate_rho and excl)
                covs.append((j, cname, rho, lo, hi, qual))
            q = [c for c in covs if c[5]]
            fin = [c for c in covs if np.isfinite(c[2])]
            if not any(np.isfinite(c[3]) and np.isfinite(c[4]) for c in fin):
                untested.append("artefact")
            rep = max(q, key=lambda c: abs(c[2])) if q else (max(fin, key=lambda c: abs(c[2])) if fin else None)
            if rep is not None:
                j, cname, rho, lo, hi, qual = rep
                row.update(max_abs_rho_covariate=abs(rho), covariate_of_max=cname, covariate_rho=rho,
                           covariate_rho_lo=lo, covariate_rho_hi=hi,
                           covariate_rho_marginal=partial_spearman(x, covmat[:, j], np.zeros(len(x))))
                if qual:
                    art_cov = cname
                elif not (np.isfinite(lo) and np.isfinite(hi)):
                    ci_undef = f"covariate_ci_undefined:{cname}"
                elif abs(rho) >= cfg.covariate_rho:
                    suspect = f"covariate_suspect:{cname}"
        else:
            untested.append("artefact")

        # gate 2
        icc_v = icc_lo = icc_hi = float("nan")
        n_rep = None
        icc_flags: list[str] = []
        if k in rep_by_kpi:
            icc_v, icc_lo, icc_hi, n_rep = icc1_bootstrap(rep_by_kpi[k], batches, idx_icc, cfg.ci_level)
            if not np.isfinite(icc_v):
                icc_flags.append("icc_undefined")
            if not np.isfinite(icc_lo):
                icc_flags.append("icc_ci_undefined")
            if icc_flags:
                untested.append("reliability")
        else:
            untested.append("reliability")
        row.update(icc=icc_v, icc_lo=icc_lo, icc_hi=icc_hi, n_rep_sites=n_rep)

        # gate 3
        rob = float("nan")
        n_set = None
        const_flag = short_flag = False
        if N is not None and k in sens_cols:
            rob, n_set, const_flag, short_flag = robustness_rho(N, k, cfg.sensitivity_params)
            if not np.isfinite(rob):
                untested.append("robustness")
        else:
            untested.append("robustness")
        row.update(robustness_rho=rob, n_settings=n_set)

        if np.isfinite(row["batch_eta2"]) and row["batch_eta2"] > cfg.eta2_flag:
            flags.append("high_batch_eta2")
        if suspect:
            flags.append(suspect)
        if ci_undef:
            flags.append(ci_undef)
        flags += icc_flags
        if const_flag:
            flags.append("robustness_constant_setting")
        if short_flag:
            flags.append("robustness_insufficient_overlap")

        # decision: first failing gate
        gate = ""
        if row["n_missing_frac"] > cfg.max_missing_frac:
            gate = "degeneracy:missing"
        elif row["n_unique"] < cfg.min_unique:
            gate = "degeneracy:few_unique"
        elif not (row["mad"] > 0):
            gate = "degeneracy:zero_mad"
        elif art_cov is not None:
            gate = f"artefact:{art_cov}"
        elif "reliability" not in untested and icc_hi < cfg.icc_upper_min:
            gate = "reliability"
        elif np.isfinite(rob) and rob < cfg.robustness_min:
            gate = "robustness"
        row.update(deciding_gate=gate, decision="drop" if gate else "keep",
                   untested_gates=";".join(untested), flags=";".join(flags))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index = list(df["kpi"])
    survivors = [k for k in kpis if df.at[k, "decision"] == "keep"]
    df["cluster_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["cluster_rep"] = ""
    if survivors:
        cl = redundancy_clusters(S, survivors, df["icc"].to_dict(), batches, cfg)
        for k in survivors:
            df.at[k, "cluster_id"] = int(cl.at[k, "cluster_id"])
            rep_name = cl.at[k, "cluster_rep"]
            df.at[k, "cluster_rep"] = rep_name
            if rep_name != k:
                df.at[k, "decision"] = "drop"
                df.at[k, "deciding_gate"] = f"redundant_with:{rep_name}"

    kept = df[df["decision"] == "keep"]
    kept = kept.assign(_i=kept["icc"].isna(), _e=kept["batch_eta2"].isna(),
                       _icc=kept["icc"].fillna(0), _eta=kept["batch_eta2"].fillna(0))
    kept = kept.sort_values(["_i", "_icc", "_e", "_eta", "kpi"], ascending=[True, False, True, True, True])
    ranks = {k: i + 1 for i, k in enumerate(kept["kpi"])}
    df["rank"] = pd.array([ranks.get(k, pd.NA) for k in df["kpi"]], dtype="Int64")
    order = list(kept["kpi"]) + [k for k in kpis if k not in ranks]
    df = df.loc[order].reindex(columns=TABLE_COLUMNS)
    for c in ("deciding_gate", "untested_gates", "flags", "covariate_of_max", "cluster_rep"):
        df[c] = df[c].fillna("")
    for c in ("n_unique", "n_rep_sites", "n_settings"):
        df[c] = df[c].astype("float").astype("Int64")
    df["cluster_id"] = df["cluster_id"].astype("Int64")
    df = df.reset_index(drop=True)
    filtered = site[[*KEY_COLS, *[k for k in kpis if k in ranks]]].reset_index(drop=True)
    return ScreenResult(table=df, filtered=filtered)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (defaults come from ``ScreenConfig()``).

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    d = ScreenConfig()
    p = argparse.ArgumentParser(prog="python -m pmdb.screen",
                                description="Screen site-level KPIs for measurement quality (see docs/kpis/screening.md).")
    p.add_argument("--kpis", nargs="+", required=True, help="site-level KPI CSV file(s): batch, site, numeric KPI columns")
    p.add_argument("--site-manifest", type=Path, default=DEFAULT_SITE_MANIFEST,
                   help="CSV with batch, site columns: the canonical site set every --kpis file must cover "
                        "exactly (default: cache/half/manifest.csv, 31 sites)")
    p.add_argument("--replicates", help="tile CSV for the reliability gate (gate 2)")
    p.add_argument("--replicate-col", default=d.replicate_col, help="replicate id column in --replicates")
    p.add_argument("--sensitivity", help="parameter-sweep CSV for the robustness gate (gate 3)")
    p.add_argument("--sensitivity-params", nargs="+", default=[],
                   help="parameter columns defining a setting (required with --sensitivity)")
    p.add_argument("--covariates", help="wide covariate CSV for the artefact gate (gate 1)")
    p.add_argument("--sentinel", nargs="+", action="extend", default=[],
                   help="COL=VALUE sentinel treated as missing (repeatable)")
    p.add_argument("--ignore-cols", nargs="+", default=[], help="extra column names to ignore (added to defaults)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    p.add_argument("--max-missing-frac", type=float, default=d.max_missing_frac,
                   help="gate 0: drop if missing fraction is greater than this")
    p.add_argument("--min-unique", type=int, default=d.min_unique,
                   help="gate 0: drop if distinct values are fewer than this")
    p.add_argument("--eta2-flag", type=float, default=d.eta2_flag,
                   help="gate 1: flag (never drop) if batch eta-squared is greater than this")
    p.add_argument("--covariate-rho", type=float, default=d.covariate_rho,
                   help="gate 1: drop if |within-batch rho| >= this and CI excludes 0")
    p.add_argument("--ci-level", type=float, default=d.ci_level, help="confidence level of bootstrap CIs")
    p.add_argument("--n-boot-cov", type=int, default=d.n_boot_cov, help="gate 1: bootstrap resamples")
    p.add_argument("--icc-upper-min", type=float, default=d.icc_upper_min,
                   help="gate 2: drop if ICC CI upper bound is less than this")
    p.add_argument("--n-boot-icc", type=int, default=d.n_boot_icc, help="gate 2: bootstrap resamples")
    p.add_argument("--robustness-min", type=float, default=d.robustness_min,
                   help="gate 3: drop if min pairwise Spearman rho across settings is less than this")
    p.add_argument("--redundancy-rho", type=float, default=d.redundancy_rho,
                   help="gate 4: |rho| at or above which KPIs cluster together")
    p.add_argument("--cocluster-frac", type=float, default=d.cocluster_frac,
                   help="gate 4: link KPIs co-clustered in at least this fraction of resamples")
    p.add_argument("--n-boot-cluster", type=int, default=d.n_boot_cluster, help="gate 4: bootstrap resamples")
    p.add_argument("--seed", type=int, default=d.seed, help="base random seed")
    return p


def write_outputs(result: ScreenResult, cfg: ScreenConfig, out_dir: Path, inputs: Mapping[str, object],
                  ignored: Mapping[str, list[str]]) -> None:
    """Write ``kpi_screen.csv``, ``site_kpis_filtered.csv`` and ``screen_config.json``.

    Parameters
    ----------
    result : ScreenResult
        Output of :func:`screen`.
    cfg : ScreenConfig
        Configuration to echo.
    out_dir : Path
        Output directory (created if needed).
    inputs : mapping
        Input paths: ``kpis`` (list), ``site_manifest``, ``replicates``, ``sensitivity``, ``covariates``.
    ignored : mapping of str to list of str
        Ignored non-numeric columns per source file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out_dir / "kpi_screen.csv", index=False)
    result.filtered.to_csv(out_dir / "site_kpis_filtered.csv", index=False)
    meta = {
        "screen_version": SCREEN_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "inputs": dict(inputs),
        "seeds": {"seed": cfg.seed, "streams": {"covariate": STREAM_COVARIATE, "icc": STREAM_ICC,
                                                 "cluster": STREAM_CLUSTER}},
        "n_sites": int(len(result.filtered)),
        "n_kpis": int(len(result.table)),
        "n_kept": int((result.table["decision"] == "keep").sum()),
        "ignored_columns": {k: list(v) for k, v in ignored.items()},
    }
    (out_dir / "screen_config.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (default ``sys.argv[1:]``).

    Returns
    -------
    int
        0 on success, 2 on an input error.
    """
    args = build_parser().parse_args(argv)
    special = ("sentinels", "sensitivity_params", "ignore_cols")
    try:
        cfg = ScreenConfig(
            **{f.name: getattr(args, f.name) for f in fields(ScreenConfig)
               if f.name not in special},
            sentinels=parse_sentinels(args.sentinel),
            sensitivity_params=tuple(args.sensitivity_params),
            ignore_cols=tuple(DEFAULT_IGNORE_COLS) + tuple(args.ignore_cols),
        )
        expected = read_site_manifest(args.site_manifest)
        frames = [read_table(p) for p in args.kpis]
        site, kpis, ignored = merge_kpi_tables(frames, args.kpis, cfg.ignore_cols, expected_sites=expected)
        for src, cols in ignored.items():
            print(f"warning: ignoring non-numeric column(s) in {src}: {cols}", file=sys.stderr)
        rep = read_table(args.replicates) if args.replicates else None
        sens = read_table(args.sensitivity) if args.sensitivity else None
        cov = read_table(args.covariates) if args.covariates else None
        result = screen(site, kpis, cfg, rep, sens, cov)
    except ScreenInputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    inputs = {"kpis": [str(p) for p in args.kpis], "site_manifest": str(args.site_manifest),
              "replicates": args.replicates,
              "sensitivity": args.sensitivity, "covariates": args.covariates}
    write_outputs(result, cfg, args.out_dir, inputs, ignored)
    t = result.table
    drops: dict[str, int] = {}
    for g in t.loc[t["decision"] == "drop", "deciding_gate"]:
        key = g.split(":")[0]
        drops[key] = drops.get(key, 0) + 1
    print(f"kept {int((t['decision'] == 'keep').sum())}/{len(t)} KPIs; drops by gate: {drops} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
