"""Batch statistics. Reads only results/kpis.parquet.

Per KPI: baseline mean/std/95% interval, per-batch effect size vs
baseline, permutation p-value. Multivariate: Hotelling T2 with a
permutation p-value. Plus within-batch CV and a leave-one-out
false-alarm check on the baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import ROOT, load_config

N_PERMUTATIONS = 10_000


def kpi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per image, one column per KPI."""
    # pivot (not pivot_table) so duplicate (image, kpi) rows raise instead
    # of being silently averaged
    return df.pivot(index=["batch", "image"], columns="kpi", values="value")


def effect_sizes(wide: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Per KPI, per non-baseline batch: effect size and permutation p."""
    rng = np.random.default_rng(0)
    base = wide.loc[baseline]
    rows = []
    for batch in wide.index.get_level_values("batch").unique():
        if batch == baseline:
            continue
        incoming = wide.loc[batch]
        for kpi in wide.columns:
            b, x = base[kpi].dropna().values, incoming[kpi].dropna().values
            if len(b) < 2 or len(x) < 1:
                continue
            effect = (x.mean() - b.mean()) / b.std(ddof=1)
            pooled = np.concatenate([b, x])
            obs = abs(x.mean() - b.mean())
            n_x = len(x)
            perm = np.array([
                abs(s[:n_x].mean() - s[n_x:].mean())
                for s in (rng.permutation(pooled) for _ in range(N_PERMUTATIONS))
            ])
            p = (1 + (perm >= obs).sum()) / (1 + N_PERMUTATIONS)
            rows.append({"batch": batch, "kpi": kpi, "effect_size": effect,
                         "p_value": p,
                         "baseline_mean": b.mean(), "baseline_std": b.std(ddof=1),
                         "baseline_lo": np.percentile(b, 2.5),
                         "baseline_hi": np.percentile(b, 97.5),
                         "batch_mean": x.mean()})
    return pd.DataFrame(rows)


def hotelling_t2_p(wide: pd.DataFrame, baseline: str, batch: str,
                   rng: np.random.Generator | None = None) -> float:
    """Permutation p-value for the Hotelling T2 statistic between the
    baseline images and one incoming batch, on standardised KPIs.

    With ~7 images per group and more KPIs than images the covariance is
    singular, so the pooled covariance uses a pseudo-inverse; the
    permutation test keeps the p-value exact regardless.
    """
    rng = rng or np.random.default_rng(0)
    common = wide.loc[baseline].dropna(axis=1).columns.intersection(
        wide.loc[batch].dropna(axis=1).columns)
    b = wide.loc[baseline, common].values
    x = wide.loc[batch, common].values
    pooled = np.vstack([b, x])
    mu, sd = pooled.mean(0), pooled.std(0, ddof=1)
    sd[sd == 0] = 1.0
    pooled = (pooled - mu) / sd

    def t2(stack: np.ndarray, n1: int) -> float:
        g1, g2 = stack[:n1], stack[n1:]
        diff = g1.mean(0) - g2.mean(0)
        cov = np.cov(stack.T) + 1e-9 * np.eye(stack.shape[1])
        return float(diff @ np.linalg.pinv(cov) @ diff)

    obs = t2(pooled, len(b))
    count = sum(
        t2(pooled[rng.permutation(len(pooled))], len(b)) >= obs
        for _ in range(N_PERMUTATIONS // 10)  # T2 is costlier; 1000 perms
    )
    return (1 + count) / (1 + N_PERMUTATIONS // 10)


def loo_false_alarms(wide: pd.DataFrame, baseline: str, verdict_fn) -> pd.DataFrame:
    """Leave-one-out on baseline images: treat each image as a one-image
    incoming batch against the remaining baseline. Returns the verdict per
    held-out image; anything other than accept is a false alarm."""
    base = wide.loc[baseline]
    rows = []
    for img in base.index:
        rest = base.drop(index=img)
        held = base.loc[[img]]
        tmp = pd.concat([rest, held], keys=["__base__", "__held__"],
                        names=["batch", "image"])
        eff = effect_sizes(tmp, "__base__")
        p_mv = hotelling_t2_p(tmp, "__base__", "__held__")
        verdict, _ = verdict_fn(eff[eff.batch == "__held__"], p_mv)
        rows.append({"image": img, "verdict": verdict, "p_multivariate": p_mv,
                     "max_abs_effect": eff.effect_size.abs().max()})
    return pd.DataFrame(rows)


def consistency(wide: pd.DataFrame) -> pd.DataFrame:
    """Within-batch coefficient of variation per KPI per batch."""
    g = wide.groupby(level="batch")
    cv = g.std(ddof=1) / g.mean().abs()
    return cv


def main() -> None:
    cfg = load_config()
    baseline = cfg["data"]["baseline_batch"]
    df = pd.read_parquet(ROOT / "results" / "kpis.parquet")
    wide = kpi_matrix(df)

    from src.verdict import decide

    eff = effect_sizes(wide, baseline)
    out = ROOT / "results"
    eff.to_parquet(out / "effects.parquet", index=False)

    # The report is written to a file alongside effects.parquet (and echoed
    # to stdout) so it can never go stale relative to the parquets, as it
    # did when it only existed via shell redirection.
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit(f"baseline: {baseline} "
         f"({len(wide.loc[baseline])} images, {wide.shape[1]} KPIs)\n")
    for batch in eff.batch.unique():
        p_mv = hotelling_t2_p(wide, baseline, batch)
        sub = eff[eff.batch == batch]
        verdict, drivers = decide(sub, p_mv)
        emit(f"=== {batch}: {verdict.upper()} (multivariate p={p_mv:.4f}) ===")
        emit(sub.reindex(sub.effect_size.abs().sort_values(ascending=False).index)
             [["kpi", "effect_size", "p_value", "baseline_mean", "batch_mean"]]
             .head(5).to_string(index=False))
        emit()

    emit("--- within-batch CV (per KPI) ---")
    emit(consistency(wide).T.round(3).to_string())

    emit("\n--- leave-one-out false alarms on baseline ---")
    loo = loo_false_alarms(wide, baseline, decide)
    emit(loo.to_string(index=False))
    n_bad = (loo.verdict != "accept").sum()
    emit(f"\n{n_bad}/{len(loo)} baseline images would be flagged "
         f"(this number goes on a slide)")

    report = out / f"stats_baseline_{baseline}.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
