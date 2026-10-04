# Negative results: FEM features for batch classification

> Paths under `.claude/` cited here (plans, agent reports) were removed from the tree in the release cleanup (#169). Read them from history: `git show ed0a840:<path>`.

Provenance record. Not part of the pitch. Classifier of record: Leo's spatial fingerprint (`pmdb/fingerprint.py`, LOO 21/31).

## Why the FEM features did not help (diagnosis)
- Simulated swelling is set by Si content: R² = 0.971 against K01, no batch signal left in the residual (Kruskal-Wallis p = 0.699). Si content itself does not separate the batches (p = 0.377). Figure: `outputs/fem/figures/swelling_vs_si.png`.
- Stress features saturate under the linear small-strain model (D20): Si von Mises ≈ 24 GPa everywhere, yield fraction 1.0.
- No pipeline bug: FEM/KPI tile joins complete, no NaNs, tile bounds agree.
- Full evidence: `.claude/checkpoint/reports/fem-why.*.md`.

## D21: FEM added to the fingerprint (pre-registered, `.claude/plans/fem-swelling.md`)
Rule: an FEM arm adds value iff LOO ≥ 23/31 and permutation p ≤ 0.05. Results in `outputs/fem_fingerprint/{main,edge5}/verdict.md`.

| Arm | Correct | Permutation p |
|---|---|---|
| A0 fingerprint | 21/31 | 0.006 |
| A1 + 2 physics FEM features | 20/31 (edge5: 21/31) | 0.006 |
| A2 + top-2 raw FEM, in-fold | 17/31 (edge5: 18/31) | 0.081 (edge5: 0.037) |

Verdict: no FEM arm passes.

## D22: XGBoost on the screened KPIs, then + FEM (pre-registered)
Rule: an arm beats the fingerprint iff LOO ≥ 22/31 and permutation p ≤ 0.05. Inputs: `outputs/kpis/screen/site_kpis_filtered.csv` minus A02/A03 (imaging covariates), 24 KPIs. Results in `outputs/xgb_kpi/{main,edge5}/verdict.md`.

| Arm | Correct | Permutation p |
|---|---|---|
| X1 KPIs | 10/31 | 0.849 |
| X2 + physics FEM | 15/31 (edge5: 16/31) | 0.232 (edge5: 0.139) |
| X3 + top-2 raw FEM, in-fold | 11/31 (edge5: 11/31) | 0.711 (edge5: 0.722) |

Verdict: no arm beats the fingerprint. X1 overfits (100% train, below the 17/31 majority baseline in LOO); physics FEM features lift XGBoost by 5-6 sites, not significant.

## Also superseded
The two-stage random forest (D18, `docs/classifier/`): end-to-end balanced accuracy KPI 0.437, FEM 0.370, KPI+FEM 0.350.
