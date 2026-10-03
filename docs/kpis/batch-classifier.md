# Batch classifier on combined KPIs

Classic (non-deep) classifiers predicting `batch` from per-site KPIs, using the three KPI lanes combined. Caveat: n = 31 sites (7 / 7 / 17); Anna's lane comes from unmerged branch `analytical-benchmarks` @ b465ed7. Every score is reported against a label-shuffle null.

Regenerate:

```
python scripts/batch_classifier.py
```

## Inputs

| lane | file | owner | join key |
|---|---|---|---|
| geometric (K/D/A) | `outputs/kpis/site_kpis.csv` | Kevin | batch, site |
| materials (`mat_`) | `outputs/kpis/materials_site_kpis.csv` | Leo | batch, site |
| analytical (`ab_`) | `outputs/kpis/anna_site_kpis.csv` (from `origin/analytical-benchmarks` @ b465ed7 `analytical_benchmarks/site_kpis.csv`) | Anna | batch, site |

## Feature sets

- `kevin`: columns matching `^(K\d|D\d|A\d)` (K08 `-1` sentinel set to NaN).
- `leo`: `^mat_`.
- `anna`: `^ab_` (acquisition / calibration columns and `__err` / `__naive_err` dropped).
- `combined`: union of the three.
- `combined_no_artefact`: `combined` minus `^A\d`, `mat_porosity`, `mat_rim_coverage`, `ab_inlens_texture.*`, `ab_inlens_edge_binder`, `ab_porosity.*`.

## Models and CV

Each model is `SimpleImputer(median)` -> `StandardScaler` -> estimator:

- `logreg`: `LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)`
- `lda`: `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`
- `rf`: `RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0)`

Score: balanced accuracy, `RepeatedStratifiedKFold(5, 20)`. Null: `permutation_test_score`, `StratifiedKFold(5)`, 200 permutations. Leave-one-out predictions and combined-set importances are also written.

## Results

Results: not yet run.
