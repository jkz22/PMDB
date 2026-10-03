# Batch classifier on combined KPIs

Classic (non-deep) classifiers predicting `batch` from per-site KPIs, using the three KPI lanes combined. Caveat: n = 31 sites (7 / 7 / 17); Anna's lane comes from unmerged branch `analytical-benchmarks` @ b465ed7. Each (feature set, model) has a label-shuffle null computed for `perm_score` (single shuffled 5-fold split); `cv_bal_acc_mean` (5x20 repeated CV) has no null of its own. p-values come from that single 5-fold split and are uncorrected for multiple testing across the 20 feature-set x model combinations; minimum attainable p = 1/201.

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
- `xgb`: `XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, random_state=0)`; no class weighting

Score: balanced accuracy, `RepeatedStratifiedKFold(5, 20)`; `cv_bal_acc_fold_std` is the std across the 100 folds (fold-to-fold spread, not the standard error of the mean). K08 = -1 ("no peak") is passed to the models through a missing-value indicator (appended indicator columns). Null: `permutation_test_score`, `StratifiedKFold(5)`, 200 permutations. Leave-one-out predictions and combined-set importances are also written.

## Results

Results: not yet run.
