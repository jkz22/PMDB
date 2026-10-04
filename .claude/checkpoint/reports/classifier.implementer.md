# classifier implementer report (steps 1-6 done, step 7 skipped; nothing committed)

## Files changed (uncommitted, in worktree)
- pmdb/classify/features.py (modified): loader aligned to fem-build s2.1 (FEM_TILE_COL = "tile" only; `config` handling and FEM_CONFIG removed; orientation filter `sym`; other checks unchanged). All 12 required metric names exist unchanged in the s2.1 81-metric list (no rename needed).
- pmdb/classify/model.py, pmdb/classify/explain.py (new, Steps 3-4)
- scripts/classify_batches.py (`run` added: --fem-tiles, --require-fem, --out, --docs)
- tests/test_classify.py (fixture updated to s2.1: no config, `tile` column; loader test now covers missing tile column and the drop-top layout; +7 model/explain tests), tests/test_classify_data.py (+test_outputs_consistent)
- docs/classifier/method.md (hand-written), docs/classifier/results.md (script-written), outputs/classifier/* (kpi_tiles6.csv 204 rows plus all C17 files)
- Kept from the earlier partial work after checking against the plan: __init__.py, kpi_tiles.py, kpi-tiles subcommand, data tests, the rest of features.py.

## Verification
- kpi-tiles: exit 0, 204 rows. NaN counts: only K05_voronoi_sigma = 6 of 204 tiles (under the 10% limit); all other columns 0. K16 is constant 0 and dropped by the C5 rule, leaving 15 KPI features.
- pytest -q -m "not data": 149 passed, 1 skipped. pytest -q -m data: 17 passed, 1 skipped.
- Determinism: two consecutive `run`s, md5 of all 11 CSVs plus report.md plus docs/classifier/results.md identical (13 hashes, cmp IDENTICAL).
- `run --require-fem` exits 2. Plain run prints "FEM tables not found: KPI arm only".
- FEM-path smoke: synthetic s2.1-shaped fixture for the real 34 sites (bottom/top/sym) at /tmp/fem_fixture.csv, `run --out /tmp/clf_smoke --docs /tmp/clf_smoke/results.md`: metrics.csv has KPI, FEM, KPI+FEM x 3 levels. Committed outputs are the KPI-only real run.
- `git status --porcelain data data_heldout cache cache_heldout outputs/kpis pmdb/kpis pmdb/fem outputs/fem`: empty.

## KPI-only LOSO results (verbatim from docs/classifier/results.md)
| arm | level | n_sites | site_acc | balanced_acc | bacc_ci_lo | bacc_ci_hi | macro_f1 | brier |
|---|---|---|---|---|---|---|---|---|
| KPI | stage1 | 31 | 0.581 | 0.548 | 0.418 | 0.702 | 0.507 | 0.233 |
| KPI | stage2 | 14 | 0.357 | 0.357 | 0.125 | 0.625 | 0.354 | 0.277 |
| KPI | end_to_end | 31 | 0.581 | 0.437 | 0.292 | 0.630 | 0.447 | 0.191 |

End-to-end confusion (rows true, cols predicted B1/B2/B3): B1 2/0/5; B2 0/1/6; B3 0/2/15.
Low numbers match Expected surprise 6 (not tuned, not a bug). Check-3 R1 multiclass balanced accuracy was 0.493.

## Provisional held-out predictions (KPI arm; provisional, FEM arms not run)
| site | predicted | confidence | p_b3 | q_b1 | tiles voting not-B3 |
|---|---|---|---|---|---|
| 3e122cbj | Batch_1 | 0.855 | 0.086 | 0.935 | 6/6 |
| fn0mhxef | Batch_3 | 0.562 | 0.562 | 0.476 | 1/6 |
| xrv9xvzb | Batch_3 | 0.586 | 0.586 | 0.354 | 1/6 |

Explanations vs Batch_3: 3e122cbj has much higher Si density (K02 116 vs 27.7, z +8.4), higher Si loading (K01 0.139 vs 0.062, z +8.0) and lower Si-graphite contact (z -5.9). fn0mhxef and xrv9xvzb deviate by at most |z| about 1.4 on any top feature (near-Batch_3 profile, low confidence). Full text in docs/classifier/results.md.

## Notes
- Loader alignment needed only the tile column / `config` removal (no escalation items). Missing-case NaN rows and the drop-top layout (sym = bottom) load fine since only `sym` rows are read (tested with top rows removed).
- Mechanical test fix: my calibration row count assert was 11, correct is 10 (3+3+4 bins).
- Classifier unit tests are slow (about 2 min for tests/test_classify.py); RF params untouched.
- Step 7 skipped: outputs/fem/tile_curves.csv absent. Per override, nothing committed; .claude/plans/classifier.md untouched.
