# D21 Step 3 (edge5 re-reduction) - implementer report
Changes: pmdb/fem/features.py (z_edge_um in region_metrics/run_curves/_first_pore_closure_s; default 0.0 keeps the original code path);
tests/test_fem_unit.py (2 new tests); modal_fem.py (refeature_case + _mode_refeature, `--mode refeature`, `--z-edge-um` default 5.0, budget rule);
scripts/fem_collect.py (`--edge RESULTS_DIR OUT_DIR`); outputs/fem/edge5/{site,tile}_curves.csv.
Definition: rows within z_edge_um of top/bottom dropped from all stats (also pore-closure and porosity baseline). Swelling when cropped =
(mean uz at upper crop node row - mean uz at lower crop node row)/cropped height (interior strain, both orientations); surface_rough = std of uz over upper crop row / cropped height.
Modal: 68/68 ok, wall 80 s, spend $0.377 (ledger 2.608 -> 2.985, cap 180). No failures.
Verification: pytest test_fem_unit+docs+modal_fixes passed; row counts 1122 x 90 / 6732 x 93, identical columns; no missing cases.
Spearman orig vs edge5 (sym, s=1): swelling 0.894, surface_rough 0.639 (top-only -0.10, bottom 0.73), porosity 0.923, vm_si_p95 0.797.
Means s=1 sym: swelling 0.1311 -> 0.1297; surface_rough 0.01088 -> 0.00675 (changed); porosity 0.0361 -> 0.0378.
