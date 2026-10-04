# Batch models: log

Every batch classifier built so far. Each writes its own per-site call + confidence; the submission combines them.
Add a row when a model is added. Held-out truths (organiser, 2026-10-04): 3e122cbj = Batch_2, fn0mhxef = Batch_1,
xrv9xvzb = Batch_3 (`outputs/heldout_labels.csv`).

Validation protocols:
- LOSO-31: leave-one-site-out on the 31 original sites. Optimistic, because sibling crops of the same parent image sit in the training fold.
- LOPO-34: leave-one-parent-out on 34 sites (31 + 3 held-out), 13 parents.

Confidence kinds:
- prob: top class probability.
- conformal: fingerprint credibility/confidence (p-value based, typically 0.1 to 0.5).

| model | what it is | code | outputs | validation | confidence | held-out calls (3e122cbj / fn0mhxef / xrv9xvzb) |
|---|---|---|---|---|---|---|
| fingerprint NB | Leo's 16 fingerprint features, naive-Bayes likelihood scores | `pmdb/fingerprint.py`, `scripts/run_fingerprint.py` ([doc](fingerprint.md)) | `outputs/fingerprint/` | LOSO-31; LOPO-34 in `outputs/menu/` | conformal | B1 / B3 / B2 |
| extended fingerprint | fingerprint + 10 pre-specified overnight features (not adopted) | `scripts/eval_extended_features.py` ([plan](eval-plan-oct4.md)) | `outputs/overnight/eval_extended/` | LOSO-31 | conformal | B3 / B3 / B2 |
| FEM fingerprint | fingerprint + FEM swelling/stress features, arms A0-A2 (A0 of record = fingerprint) | `scripts/fem_fingerprint_eval.py` ([docs](fem/)) | `outputs/fem_fingerprint/{main,edge5}/` | LOSO-31 | conformal | B1 / B3 / B2 (A0) |
| patch-MIL | MicroNet patch embeddings + calibrated kNN, pooled per site | `modal_patch_mil.py`, `pmdb/patch_mil.py`, `pmdb/patch_lopo.py` ([doc](patch_mil.md)) | `outputs/patch_mil/` (⚠️ LOPO files `final_heldout_predictions.csv`, `lopo_*` are stale, see [STALE.md](../outputs/patch_mil/STALE.md)) | LOSO-31; LOPO-34 | prob | B1 / B2 / B3 |
| GNN | GINE graph net on Si particles + graphite-mediated Delaunay edges | `modal_gnn.py` | `outputs/gnn/` | LOSO-31 | prob | B1 / B3 / B3 |
| XGB-KPI | XGBoost on 24 screened KPIs (X1), +FEM arms X2/X3 | `pmdb/xgb_kpi.py`, `scripts/xgb_kpi_eval.py` | `outputs/xgb_kpi/{main,edge5}/` | LOSO-31 | prob | B1 / B2 / B2 (X1) |
| tile classifier | two-stage random forest on tiles vs Batch_3; arms KPI / FEM / KPI+FEM (KPI final) | `scripts/classify_batches.py`, `pmdb/classify/` ([docs](classifier/)) | `outputs/classifier/` | LOSO-31 | prob | B1 / B3 / B3 (KPI) |
| menu ensemble (frozen pick) | mean of fingerprint `softmax(-score/0.1)` and patch-MIL probabilities | `pmdb/batch_menu.py` via `modal_patch_mil.py` ([doc](patch_mil.md#feedback-round-1-2026-10-04)) | `outputs/menu/` | LOPO-34 | prob + flag rule | B3 / B3 / B2 (LOPO) |
| lasso logreg | L1 multinomial logistic regression on all 82 combined KPIs (negative result: LOPO balanced accuracy 0.27, below chance) | `scripts/lasso_logreg.py` | `outputs/kpis/lasso_logreg/` | nested LOPO-31 (13 parents); each held-out site predicted by a model trained without its parent's sibling crops | prob | B3 / B2 / B1 |
| KPI-PCA | 82 combined KPIs, then 5 PCs, then balanced logistic regression; Batch_1 outliers 5n1q8atc and 4ih2ggld excluded from training | `pmdb/kpi_pca.py`, `scripts/kpi_menu.py` | `outputs/kpis/kpi_pca/lopo_predictions.csv` | LOPO-34 | prob + flag rule | B3 / B2 / B3 (LOPO) |
| fp + patch + KPI-PCA | mean of the three probability vectors (post-hoc menu option) | `scripts/kpi_menu.py` | `outputs/menu/kpi_menu_{predictions,summary}.csv` | LOPO-34 | prob + flag rule | B3 / B3 / B2 (LOPO) |
| patch probe | supervised linear probe on frozen MicroNet patch embeddings, pooled per site | `pmdb/patch_probe.py`, `modal_patch_mil.py --mode probe` ([doc](patch_mil.md)) | `outputs/patch_probe/` | LOPO-34 (24/34 = 0.71) | prob | B2 / B2 / B3 |
| probe + ensemble (adopted candidate) | probe probabilities combined with the menu ensemble; adopted over the ensemble on LOPO rubric (1.412 vs 1.235) and balanced accuracy (0.630 vs 0.597) | `pmdb/patch_probe.py` ([verdict](patch_mil.md)) | `outputs/patch_probe/predictions.csv` (`comb_*`) | LOPO-34 (24/34 = 0.71) | prob | B3 / B2 / B3 |
| dark-level cut rule (cross-check only) | pore-floor dark level `SE_type_D`/`BSE_D` (`pmdb/clean.py::dark_level`), two-cut ordinal rule B1<B2<B3; sibling-anchored variant tested and rejected (0.37 on mixed parents) | `scripts/run_sibling_anchor.py` ([readme](../outputs/sibling_anchor/README.md)) | `outputs/sibling_anchor/{SE_type_D,BSE_D}/` | LOPO-34 (0.65 SE / 0.68 BSE); rides on the Batch 3 black-level offset | margin to cut (high/low) | B2 / B2 / B3 (SE rule) |

Combined KPIs (82) = spatial K/D/A (`outputs/kpis/site_kpis.csv`) + materials `mat_` (`materials_site_kpis.csv`) +
analytical `ab_` (`anna_site_kpis.csv`); held-out copies under `outputs/heldout/kpis/`, loaded with
`scripts/batch_classifier.py::load_combined`.
