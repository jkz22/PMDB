# Q&A cheat sheet (2 min, so keep each answer to about 15 s)

**"Only {{fp_n}} sites. Isn't that too few?"**
Yes, which is why we fit no weights: {{fp_features}} curve-shape features and per-batch robust statistics.
Every claim is stress-tested instead: {{perm_n}} label permutations (p = {{perm_p}}; the null p95 is the
majority baseline, {{perm_null_p95}}), {{jk_refits}} jackknife refits with the held-out calls stable in {{jk_stable}}, and
{{hp_configs}} hyperparameter configs.

**"Why not deep learning / XGBoost?"**
We tried it, pre-registered. XGBoost on the 24 screened KPIs got {{xgb_kpi_only}} (p {{xgb_kpi_only_p}}) and overfits
(100% on train). With physics FEM features added it reached {{xgb_best}}, still not significant. A CNN on
pixels would learn the black level, which is exactly the trap from view 1.

**"Couldn't you just harmonise the intensities?"**
We did: six LUT methods, `harmonise="hybrid"` recommended (`docs/harmonisation.md`). After
offset and gain correction, the Si/graphite contrast of the affected Batch 3 sites equals that of
Batches 1 and 2, which shows it's an imaging setting, not material. The fingerprint never
depended on it, because it uses geometry only.

**"What did FEM add, if it didn't improve accuracy?"**
It explains *why*: swelling follows Si amount (R² {{fem_r2}}), and amount doesn't differ by batch
(p {{si_kw_p}}). So no amount-based feature can work, and arrangement is the physically meaningful
signal. {{fem_runs}} runs, {{fem_missing}} failures, {{fem_cost}} on Modal.

**"How do you know the dashboard isn't faked?"**
The model is ported to JS and recomputes the held-out calls from `outputs/fingerprint/features.csv`
in the browser, identical to the Python output to 1e-15 (`node --test demo/test`). Edit `outputs/`
and the page reloads within 2 s.

**"Confidence {{call_3e122cbj_conf_plain}} on 3e122cbj? Isn't that a failure?"**
No, it's honest. Its arrangement is typical of every batch, so the conformal p-values can't
separate them. {{call_3e122cbj_batch}} is still the stable call, and it differs from Batch 3 in amount (Si fraction
z ≈ +8, a loading only Batch 1 reaches).

**"xrv9xvzb has a Batch-3-style black level, yet you call it {{call_xrv9xvzb_batch}}?"**
Its mid-depth Si dip sits inside the Batch 2 cluster. We bet on the material over the microscope
and recorded that before scoring.

**"What would a manufacturer do with this?"**
Incoming QC: accept the shipment, flag it as a *kind* of drift (sedimentation-like vs drying
migration), or reject it as out of distribution. View 5.

**"Weaknesses?"**
The features were explored on labelled data. The conformal grid is coarse with n = 7 per batch.
The FEM model is linear small-strain, so stress saturates. All of this is stated in
`docs/fingerprint.md` and `docs/fem/negative-results.md`.

## Results from teammates' branches (not on main yet — cite the branch if asked)

**"Why not deep learning? A CNN must do better."**
Jihan's representation track (branch `devin/1791038010-v2-representation`, `outputs/v2/off/RESULTS_TABLE.md`): fine-tuned ResNet-18 / DINOv2 separate Batch 3 from Batch 1+2 at 0.84–0.87 field accuracy (AUROC 0.88–0.95) on every harmonisation route. But two scalar microscope statistics — image noise σ and sharpness — reproduce it at 0.90 with the same wrong fields, while the material KPIs are at chance (0.55) for that label. The deep model is an excellent microscope-session detector: our trap again, at scale.

**"What's the ceiling? Why only 0.68?"**
Branch `devin/1791068930-functional-morphology`, `docs/decision.md`: every model family on material features — naive-Bayes fingerprint, two-stage RF, frozen DINOv2/MicroNet embeddings, XGBoost, tile voting — lands at 0.55–0.68. The simplest "nearest batch mean" on grey levels *inside graphite pixels only* scores 0.74 (perm p 0.002); no material information can enter there, so any image model above ~0.68 should be suspected of reading the microscope. The ceiling is the data: a power analysis puts confirmation of the strongest effects at ~10–15 fields per batch (we have 7/7/17).

**"How many images would a plant need?"**
Leandro's ImageRep extension (separate `imagerep-repro` repo): a single image's error bar is honest for that spot, but spots vary ~2× more, so judging by it alone gives 5/17 false alarms on good Batch 3 fields; detecting a 0.01 Si-fraction shift needs ~30 fields per batch, not ~8.

**"Another model calls fn0mhxef off-baseline — who's right?"**
The deep off-detector calls fn0mhxef off on every route, but its KPIs sit at baseline (Si −0.3 SD) and its only deviations are imaging (sharpness −4 SD, noise −2 SD) — exactly the cue that detector reads. Our fingerprint says Batch 3 with credibility {{call_fn0mhxef_cred}} and confidence {{call_fn0mhxef_conf}}: we report it as uncertain, not as a confident accept.
