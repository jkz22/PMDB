# Q&A cheat sheet (2 min, so keep each answer to about 15 s)

**"Only 31 sites. Isn't that too few?"**
Yes, which is why we fit no weights: 16 curve-shape features and per-batch robust statistics.
Every claim is stress-tested instead: 10,000 label permutations (p = 0.0025; the null p95 is the
majority baseline, 0.548), 1,496 jackknife refits with the held-out calls stable in 87–100%, and
45 hyperparameter configs.

**"Why not deep learning / XGBoost?"**
We tried it, pre-registered. XGBoost on the 24 screened KPIs got 10/31 (p 0.85) and overfits
(100% on train). With physics FEM features added it reached 15/31, still not significant. A CNN on
pixels would learn the black level, which is exactly the trap from view 1.

**"Couldn't you just harmonise the intensities?"**
We did: six LUT methods, `harmonise="hybrid"` recommended (`docs/harmonisation.md`). After
offset and gain correction, the Si/graphite contrast of the affected Batch 3 sites equals that of
Batches 1 and 2, which shows it's an imaging setting, not material. The fingerprint never
depended on it, because it uses geometry only.

**"What did FEM add, if it didn't improve accuracy?"**
It explains *why*: swelling follows Si amount (R² 0.97), and amount doesn't differ by batch
(p 0.38). So no amount-based feature can work, and arrangement is the physically meaningful
signal. 68 runs, 0 failures, $2.61 on Modal.

**"How do you know the dashboard isn't faked?"**
The model is ported to JS and recomputes the held-out calls from `outputs/fingerprint/features.csv`
in the browser, identical to the Python output to 1e-15 (`node --test demo/test`). Edit `outputs/`
and the page reloads within 2 s.

**"Confidence 0 on 3e122cbj? Isn't that a failure?"**
No, it's honest. Its arrangement is typical of every batch, so the conformal p-values can't
separate them. Batch 1 is still the stable call, and it differs from Batch 3 in amount (Si fraction
z ≈ +8, a loading only Batch 1 reaches).

**"xrv9xvzb has a Batch-3-style black level, yet you call it Batch 2?"**
Its mid-depth Si dip sits inside the Batch 2 cluster. We bet on the material over the microscope
and recorded that before scoring.

**"What would a manufacturer do with this?"**
Incoming QC: accept the shipment, flag it as a *kind* of drift (sedimentation-like vs drying
migration), or reject it as out of distribution. View 5.

**"Weaknesses?"**
The features were explored on labelled data. The conformal grid is coarse with n = 7 per batch.
The FEM model is linear small-strain, so stress saturates. All of this is stated in
`docs/fingerprint.md` and `docs/fem/negative-results.md`.
