# Submission pack (iterate.inc, deadline 15:00)

- **Repo:** https://github.com/jkz22/PMDB (public)
- **Video (≤ 2 min):** `demo/backup/pmdb-submission.mp4`, a captioned walkthrough that works muted. Upload it to YouTube (unlisted is fine) or Loom.
  Regenerate with `node demo/server.mjs &` then `python demo/submission/build.py --video`.

## Short description (paste into the form)

PMDB identifies which manufacturing batch a battery electrode came from, using SEM images of a silicon/graphite anode. The obvious approach (an intensity model, {{intensity_acc}} accuracy) turns out to learn the microscope: one grey level of brightness flips its calls. We instead built a geometry-only fingerprint of *where* the silicon sits through the electrode depth. It scores {{fp_score}} leave-one-out (permutation p = {{perm_p}}), its held-out calls carry calibrated confidence, and it rejects out-of-distribution shipments. {{fem_runs}} FEM charging simulations ({{fem_cost}} on Modal) show why: swelling follows the silicon amount, and the batches hold the same amount, so arrangement is the signal. Everything runs live in a Node/JS dashboard that matches the Python model to 1e-15.

## One-liner

A battery-electrode batch fingerprint that can't be fooled by a brightness knob and knows when it doesn't know.
