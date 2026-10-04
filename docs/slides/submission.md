# Submission pack (iterate.inc, deadline 15:00)

- **Repo:** https://github.com/jkz22/PMDB (public)
- **Video (≤ 2 min):** `demo/backup/pmdb-submission.mp4`, a captioned walkthrough that works muted. Upload it to YouTube (unlisted is fine) or Loom.
  Regenerate with `node demo/server.mjs & python demo/record_submission.py`.

## Short description (paste into the form)

PMDB identifies which manufacturing batch a battery electrode came from, using SEM images of a silicon/graphite anode. The obvious approach (an intensity model, 0.71 accuracy) turns out to learn the microscope: one grey level of brightness flips its calls. We instead built a geometry-only fingerprint of *where* the silicon sits through the electrode depth. It scores 21/31 leave-one-out (permutation p = 0.0025), its held-out calls carry calibrated confidence, and it rejects out-of-distribution shipments. 68 FEM charging simulations ($2.61 on Modal) show why: swelling follows the silicon amount, and the batches hold the same amount, so arrangement is the signal. Everything runs live in a Node/JS dashboard that matches the Python model to 1e-15.

## One-liner

A battery-electrode batch fingerprint that can't be fooled by a brightness knob and knows when it doesn't know.
