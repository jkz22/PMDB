## Hackathon demo (start here)

**Which batch did this battery electrode come from, and is a new shipment out of spec?** The obvious intensity
model ({{intensity_acc}} accuracy) has learned the microscope: one grey level of brightness flips its calls. PMDB answers from
*where the silicon sits* through the electrode depth instead:
- {{fp_score}} leave-one-out with a geometry-only fingerprint, permutation p = {{perm_p}} over {{perm_n}} shuffles;
- held-out calls with calibrated confidence;
- out-of-distribution shipments are rejected;
- {{fem_runs}} FEM charging simulations show swelling follows the Si *amount*, which does not differ by batch.

- Live dashboard: `node demo/server.mjs` → http://localhost:8080 (Node ≥ 18, no `npm install`; models ported to JS
  with parity tests, `node --test demo/test`)
- Video: `demo/backup/pmdb-submission.mp4` (captioned, < 2 min)
- Pitch: [`docs/slides/pitch.html`](docs/slides/pitch.html) · [`docs/slides/run-of-show.md`](docs/slides/run-of-show.md) · [`docs/slides/qa.md`](docs/slides/qa.md)
