# ⚠️ These outputs are stale. Do not submit them as they are.

`final_heldout_predictions.csv`, `lopo_predictions.csv` and `lopo_evaluation.json` were produced
before PR #66 changed the patch + fingerprint ensemble in `pmdb/patch_lopo.py`:

- The fingerprint vote now uses the same likelihood scores as `fp.predict`.
- A unanimous patch/fingerprint call can no longer be overturned.
- Explanations only claim the evidence agrees when it actually does.

Known wrong row: **fn0mhxef** is listed as `Batch_2 (high)`, but both component calls
(`patch_call`, `fingerprint_call`) are `Batch_3`, so the current code assigns `Batch_3`.
Other rows, the ensemble probabilities, the confidence strata and the CV metrics may also change.

Regenerate (this spends Modal budget) and then delete this file:

    modal run modal_patch_mil.py --mode lopo   # or --mode all if distances.npz is not on the volume
