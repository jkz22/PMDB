# ⚠️ These outputs are stale. Do not submit them as they are.

`evaluation.json` and `predictions.csv` were produced before the Devin review fixes on PR #79:

- Training weights are now site-balanced: each site sums to 1/(sites in its batch), so every batch has equal
  total weight regardless of patch counts (`class_weight="balanced"` is dropped in `pmdb/patch_probe.py`).
- `--n-perm` is now passed through to `probe_lopo` (the old run used 200 permutations whatever was requested).

Observed accuracy, per-site probabilities, held-out calls and the permutation p-value may all change.

Regenerate (this spends Modal budget) and then delete this file:

    modal run modal_patch_mil.py --mode probe --n-perm 1000
