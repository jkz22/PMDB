# D22 verdict (tag `main`)



Rule: an XGB arm BEATS the fingerprint iff LOO correct >= 22/31 (fingerprint 21/31) and permutation p <= 0.05 (1000 perms, seed 0, all in-fold steps rerun, Modal). Highest correct wins, tie -> fewer features.

| arm | correct | accuracy | balanced acc | perm p | null mean | beats fingerprint |
|---|---|---|---|---|---|---|
| X1 | 10/31 | 0.323 | 0.280 | 0.8492 | 0.401 | False |
| X2 | 15/31 | 0.484 | 0.378 | 0.2318 | 0.401 | False |
| X3 | 11/31 | 0.355 | 0.272 | 0.7113 | 0.395 | False |
| fingerprint A0 | 21/31 | 0.677 | 0.636 | - | - | reference |

**Verdict: no XGB arm beats the fingerprint (21/31); the fingerprint stays classifier of record**

Per-site agreement with the fingerprint (LOO):

```
category  both right  only XGB right  only fingerprint right  both wrong
arm                                                                     
X1                 7               3                      14           7
X2                11               4                      10           6
X3                10               1                      11           9
```

Confusion (true rows -> assigned):

X1: {"Batch_1": {"Batch_1": 2, "Batch_2": 1, "Batch_3": 4}, "Batch_2": {"Batch_1": 2, "Batch_2": 1, "Batch_3": 4}, "Batch_3": {"Batch_1": 8, "Batch_2": 2, "Batch_3": 7}}

X2: {"Batch_1": {"Batch_1": 2, "Batch_2": 2, "Batch_3": 3}, "Batch_2": {"Batch_1": 4, "Batch_2": 1, "Batch_3": 2}, "Batch_3": {"Batch_1": 3, "Batch_2": 2, "Batch_3": 12}}

X3: {"Batch_1": {"Batch_1": 1, "Batch_2": 1, "Batch_3": 5}, "Batch_2": {"Batch_1": 2, "Batch_2": 1, "Batch_3": 4}, "Batch_3": {"Batch_1": 4, "Batch_2": 4, "Batch_3": 9}}

Selected FEM features (count over 31 folds):

- X3 q25_vm_binder@s0.5: 31
- X3 q5_vm_binder@s0.5: 13
- X3 q75_J_pore@s0.5: 9
- X3 band_J_maxdev@s0.5: 3
- X3 band_vm_maxdev@s0.5: 2
- X3 band_J_maxdev@s1.0: 1
- X3 q50_J_pore@s0.5: 1
- X3 q75_J_binder@s0.5: 1
- X3 q95_J_pore@s1.0: 1

Held-out predictions:

```
        batch     site arm assigned  confidence
Batch_heldout 3e122cbj  X1  Batch_1    0.903732
Batch_heldout fn0mhxef  X1  Batch_2    0.701723
Batch_heldout xrv9xvzb  X1  Batch_2    0.714231
```
