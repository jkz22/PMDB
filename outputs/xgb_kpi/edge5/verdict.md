# D22 verdict (tag `edge5`)

**Sensitivity run (tag `edge5`, table `outputs/fem/edge5/site_curves.csv`): never the final arm.**

Rule: an XGB arm BEATS the fingerprint iff LOO correct >= 22/31 (fingerprint 21/31) and permutation p <= 0.05 (1000 perms, seed 0, all in-fold steps rerun, Modal). Highest correct wins, tie -> fewer features.

| arm | correct | accuracy | balanced acc | perm p | null mean | beats fingerprint |
|---|---|---|---|---|---|---|
| X2e | 16/31 | 0.516 | 0.370 | 0.1389 | 0.403 | False |
| X3e | 11/31 | 0.355 | 0.272 | 0.7223 | 0.402 | False |
| fingerprint A0 | 21/31 | 0.677 | 0.636 | - | - | reference |

**Verdict: no XGB arm beats the fingerprint (21/31); the fingerprint stays classifier of record**

Per-site agreement with the fingerprint (LOO):

```
category  both right  only XGB right  only fingerprint right  both wrong
arm                                                                     
X2e               13               3                       8           7
X3e                9               2                      12           8
```

Confusion (true rows -> assigned):

X2e: {"Batch_1": {"Batch_1": 2, "Batch_2": 4, "Batch_3": 1}, "Batch_2": {"Batch_1": 6, "Batch_2": 0, "Batch_3": 1}, "Batch_3": {"Batch_1": 0, "Batch_2": 3, "Batch_3": 14}}

X3e: {"Batch_1": {"Batch_1": 1, "Batch_2": 1, "Batch_3": 5}, "Batch_2": {"Batch_1": 2, "Batch_2": 1, "Batch_3": 4}, "Batch_3": {"Batch_1": 3, "Batch_2": 5, "Batch_3": 9}}

Selected FEM features (count over 31 folds):

- X3e q25_vm_binder@s0.5: 30
- X3e q95_p_gr@s0.5: 8
- X3e q75_J_pore@s0.5: 6
- X3e q5_J_gr@s0.5: 4
- X3e band_J_maxdev@s1.0: 2
- X3e q5_vm_binder@s1.0: 2
- X3e vm_gr_p95_MPa@s1.0: 2
- X3e q25_J_gr@s1.0: 1
- X3e q25_vm_binder@s1.0: 1
- X3e q50_J_pore@s0.5: 1
- X3e q5_vm_gr@s0.5: 1
- X3e q75_J_binder@s0.5: 1
- X3e q75_p_gr@s1.0: 1
- X3e q99_p_gr@s1.0: 1
- X3e swelling@s0.5: 1

Held-out predictions:

```
        batch     site arm assigned  confidence
Batch_heldout 3e122cbj X2e  Batch_1    0.960039
Batch_heldout fn0mhxef X2e  Batch_2    0.778133
Batch_heldout xrv9xvzb X2e  Batch_2    0.826641
Batch_heldout 3e122cbj X3e  Batch_1    0.890449
Batch_heldout fn0mhxef X3e  Batch_1    0.477027
Batch_heldout xrv9xvzb X3e  Batch_2    0.767363
```
