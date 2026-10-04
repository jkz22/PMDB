# D21 verdict (tag `main`)



Rule: FEM arm adds value iff LOO correct >= 23/31 and permutation p <= 0.05 (1000 perms, seed 0, in-fold selection rerun).

| arm | correct | accuracy | balanced acc | perm p | null mean | passes |
|---|---|---|---|---|---|---|
| A0 | 21/31 | 0.677 | 0.636 | 0.0060 | 0.372 | reference |
| A1 | 20/31 | 0.645 | 0.616 | 0.0060 | 0.377 | False |
| A2 | 17/31 | 0.548 | 0.557 | 0.0809 | 0.378 | False |

**Verdict: classifier of record = A0** (no FEM arm passed the rule)

A2 selected features (count over 31 folds):

- q25_vm_binder@s0.5: 31
- q5_vm_binder@s0.5: 13
- q75_J_pore@s0.5: 9
- band_J_maxdev@s0.5: 3
- band_vm_maxdev@s0.5: 2
- band_J_maxdev@s1.0: 1
- q50_J_pore@s0.5: 1
- q75_J_binder@s0.5: 1
- q95_J_pore@s1.0: 1

Held-out predictions:

```
        batch     site arm assigned  credibility  confidence   ood
Batch_heldout 3e122cbj  A0  Batch_1     0.875000       0.000 False
Batch_heldout fn0mhxef  A0  Batch_3     0.444444       0.125 False
Batch_heldout xrv9xvzb  A0  Batch_2     1.000000       0.500 False
```
