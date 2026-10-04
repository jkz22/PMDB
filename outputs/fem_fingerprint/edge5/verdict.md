# D21 verdict (tag `edge5`)

**Sensitivity run (A3, tag `edge5`, table `outputs/fem/edge5/site_curves.csv`): never the final arm.**

Rule: FEM arm adds value iff LOO correct >= 23/31 and permutation p <= 0.05 (1000 perms, seed 0, in-fold selection rerun).

| arm | correct | accuracy | balanced acc | perm p | null mean | passes |
|---|---|---|---|---|---|---|
| A0 | 21/31 | 0.677 | 0.636 | 0.0060 | 0.372 | reference |
| A1 | 21/31 | 0.677 | 0.636 | 0.0060 | 0.377 | False |
| A2 | 18/31 | 0.581 | 0.521 | 0.0370 | 0.379 | False |

**Verdict: classifier of record = A0** (no FEM arm passed the rule)

A2 selected features (count over 31 folds):

- q25_vm_binder@s0.5: 30
- q95_p_gr@s0.5: 8
- q75_J_pore@s0.5: 6
- q5_J_gr@s0.5: 4
- band_J_maxdev@s1.0: 2
- q5_vm_binder@s1.0: 2
- vm_gr_p95_MPa@s1.0: 2
- q25_J_gr@s1.0: 1
- q25_vm_binder@s1.0: 1
- q50_J_pore@s0.5: 1
- q5_vm_gr@s0.5: 1
- q75_J_binder@s0.5: 1
- q75_p_gr@s1.0: 1
- q99_p_gr@s1.0: 1
- swelling@s0.5: 1

Held-out predictions:

```
        batch     site arm assigned  credibility  confidence   ood
Batch_heldout 3e122cbj  A0  Batch_1     0.875000       0.000 False
Batch_heldout fn0mhxef  A0  Batch_3     0.444444       0.125 False
Batch_heldout xrv9xvzb  A0  Batch_2     1.000000       0.500 False
```
