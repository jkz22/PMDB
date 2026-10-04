# D21 verdict (tag `free_lateral`)

**Sensitivity run (A3, tag `free_lateral`, table `outputs/fem/free_lateral/site_curves.csv`): never the final arm.**

Rule: FEM arm adds value iff LOO correct >= 23/31 and permutation p <= 0.05 (1000 perms, seed 0, in-fold selection rerun).

| arm | correct | accuracy | balanced acc | perm p | null mean | passes |
|---|---|---|---|---|---|---|
| A0 | 21/31 | 0.677 | 0.636 | 0.0060 | 0.372 | reference |
| A1 | 24/31 | 0.774 | 0.695 | 0.0010 | 0.374 | True |
| A2 | 21/31 | 0.677 | 0.664 | 0.0050 | 0.382 | False |

**Verdict: classifier of record = A0** (no FEM arm passed the rule)

A2 selected features (count over 31 folds):

- q25_vm_binder@s0.5: 29
- q50_J_binder@s0.5: 14
- q50_p_binder@s0.5: 14
- band_J_maxdev@s1.0: 1
- q25_vm_binder@s1.0: 1
- q50_J_binder@s1.0: 1
- q50_J_pore@s0.5: 1
- q95_p_gr@s0.5: 1

Held-out predictions:

```
        batch     site arm assigned  credibility  confidence   ood
Batch_heldout 3e122cbj  A0  Batch_1     0.875000       0.000 False
Batch_heldout fn0mhxef  A0  Batch_3     0.444444       0.125 False
Batch_heldout xrv9xvzb  A0  Batch_2     1.000000       0.500 False
```
