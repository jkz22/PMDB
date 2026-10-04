# LOSO vs leave-one-parent-out (LOPO), every family on this branch

Majority baseline: 17/31 = 0.548, 18/34 = 0.529. Permutation p: site labels permuted, parent groups fixed.

## 31 sites

| method | LOSO acc (p) | LOPO acc (p) | LOPO balanced | LOPO recall B1/B2/B3 |
|---|---|---|---|---|
| fingerprint16 NB | 0.677 (0.003) | 0.677 (0.007) | 0.636 | 0.71/0.43/0.76 |
| nearest mean: fingerprint16 | 0.645 (0.014) | 0.645 (0.008) | 0.616 | 0.71/0.43/0.71 |
| nearest mean: overlay fractions (material) | 0.484 (0.118) | 0.452 (0.170) | 0.359 | 0.00/0.43/0.65 |
| nearest mean: overlay depth profiles (material) | 0.484 (0.144) | 0.516 (0.086) | 0.538 | 0.57/0.57/0.47 |
| nearest mean: raw BSE grey (acquisition) | 0.613 (0.008) | 0.516 (0.086) | 0.454 | 0.29/0.43/0.65 |
| nearest mean: harmonised BSE grey (acquisition) | 0.645 (0.006) | 0.645 (0.002) | 0.588 | 0.71/0.29/0.76 |
| nearest mean: graphite-only grey (acquisition) | 0.742 (0.002) | 0.581 (0.020) | 0.633 | 0.71/0.71/0.47 |
| nearest mean: pore-only grey (acquisition) | 0.710 (0.002) | 0.677 (0.002) | 0.664 | 0.86/0.43/0.71 |
| XGBoost: reliable KPIs | 0.452 (0.455) | 0.161 (1.000) | 0.098 | 0.00/0.00/0.29 |
| XGBoost: reliable KPIs + functional | 0.516 (0.238) | 0.355 (0.891) | 0.244 | 0.00/0.14/0.59 |
| XGBoost: fingerprint16 | 0.677 (0.010) | 0.645 (0.020) | 0.588 | 0.29/0.71/0.76 |
| XGBoost: all reliable (KPIs + fingerprint16 + functional) | 0.613 (0.030) | 0.516 (0.238) | 0.454 | 0.29/0.43/0.65 |

## 34 sites

| method | LOSO acc (p) | LOPO acc (p) | LOPO balanced | LOPO recall B1/B2/B3 |
|---|---|---|---|---|
| fingerprint16 NB | 0.500 (0.076) | 0.471 (0.100) | 0.435 | 0.50/0.25/0.56 |
| nearest mean: fingerprint16 | 0.588 (0.008) | 0.559 (0.024) | 0.537 | 0.50/0.50/0.61 |
| nearest mean: overlay fractions (material) | 0.559 (0.032) | 0.471 (0.144) | 0.343 | 0.00/0.25/0.78 |
| nearest mean: overlay depth profiles (material) | 0.441 (0.198) | 0.412 (0.279) | 0.421 | 0.50/0.38/0.39 |
| nearest mean: raw BSE grey (acquisition) | 0.588 (0.028) | 0.500 (0.110) | 0.454 | 0.38/0.38/0.61 |
| nearest mean: harmonised BSE grey (acquisition) | 0.647 (0.004) | 0.618 (0.004) | 0.551 | 0.75/0.12/0.78 |
| nearest mean: graphite-only grey (acquisition) | 0.588 (0.012) | 0.500 (0.082) | 0.523 | 0.62/0.50/0.44 |
| nearest mean: pore-only grey (acquisition) | 0.676 (0.002) | 0.676 (0.002) | 0.657 | 0.88/0.38/0.72 |
| XGBoost: reliable KPIs | 0.382 (0.703) | 0.206 (1.000) | 0.153 | 0.00/0.12/0.33 |
| XGBoost: reliable KPIs + functional | 0.441 (0.426) | 0.294 (0.970) | 0.231 | 0.00/0.25/0.44 |
| XGBoost: fingerprint16 | 0.588 (0.020) | 0.559 (0.059) | 0.491 | 0.25/0.50/0.72 |
| XGBoost: all reliable (KPIs + fingerprint16 + functional) | 0.471 (0.277) | 0.471 (0.297) | 0.389 | 0.25/0.25/0.67 |

