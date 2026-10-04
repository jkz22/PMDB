# Sibling-anchored calls from `SE_type_D` (pore-floor dark level)

Labelled sites 34 (31 + 3 released truths), majority 0.529. delta (within pure-parent SD of SE_type_D) = 0.49 DN.

| predictor | n | acc | recall B1/B2/B3 |
|---|---|---|---|
| two-cut rule, full fit (cuts -8.82, 0.69) | 34 | 0.794 | - |
| two-cut rule, LOPO | 34 | 0.647 | 0.38/0.62/0.78 |
| sibling-anchored (sites with labelled siblings) | 32 | 0.531 | 0.17/0.50/0.67 |
| sibling-anchored, mixed parents only | 19 | 0.368 | 0.17/0.50/0.40 |
| LOPO cut rule on the same mixed-parent sites | 19 | 0.526 | - |

## Test-site calls

| site | parent | D | assigned | conf | p(B1) | p(B2) | p(B3) | cut-rule | labelled siblings (label) D |
|---|---|---|---|---|---|---|---|---|---|
| 0eryguqq | G1612 | +4.78 | Batch_2 | low | 0.00 | 0.58 | 0.42 | Batch_3 | ptg8lmto (3) +5.44; xgj4xftb (3) +6.21 |
| 4hq27w4c | G2148 | -10.25 | Batch_1 | high | 1.00 | 0.00 | 0.00 | Batch_1 | f1vzngrs (1) -9.01; epqdaau9 (2) -8.80 |
| fhwrjtet | G1612 | +5.88 | Batch_3 | high | 0.00 | 0.00 | 1.00 | Batch_3 | ptg8lmto (3) +5.44; xgj4xftb (3) +6.21 |
| fspqbkxl | G2148 | -11.25 | Batch_1 | high | 1.00 | 0.00 | 0.00 | Batch_1 | f1vzngrs (1) -9.01; epqdaau9 (2) -8.80 |
| soo2ax3r | G2156 | +4.05 | Batch_2 | high | 0.00 | 0.69 | 0.31 | Batch_3 | fzrt2k6r (1) +3.55; b3esycq1 (2) +4.35 |
| y59rxmxl | G1880 | +0.07 | Batch_1 | low | 0.60 | 0.40 | 0.00 | Batch_2 | uhdslk0o (1) -0.09 |

## Per-site labelled evaluation

| site | parent | truth | D | mixed | LOPO cut | anchored | p(B1) | p(B2) | p(B3) |
|---|---|---|---|---|---|---|---|---|---|
| ptg8lmto | G1612 | Batch_3 | +5.44 |  | Batch_3 | Batch_3 | 0.00 | 0.44 | 0.56 |
| xgj4xftb | G1612 | Batch_3 | +6.21 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| iv6g2oq0 | G1780 | Batch_1 | -10.07 |  | Batch_1 | Batch_1 | 1.00 | 0.00 | 0.00 |
| uhdslk0o | G1880 | Batch_1 | -0.09 |  | Batch_2 | Batch_2 | 0.00 | 1.00 | 0.00 |
| 0grcilhi | G1904 | Batch_3 | +4.42 |  | Batch_2 | Batch_2 | 0.00 | 0.55 | 0.45 |
| hawkfj64 | G1904 | Batch_3 | +4.59 |  | Batch_2 | Batch_2 | 0.00 | 0.57 | 0.43 |
| mgxahqnk | G1904 | Batch_3 | +5.10 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| fn0mhxef | G2048 | Batch_1 | -4.92 | y | Batch_2 | Batch_2 | 0.36 | 0.64 | 0.00 |
| avn74qx1 | G2048 | Batch_2 | -2.59 | y | Batch_2 | Batch_2 | 0.23 | 0.77 | 0.00 |
| 3806gxp0 | G2048 | Batch_2 | -0.68 | y | Batch_2 | Batch_2 | 0.00 | 0.75 | 0.25 |
| 71vgq3fw | G2060 | Batch_3 | +22.70 |  | Batch_3 | Batch_2 | 0.00 | 0.70 | 0.30 |
| tuy3zymq | G2060 | Batch_3 | +23.20 |  | Batch_3 | Batch_3 | 0.00 | 0.23 | 0.77 |
| kbdh4tri | G2060 | Batch_3 | +23.60 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| x7u69zsw | G2060 | Batch_3 | +23.76 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| rxax5ozo | G2068SE | Batch_2 | +4.59 | y | Batch_3 | Batch_3 | 0.00 | 0.34 | 0.66 |
| x77cy643 | G2068SE | Batch_3 | +4.83 | y | Batch_3 | Batch_2 | 0.00 | 0.73 | 0.27 |
| utfgcjfa | G2068SE | Batch_3 | +5.61 | y | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| vc2whyaq | G2068SE | Batch_3 | +5.92 | y | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| ffwubibz | G2080 | Batch_1 | -0.57 | y | Batch_2 | Batch_2 | 0.29 | 0.71 | 0.00 |
| r17byphk | G2080 | Batch_2 | +0.55 | y | Batch_2 | Batch_2 | 0.00 | 0.56 | 0.44 |
| cfe5vt7s | G2080 | Batch_3 | +0.83 | y | Batch_2 | Batch_2 | 0.00 | 1.00 | 0.00 |
| ufdvpb81 | G2088 | Batch_3 | +6.35 |  | Batch_3 | Batch_3 | 0.00 | 0.30 | 0.70 |
| hzumfsms | G2088 | Batch_3 | +6.48 |  | Batch_3 | Batch_3 | 0.00 | 0.32 | 0.68 |
| 9luzk4jm | G2088 | Batch_3 | +7.37 |  | Batch_3 | Batch_3 | 0.00 | 0.29 | 0.71 |
| xrv9xvzb | G2088 | Batch_3 | +7.92 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| f1vzngrs | G2148 | Batch_1 | -9.01 | y | Batch_1 | Batch_2 | 0.41 | 0.59 | 0.00 |
| epqdaau9 | G2148 | Batch_2 | -8.80 | y | Batch_1 | Batch_1 | 1.00 | 0.00 | 0.00 |
| fzrt2k6r | G2156 | Batch_1 | +3.55 | y | Batch_3 | Batch_3 | 0.43 | 0.00 | 0.57 |
| b3esycq1 | G2156 | Batch_2 | +4.35 | y | Batch_3 | Batch_3 | 0.00 | 0.43 | 0.57 |
| pl8uabbv | G2272 | Batch_3 | -2.27 | y | Batch_2 | Batch_2 | 0.00 | 1.00 | 0.00 |
| i9jiqjwl | G2272 | Batch_2 | -1.83 | y | Batch_2 | Batch_3 | 0.00 | 0.48 | 0.52 |
| 4ih2ggld | G2316 | Batch_1 | -9.89 | y | Batch_1 | Batch_1 | 1.00 | 0.00 | 0.00 |
| 5n1q8atc | G2316 | Batch_1 | -8.84 | y | Batch_2 | Batch_2 | 0.00 | 1.00 | 0.00 |
| 3e122cbj | G2316 | Batch_2 | -8.46 | y | Batch_2 | Batch_2 | 0.43 | 0.57 | 0.00 |
