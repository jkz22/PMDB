# Sibling-anchored calls from `BSE_D` (pore-floor dark level)

Labelled sites 34 (31 + 3 released truths), majority 0.529. delta (within pure-parent SD of BSE_D) = 0.73 DN.

| predictor | n | acc | recall B1/B2/B3 |
|---|---|---|---|
| two-cut rule, full fit (cuts 1.48, 8.70) | 34 | 0.765 | - |
| two-cut rule, LOPO | 34 | 0.676 | 0.50/0.38/0.89 |
| sibling-anchored (sites with labelled siblings) | 32 | 0.594 | 0.50/0.25/0.78 |
| sibling-anchored, mixed parents only | 19 | 0.368 | 0.50/0.25/0.40 |
| LOPO cut rule on the same mixed-parent sites | 19 | 0.474 | - |

## Test-site calls

| site | parent | D | assigned | conf | p(B1) | p(B2) | p(B3) | cut-rule | labelled siblings (label) D |
|---|---|---|---|---|---|---|---|---|---|
| 0eryguqq | G1612 | +16.25 | Batch_3 | high | 0.00 | 0.15 | 0.85 | Batch_3 | ptg8lmto (3) +16.50; xgj4xftb (3) +18.86 |
| 4hq27w4c | G2148 | -1.63 | Batch_1 | high | 1.00 | 0.00 | 0.00 | Batch_1 | f1vzngrs (1) -0.95; epqdaau9 (2) -0.09 |
| fhwrjtet | G1612 | +18.29 | Batch_3 | high | 0.00 | 0.00 | 1.00 | Batch_3 | ptg8lmto (3) +16.50; xgj4xftb (3) +18.86 |
| fspqbkxl | G2148 | +0.67 | Batch_1 | low | 0.38 | 0.24 | 0.38 | Batch_1 | f1vzngrs (1) -0.95; epqdaau9 (2) -0.09 |
| soo2ax3r | G2156 | +10.55 | Batch_1 | high | 0.70 | 0.00 | 0.30 | Batch_3 | fzrt2k6r (1) +10.29; b3esycq1 (2) +11.48 |
| y59rxmxl | G1880 | +5.05 | Batch_1 | low | 0.58 | 0.42 | 0.00 | Batch_2 | uhdslk0o (1) +5.37 |

## Per-site labelled evaluation

| site | parent | truth | D | mixed | LOPO cut | anchored | p(B1) | p(B2) | p(B3) |
|---|---|---|---|---|---|---|---|---|---|
| ptg8lmto | G1612 | Batch_3 | +16.50 |  | Batch_3 | Batch_3 | 0.00 | 0.32 | 0.68 |
| xgj4xftb | G1612 | Batch_3 | +18.86 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| iv6g2oq0 | G1780 | Batch_1 | +0.94 |  | Batch_1 | Batch_1 | 1.00 | 0.00 | 0.00 |
| uhdslk0o | G1880 | Batch_1 | +5.37 |  | Batch_2 | Batch_2 | 0.00 | 1.00 | 0.00 |
| 0grcilhi | G1904 | Batch_3 | +12.03 |  | Batch_3 | Batch_3 | 0.00 | 0.25 | 0.75 |
| hawkfj64 | G1904 | Batch_3 | +12.49 |  | Batch_3 | Batch_3 | 0.00 | 0.31 | 0.69 |
| mgxahqnk | G1904 | Batch_3 | +13.24 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| fn0mhxef | G2048 | Batch_1 | +0.84 | y | Batch_1 | Batch_1 | 1.00 | 0.00 | 0.00 |
| avn74qx1 | G2048 | Batch_2 | +2.02 | y | Batch_1 | Batch_1 | 0.63 | 0.37 | 0.00 |
| 3806gxp0 | G2048 | Batch_2 | +5.87 | y | Batch_1 | Batch_1 | 0.64 | 0.16 | 0.20 |
| 71vgq3fw | G2060 | Batch_3 | +31.27 |  | Batch_3 | Batch_2 | 0.00 | 0.70 | 0.30 |
| tuy3zymq | G2060 | Batch_3 | +32.05 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| x7u69zsw | G2060 | Batch_3 | +32.66 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| kbdh4tri | G2060 | Batch_3 | +32.67 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| rxax5ozo | G2068SE | Batch_2 | +11.35 | y | Batch_3 | Batch_3 | 0.00 | 0.34 | 0.66 |
| x77cy643 | G2068SE | Batch_3 | +11.38 | y | Batch_3 | Batch_2 | 0.00 | 0.78 | 0.22 |
| utfgcjfa | G2068SE | Batch_3 | +12.23 | y | Batch_3 | Batch_3 | 0.00 | 0.26 | 0.74 |
| vc2whyaq | G2068SE | Batch_3 | +12.97 | y | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| ffwubibz | G2080 | Batch_1 | +6.71 | y | Batch_2 | Batch_2 | 0.33 | 0.67 | 0.00 |
| r17byphk | G2080 | Batch_2 | +8.01 | y | Batch_2 | Batch_2 | 0.00 | 1.00 | 0.00 |
| cfe5vt7s | G2080 | Batch_3 | +9.38 | y | Batch_2 | Batch_2 | 0.00 | 0.67 | 0.33 |
| ufdvpb81 | G2088 | Batch_3 | +16.87 |  | Batch_3 | Batch_3 | 0.00 | 0.39 | 0.61 |
| hzumfsms | G2088 | Batch_3 | +17.35 |  | Batch_3 | Batch_3 | 0.00 | 0.19 | 0.81 |
| 9luzk4jm | G2088 | Batch_3 | +18.01 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| xrv9xvzb | G2088 | Batch_3 | +18.49 |  | Batch_3 | Batch_3 | 0.00 | 0.00 | 1.00 |
| f1vzngrs | G2148 | Batch_1 | -0.95 | y | Batch_1 | Batch_1 | 1.00 | 0.00 | 0.00 |
| epqdaau9 | G2148 | Batch_2 | -0.09 | y | Batch_1 | Batch_1 | 0.52 | 0.48 | 0.00 |
| fzrt2k6r | G2156 | Batch_1 | +10.29 | y | Batch_3 | Batch_3 | 0.43 | 0.00 | 0.57 |
| b3esycq1 | G2156 | Batch_2 | +11.48 | y | Batch_3 | Batch_3 | 0.00 | 0.43 | 0.57 |
| pl8uabbv | G2272 | Batch_3 | +4.20 | y | Batch_2 | Batch_2 | 0.31 | 0.69 | 0.00 |
| i9jiqjwl | G2272 | Batch_2 | +6.72 | y | Batch_2 | Batch_2 | 0.00 | 0.69 | 0.31 |
| 4ih2ggld | G2316 | Batch_1 | +0.69 | y | Batch_1 | Batch_1 | 1.00 | 0.00 | 0.00 |
| 5n1q8atc | G2316 | Batch_1 | +2.12 | y | Batch_2 | Batch_2 | 0.00 | 1.00 | 0.00 |
| 3e122cbj | G2316 | Batch_2 | +2.21 | y | Batch_2 | Batch_1 | 0.52 | 0.48 | 0.00 |
