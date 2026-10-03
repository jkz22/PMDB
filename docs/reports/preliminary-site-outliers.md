# Preliminary site outlier check (Batches 1–3)

2026-10-03 · preliminary, not the final pipeline

**Verdict: two Batch_1 sites, `5n1q8atc` and `4ih2ggld`, are outliers (distance 3.06 and 2.97, about 1.6× the next site; confidence p = 0.032 and 0.065). With them set aside, no batch differs from the others.**

**Next action:** ask whoever collected the data whether `5n1q8atc` and `4ih2ggld` come from the same electrode piece.

## 1. Batch-level result

Permutation test (energy distance, 2000 label shuffles, verdict KPIs); the p-value is how often random relabelling gives a gap this large.

1. Batch_1 (remaining 5) vs rest p = 0.74; Batch_2 p = 0.41; Batch_3 p = 0.28; pairwise p = 0.56–0.88.
2. With all 7 Batch_1 sites, Batch_1 vs rest was p = 0.046, entirely from the two sites.
3. Dropping the 4 SE-detector sites changes nothing (Batch_1 p = 0.64).
4. The two sites vs rest give p = 0.003, but that group was picked after looking, so the number is circular.

## 2. Per-site ranking

Each site vs the other 30. Distance = root-mean-square robust z over the verdict KPIs (robust z: how many typical spreads a value sits from the median of the other 30). Confidence = conformal p-value, the share of normal-site scores at least as high (minimum 1/31 = 0.032).

| rank | site | batch | distance (RMS robust z) | conformal p |
|---|---|---|---|---|
| 1 | `5n1q8atc` | Batch_1 | 3.06 | 0.032 |
| 2 | `4ih2ggld` | Batch_1 | 2.97 | 0.065 |
| 3 | `avn74qx1` | Batch_2 | 1.82 | 0.129 |
| 4 | `r17byphk` | Batch_2 | 1.79 | 0.129 |
| 5 | `tuy3zymq` | Batch_3 | 1.66 | 0.194 |
| 6 | `kbdh4tri` | Batch_3 | 1.66 | 0.194 |

Below the two outliers, scores form a continuum.

## 3. Why the p-values are not smaller

With 31 sites, 0.032 is the floor. The two outliers also mask each other (each sits in the other's reference set), hence 0.065 for `4ih2ggld`. A block-PCA + Hotelling F-test masked them further (p ≈ 0.10) and is not used. Corrected for 31 tests, neither site is formally significant; the evidence is the gap plus the batch result. A selection-adjusted simulation test (~1.5 h) would fix this; deferred.

## 4. What is different

Robust z vs all sites, capped at ±5; pairs are `5n1q8atc` / `4ih2ggld`:

1. More Si: area fraction (K01) +5.0 in both.
2. More, larger Si particles: number density (K02) and largest particle (K03_max) +5.0 in both; agglomerate fraction (K04) +4.0 / +3.1.
3. Si packed closer: MST edge length (K09) −5.0 in both; empty-space P95 (K14) −3.1 / −3.0.
4. Less Si boundary touching graphite: Si–graphite contact (K15) −4.5 / −5.0.

## 5. Not an imaging artefact

1. Visible in raw BSE: many more large angular bright particles (`outputs/overlays/Batch_1__5n1q8atc.png` vs `Batch_1__fzrt2k6r.png`).
2. BSE brightness normal (p99 119 / 116); Batch_3 sites with similar p99 are not outliers.
3. Process/artefact diagnostic KPIs: no batch differs.
4. Unknown: whether both sites come from one electrode piece (one event or two). Segmentation is v0, no EDS ground truth.

## 6. Plan to Sunday 14:00

1. Sat evening (~3 h), top priority (the unseen batch is judged on accuracy): one command scoring a new batch folder against the 29 non-outlier sites (provisional baseline) → accept / investigate / reject, with driving KPIs.
2. Sat (~1 h): autoencoder team runs the same ranking on latent features; do they flag the same two sites?
3. Sun 9–12: demo page per batch (verdict, distance, confidence, top KPIs, overlay).
4. Sun 12–14: dry run on the unseen batch, plus buffer.
5. Deferred: selection-adjusted test; electrode-piece check.

Reproduce: `python scripts/prelim_site_outliers.py` → `outputs/outliers/prelim_site_ranking.csv` (per-site table). Batch-level permutation numbers came from throwaway scripts.

**Next action:** ask whoever collected the data whether `5n1q8atc` and `4ih2ggld` come from the same electrode piece.
