# Final predictions: 6 test images

**Model:** supervised linear probe on frozen MicroNet patch embeddings. Leave-one-parent-out accuracy 0.71 (balanced 0.63, permutation p = 0.005). Single model, no ensemble.
**Confidence:** the probe's probability for the called batch, labelled HIGH (≥ 0.8), MEDIUM (0.67–0.8) or LOW (< 0.67).

| Image | Batch | Confidence | Label |
|---|---|---|---|
| `0eryguqq` | **Batch 3** | 0.999 | HIGH |
| `fhwrjtet` | **Batch 3** | 0.994 | HIGH |
| `fspqbkxl` | **Batch 2** | 0.72 | MEDIUM |
| `y59rxmxl` | **Batch 1** | 0.62 | LOW |
| `soo2ax3r` | **Batch 1** | 0.59 | LOW |
| `4hq27w4c` | **Batch 2** | 0.57 | LOW |

## Explanations (vs Batch 3, the supplier baseline)

**`0eryguqq` → Batch 3 (0.999, HIGH)**
1. A denser Si particle population, with more Si area and porosity than a Batch 2 crop.
2. Consistent through the full coating depth.
3. 54% of the judgement maps onto measured microstructure.

**`fhwrjtet` → Batch 3 (0.994, HIGH)**
1. Same picture: Batch-3-like Si area fraction, particle density and porosity.
2. Holds through the depth.
3. 58% maps onto measured microstructure.

**`fspqbkxl` → Batch 2 (0.72, MEDIUM)**
1. Differs from Batch 3 mainly in fine texture.
2. Measurable differences (Si fraction, Si–graphite contact) are small.
3. Only 44% maps onto measured microstructure.

**`y59rxmxl` → Batch 1 (0.62, LOW)**
1. Lower Si area fraction and fewer Si particles than Batch 2.
2. Less Si–graphite contact.
3. 54% maps onto measured microstructure.

**`soo2ax3r` → Batch 1 (0.59, LOW)**
1. Fewer but larger Si particles.
2. Most visible deeper in the coating.
3. 60% maps onto measured microstructure.

**`4hq27w4c` → Batch 2 (0.57, LOW)**
1. Near tie with Batch 1 (0.43).
2. Slightly more porosity and Si area, fewer Si particles.
3. 53% maps onto measured microstructure.

## Method note
Predictions use the probe trained with each test image's source image excluded, the validated setting. Sources: `outputs/probe_explain/test_final_predictions.csv`, `test_contributions.csv`.

## Evidence maps
Each test image has a map in `demo/public/evidence/<site>.png` (dashboard tab "Test calls", click a card to enlarge). Si particles and pores are tinted red where their patch agrees with the call and blue where it argues against; graphite stays grey. Both Batch 3 calls are red throughout. `4hq27w4c` and `soo2ax3r` show visible blue clusters (top-right/centre pores; left third), matching their LOW confidence.
