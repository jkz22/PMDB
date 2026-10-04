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
1. Every measured trait except Si–graphite contact points to Batch 3 over Batch 2, led by Si particle density and Si area fraction; the support is spread across the whole image.
2. Evidence map: red Si/pores support the call, blue argue against.
3. 54% of the judgement maps onto measured microstructure.

**`fhwrjtet` → Batch 3 (0.994, HIGH)**
1. Same picture as 0eryguqq: Si area fraction, Si particle density and the depth pattern all point to Batch 3; only Si–graphite contact dissents.
2. Evidence map: red Si/pores support the call, blue argue against.
3. 58% maps onto measured microstructure.

**`fspqbkxl` → Batch 2 (0.72, MEDIUM)**
1. Porosity, Si–graphite contact and Si area fraction point to Batch 2; the depth pattern, Si particle size and density point back toward Batch 1. Over half of the call rests on texture we do not measure.
2. Evidence map: red Si/pores support the call, blue argue against.
3. Only 44% maps onto measured microstructure.

**`y59rxmxl` → Batch 1 (0.62, LOW)**
1. Mixed evidence: the depth pattern and Si particle size point to Batch 1, but Si area fraction, particle density, contact and porosity look more like Batch 2.
2. Evidence map: red Si/pores support the call, blue argue against.
3. 54% maps onto measured microstructure.

**`soo2ax3r` → Batch 1 (0.59, LOW)**
1. The depth pattern, Si area fraction, particle size and porosity point to Batch 1; Si particle density and Si–graphite contact look more like Batch 2, mostly in the left third of the image.
2. Evidence map: red Si/pores support the call, blue argue against.
3. 60% maps onto measured microstructure.

**`4hq27w4c` → Batch 2 (0.57, LOW)**
1. Near tie: porosity, Si–graphite contact and Si area fraction point to Batch 2; Si particle density, size and the depth pattern point to Batch 1.
2. Evidence map: red Si/pores support the call, blue argue against.
3. 53% maps onto measured microstructure.

## Method note
Predictions use the probe trained with each test image's source image excluded, the validated setting. Sources: `outputs/probe_explain/test_final_predictions.csv`, `test_contributions.csv`.

## Evidence maps
Each test image has a map in `demo/public/evidence/<site>.jpg` (dashboard tab "Test calls", click a card to enlarge). Si particles and pores are tinted red where their patch agrees with the call and blue where it argues against; graphite stays grey. Both Batch 3 calls are red throughout. `4hq27w4c` and `soo2ax3r` show visible blue clusters (top-right/centre pores; left third), matching their LOW confidence.
