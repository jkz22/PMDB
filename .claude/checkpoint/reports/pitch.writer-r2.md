# pitch.writer-r2 report

## Draft
- `docs/pitch-brief.md` (whole file, 12 slides + headline + numbers checklist), rewritten around the six-part story:
  - S1 problem + data
  - S2 the question (can mechanics reveal batch differences)
  - S3 simulation (68/68, $2.608, GIFs; finite-strain to D20 switch moved to the speaker note)
  - S4 swelling in the literature band
  - S5 finding: swelling follows Si content, Si content is the same across batches, so batches differ in arrangement, not amount
  - S6 Leo's fingerprint, credited by name
  - S7 pre-registered D21 test table (main + edge5)
  - S8 held-out calls, with credibility and confidence explained plainly
  - S9 how each site differs from Batch_3
  - S10 cross-check against our RF arms (xrv9xvzb disagreement)
  - S11 limitations (linear mechanics, interior windows/BCs + edge5 result, n=31, composition blind spot)
  - S12 next steps (periodic BCs, finite strain + contact, 3D) + close
- Removed or corrected superseded material:
  - The "KPI arm selected" headline is gone. The RF is now a cross-check, and its 0.437/0.370/0.350 appears only as supporting evidence (S5 note).
  - The full RF ablation table, the old approach slide, the calibration bullet and the "under $3" claim are gone.
  - The old final calls (xrv9xvzb Batch_3 0.586) were replaced by the fingerprint calls.

## Rules applied
- No project CLAUDE.md. I applied the operator's rules: plain, direct, technical English, every number copied from a cited file with its path, nothing invented. I kept the r1 conventions: `→`, en-dash ranges, the `−` minus sign, a source path after each claim, no em dashes in prose.

## Grounding (all numbers are in the Numbers checklist with paths)
- Values I computed myself from CSVs, labelled "min/max over batch rows":
  - Batch_2 max si_depth_rel_band2 0.705665; Batch_3 range 0.482759–1.394753; Batch_1 max 1.100702
  - mid-dip Batch_3 min −0.692301; Batch_2 range −1.035662 to −0.400299
  - site K01 max per batch, B1 0.164500, B2 0.083739, B3 0.089180 (`outputs/kpis/site_kpis.csv`)
- I recomputed the edge5 Spearman values (0.894, 0.639) and means from `outputs/fem/edge5/site_curves.csv` and `outputs/fem/site_curves.csv`. They match `.claude/reports/fem-edge5.implementer.md`.
- dev values come from the `dev_Batch_3` column of `outputs/fingerprint/heldout_explain.csv`. The definition is |z − centre| / scale (`pmdb/fingerprint.py` explain()).
- Unsourced (labelled as proposals or inference in the text):
  - "3D microstructures" next step: proposed, not in the repo docs.
  - "pair fingerprint with a composition check": my proposal, from S9.
  - "swelling roughly proportional to Si fraction in a linear model": labelled as our reading of R² 0.971 (synthesiser C2 marks it INFERRED).
- Figure gap: the S5 swelling-vs-K01 scatter does not exist yet. The brief says so and names the data files.

## Contradictions / things to check before presenting
1. `docs/fingerprint.md` section 4 says every Batch 2 site is mid-depth depleted "(< 0.7)". `outputs/fingerprint/features.csv` gives Batch_2 max si_depth_rel_band2 = 0.705665. The brief uses the CSV value.
2. The fingerprint has three permutation p-values:
   - 0.002 (500 perms, `evaluation.json`, used on S6)
   - 0.0060 (1000 perms, D21 A0, used in the headline and S7)
   - 0.0025 (10k perms, explorer-leo, not used)
   Null means are 0.369 vs 0.372. They are consistent, but expect a question if both p-values appear on screen.
3. Ledger: `outputs/fem/run_log.json` still says 2.6081. The edge5 spend ($0.377, ledger → 2.985) is recorded only in `.claude/reports/fem-edge5.implementer.md` and the d686139 commit message ($0.38). No committed output records it.
4. 3e122cbj: the fingerprint finds its arrangement typical of every batch (confidence 0.0, ood False). The RF z-scores show Si loading at z +8.0 vs Batch_3. This is not a contradiction: the fingerprint cancels composition by design. It is a real blind spot, though, and it is flagged in S9/S11.
5. fn0mhxef is "Batch_3", but its largest deviations from the Batch_3 centre (dev 2.12, 1.87) are larger than any of 3e122cbj's (max 0.84). S9 states this openly.

## Notes
- No brief file is named by the project. I edited only docs/pitch-brief.md.
