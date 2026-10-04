## Plan ready (round 2 revision)
- **Plan file**: `/Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/plans/fem-build.md`
- **Steps**: 11 steps (1-6, new 6b, 7-10). Complexity is HIGH: the physics is unchanged, D17 shrank the scope and D19 adds a little docs work.
- **Key decisions**:
  - **D17 cut, applied in full.** There is one parameter set only. The following are deleted everywhere: the `config` argument, named configs, the `configs:`/`sweeps:`/`siox:`/`bc:` YAML blocks and the `config` column. Keeping them would not have been free, because they ran through `config.py`, `run_case`, the remote function, the entrypoint, the volume paths, the ledger and both CSVs. P18 and ES13 stay as numbered stubs so cross-references still resolve. Production is 68 cases in 2 chunks of 34. STOP B gives expected and worst-case projections over the 68 cases, plus the worst case of one 34-case chunk.
  - **STOP C is now "checkpoint C"** (Step 9) and is not a user stop. Its report content is recorded for the final hand-back. The run still halts on:
    - a G4 failure;
    - a G5 failure;
    - a cap refusal (exit 3);
    - any case that still errors or is lost after its single rerun.
    The per-chunk P22 cap re-check is kept. STOP A and STOP B are unchanged.
  - **Minor 2.** P20 adds build step (0) before `pip install uv`. It asserts the image's own `numpy.__version__ == "2.2.6"` and fails the image build otherwise; ES2 covers that failure.
  - **Minor 3.** The probe now checks image-vs-local label identity on `PROBE_SITES`: vc2whyaq plus 71vgq3fw, kbdh4tri, tuy3zymq and x7u69zsw. To fit the extra sites, probe memory is 4096 MB and the timeout 1800 s. ES3 and STOP A now report per-site results.
  - **D19 (your mid-round message), kept lean.**
    - New P28: docs use `<!-- AUTO:name -->` blocks filled by one script, `scripts/fem_docs.py`. Outside the blocks, `docs/fem/method.md` and `docs/fem/results.md` contain no numbers. A block whose inputs do not exist yet shows a `_pending_` line. Parameter citations (Source and Evidence cells) are copied at runtime from the `## 1. Final parameter table` in `docs/fem/literature-review.md`.
    - New Step 6b (local): `fem_docs.py --method`, `docs/fem/method.md` and `tests/test_fem_docs.py`. method.md covers:
      - the mermaid pipeline;
      - physics assumptions;
      - the parameter table with citations;
      - BCs;
      - a generated SOC table;
      - gates;
      - runtime/cost, which stays pending until the run;
      - limitations.
    - Step 10 adds `--results`, which writes `docs/fem/results.md`:
      - validation against the stop window and the literature band;
      - G1/G2 results;
      - a convergence summary;
      - the actual cost;
      - figures and the 34 GIF paths;
      - a Kruskal-Wallis table of the top-8 per-batch differences (descriptive).
    - Step 10 also writes 3 PNGs to `outputs/fem/figures/`: `swelling_vs_soc`, `swelling_by_batch` and `example_frames`.
    - `run_log.json` gains `cost_model` and `production`, so the cost blocks can be generated. The collector commits `outputs/fem/g2.csv`.
    - `docs/fem/README.md` becomes operational only. `docs/classifier/*` and `docs/pitch-brief.md` are out of scope.
  - **Final D10 wording applied:** class-weighted mean of tile probabilities, in a separate classifier plan. The preamble names D18 as the consumer of §2.1.
- **Revision log**: Round 2 appended, with point-by-point responses to review comments 1-3, your items 1-3, the final D10 and D19.
- **Open questions for dispatcher**: none blocking.

### Notes for the parallel classifier plan
1. **Two schema facts beyond "remove config variants":**
   - There is no `config` column.
   - Tile rows carry an integer `tile` column (0-5) instead of a mixed `region` column. Site rows have no region or tile column.
2. **Some D18 example scalars have no direct column.**
   - "Interface stress" has no direct column. The nearest columns are `p_si_mean_MPa` and the `q*_vm_*` quantiles.
   - Curve slopes are not stored. They must be derived from the per-frame rows; for example, the 0-25% SOC slope uses frames 0 and 2 plus the s* substep, which is not a stored frame.
   - The classifier plan has to pick its 10-20 scalars from the 81 columns below.

### Final output contract, verbatim from plan §2.1

### 2.1 Output contract: `site_curves.csv` and `tile_curves.csv` (P27)

Both files: one header row, comma-separated, floats written with `float_format="%.6g"`, booleans as `True`/`False`, NaN as an empty field (pandas default). They contain the 31 labelled sites and the 3 held-out sites (34 sites), production tag `full` only. Rows are sorted by `batch`, `site`, `orientation` in the order (bottom, top, sym), then `tile` (tile file only), then `frame`. Consumers must select metric columns by name, not by position.

Key and metadata columns, in this order:

| column | type | values / meaning |
|---|---|---|
| `batch` | str | `Batch_1`, `Batch_2`, `Batch_3`, `Batch_heldout` |
| `site` | str | site id, e.g. `vc2whyaq` |
| `heldout` | bool | `True` iff `batch == "Batch_heldout"` |
| `orientation` | str | `bottom` (collector at image last row), `top` (collector at image row 0), `sym` (P17 mean of the two; primary feature set) |
| `tile` | int | **tile_curves only**: 0..5, left to right in image columns |
| `tile_x0_um`, `tile_x1_um` | float | **tile_curves only**: tile column bounds in µm from image column 0 on the 100 nm grid (P15) |
| `frame` | int | 0..10 |
| `s` | float | SOC = frame/10 (0.0, 0.1, …, 1.0) |
| `converged` | bool | frame solved (sym: both orientations solved) |
| `failed_at_s` | float | SOC at which the solve failed, NaN if it reached s = 1 (sym: min); same on every row of a (site, orientation) |
| `first_pore_closure_s` | float | first frame s with any PORE cell J < 0.1 over the whole domain, NaN if never (sym: min); same on every row of a (site, orientation) |

Then the 81 metric columns (§Step 5 definitions), in this order:
1. Tier 1 (17): `swelling, surface_rough, sxx_mean_MPa, syy_mean_MPa, porosity, porosity_change, porosity_rel_change, J_si_mean, J_gr_mean, J_binder_mean, vm_si_p50_MPa, vm_si_p95_MPa, vm_gr_p95_MPa, vm_binder_p95_MPa, p_si_mean_MPa, si_yield_frac, pore_closed_frac`.
2. Tier 3 (60): for field in (`vm`, `p`, `J`), for phase in (`si`, `gr`, `binder`), for q in (5, 25, 50, 75, 95, 99): `q{q}_{field}_{phase}` (54 columns, e.g. `q5_vm_si`), then `q5_J_pore, q25_J_pore, q50_J_pore, q75_J_pore, q95_J_pore, q99_J_pore`. `vm` and `p` in MPa; `J` dimensionless.
3. Tier 4 (4): `band_vm_maxdev, band_vm_absslope, band_J_maxdev, band_J_absslope`.

Metric NaN rules: unconverged frame → all 81 NaN; empty phase in the region → that phase's metrics NaN; `porosity_rel_change` NaN if porosity at frame 0 is 0; `sym` NaN if either orientation is NaN.

Unique keys and row counts:
- `site_curves.csv`: key (`batch`, `site`, `orientation`, `frame`); region = union of the 6 tiles (P15); 34 × 3 × 11 = **1122 rows**; 9 key/meta columns (`batch, site, heldout, orientation, frame, s, converged, failed_at_s, first_pore_closure_s`; no `tile`, `tile_x0_um`, `tile_x1_um`) + 81 metrics = **90 columns**.
- `tile_curves.csv`: key (`batch`, `site`, `orientation`, `tile`, `frame`); 34 × 3 × 6 × 11 = **6732 rows**; 12 key/meta columns + 81 metrics = **93 columns**.

No `config`, `region` or `tag` column exists in either file.
