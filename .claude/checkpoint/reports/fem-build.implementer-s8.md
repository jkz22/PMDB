# fem-build Steps 8-10 (implementer-s8) -- running report

## Step 8: code
- modal_fem.py: added `_manifest_cases`, `_reference`, `_expected`, full P22 `budget_check` (rules a-d, `launched` list in budget.json), `_mode_bench`, `_mode_full`, entrypoint modes `bench`/`full`.
- scripts/fem_collect.py: `--results` full collector (site_curves, tile_curves, validation, run_log, GIF copy), `--g2` kept.
- scripts/fem_docs.py: `results_blocks` + figures + `--results` (Step 10 code written ahead of the run).

## Step 8: benchmark (Batch_3/vc2whyaq, full site, 100 nm, 903,716 cells = 517 x 1748, linear mechanics)
Launch: `modal run --detach modal_fem.py --mode bench` (app ap-Oqi15OQXmfBLgvvt0qqOme), SYNC DONE bench printed, 3 results no error.
| case | wall_s | substeps(=solves) | newton | peak_rss_mb | cost_usd | failed_at_s | swelling(s=1) |
|---|---|---|---|---|---|---|---|
| cpu4 bottom | 118.6 | 11 | 11 | 6283 | 0.0210 | NaN | 0.1265 |
| cpu4 top | 125.4 | 11 | 11 | 6289 | 0.0216 | NaN | 0.1276 |
| cpu8 bottom | 157.4 | 11 | 11 | 6324 | 0.0390 | NaN | 0.1265 |
- swelling_sym(s=1) = 0.1271 gate_ok True, lit_band_ok True.
- VT2 sxx_mean/(-10) = 260.8 / 260.1 (bottom/top), VT4 porosity_change -0.0046 / -0.0054, rel_change -0.106 / -0.124, VT5 J_si_mean 1.676 / 1.687, vm_si_p95 ~24.7-24.9 GPa, si_yield_frac 1.0, first_pore_closure_s 0.1, no gif_error.
- Bench GIF 1,898,436 bytes (<= 2 MB). fields npz from the volume loads: J (11, 517, 1748), u_nodes (11, 518, 1749, 2) (the plan's "H/2" = half-res grid H already coarsened).
- vm range [1, 1e4] MPa: Si vm p50 18-20 GPa exceeds the 1e4 top of the colour range (saturated colour, as in s7; ES9 note: stress colour saturates in Si, known consequence of D20).
- Extrapolation check: window 27 s (206,800 cells) vs full 119 s (903,716): ratio 4.4 for 4.37x cells (exponent ~1.0, below the plan's 1.5; the projection is conservative).
- Chosen production settings (plan rules): cpu 4 (cost 0.021 < 0.039), memory = max(4096, ceil(1.5*6289/1024)*1024) = 10240 MB, timeout = min(28800, max(7200, 3*125 s rounded to hour)) = 7200 s.
- Expected projection: c_ref ~0.021, 68 cases x 1.25 x c_ref x (n/n_ref)^1.5 ~ $2 + Stage 2 worst (2 x (7200+120)/3600 x rate(4, 16 GiB) ~ $1.3 at 4 cpu) + ledger 1.26 + allowance 5 = ~$10, far below $180: no drop.
- Worst case: 68 x (2.5 x 7200 + 240)/3600 x rate(4 cpu, 10 GiB = $0.2688/h) = ~$92.7 + ledger 1.26 = ~$94 (reported, not gated); one 34-case chunk worst ~$46.
- Ledger after bench: $1.261.

## Step 9 (production)
- Launched `modal run --detach modal_fem.py --mode full --orientation bottom|top --cpu 4 --memory 10240 --timeout 7200` for both orientations in parallel (budget.json: no drops). Both printed SYNC DONE full; 68/68 ok, 0 error/lost, no reruns needed (failed_at_s NaN everywhere). Mean wall 138 s, max 191 s, tag cost $1.316.
- Stage 2 (G2) run: window 20 um, 100 vs 50 nm: PASS (max rel diff: porosity_change 0.065 (limit 0.10), others <= 0.016, vm <= 0.014 (limit 0.20)). outputs/fem/g2.csv.
- Collector: site_curves 1122 x 90, tile_curves 6732 x 93, validation 34, GIFs 34 (max 1,995,302 bytes), missing none, orientations_run [bottom, top].
- G4: median swelling_sym(s=1) 0.1268, gate_ok True, lit_band_ok True; no site outside [0.03, 0.39]; G5 = 0. By batch medians: B1 0.132, B2 0.125, B3 0.125, held-out 0.127 (individual held-out 0.168, 0.127, 0.118).
- VT2 medians (sxx/-10): ~280-300 (MPa scale); VT4 porosity_change ~ -0.005, rel ~ -0.12 to -0.17.
- Ledger total $2.61 (cap $180).

## Step 10
- scripts/fem_docs.py: `results_blocks`, figure generation, `--results`; docs/fem/results.md created (6 AUTO blocks + Limitations covering D20 evidence); method.md cost block filled; 0 `_pending:` in both. Figures: outputs/fem/figures/{swelling_vs_soc,swelling_by_batch,example_frames}.png.
- tests/test_fem_docs.py: +1 test (4 passed). pytest -m "not data": 150 passed, 1 skipped; -m data: 17 passed, 1 skipped.
- NOT done (not in the dispatch): docs/fem/README.md, top-level README pointer, `modal run --mode unit` re-run on final tree (modal_fem.py changed only in orchestrator/budget parts; last unit run was 49 passed at s7).
