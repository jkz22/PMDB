## Plan ready

- **Plan file**: `/Users/Kevin/Documents/GitHub/PMDB/.claude/worktrees/heldout-data/.claude/plans/fem-build.md`
- **Steps**: 10 steps, complexity HIGH.
  - Steps 1-6: local pure-numpy and dolfinx code with tests.
  - Step 7: the local stages 2-4, ending at STOP POINT A.
  - Step 8: the Modal benchmark, ending at STOP POINT B.
  - Step 9: the full Si run, ending at STOP POINT C.
  - Step 10: SiOx, sweeps, collection, docs and commit.
  - The implementer reports back at every stop point before any Modal spend continues.

**Key decisions**
- **Mesh and solver.** One quad cell per 100 nm pixel: the 50 nm segmentation is coarsened 2×2 by majority vote, with ties going Si > pore > graphite > binder > artefact. Displacement is vector Q1, materials are cell-wise constants, and the solve is serial dolfinx (≥ 0.10) with SNES plus a MUMPS direct solve. Parallelism comes from running many cases at once on Modal, one case per container (cpu 4 by default; the benchmark tries cpu 4 against cpu 8). A full site is about 1.0 M cells and 2.0 M unknowns. A 100 vs 50 nm check on a 20 µm window is a gate (≤ 10% on swelling, σxx, porosity and Si J; ≤ 20% on von Mises quantiles).
- **Physics.** Neo-Hookean energy weighted by det Fλ. The 3D Fλ is diag(λx, λz, λy) with total out-of-plane stretch 1, so Ce33 = 1/λy². The out-of-plane eigenstretch is J^(1/3) for Si and the a-axis value for graphite. Collector u_z = 0, lateral u_x = 0, and the separator edge is free, confined, or under a solid-only dead-load pressure. Load stepping is adaptive: Δs from 0.05 down to 0.1/64, with a forced substep at s*, a linear predictor, and NaN frames after a failure.
- **Features and outputs.** Six fixed tiles over the central region per D9; site-level metrics use the union of those tiles. There are 81 metrics per region and frame: 17 Tier-1, 60 Tier-3 quantiles, and 4 Tier-4 depth-band measures that reuse the existing K12 helpers; Tier 5 is not built. Orientations are combined as `sym`, the mean of the two collector orientations, which is the same if the image is flipped vertically; bottom and top rows are kept too. Sweeps run collector-at-bottom only and are scored by Spearman ρ ≥ 0.8 across all 12 variants. The 6 sweep sites were selected now and pinned in the config: Batch_1/fzrt2k6r, uhdslk0o; Batch_2/3806gxp0, rxax5ozo; Batch_3/vc2whyaq, utfgcjfa.

**Open questions for dispatcher**: none blocking. Three things you should know:

1. **The swelling gate may trip on the low side (Expected surprise 4).** A back-of-envelope estimate for a typical site is about 76% graphite × 9.4% c-axis strain ≈ 7%, plus 1-3% from Si. That is around the 9% lower edge of the band I fixed for the gate, [9%, 39%]; the literature review already expects a dense 2D model to under-predict. The plan treats this as the user's gate (stop and report). It does not widen the band.

2. **Cost estimate.** The plan has 208 production cases: 68 Si, 68 SiOx and 72 sweep cases, the sweeps being single-orientation by design. My unmeasured guess is about $0.35 per case (about 1 h at 4 cores and 16 GiB), roughly $75 in total. That is above the $40-70 budget note but under the $180 hard cap. STOP POINT B projects the real figure from the benchmark against the cap. The plan assumes Modal rates of $0.0000131 per core-second and $0.00000222 per GiB-second, and the implementer checks them before launching.

3. **Unverified assumptions.** The `pmdb-fem` env did not exist yet when I checked, so the dolfinx version and its API are unconfirmed. The plan targets dolfinx ≥ 0.10 (SNES-based `NonlinearProblem`), and Expected surprise 1 says what to do if the version differs. The Modal `Image.micromamba` and `Function.with_options` calls do exist in the locally installed modal 1.6.0.

The decision list in `.claude/plans/fem-swelling.md` changed while I was planning: a pre-approval section and the $180 hard cap were added. The plan's gates (P23) and stop points follow it.
