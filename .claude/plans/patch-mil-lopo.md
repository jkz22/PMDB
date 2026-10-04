# Plan: leave-one-parent-out (LOPO) evaluation, patch + fingerprint ensemble, final held-out calls

## 1. Context

The 31 labelled sites are crops of 13 parent electrode images regrouped into artificial batches, so leave-one-site-out (LOSO) leaks through sibling crops: patch kNN can match a test patch to its own parent's other crops. Current LOSO numbers (patch top-10 acc 0.710, fingerprint 0.677) are therefore optimistic. This task adds a parent table, generalises the patch classifier's CV to grouped (leave-one-parent-out) exclusion, evaluates the fingerprint model the same way, combines both into a probability-averaging ensemble with a rubric-optimal confidence flag, and writes final held-out calls with plain-language explanations. Parent membership is used ONLY to exclude siblings (evaluation folds and held-out banks); it is never evidence for a batch call or confidence (user decision).

## 2. Design decisions

1. **Parent key** = (full-res `height` from `outputs/clean{,_heldout}/summary.csv`, `se_detector` from `cache{,_heldout}/half/manifest.csv`, `BSE_grey_step` from the summaries); `parent_id = f"h{height}_{se_detector}_s{bse_grey_step}"`. Verified on the committed CSVs: 13 labelled parents; held-out 3e122cbj -> h2316_ETD_s2, fn0mhxef -> h2048_ETD_s2, xrv9xvzb -> h2088_ETD_s3. Use summary `height` (full-res), NOT manifest `height` (half-res).
2. **Grouped CV is a generalisation, not a fork**: every "exclude own site" becomes "exclude own group" when `site_groups` is passed; with `site_groups=None` the existing code path runs unchanged, so the saved LOSO results stay reproducible bit-for-bit. Regression test: `site_groups=np.arange(n_sites)` (new code path) must equal `site_groups=None` (old code path) on a synthetic D.
3. **Calibration stays matched under grouping**: the reference distribution for a bank site is its patches' min distance to bank sites of OTHER groups; the test-side score averages over "drop one bank GROUP" (instead of one bank site). Both sides therefore see a bank with one group removed. Any bank with <2 distinct groups raises `ValueError` (verified: every batch has >=6 parents, so real LOPO never trips it).
4. **Held-out patch bank** = all labelled sites except those sharing the held-out site's parent_id (no sibling-inclusive variant). A held-out site whose parent_id matches no labelled parent uses the full bank.
5. **Fingerprint** is used via the public `pmdb.fingerprint.fit` / `predict` / `explain` only (no edits to `pmdb/fingerprint.py`). Fingerprint call = `predict(...)["assigned"]` (lowest score, the classifier of record). Fingerprint probability vector `q_b = p_b / sum_b p_b` (conformal p-values normalised; documented as a heuristic, p-values are not posteriors). p-values are >= 1/(n_b+1) > 0, so the sum is never 0.
6. **Ensemble**: `p_ens = 0.5 * (p_patch + q_fp)` where `p_patch = pm.softmax_conf(top10 pooled)`; ensemble call = argmax `p_ens` (ties -> lowest batch index, i.e. `np.argmax`).
7. **Confidence flag** (rubric: high-correct 2, low (either) 1, high-wrong 0 -> high is score-optimal iff P(correct) > 0.5): stratum = `agree` (patch_call == fingerprint_call). For held-out sites: high iff ensemble LOPO accuracy over ALL labelled sites in the same stratum > 0.5. For the LOPO rubric estimate each labelled site's flag is computed from the stratum accuracy over labelled sites of OTHER parents (nested, so the flag rule is not scored on the sites that set it); empty stratum -> low. Single models (patch alone, fingerprint alone) get the same rule using their own correctness. FINGERPRINT-CONFIDENCE ADDENDUM (dispatcher): (a) override — if the fingerprint marks a site OOD (`ood == True`, every batch rejected), its flag is forced to low, in LOPO rubric scoring and held-out alike; (b) diagnostic only, not used in the flag — in lopo_evaluation.json under `confidence.fp_confidence_diagnostic`, report fingerprint LOPO accuracy and n for conformal `confidence >= 0.5` vs `< 0.5`, and ensemble LOPO accuracy and n for the 2x2 of agree x (fp confidence >= 0.5). Also add rubric rows `fp_conf_flag` (high iff fp confidence >= 0.5, fingerprint call) for comparison. Rubric report: ensemble_flag, ensemble_all_high, all_low (=1.0 by construction), patch_flag, patch_all_high, fingerprint_flag, fingerprint_all_high.
8. **Both LOSO and LOPO are computed by the same function** (`groups = site index` vs `groups = parent code`) for patch, fingerprint and ensemble, giving the LOSO-vs-LOPO table from one code path. Sanity: LOSO patch must reproduce `outputs/patch_mil/evaluation.json` loo accuracy 0.7096774; LOSO fingerprint must reproduce `outputs/fingerprint/evaluation.json` loo accuracy.
9. **Permutation test (LOPO only)**: permute SITE labels, keep parent groups fixed (stated in JSON `scheme`). A permutation is redrawn if it makes any fold infeasible (some batch with <2 distinct training groups after removing a parent); `n_rejected` is reported. The same permutation is used for patch, fingerprint and ensemble each draw. 1000 perms, seed 0, p = (1 + #null>=obs)/(n_perm+1) (matches existing pattern in `pm.permutation_test`).
10. **New module `pmdb/patch_lopo.py`** holds fingerprint-LOPO, ensemble, flag, rubric, permutation, explanation and output assembly; `pmdb/patch_mil.py` only gets the grouped-CV generalisation. Keeps patch_mil.py focused (pure numpy/pandas, no fingerprint import).
11. **Modal data shipping**: the parent table and fingerprint feature tables are computed/read locally in the entrypoint and passed to the remote function as `list[dict]` arguments (not `add_local_file`). Rationale: no image rebuild, no new file paths in the container, `pmdb` is already shipped via `add_local_python_source`. This deliberately replaces brief item 7's "add_local_file".
12. **Held-out site list** = `cache_heldout/half/manifest.csv` (full run), replacing the `HELDOUT` constant. New `--mode heldout` embeds only held-out sites whose embedding npz is missing on the volume, recomputes distances, then runs LOSO eval + LOPO eval. Smoke runs skip LOPO (3 sites/batch cannot give >=2 groups per batch per fold).
13. **Missing fingerprint features for a held-out site -> raise `ValueError`** naming the site and the command to generate them; never silently fall back to patch-only.
14. **Explanation** uses a fixed template (Step 4) with feature phrases from a dict; Batch 3 contrast = top 3 features by `dev_Batch_3` from `fp.explain`, direction = sign(z - center_Batch_3); patch contrast = fraction of patches with `u_Batch_3 > pm.ANOM_U` and the image third (by patch row index) holding most of them, with the montage path `outputs/patch_mil/figures/heldout_<site>.png`.

## 3. Out of scope

- Do NOT modify `pmdb/fingerprint.py`, `scripts/run_fingerprint.py`, `pmdb/patch_embed.py`, anything in `data/`, `data_heldout/`, `cache*/`.
- Do NOT change `loo_bank_scores`, `loo_scores`, `build_outputs`, `explain_heldout` behaviour or the LOSO output files (`outputs/patch_mil/{evaluation.json,loo_predictions.csv,heldout_predictions.csv,patch_scores.csv}`); they stay as written by the current LOSO run (re-running the LOSO evaluate stage must reproduce them).
- No re-embedding of labelled sites; no hyper-parameter changes (K_NN, TOP_FRAC, SOFTMAX_TAU, ANOM_U).
- Parent membership must not enter any probability, call or flag computation (only bank/fold exclusion).
- Do not start until the `patch-mil.md` implementer's commit has landed (`git log -1` shows it and `git status` shows no modified tracked files under `pmdb/`, `modal_patch_mil.py`, `docs/patch_mil.md`). If it has not landed, STOP and escalate.

## 4. Steps

### Step 1 — `pmdb/parents.py` + test

New module (match `pmdb` style: `from __future__ import annotations`, module docstring pointing to docs/patch_mil.md).

```python
ROOT = Path(__file__).resolve().parents[1]

def parent_id(height: int, se_detector: str, bse_grey_step: int) -> str:
    return f"h{int(height)}_{se_detector}_s{int(bse_grey_step)}"

def parent_groups(root: Path | str | None = None, include_heldout: bool = True) -> pd.DataFrame:
    """Columns [batch, site, parent_id, height, se_detector, bse_grey_step]; labelled rows first
    (cache/half/manifest.csv order), then held-out rows (cache_heldout/half/manifest.csv order)."""
```
Logic: for each (manifest, summary) pair — (`cache/half/manifest.csv`, `outputs/clean/summary.csv`) and, if `include_heldout`, (`cache_heldout/half/manifest.csv`, `outputs/clean_heldout/summary.csv`) — left-merge manifest[batch, site, se_detector] with summary[batch, site, height, BSE_grey_step] on (batch, site); if any manifest site has no summary row raise `ValueError(f"no clean summary for {sites}; run scripts/build_clean.py for them first")` (implementer: check the held-out clean build command in docs/clean.md and use the exact command in the message). Rename `BSE_grey_step` -> `bse_grey_step`, cast height/step to int, build `parent_id`. Read CSVs with `dtype={"site": str}`.

Also `def write_parent_groups(path: Path | str | None = None) -> pd.DataFrame` writing `outputs/parent_groups.csv` (default) with `index=False`, returning the frame.

Test `tests/test_parents.py` (reads committed CSVs only, no `data` marker):
- labelled rows: 31 sites, `nunique(parent_id) == 13`; per-parent counts equal `{h1612_ETD_s1: 2, h1780_ETD_s1: 1, h1880_ETD_s1: 1, h1904_ETD_s1: 3, h2048_ETD_s2: 2, h2060_ETD_s1: 4, h2068_SE_s1: 4, h2080_ETD_s1: 3, h2088_ETD_s3: 3, h2148_ETD_s1: 2, h2156_ETD_s2: 2, h2272_ETD_s2: 2, h2316_ETD_s2: 2}`.
- held-out: 3e122cbj -> h2316_ETD_s2, fn0mhxef -> h2048_ETD_s2, xrv9xvzb -> h2088_ETD_s3.
- every batch has >= 6 distinct labelled parents.
- `parent_id(2068, "SE", 1) == "h2068_SE_s1"`.

Then run `python -c "from pmdb.parents import write_parent_groups; write_parent_groups()"` to create `outputs/parent_groups.csv`.

Verify: `pytest -q tests/test_parents.py` — 4 tests pass; `wc -l outputs/parent_groups.csv` -> 35.

### Step 2 — grouped CV in `pmdb/patch_mil.py`

Changes (all additive; `site_groups=None` => current behaviour byte-identical):

```python
def reference_distribution(D, patch_site, bank_sites, site_groups=None) -> np.ndarray:
    # site_groups: (n_sites,) group code per labelled site, or None
    # None: unchanged existing body.
    # grouped: if len(np.unique(site_groups[bank_sites])) < 2: raise ValueError(
    #     "calibration needs at least 2 training groups per batch bank")
    #   rows = np.isin(patch_site, bank_sites)
    #   sub = D[np.ix_(rows, bank_sites)].copy()
    #   same = site_groups[patch_site[rows]][:, None] == site_groups[bank_sites][None, :]
    #   sub[same] = np.inf
    #   return np.sort(sub.min(axis=1))

def logo_bank_scores(D_rows, bank_sites, bank_groups, ref_sorted) -> np.ndarray:
    """(n_groups, n) calibrated u for every dropped bank GROUP (grouped analogue of loo_bank_scores)."""
    sub = D_rows[:, bank_sites]
    out = []
    for g in np.unique(bank_groups):
        keep = bank_groups != g
        out.append(calibrate(sub[:, keep].min(axis=1), ref_sorted))
    return np.stack(out)

def score_site(D_rows, D, patch_site, site_labels, train_mask, n_batches, site_groups=None) -> SiteScore:
    # None: unchanged. Grouped: ref = reference_distribution(D, patch_site, bank, site_groups);
    #   uj = logo_bank_scores(D_rows, bank, site_groups[bank], ref); rest identical
    #   (u = uj.mean(0), top = top_mean(uj).mean(), mean = uj.mean()).

def lopo_scores(D, patch_site, site_labels, site_groups, n_batches) -> list[SiteScore]:
    site_groups = np.asarray(site_groups)
    return [score_site(D[patch_site == i], D, patch_site, site_labels,
                       site_groups != site_groups[i], n_batches, site_groups)
            for i in range(len(site_labels))]

def heldout_scores(Dh, patch_site_h, n_heldout, D, patch_site, site_labels, n_batches,
                   site_groups=None, heldout_groups=None) -> list[SiteScore]:
    # None/None: unchanged (train = all). Otherwise train_h = site_groups != heldout_groups[h]
    # (a held-out site with no labelled parent gets group code -1 -> full bank) and score_site(..., site_groups).
```

Tests appended to `tests/test_patch_mil.py` (use existing `make_D`):
- `test_groups_equal_sites_reproduces_loso`: labels `[0]*4+[1]*4+[2]*4`, `make_D(labels, 7, seed=3, separable=False)`; for every site, `lopo_scores(..., np.arange(12), 3)[i]` vs `loo_scores(...)[i]`: `np.allclose` on `u`, `pooled["top10"]`, `pooled["mean"]` (atol 1e-6).
- `test_grouped_fold_ignores_own_group`: labels as above, groups `[0,0,1,2, 3,3,4,5, 6,6,7,8]`; compute `lopo_scores` for site 0; then set every column of group 0 (sites 0 and 1) to `np.nan`-free garbage (e.g. `-5.0`, which would win any min) for ALL rows except own-column inf, recompute: site 0's scores unchanged. Also: setting site 1's rows (its patches as bank references) to garbage leaves site 0's scores unchanged.
- `test_grouped_reference_requires_two_groups`: bank whose sites all share one group -> `pytest.raises(ValueError)`.
- `test_heldout_excludes_sibling_group`: heldout rows' distance to group-g columns set to -5.0, heldout_groups=[g]; result equals the result with those columns at their original values.

Verify: `pytest -q tests/test_patch_mil.py` — all existing tests plus 4 new pass.

### Step 3 — `pmdb/patch_lopo.py`: fingerprint LOPO, ensemble, flags, rubric

```python
from pmdb import fingerprint as fp
from pmdb import patch_mil as pm
BATCHES = pm.BATCHES

def normalise_p(p: np.ndarray) -> np.ndarray          # rows / row sums
def fingerprint_cv(X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> pd.DataFrame
    # X index (batch, site) in the SAME order as the patch site order; groups aligned to X rows.
    # For each unique group g: m = fp.fit(X[groups != g], y[groups != g]);
    #   assert m.batches == list(BATCHES); pr = fp.predict(m, X[groups == g]).
    # Return frame indexed like X with: fp_call (pr["assigned"]), p_fp_<b> (raw), q_fp_<b> (normalise_p).
def fingerprint_heldout(X, y, groups_lab: np.ndarray, H: pd.DataFrame, groups_h: np.ndarray
                        ) -> tuple[pd.DataFrame, pd.DataFrame]
    # per held-out row h: fit on X[groups_lab != groups_h[h]]; predict + fp.explain on H.iloc[[h]].
    # returns (pred frame as fingerprint_cv columns, concatenated explain frame).
def patch_cv(D, patch_site, site_labels, groups) -> list[pm.SiteScore]   # pm.lopo_scores
def ensemble(p_patch: np.ndarray, q_fp: np.ndarray) -> np.ndarray       # 0.5 * (a + b)
def stratum_flags(correct: np.ndarray, agree: np.ndarray, groups: np.ndarray) -> np.ndarray
    # flag_i = mean(correct[j]) > 0.5 over j with groups[j] != groups[i] and agree[j] == agree[i];
    # no such j -> False
def stratum_table(correct, agree) -> dict   # {"agree": {"n", "accuracy"}, "disagree": {...}} over all sites
def heldout_flag(correct, agree, query_agree: bool) -> bool   # stratum accuracy over ALL sites > 0.5; n=0 -> False
def expected_rubric(correct: np.ndarray, high: np.ndarray) -> float
    # mean of where(high, where(correct, 2, 0), 1)
def cv_feasible(labels: np.ndarray, groups: np.ndarray) -> bool
    # for every group g and batch b: len(unique(groups[(labels == b) & (groups != g)])) >= 2
```

`run_cv(D, patch_site, site_labels, groups, X, y) -> pd.DataFrame` (one row per labelled site, site order of D): columns `batch, site, true, patch_call, fp_call, ens_call, agree, p_patch_<b>, q_fp_<b>, p_ens_<b>, correct_patch, correct_fp, correct_ens` — patch probs via `pm.softmax_conf(top10)`, patch_call = argmin top10 (existing rule), ens_call = argmax p_ens.

`lopo_permutation(D, patch_site, site_labels, groups, X, n_perm=1000, seed=0) -> dict`:
```python
rng = np.random.default_rng(seed); null = {"patch": [], "fingerprint": [], "ensemble": []}; rejected = 0
while len(null["patch"]) < n_perm:
    perm = rng.permutation(site_labels)
    if not cv_feasible(perm, groups):
        rejected += 1
        if rejected > 100 * n_perm: raise RuntimeError("permutation feasibility rejection limit")
        continue
    df = run_cv(D, patch_site, perm, groups, X, pd.Series([BATCHES[c] for c in perm], index=X.index))
    for k, col in (("patch", "correct_patch"), ("fingerprint", "correct_fp"), ("ensemble", "correct_ens")):
        null[k].append(df[col].mean())
```
Return per model `{observed_accuracy, null_mean, null_p95, p_value}` plus `n_perm, seed, n_rejected, scheme: "site labels permuted, parent groups fixed, infeasible draws (a batch with <2 training parents in some fold) redrawn"`.

Tests in new `tests/test_patch_lopo.py` (synthetic, no data):
- `normalise_p` rows sum to 1; `ensemble` of [1,0,0] and [0,0,1] = [0.5,0,0.5].
- `stratum_flags`: correct `[1,1,0,1,0,0]`, agree `[T,T,T,F,F,F]`, groups `[0,1,2,3,4,5]` -> `[False, False, True, False, False, False]` (site 0 sees agree-others {1,0} -> 0.5 not > 0.5; site 2 sees {1,1} -> 1.0; site 3 sees {0,0}).
- `expected_rubric([T,F,T,F],[T,T,F,F]) == (2+0+1+1)/4`.
- `cv_feasible`: labels `[0,0,1,1,2,2]` groups `[0,1,2,3,4,5]` -> False (dropping group 0 leaves batch 0 one group); labels `[0,0,0,1,1,1,2,2,2]` groups `[0..8]` -> True.
- `fingerprint_cv` leakage: synthetic X (9 sites x 3 features, 3 per batch... use 12 sites, 4 per batch, groups pairs) — assert that changing feature values of a test group's sibling rows... simpler and decided: monkeypatch `fp.fit` with a wrapper recording the training index; assert no fit call ever contains a row of the predicted group.

Verify: `pytest -q tests/test_patch_lopo.py` — 6 tests pass.

### Step 4 — held-out assembly, explanations, `build_lopo_outputs`

In `pmdb/patch_lopo.py`:

```python
FEATURE_PHRASES = {
    "si_depth_slope": "the trend of Si fraction with electrode depth",
    "si_depth_mid_dip": "the dip in Si fraction mid-depth",
    "k15_contact_tilestd": "the variability of Si-graphite contact between tiles",
}
def feature_phrase(name: str) -> str:
    # si_depth_rel_band{k} -> f"the relative Si fraction in depth band {k} (of 5)"
    # gx_{a}_{b} -> f"graphite lateral texture at {a}-{b} um scale"
    # gz_{a}_{b} -> f"graphite through-depth texture at {a}-{b} um scale"
    # else FEATURE_PHRASES[name]; unknown name -> the raw name (no exception)
def image_third(rows: np.ndarray, n_rows: int) -> str   # argmax count over min(2, r*3//n_rows) -> "top"/"middle"/"bottom"; ties -> first
```

Explanation template (exact; `{}` filled, probabilities `.2f`, fractions `.0%`, dev `.1f`):

AUDIENCE RULE (user, overrides earlier wording): the explanation is read by the organiser, who knows nothing about our models. It must NOT mention "model", "patch model", "fingerprint", "ensemble", "kNN", "embedding", "score", "agree/disagree", "stratum", "LOPO" or probabilities. Every reason, including the confidence reason, is stated as microstructure evidence and which batch that evidence resembles. The agreement-stratum flag rule stays the internal decision rule; only the wording changes.

Per evidence line, "resembles {b}" = the batch with the smallest `dev_<b>` for that feature (fingerprint lines), and for the texture line the batch with the lowest patch top-10 calibrated score (`patch_call`). Template (exact; fractions `.0%`):

```
{site}: Batch {k} ({flag} confidence). Compared with Batch 3 (supplier baseline): (1) {phrase1} is {higher|lower} than in Batch 3, typical of Batch {r1}; (2) {phrase2} is {higher|lower} than in Batch 3, typical of Batch {r2}; (3) {phrase3} is {higher|lower} than in Batch 3, typical of Batch {r3}; (4) local microstructure appearance: {frac} of the image's 11.2 um areas look unlike any Batch 3 image (Batch 3 images: typically {b3_median_frac}), {where}; overall it looks most like Batch {rt}. {confidence_sentence}
```
- `{where}` = `mostly in the {third} third of the image (see outputs/patch_mil/figures/heldout_{site}.png)`, or `spread across the whole image` if no third holds >50% of the anomalous areas, or omitted with `none do` wording if frac == 0.
- `{b3_median_frac}` = median over LOO Batch 3 sites of the anomalous fraction (existing evaluation value, ~4%).
- `{confidence_sentence}`: if flag high → `Confidence is high because the depth-profile, graphite-arrangement and local-appearance evidence consistently point to Batch {k}.` If low → `Confidence is low because the evidence is mixed: {list of evidence items resembling Batch k} point to Batch {k}, but {list of items resembling another batch} look like Batch {other}.` (Items named by their plain phrase; if every item resembles k but flag is low, use `Confidence is low because the differences from the other batches are small relative to normal variation between images of the same batch.`)
- Phrase dict: si_depth_rel_band{i} → "Si fraction in depth band {i+1} of 5 (1 = top)", si_depth_slope → "the top-to-bottom trend in Si fraction", si_depth_mid_dip → "Si depletion at mid-depth", gx_a_b → "graphite clustering along the layer at {a}-{b} um spacing", gz_a_b → "graphite clustering through the depth at {a}-{b} um spacing", k15_contact_tilestd → "variability of Si-graphite contact across the image". (This replaces the earlier `feature_phrase` examples; update the step-4 test expectation accordingly.)
The explanation must not mention parents/siblings. Test: explanation contains "Batch 3" and none of the forbidden words above (case-insensitive).

`build_lopo_outputs(D, patch_site, site_labels, sites: pd.DataFrame, parents: pd.DataFrame, X, H, Dh, patch_site_h, heldout_ids, coords_h, n_perm=1000, seed=0) -> dict` returns:
- `lopo_predictions`: `run_cv` with parent groups, plus columns `parent_id`, `flag_high` (stratum_flags on correct_ens), `fold="lopo"`; and the LOSO run (groups = `np.arange(n_sites)`) appended with `fold="loso"` (flag_high computed the same way within LOSO).
- `evaluation` (-> lopo_evaluation.json): `{n_sites, n_parents, parents: {parent_id: [sites]}, loso: {patch, fingerprint, ensemble}, lopo: {patch, fingerprint, ensemble}` (each `pm.classification_metrics`), `permutation: lopo_permutation(...)`, `confidence: {rule, strata: stratum_table(correct_ens, agree)}`, `rubric_expected_score: {ensemble_flag, ensemble_all_high, all_low, patch_flag, patch_all_high, fingerprint_flag, fingerprint_all_high}` (LOPO), `fingerprint_probability: "conformal p-values normalised to sum 1 (heuristic, not a posterior)"`, `parents_used_as_evidence: false}`.
- `final_heldout`: columns exactly `site, assigned, confidence_flag, p_ens_Batch_1, p_ens_Batch_2, p_ens_Batch_3, patch_call, fingerprint_call, explanation` (`confidence_flag` in {"high","low"}). Patch side: `pm.heldout_scores(..., site_groups=parent codes, heldout_groups=held-out parent codes or -1)`; anomalous fraction from `u[:, 2] > pm.ANOM_U`; third from `coords_h[sel][:, 0]` of anomalous patches with `n_rows = coords_h[sel][:, 0].max() + 1`. Fingerprint side: `fingerprint_heldout`. Flag: `heldout_flag(correct_ens of LOPO rows, agree of LOPO rows, this site's agree)`.

Parent codes: `pd.factorize` over labelled `parent_id` (aligned to D site order by (batch, site) merge, assert no NaN); held-out code = labelled code of the same parent_id, else -1. `X` must be reindexed to D's (batch, site) order (assert identical set); `H` reindexed to `heldout_ids`, missing -> `ValueError` per Decision 13 (message names `scripts/run_fingerprint.py --heldout-dir outputs/heldout/kpis`).

Test (append to `tests/test_patch_lopo.py`): `test_build_lopo_outputs_synthetic` — 12 labelled sites (4/batch, groups of 2, separable `make_D`-style D copied into the test), X with 3 informative synthetic features named `si_depth_slope, gx_0.5_2.0, k15_contact_tilestd`, 1 held-out site whose parent equals labelled group 0; assert final table columns as listed, `confidence_flag` in {high, low}, `p_ens` rows sum to 1, explanation contains "Batch 3" and not "parent"/"sibling" or any AUDIENCE RULE forbidden word; `n_perm=5`. Plus `feature_phrase("gz_2.0_4.0") == "graphite clustering through the depth at 2.0-4.0 um spacing"` and `image_third(np.array([0,0,5]), 6) == "top"`.

Verify: `pytest -q tests/test_patch_lopo.py` — 8 tests pass.

### Step 5 — Modal orchestration (`modal_patch_mil.py`)

- `embed_sites(sites, tag, skip_existing: bool = False)`: when `skip_existing` and `/out/{tag}/emb/{batch}__{site}.npz` exists, skip that site (print "skip").
- New `@app.function(volumes={"/out": out_vol}, cpu=4.0, memory=8192, timeout=3600) def evaluate_lopo(tag: str, n_perm: int, parents: list[dict], fp_features: list[dict], fp_heldout: list[dict], seed: int = 0) -> dict`: loads `/out/{tag}/distances.npz` like `evaluate`, rebuilds DataFrames (`set_index(["batch","site"])` for features), calls `build_lopo_outputs`, returns `{"evaluation", "lopo_predictions" (records), "final_heldout" (records), "elapsed_s"}`.
- `main`: `mode` accepts `embed | distances | eval | lopo | heldout | all`. Full-run held-out list = `cache_heldout/half/manifest.csv` `site` column (string), replacing `HELDOUT` (delete the constant). `lopo` and `all` run `evaluate_lopo` after `eval`; `heldout` = embed held-out only with `skip_existing=True`, then distances, eval, lopo. Locally before the remote call: `parents = write_parent_groups()` (writes outputs/parent_groups.csv), `fp_features = pd.read_csv("outputs/fingerprint/features.csv", dtype={"site": str})`, `fp_heldout = pd.read_csv("outputs/fingerprint/heldout_features.csv", dtype={"site": str})`. Write `outputs/patch_mil/lopo_evaluation.json`, `lopo_predictions.csv`, `final_heldout_predictions.csv` via `_atomic_write`; print LOSO/LOPO accuracy per model, permutation p, rubric ensemble_flag, and each held-out `site -> assigned (flag)`. Smoke: skip lopo with a printed notice. Update `METHOD`/module docstring run lines to include `--mode lopo` and `--mode heldout`.

Verify locally: `python -c "import modal_patch_mil"` imports; `pytest -q -m "not data"` passes.

### Step 6 — run on Modal

`modal run modal_patch_mil.py --mode lopo` (D already on volume `pmdb-patch-out`, tag `full`). Follow the Modal-debugging memory: on failure read logs before relaunching.

Verify: the three output files exist; `lopo_evaluation.json["loso"]["patch"]["accuracy"] == 0.7096774193548387` and its confusion equals `outputs/patch_mil/evaluation.json["loo"]["confusion"]`; `["loso"]["fingerprint"]["accuracy"]` equals `outputs/fingerprint/evaluation.json["loo"]["accuracy"]`; `final_heldout_predictions.csv` has 3 rows; `git diff --stat outputs/patch_mil/evaluation.json outputs/patch_mil/loo_predictions.csv` empty.

### Step 7 — docs + AGENTS.md

`docs/patch_mil.md` new sections (after "Results", before "Limitations"):
- **Parent images (identified confound)**: parent table (parent_id, batches with counts, held-out members) from outputs/parent_groups.csv; how found (shared full-res height + SE detector + BSE grey-level step); why LOSO is optimistic (sibling crops in the bank / training set); explicit statement "parent membership is used only to exclude siblings from training banks and CV folds; it never contributes evidence to a batch call or confidence".
- **LOSO vs LOPO**: table rows patch / fingerprint / ensemble, columns LOSO acc, LOPO acc, LOPO balanced acc, LOPO macro-F1, LOPO permutation p; plus LOPO confusion for the ensemble; note the permutation scheme.
- **Confidence flag and rubric**: rule, stratum sizes/accuracies, expected scores table (all seven entries) with the note that all-low scores 1.0 by construction.
- **Final held-out calls**: table from final_heldout_predictions.csv with explanations verbatim.
- **Scoring new test images**: (1) add raw TIFFs and rebuild held-out cache/clean summary (cite exact commands found in docs/clean.md / AGENTS.md), (2) compute held-out KPIs + `python scripts/run_fingerprint.py --heldout-dir outputs/heldout/kpis` (implementer: locate the KPI command that writes `outputs/heldout/kpis/curves.csv` by grepping docs/ and scripts/; if none exists, STOP and escalate), (3) `modal volume put pmdb-data ...` for new held-out cache files (match the existing docstring commands), (4) `modal run modal_patch_mil.py --mode heldout`.
- Note the fingerprint probability normalisation heuristic.

`AGENTS.md`: one line under "Known Confounds & QC Data": `- **Parent images**: the 31 labelled sites are crops of 13 parent images (plus held-out siblings); mapping in [outputs/parent_groups.csv](outputs/parent_groups.csv), see docs/patch_mil.md. Use leave-one-parent-out CV.`

Verify: `pytest -q -m "not data"` and `pytest -q -m data` both pass (data suite may skip; report counts).

### Step 8 — commit

On branch `worktree-patch-anomaly-mil`: stage `pmdb/parents.py pmdb/patch_lopo.py pmdb/patch_mil.py modal_patch_mil.py tests/test_parents.py tests/test_patch_lopo.py tests/test_patch_mil.py docs/patch_mil.md AGENTS.md outputs/parent_groups.csv outputs/patch_mil/lopo_evaluation.json outputs/patch_mil/lopo_predictions.csv outputs/patch_mil/final_heldout_predictions.csv .claude/plans/patch-mil-lopo.md`; commit `feat(patch-mil): leave-one-parent-out eval, patch+fingerprint ensemble, final held-out calls`. No push/PR (dispatcher handles).

Verify: `git status --short` shows none of the staged paths.

## 5. Expected surprises

- **LOSO patch reproduction differs from 0.7097**: means the `site_groups=None` path was altered or the volume D changed. Do not proceed to docs; STOP and escalate.
- **LOSO fingerprint differs from `outputs/fingerprint/evaluation.json`**: likely feature-row ordering or dtype (site read as int/float). Pre-authorised: fix ordering/dtype in the wrapper only; if still different, escalate.
- **`outputs/fingerprint/heldout_features.csv` is untracked/absent** (it exists locally now but may not be committed): pre-authorised to read it as-is; if absent, STOP (needs the KPI pipeline).
- **Permutation runtime > Modal timeout (3600 s)**: pre-authorised to raise `timeout` to 7200 and `cpu` to 8.0; do not reduce n_perm without escalating.
- **`n_rejected` very large (> 10 * n_perm)**: still valid; report it in docs, no change.
- **LOPO accuracy drops below the majority baseline (0.548) for some model**: expected possibility, report honestly; no tuning.
- **A held-out site's fingerprint and patch disagree and stratum accuracy <= 0.5** -> low flag; this is the intended outcome, not a bug.
- **`fp.fit` emits RuntimeWarnings (nanmedian of small batches)**: ignore.

## 6. Done criteria

- `pytest -q -m "not data"` and `pytest -q -m data` pass.
- `outputs/parent_groups.csv` (34 rows), `outputs/patch_mil/lopo_evaluation.json`, `lopo_predictions.csv`, `final_heldout_predictions.csv` (3 rows, exact columns) exist; LOSO patch accuracy in lopo_evaluation.json == 0.7096774193548387; LOSO fingerprint accuracy matches outputs/fingerprint/evaluation.json; existing LOSO files unchanged.
- docs/patch_mil.md has the five new sections; AGENTS.md pointer line present; nothing in outputs or explanations uses parent membership as evidence.
- Committed on `worktree-patch-anomaly-mil`.

## 7. Revision log

(empty)
