# Plan r3: held-out truths as labels, within-parent contrasts, parent-centred fingerprint, model menu, test-day runbook

## 1. Context

Organiser feedback (2026-10-04): held-out truths are 3e122cbj = Batch_2, fn0mhxef = Batch_1, xrv9xvzb = Batch_3. Our submission (fingerprint classifier of record: 3e122cbj -> Batch_1, fn0mhxef -> Batch_3, xrv9xvzb -> Batch_2) got 0/3. Parent images are deliberately split across batches, so the designed batch signal is what differs between crops of the same parent. This round (a) adds the 3 truths as labelled sites (34 total, 13 parents), (b) measures which fingerprint features differ consistently between batches within the same parent, (c) adds a parent-centred fingerprint variant, (d) picks a model by pre-registered LOPO rubric score from a 5-option menu, and (e) builds a one-command runbook for the 4-6 new test images released this afternoon. Builds on `.claude/plans/patch-mil-lopo.md` (pmdb/parents.py, pmdb/patch_lopo.py, grouped CV in pmdb/patch_mil.py, `modal run modal_patch_mil.py --mode lopo`).

## 2. Design decisions

1. **Gate**: start only after the patch-mil-lopo implementer's commit has landed (`git log -1` shows `feat(patch-mil): leave-one-parent-out eval...` and `git status --short` shows no modified tracked file under `pmdb/`, `modal_patch_mil.py`, `tests/`). Otherwise STOP and escalate.
2. **Labels file** `outputs/heldout_labels.csv`, columns `site,batch,source`, rows in order 3e122cbj/Batch_2, fn0mhxef/Batch_1, xrv9xvzb/Batch_3, source `organiser feedback 2026-10-04`. Sites keep their storage key `("Batch_heldout", site)` everywhere (cache, embeddings, feature tables); the truth lives in a separate `label` column. Rationale: no re-keying of caches/embeddings on Modal; one merge function owns the mapping.
3. **New module `pmdb/batch_menu.py`** holds labels, 34-site feature table, centring, centred-fingerprint CV, menu CV, flags, selection, test-site predictions. It only CALLS `pmdb.patch_lopo` / `pmdb.patch_mil` / `pmdb.fingerprint` public-or-underscore helpers (`plo._run_cv`, `plo._pred_frame`, `plo.fingerprint_cv`, `plo.fingerprint_heldout`, `plo.ensemble`, `plo.normalise_p`, `plo.stratum_flags`, `plo.heldout_flag`, `plo.expected_rubric`, `plo.make_explanation`, `pm.heldout_scores`, `pm.softmax_conf`, `pm.classification_metrics`, `pm.ANOM_U`); it does not modify them. Rationale: the lopo outputs stay reproducible.
4. **Within-parent contrast** (pmdb/within_parent.py): pool = 34 labelled sites, 16 features from `outputs/fingerprint/features.csv` + `heldout_features.csv`. For each parent and each batch pair (a, b) in `(B1,B2), (B1,B3), (B2,B3)` with both present: `diff = mean_a - mean_b` per feature. Per feature x pair: `n_parents`, `n_pos`, `n_neg`, two-sided exact sign-test p (zeros dropped; `p = min(1, 2 * sum_{k<=min(n_pos,n_neg)} C(n,k) / 2^n)`, `math.comb`, NaN if n = 0), `median_diff`, `median_diff_sd = median_diff / SD_feature` (SD over the 34 sites, ddof=1), `consistency = max(n_pos, n_neg) / n_parents`, `is_signal = (n_parents >= 3) & (consistency == 1.0)`. Pre-registered criterion; expected n_parents (verified from outputs/parent_groups.csv + truths): B1-B2 = 5 (h2048, h2080, h2148, h2156, h2316), B1-B3 = 1 (h2080), B2-B3 = 3 (h2068, h2080, h2272). Minimum achievable p with n=5 is 0.0625, so p is reported, not thresholded.
5. **Parent-centred features**: `Xc = X - groupby(parent_id).transform("mean")` over the whole pool passed in (all labelled sites + any test sites; labels never read), so a test site's own row is in its parent mean. **Singleton rule (decided)**: a site that is the only member of its parent in the pool is (i) excluded from centred-model training and (ii) predicted with the uncentred fingerprint (its row copied from the uncentred CV / test prediction). Rationale: a singleton's centred row is identically zero, carries no information, and would be assigned to whichever batch centre sits nearest 0 (an artefact); its uncentred row is the only evidence available, and zero rows in training would shrink every batch centre and scale toward 0. Currently 2 labelled singletons (h1780, h1880).
6. **Menu (5 options, fixed order)** — calls, flags, component counts:
   | option | call | flag rule | OOD override | components |
   |---|---|---|---|---|
   | `fingerprint` | `fp_call` | global | `fp_ood` | 1 |
   | `fingerprint_centred` | `fpc_call` | global | `fpc_ood` | 1 |
   | `patch` | `patch_call` | global | none | 1 |
   | `ensemble` | argmax `0.5*(p_patch + q_fp)` | stratum on `agree_fp = patch_call == fp_call` | `fp_ood` | 2 |
   | `ensemble_centred` | argmax `0.5*(p_patch + q_fpc)` | stratum on `agree_fpc = patch_call == fpc_call` | `fpc_ood` | 2 |
   - **global** (single models): LOPO flag_i = mean(correct over sites of OTHER parents) > 0.5; test-site flag = mean(correct over all labelled LOPO rows) > 0.5. Rationale: a single model should not need the other model for its confidence (keeps "fewer components" honest).
   - **stratum** (ensembles): exactly the lopo plan's rule (`plo.stratum_flags` nested by parent; `plo.heldout_flag` for test sites).
   - OOD override: flag forced low if the override column is True (lopo plan addendum (a)).
   - Probabilities reported for test sites: ensembles `p_ens`/`p_ensc`; fingerprint options `q_fp`/`q_fpc`; patch `pm.softmax_conf(top10)`.
7. **Pre-registered selection** (`select_option`): score = LOPO expected rubric with the flag rule (`plo.expected_rubric`), on 34 sites. Pick max score; ties (equal after `round(x, 9)`) -> fewer components, then menu order. If the winner's score <= 1.0 (all-low baseline), confidence_mode = `all_low` and the option = the one with highest LOPO accuracy (same tie-break); else confidence_mode = `rule`. Written once to `outputs/menu/selection.json` by the r3 run and READ (never recomputed) on test day. Rationale: freezes the choice before seeing test images.
8. **No permutation test in the menu** (hackathon scope; the 31-site LOPO permutation from the lopo plan stands). Report per-option rubric SE = `std(per-site rubric points, ddof=1)/sqrt(n)` and the selection-bias caveat text (Step 3).
9. **Patch side on 34 sites**: `compute_distances` gets `out_name` (r3 uses `distances_r3.npz`, leaving `distances.npz` untouched) and `heldout_batch`; the 34 labelled = 31 manifest sites + `("Batch_heldout", s)` for the 3 labelled held-out sites (embeddings already on volume `pmdb-patch-out` under `/full/emb/`). Empty test list allowed (Dh shape `(0, n_lab)`).
10. **Test-site storage**: organiser TIFFs go locally to `data_test/Batch_test/` (same filenames as `data_heldout/Batch_heldout/`: `img_<site>_<BSE|Inlens|ETD|SE>.tif`; gitignored), batch key `Batch_test`. On the Modal data volume `pmdb-data` (mounted at `/data`) everything lives under `/test/` and mirrors the verified `/heldout/` layout (`/heldout/half/{manifest.csv,Batch_heldout__<site>.npz}`, `/heldout/harmonised/affine2/luts.npz`):
    | volume path | written by | pulled back to (committed) |
    |---|---|---|
    | `/test/raw/Batch_test/img_*.tif` | local entrypoint upload | — (never) |
    | `/test/half/manifest.csv`, `/test/half/Batch_test__<site>.npz` | `build_cache.build_cache` | `cache_test/half/manifest.csv` only (`path` column rewritten to `cache_test/half/<batch>__<site>.npz`) |
    | `/test/raw_intensity_stats.csv` | `build_cache.build_cache` | `outputs/test/raw_intensity_stats.csv` |
    | `/test/harmonised/anchors.csv`, `/test/harmonised/affine2/{luts.npz,params.csv,reference.json}` | `build_harmonised._collect_anchors` + `_fit_all` | `cache_test/harmonised/...` (same relative paths) |
    | `/test/clean/{summary.csv,material_thresholds.json}`, `/test/clean/Batch_test/<site>/params.json` (+ TIFFs, previews, qc_report.html stay on the volume) | `build_clean.main` | `outputs/clean_test/...` (same relative paths) |
    | `/test/kpis/{site_kpis,tile_kpis,curves,sensitivity}.csv`, `run_log.json` | `run_kpis.main` | `outputs/test/kpis/...` |
    | `/test/features.csv` (columns `batch,site,<16 features>`) | `fp.read_feature_inputs` | `outputs/test/features.csv` |
    Rationale: `data_heldout/` is read-only and its README forbids mixing; the `Batch_` prefix keeps `list_sites` and the scripts working with root arguments only; `load_site(..., cache_root="/data/test")` needs exactly `half/` + `harmonised/affine2/luts.npz`.
11. **Compute placement (user decision 2026-10-04, replaces the r1 "local preprocessing" exception)**: ALL test-day compute runs on Modal. New module `modal_test_prep.py` (app `pmdb-test-prep`, CPU image, `pmdb-data` mounted WRITABLE at `/data`) does the per-site preprocessing; `modal_patch_mil.py --mode test` does GPU embedding + menu evaluation as before. Local work on test day is only: file copy, upload (inside the entrypoint), writing the pulled-back bytes, and the CSV/markdown assembly already in Step 4's `test` mode. One container processes all test sites (stages run in sequence, with the scripts' own process pools over sites), not a per-site `.map`. Rationale: `build_cache`, `build_clean.main` and `run_kpis.main` each write one multi-site table; a per-site map would need new table-merging code, while 4-6 sites fit one 8-CPU container in ~10 min.
14. **Reuse, not re-implementation, of the preprocessing**: the container imports the shipped scripts (`scripts/build_cache.py`, `scripts/build_harmonised.py`, `scripts/build_clean.py`, `scripts/run_kpis.py`, via `add_local_file` to `/root/scripts/` and `sys.path.insert(0, "/root/scripts")`) and calls `build_cache.build_cache(...)`, `build_harmonised._collect_anchors` / `_fit_all`, `build_clean.main()` (argv via `sys.argv`), `run_kpis.main(argv)`; the scripts are not edited. Labelled LUTs are NOT refitted: the affine2 reference is loaded with `H.load_reference(Path("/root/ref"), "affine2")` from the shipped labelled `cache/harmonised/affine2/reference.json` (JSON float round-trip is exact, so held-out/test LUTs equal what `build_harmonised.py --heldout-cache-root` would produce; the rehearsal checks it). Labelled-reference artefacts shipped into the image with `add_local_file` (not the volume, so they cannot go stale relative to the code): `cache/harmonised/affine2/reference.json` -> `/root/ref/harmonised/affine2/reference.json`; `outputs/clean/targets.json` -> `/root/ref/clean/targets.json`; `outputs/clean/summary.csv` -> `/root/ref/clean/summary.csv` (for `--labelled-summary`: material-flag thresholds from the labelled distribution instead of from 4-6 test sites); `docs/kpis/kpi_catalogue.csv` -> `/root/docs/kpis/kpi_catalogue.csv` (`pmdb.kpis.CATALOGUE_PATH` resolves to `<parents[2]>/docs/kpis/`, i.e. `/root/docs/kpis/`, same as `modal_app.py`). No labelled raw data or labelled half cache is uploaded or read by the prep.
15. **Package pins for the prep image = the labelled KPI run** (`outputs/kpis/run_log.json`: python 3.10, numpy 1.26.4, scipy 1.14.1, scikit-image 0.25.2, pandas 2.1.4, Pillow 12.3.0; these equal `modal_app.py`'s pins) + tifffile 2025.5.10, imagecodecs 2025.3.30, matplotlib 3.10.7 (needed by `run_kpis` -> `pmdb.overlay` import and `build_clean._thumbs`). Rationale: test features must come from the same numeric stack as the 31 labelled training rows. Note: the 3 held-out KPI rows were computed locally with a newer stack (numpy 2.5.3, scipy 1.18.1, scikit-image 0.26.0, pandas 3.0.6, python 3.14 per `outputs/heldout/kpis/run_log.json`), so the rehearsal comparison also measures that stack difference.
12. **Explanations**: per-site text = `plo.make_explanation(...)` (AUDIENCE RULE unchanged; evidence from the uncentred `fp.explain` + patch anomaly, regardless of which option made the call) + one run-level sentence from the within-parent `is_signal` rows (Step 2 `signal_sentence`). That sentence says "crops of the same source image"; it is a general finding, not evidence about the site, and contains none of `plo.FORBIDDEN` (which includes "parent"/"sibling"). The figure reference `" (see outputs/patch_mil/figures/heldout_{site}.png)"` is stripped from test-site explanations (no test montages are generated).
13. **Do not tune on the 3 truths**: no hyper-parameter, feature or rule is changed in response to their individual results; they enter LOPO as ordinary sites.

## 3. Out of scope

- No edits to `pmdb/fingerprint.py`, `pmdb/patch_embed.py`, `pmdb/patch_lopo.py`, `pmdb/harmonise.py`, `pmdb/clean.py`, `pmdb/io.py`, `scripts/run_fingerprint.py`, `scripts/build_*.py`, `scripts/run_kpis.py`, `modal_app.py`, anything in `data/` or `data_heldout/` (including its README), `cache/`, `cache_heldout/`.
- Never upload anything from `data/` or `data_heldout/` to Modal (the rehearsal uploads fn0mhxef's TIFFs only via the `data_test/` copy). No labelled LUT refit; no edits to `/half`, `/harmonised` or `/heldout` on the `pmdb-data` volume.
- Do not overwrite `outputs/patch_mil/*` (lopo/LOSO outputs) or `/out/full/distances.npz` on the volume.
- No re-embedding of labelled or held-out sites; no hyper-parameter changes; no new permutation tests; no sweeps.
- Parent membership enters only fold/bank exclusion, centring (labels unread) and the within-parent analysis; never a call or flag directly.
- No push / PR.

## 3b. Dispatcher addendum (round 2, 2026-10-04) — overrides conflicting text above/below

Findings from the lopo run (commit 6876910) that this round must fix:

**A. Fingerprint probability must agree with its call.** In lopo the fingerprint call is argmin of its likelihood scores but its probability vector was normalised conformal p-values, so for fn0mhxef both models said B3 and the ensemble said B2. In `pmdb/batch_menu.py` (not patch_lopo.py) define `fp_prob(scores) = softmax(-scores / 0.1)` over the per-batch `score_Batch_*` returned by `fingerprint.predict` (same tau as patch `softmax_conf`), and use it in EVERY ensemble option (centred and uncentred). Test: `argmax(fp_prob) == fp_call` on synthetic scores. Report in the menu output which held-out ensemble calls change vs the lopo run.

**B. Confidence sentence comes from the evidence, not the flag.** Count the 4 evidence items (3 fingerprint lines + texture line) whose "typical of Batch r" equals the call k. Sentence:
- flag high: `Confidence is high: {items resembling k} point to Batch {k}` + (if any others) `, while {other items} look like Batch {x}` + `.`
- flag low: `Confidence is low because the evidence is mixed: {items resembling k} point to Batch {k}, but {other items} look like Batch {x}.`; if all 4 resemble k use the existing "differences ... small relative to normal variation" sentence.
- The word "consistently" may appear only when all 4 items resemble k. Also emit `n_evidence_for_call` (0–4) as a column. Test the three branches.

**C. Recompute the 3 held-out sites' fingerprint features on Modal before using them as training rows.** Their KPI rows were computed locally on a different stack (numpy 2.5 / python 3.14). Execution order becomes: Step 5a (modal_test_prep.py) → Step 5c rehearsal extended to ALL THREE held-out sites (copy each site's raw TIFFs from data_heldout/Batch_heldout into data_test/Batch_test — reading data_heldout is allowed, writing is not; uploading those 3 copies to pmdb-data:/test/raw is authorised) → save the Modal-computed 16 features as `outputs/heldout_features_modal.csv` and report per-feature max |Δ| / labelled SD vs the old rows → Steps 1–4 use the Modal-computed rows → Step 5b → Step 6. Rehearsal pass rules: half cache / LUT / parent key mismatches still stop-and-escalate; feature differences are informational only (report them, do not stop). Clean up /test and data_test afterwards as planned.

## 4. Steps

### Step 1 — labels, 34-site tables, parents for test sites

- Create `outputs/heldout_labels.csv` per Decision 2.
- `pmdb/batch_menu.py` (style: `from __future__ import annotations`, docstring -> docs/patch_mil.md, `ROOT = Path(__file__).resolve().parents[1]`):
```python
BATCHES = pm.BATCHES
def labelled_sites(root: Path | str | None = None) -> pd.DataFrame:
    """[batch, site, label]: 31 rows of cache/half/manifest.csv (label = batch), then the rows of
    outputs/heldout_labels.csv as batch='Batch_heldout', label=<truth>. Raises ValueError if a labels-file
    site is missing from cache_heldout/half/manifest.csv or a label is not in BATCHES. site read as str."""
def feature_table(keys: list[tuple[str, str]], root=None) -> pd.DataFrame:
    """Concat outputs/fingerprint/features.csv + heldout_features.csv (dtype site=str), index (batch, site),
    reindexed to keys; ValueError naming missing keys."""
```
- `pmdb/parents.py`: add a third pair `("cache_test/half/manifest.csv", "outputs/clean_test/summary.csv")`, used only when `include_test=True` (new kwarg, default True) AND `root/cache_test/half/manifest.csv` exists. Missing-summary error for it names `modal run modal_test_prep.py::main`. New-parent sites need no special code (unique key -> own parent_id).
- `.gitignore`: add `data_test/`.
- Tests `tests/test_batch_menu.py`: `test_labelled_sites` (34 rows, unique (batch,site), held-out labels exactly as Decision 2, all labels in BATCHES, first 31 have label == batch); `test_parent_groups_without_test_dir_unchanged` (with no `cache_test/`, `parent_groups()` returns 34 rows).

Verify: `pytest -q tests/test_batch_menu.py tests/test_parents.py` — all pass.

### Step 2 — within-parent contrast analysis

`pmdb/within_parent.py`:
```python
PAIRS = (("Batch_1", "Batch_2"), ("Batch_1", "Batch_3"), ("Batch_2", "Batch_3"))
def parent_contrasts(F: pd.DataFrame, label: pd.Series, parent: pd.Series) -> pd.DataFrame
    # long: parent_id, pair ("Batch_1-Batch_2"), feature, n_a, n_b, diff (mean_a - mean_b)
def sign_test_p(n_pos: int, n_neg: int) -> float     # Decision 4 formula
def summarise_contrasts(C: pd.DataFrame, sd: pd.Series) -> pd.DataFrame
    # one row per (feature, pair): Decision 4 columns, sorted by pair, then is_signal desc, n_parents desc, |median_diff_sd| desc
def signal_sentence(S: pd.DataFrame, max_items: int = 3) -> str
```
`signal_sentence`: take `is_signal` rows sorted by n_parents desc, |median_diff_sd| desc, first `max_items`; per row `"{plo.feature_phrase(f)} is {higher|lower} in Batch {a} than in Batch {b} ({n} of {n} source images)"` (higher iff median_diff > 0; batch number only, e.g. "Batch 1"); sentence = `"Differences that hold between crops of the same source image assigned to different batches: " + "; ".join(items) + "."`; no signal rows -> `""`.

`scripts/within_parent.py` (no args): builds the 34-site pool (`labelled_sites`, `feature_table`, `pmdb.parents.parent_groups(include_test=False)`), writes `outputs/within_parent/contrasts.csv`, `summary.csv`, `within_parent.png` (heatmap 16 features x 3 pairs of `median_diff_sd`, `RdBu_r`, symmetric limits, each cell annotated `"{n_pos}/{n_parents}"` with `+` or `-` majority sign, title states diff = first batch minus second), and prints the `is_signal` rows and `signal_sentence`.

Tests `tests/test_within_parent.py`: synthetic 3-feature F, parents p1 {B1,B2}, p2 {B1,B2,B3}, p3 {B3,B3}: B1-B2 n_parents == 2, B1-B3 == 1, p3 contributes no rows; `sign_test_p(5, 0) == 0.0625`, `sign_test_p(2, 2) == 1.0`; `signal_sentence` of a frame with no signal == "" and of a signal frame contains none of `plo.FORBIDDEN` (case-insensitive) and contains "Batch 1".

Verify: `pytest -q tests/test_within_parent.py` passes; `python scripts/within_parent.py` writes the 3 files; `summary.csv` has 48 rows; n_parents per pair = 5 / 1 / 3 (Decision 4).

### Step 3 — centring, centred-fingerprint CV, menu CV, flags, selection

In `pmdb/batch_menu.py`:
```python
OPTIONS = ("fingerprint", "fingerprint_centred", "patch", "ensemble", "ensemble_centred")
N_COMPONENTS = {"fingerprint": 1, "fingerprint_centred": 1, "patch": 1, "ensemble": 2, "ensemble_centred": 2}

def centre_by_parent(F: pd.DataFrame, parent: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    # parent aligned to F.index; returns (F - F.groupby(parent.values).transform("mean"), singleton bool Series)

def fingerprint_centred_cv(Xc, y, groups, singleton: np.ndarray, fallback: pd.DataFrame) -> pd.DataFrame:
    # fallback = plo.fingerprint_cv(X, y, groups) (uncentred). For each group g with non-singleton test rows:
    #   tr = (groups != g) & ~singleton; m = fp.fit(Xc[tr], y[tr]); assert m.batches == list(BATCHES)
    #   rows = plo._pred_frame(fp.predict(m, Xc[(groups == g) & ~singleton]))
    # singleton rows = fallback rows. Return in Xc.index order, same columns as fallback.

def global_flags(correct, groups) -> np.ndarray   # mean(correct[groups != groups[i]]) > 0.5; none -> False

def menu_cv(D, patch_site, site_labels, groups, X, Xc, y, singleton) -> tuple[pd.DataFrame, list]:
    # df, sc = plo._run_cv(D, patch_site, site_labels, groups, X, y)
    # fallback = plo.fingerprint_cv(X, y, groups) (re-call; cheap) ; fpc = fingerprint_centred_cv(...)
    # add: fpc_call, fpc_ood, q_fpc_<b>, ensc_call (argmax plo.ensemble(p_patch, q_fpc)), p_ensc_<b>,
    #      agree_fpc, correct_fpc, correct_ensc, parent_code (= groups), singleton
    # rename-free: keep _run_cv's columns (fp_call, ens_call, agree, fp_ood, correct_patch/fp/ens)
    # add flag_<option> for each option per Decision 6 (with OOD override). return (df, sc)

def menu_summary(df, site_labels) -> dict:
    # per option: classification_metrics, rubric (flag rule), rubric_all_high, rubric_se, n_high,
    #   accuracy_when_high; plus "all_low": 1.0
def select_option(summary: dict) -> dict   # Decision 7 -> {"option", "confidence_mode", "rubric", "rule"}
```
`menu_summary` also stores `"caveat"`: `"Selection is the maximum of 5 LOPO rubric estimates on 34 sites from 13 parent images; the winning estimate is optimistically biased (winner's curse). Differences smaller than about one SE are not meaningful. The 3 held-out sites enter LOPO as ordinary labelled sites; nothing was tuned on them."` and `"singleton_rule"` (Decision 5 text).

Test-site prediction:
```python
def predict_test(D, patch_site, site_labels, groups, X, Xc, y, singleton, menu_df, sc_lopo,
                 Dh, patch_site_h, test_keys, test_codes, H, Hc, singleton_h, coords_h, selection) -> pd.DataFrame
```
- patch: `pm.heldout_scores(Dh, patch_site_h, len(test_keys), D, patch_site, site_labels, 3, groups, test_codes)` (code -1 for new parents).
- fingerprint: `plo.fingerprint_heldout(X, y, groups, H, test_codes)` -> (fph, expl). Centred: per test row, `tr = (groups != code) & ~singleton`, fit on `Xc[tr]`, predict `Hc.iloc[[h]]`; singleton_h rows copy fph.
- per option call/probability as Decision 6; flag: global -> `menu_df[correct_col].mean() > 0.5`; stratum -> `plo.heldout_flag(menu_df[correct_col], menu_df[agree_col], this_agree)`; OOD override; if `selection["confidence_mode"] == "all_low"` every flag low.
- `b3_med` = median over LOPO patch scores `sc_lopo` of label-2 sites of `(s.u[:, 2] > pm.ANOM_U).mean()`; explanation via `plo.make_explanation` exactly as `build_lopo_outputs` calls it, with `k` = selected call, `high` = selected flag.
- Columns exactly: `site, assigned, confidence, option, p_Batch_1, p_Batch_2, p_Batch_3, patch_call, fingerprint_call, fingerprint_centred_call, parent_id, explanation` (`parent_id` is for our audit; not shown to organisers).

Tests appended to `tests/test_batch_menu.py`:
- `test_centre_by_parent`: F 5 rows x 2 cols, parents `[a,a,b,b,c]` -> per-parent column means of Xc are 0; singleton `[F,F,F,F,T]`; row c all zeros; appending a 6th row to parent a changes rows 0-1 of Xc (test row joins the mean).
- `test_fingerprint_centred_cv_singletons`: 14 synthetic sites (4/batch in parent pairs + 2 singletons), monkeypatch `fp.fit` with a recorder: no recorded centred fit contains a singleton row or a row of the predicted group; singleton rows equal the fallback rows.
- `test_global_flags`: correct `[1,1,0,0,1]`, groups `[0,0,1,2,3]` -> `[False, False, True, True, False]` (site 0 sees [0,0,1]; site 2 sees [1,1,0,1]; site 3 sees [1,1,0,1]; site 4 sees [1,1,0,0] -> 0.5 not > 0.5).
- `test_select_option`: (a) rubrics {fp 1.2, fpc 1.3, patch 1.1, ens 1.3, ensc 1.0} -> fingerprint_centred, mode rule (tie with ens broken by components); (b) all rubrics <= 1.0 with accuracies {fp .5, fpc .45, patch .5, ens .55, ensc .4} -> option ensemble, mode all_low; (c) exact tie between two 1-component options -> earlier in OPTIONS.
- `test_predict_test_synthetic`: reuse the synthetic setup style of `tests/test_patch_lopo.py::test_build_lopo_outputs_synthetic` with 1 test site whose parent is labelled group 0 and 1 test site with code -1 and singleton; columns exact; confidence in {high, low}; p rows sum to 1; explanation passes the FORBIDDEN check.

Verify: `pytest -q tests/test_batch_menu.py` all pass; `pytest -q -m "not data"` passes.

### Step 4 — Modal: 34-site distances, `evaluate_menu`, modes `menu` and `test`; run the menu

`modal_patch_mil.py`:
- `embed_sites`: `cache_root = {"Batch_heldout": "/data/heldout", "Batch_test": "/data/test"}.get(batch, "/data")`.
- `compute_distances(labelled, heldout: list[str], tag, heldout_batch: str = "Batch_heldout", out_name: str = "distances.npz")`: load test embeddings with `heldout_batch`; save to `/out/{tag}/{out_name}`; if `heldout` is empty, `Dh = np.zeros((0, len(labelled)))`, `patch_site_h = np.zeros(0, int)`, `coords_h = np.zeros((0, 2), int)`, `bse_hf_h = np.zeros(0)`. Defaults reproduce current behaviour.
- New `@app.function(volumes={"/out": out_vol}, cpu=4.0, memory=8192, timeout=3600) def evaluate_menu(tag: str, dist_name: str, labels: list[dict], parents: list[dict], fp_lab: list[dict], fp_test: list[dict], test_batch: str, selection: dict | None) -> dict`: loads the npz; builds keys/labels from `labels` (aligned to `site_batch`/`site_id` order, assert); groups = factorize of labelled parent_id; pool = labelled features + test features; `Xc, singleton` via `centre_by_parent` over the pool, then split; runs `menu_cv`, `menu_summary`; `selection = select_option(summary)` if None; if test sites present -> `predict_test`. Returns `{"summary", "selection", "menu_predictions" (records), "test_predictions" (records or []), "elapsed_s"}`.
- `main` modes add `menu` and `test`:
  - shared local prep: `lab = labelled_sites()`; `labelled34 = [(b, s) for b, s in zip(lab.batch, lab.site)]`; `parents = write_parent_groups()`; `fp_lab = feature_table(labelled34).reset_index()`.
  - `menu`: `compute_distances.remote(labelled34, [], "full", "Batch_test", "distances_r3.npz")`; `evaluate_menu.remote("full", "distances_r3.npz", lab records, parents, fp_lab, [], "Batch_test", None)`; write `outputs/menu/menu_evaluation.json` (`{"summary", "selection", "n_sites": 34}`), `outputs/menu/menu_predictions.csv`, `outputs/menu/selection.json` (`selection` + `"selected_on": "LOPO, 34 labelled sites, r3 run"`); print per-option accuracy / rubric / SE and the selection.
  - `test`: `test_sites` from `cache_test/half/manifest.csv` (site as str; SystemExit if missing); `selection = json.load(outputs/menu/selection.json)` (SystemExit if missing: "run --mode menu first"); `fp_test = pd.read_csv("outputs/test/features.csv", dtype={"batch": str, "site": str})` (written by `modal_test_prep.py`, Step 5; SystemExit naming `modal run modal_test_prep.py::main` if the file is missing or any test site is absent from it); `embed_sites.remote([("Batch_test", s) for s in test_sites], "full", True)`; `compute_distances.remote(labelled34, test_sites, "full", "Batch_test", "distances_r3.npz")`; `evaluate_menu.remote(..., fp_test, "Batch_test", selection)`. Locally: strip `" (see outputs/patch_mil/figures/heldout_{site}.png)"` from each explanation, append `" " + signal_sentence(pd.read_csv("outputs/within_parent/summary.csv"))` when non-empty; write `outputs/test/menu_evaluation.json`, `outputs/test/final_predictions.csv`, and `outputs/test/submission.md` (markdown table `site | batch | confidence | explanation`, batch as "Batch 1"); print each `site -> assigned (confidence)`.
- Update module docstring run lines with `--mode menu` and `--mode test`.

Then run: `modal run modal_patch_mil.py --mode menu`.

Verify: `python -c "import modal_patch_mil"` imports; menu run completes; `outputs/menu/menu_predictions.csv` has 34 rows; `outputs/menu/selection.json` exists; `git diff --stat outputs/patch_mil/` empty. Sanity check (not a gate): the `fingerprint` option's LOPO correctness on the 31 original sites should equal `correct_fp` in `outputs/patch_mil/lopo_predictions.csv` (fold lopo) for sites whose parent did not gain a held-out member (h2048, h2316, h2088 parents may differ because their training folds changed); report the count of mismatches outside those parents — expected 0, otherwise escalate.

### Step 5 — Modal test-site preprocessing (`modal_test_prep.py`), runbook script, rehearsal

**5a. `modal_test_prep.py`** (repo root, new; style of `modal_app.py`: module docstring with run lines, top level imports only `modal` + stdlib, all `pmdb`/numpy/pandas imports inside functions or entrypoints):

```python
APP_NAME = "pmdb-test-prep"
TEST_ROOT = "/data/test"            # volume path /test
RAW_ROOT = "/data/test/raw"         # list_sites data_root; holds Batch_test/
TEST_BATCH = "Batch_test"
image = (modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4", "scipy==1.14.1", "scikit-image==0.25.2", "pandas==2.1.4", "Pillow==12.3.0",
                 "tifffile==2025.5.10", "imagecodecs==2025.3.30", "matplotlib==3.10.7")     # Decision 15
    .env({"PMDB_CACHE": TEST_ROOT, "MPLBACKEND": "Agg"})
    .add_local_python_source("pmdb")
    .add_local_file(...)  # the 4 scripts -> /root/scripts/<name>.py and the 4 reference files of Decision 14
)
data_vol = modal.Volume.from_name("pmdb-data")          # writable (no .read_only())
app = modal.App(APP_NAME, image=image)

@app.function(volumes={"/data": data_vol}, cpu=8.0, memory=32768, timeout=3600)
def prep_test_sites(expected_sites: list[str]) -> dict:
    """Returns {"files": {<local repo-relative path>: bytes}, "sites": [...], "timings_s": {stage: s}}."""
```
`prep_test_sites` body, in this order (each stage timed; any failure raises, nothing is returned):
1. `os.environ["PMDB_CACHE"] = TEST_ROOT`; `sys.path.insert(0, "/root/scripts")`; `man = pmdb.io.list_sites(RAW_ROOT)`; assert `man.batch` is all `TEST_BATCH` and `sorted(man.site.astype(str)) == sorted(expected_sites)`, else `RuntimeError("volume /test/raw has sites {...}, expected {...}; run: modal volume rm -r pmdb-data /test/raw")`. `n = min(len(expected_sites), 6)`.
2. `build_cache.build_cache(data_root=Path(RAW_ROOT), cache_root=Path(TEST_ROOT), outputs_dir=Path(TEST_ROOT), force=True)`.
3. `ref = H.load_reference(Path("/root/ref"), "affine2")`; `anchors, hists = build_harmonised._collect_anchors(Path(TEST_ROOT))`; `(Path(TEST_ROOT)/"harmonised").mkdir(parents=True, exist_ok=True)`; `anchors.to_csv(f"{TEST_ROOT}/harmonised/anchors.csv", index=False)`; `build_harmonised._fit_all(Path(TEST_ROOT), anchors, hists, ref, ["affine2"])`.
4. `sys.argv = ["build_clean.py", "--data-root", RAW_ROOT, "--out", f"{TEST_ROOT}/clean", "--targets", "/root/ref/clean/targets.json", "--labelled-summary", "/root/ref/clean/summary.csv", "--workers", str(n)]`; `build_clean.main()`.
5. `import run_kpis` (only now, after step 1 set `PMDB_CACHE`, because `run_kpis.MANIFEST` is bound at import); `rc = run_kpis.main(["--out-dir", f"{TEST_ROOT}/kpis", "--no-overlays", "--jobs", str(n)])`; `rc != 0` -> `RuntimeError` including the text of `{TEST_ROOT}/kpis/run_attempt.json` if present.
6. `X = fp.read_feature_inputs(f"{TEST_ROOT}/kpis/curves.csv", f"{TEST_ROOT}/kpis/tile_kpis.csv")`; assert no NaN and index == the manifest's `(batch, site)` set; `X.reset_index().to_csv(f"{TEST_ROOT}/features.csv", index=False)`.
7. `data_vol.commit()`; read back the pulled files of the Decision 10 table into `files` (keys = local paths: `cache_test/half/manifest.csv`, `cache_test/harmonised/anchors.csv`, `cache_test/harmonised/affine2/{luts.npz,params.csv,reference.json}`, `outputs/test/raw_intensity_stats.csv`, `outputs/clean_test/{summary.csv,material_thresholds.json}`, `outputs/clean_test/Batch_test/<site>/params.json`, `outputs/test/kpis/{site_kpis,tile_kpis,curves,sensitivity}.csv`, `outputs/test/kpis/run_log.json`, `outputs/test/features.csv`).

```python
@app.function(volumes={"/data": data_vol}, cpu=1.0, memory=4096, timeout=600)
def rehearsal_check(test_site: str, heldout_site: str) -> dict:
    """{"half_equal", "half_n_diff", "half_max_abs_diff", "lut_equal"}: /data/test/half/Batch_test__{test_site}.npz
    vs /data/heldout/half/Batch_heldout__{heldout_site}.npz ("image" arrays, shape must match), and
    /data/test/harmonised/affine2/luts.npz[Batch_test__{test_site}] vs /data/heldout/harmonised/affine2/luts.npz[Batch_heldout__{heldout_site}]."""
```

Local entrypoints:
- `@app.local_entrypoint() def main():` (run as `modal run modal_test_prep.py::main`) — `man = pmdb.io.list_sites(ROOT / "data_test")` (validates filenames/detectors locally before any upload); SystemExit if any batch other than `Batch_test`. Upload ONLY those sites' TIFFs: `with data_vol.batch_upload(force=True) as up: up.put_file(str(p), f"/test/raw/Batch_test/{p.name}")` for every `path_bse/path_inlens/path_se_type`. `res = prep_test_sites.remote(sites)`. Write every `res["files"]` item atomically to `ROOT / key` (mkdir parents; same tmp+replace pattern as `modal_patch_mil._atomic_write`); before writing `cache_test/half/manifest.csv`, set its `path` column to `cache_test/half/{batch}__{site}.npz`. Print sites, per-stage timings, and per site `height`, `BSE_grey_step` and `pmdb.parents.parent_id(...)` from the pulled summary + manifest.
- `@app.local_entrypoint() def check(test_site: str = "fn0mhxef", heldout_site: str = "fn0mhxef"):` (rehearsal only) — `r = rehearsal_check.remote(...)`, then local comparisons, print one PASS/FAIL line per check, `SystemExit(1)` on any FAIL. Checks and tolerances (decided):
  | check | reference | pass rule |
  |---|---|---|
  | half cache | volume `/heldout/half` | `half_equal` (exact, uint8). Pre-authorised: if not exact but `half_max_abs_diff <= 1` and `half_n_diff / size < 1e-3`, PASS with a printed warning |
  | affine2 LUT | volume `/heldout/harmonised/affine2/luts.npz` | `lut_equal` (exact) |
  | parent key | `pmdb.parents.parent_groups()` row for `(Batch_heldout, heldout_site)` | `height`, `se_detector`, `bse_grey_step`, `parent_id` exactly equal to the `(Batch_test, test_site)` row (expected `h2048_ETD_s2`) |
  | fingerprint features | `outputs/fingerprint/heldout_features.csv` row | for each of the 16 features, `abs(test - ref) <= 0.05 * SD_f`, `SD_f` = std (ddof=1) of that feature over the 31 rows of `outputs/fingerprint/features.csv`; print `max abs(diff)/SD_f` and the feature it occurs on |
  Rationale for 0.05 SD: KPIs come from a different numeric stack than the held-out run (Decision 15), so bit-equality is not expected; 5% of a between-site SD is far below anything that moves a nearest-centre call, and anything larger means the stacks disagree materially.

**5b. `scripts/score_test_sites.sh`** (`#!/usr/bin/env bash`, `set -euo pipefail`, `cd "$(dirname "$0")/.."`, chmod +x):
```bash
ls data_test/Batch_test/img_*_BSE.tif >/dev/null 2>&1 || { echo "copy organiser TIFFs to data_test/Batch_test/img_<site>_<BSE|Inlens|ETD|SE>.tif"; exit 1; }
test -f outputs/menu/selection.json || { echo "outputs/menu/selection.json missing: the menu must be frozen (modal run modal_patch_mil.py --mode menu) before test images"; exit 1; }
modal run modal_test_prep.py::main
modal run modal_patch_mil.py --mode test
```
Update `modal_patch_mil.py`'s docstring with the `--mode test` prerequisite (`modal run modal_test_prep.py::main`).

Verify 5a/5b statically: `python -c "import modal_test_prep"` imports; `bash -n scripts/score_test_sites.sh` passes.

**5c. Rehearsal (mandatory, then clean up)**: `mkdir -p data_test/Batch_test && cp data_heldout/Batch_heldout/img_fn0mhxef_*.tif data_test/Batch_test/`; `bash scripts/score_test_sites.sh`; then `modal run modal_test_prep.py::check --test-site fn0mhxef --heldout-site fn0mhxef`. Expected: prep completes; all 4 checks PASS; `outputs/test/final_predictions.csv` has 1 row with `parent_id == h2048_ETD_s2` (its labelled twin is excluded from bank/training — no leak); explanation has no FORBIDDEN word and no figure path. Record in the report: per-stage prep timings, total wall time of each `modal run`, the max feature diff/SD, and the predicted batch for the rehearsal site (truth Batch_1; informational, not a gate). Then clean up: delete `data_test/`, `cache_test/`, `outputs/test/`, `outputs/clean_test/`; `modal volume rm -r pmdb-data /test`; `modal volume rm pmdb-patch-out /full/emb/Batch_test__fn0mhxef.npz`; re-run `modal run modal_patch_mil.py --mode menu` only if `outputs/menu/*` changed (it should not; `test` writes to outputs/test/).

Verify: rehearsal expectations above; after cleanup `git status --short` shows no data_test/cache_test/outputs/test/outputs/clean_test paths and `modal volume ls pmdb-data /` lists only `half`, `harmonised`, `heldout`.

**Expected runtime/cost (test day, 4-6 sites)**: upload ~60 MB/site (~1-3 min total); prep container ~8-12 min (cache <1 min, LUTs <1 min, clean ~2-4 min with one worker per site, KPIs ~3-5 min); first run also builds the prep image (~3-5 min; the rehearsal pays this, so test day reuses the cached image as long as the image inputs are unchanged); `--mode test` (T4 embedding of 4-6 sites + 34-site distances + menu evaluation) ~10-15 min. Total ~25-35 min wall. Cost: prep 8 CPU + 32 GiB for ~0.2 h is roughly $0.10-0.30; GPU + evaluation under $1; whole chain well under $2.

### Step 6 — docs, AGENTS.md, commit

`docs/patch_mil.md`, new section **Feedback round 1 (2026-10-04)** (after the lopo sections):
- Submission (fingerprint classifier of record, `outputs/fingerprint/heldout_predictions.csv`): 3e122cbj Batch_1, fn0mhxef Batch_3, xrv9xvzb Batch_2; truths Batch_2 / Batch_1 / Batch_3; 0/3 correct. The majority batch of the labelled siblings would have been right on 1/3, so parents are deliberately split across batches. Lesson: the designed batch signal is what differs between crops of the same parent; the 3 truths are now labelled sites (`outputs/heldout_labels.csv`).
- **Within-parent contrasts**: method (Decision 4), n_parents per pair, the `is_signal` rows as a table (feature phrase, pair, k/n, sign p, median diff in SD), figure `outputs/within_parent/within_parent.png`; plain-language summary = one line per `is_signal` row in the `signal_sentence` item wording; if none, write "No feature x pair met the pre-registered criterion (>= 3 source images, all the same sign)" and list the top 3 by consistency then n_parents.
- **Parent-centred fingerprint**: definition, singleton rule with rationale.
- **Model menu (LOPO, 34 sites)**: table option | accuracy | balanced acc | rubric (flag rule) | rubric SE | rubric all-high | n high; all-low row = 1.000; selection + mode; caveat text verbatim.
- **Test day runbook**: the script, what each line does, storage layout (Decision 10 table), compute placement (Decision 11: everything on Modal; `modal_test_prep.py` reuses the build scripts, labelled reference artefacts shipped in the image, package pins of the labelled KPI run), rehearsal result (the 4 checks with the measured max feature diff/SD), expected runtime/cost (Step 5), outputs (`outputs/test/final_predictions.csv`, `submission.md`), and "selection is frozen in outputs/menu/selection.json; do not re-run --mode menu after test images arrive".

`AGENTS.md`, "Held-back Test Sites" section: append `Truths released by the organisers on 2026-10-04 are in outputs/heldout_labels.csv; since then the patch-MIL menu (docs/patch_mil.md) uses these 3 sites as labelled sites. New organiser test images go to data_test/Batch_test/ (see docs/patch_mil.md, Test day runbook).`

Run `pytest -q -m "not data"` and `pytest -q -m data`.

Commit on `worktree-patch-anomaly-mil`: `outputs/heldout_labels.csv pmdb/batch_menu.py pmdb/within_parent.py pmdb/parents.py scripts/within_parent.py scripts/score_test_sites.sh modal_patch_mil.py modal_test_prep.py tests/test_batch_menu.py tests/test_within_parent.py outputs/within_parent/ outputs/menu/ outputs/parent_groups.csv docs/patch_mil.md AGENTS.md .gitignore .claude/plans/patch-mil-r3.md`; message `feat(patch-mil): r3 — held-out truths as labels, within-parent contrasts, parent-centred fingerprint, LOPO model menu, test-day runbook`. No push.

Verify: both pytest runs pass (report counts); `git status --short` shows none of the committed paths.

## 5. Expected surprises

- **`fp.fit` fails on centred training** (e.g. a batch with too few non-singleton rows in some fold, or a feature with zero spread): pre-authorised to let that fold's centred predictions fall back to the uncentred rows for that fold, recorded per site in a `fpc_fallback` boolean column and counted in the summary. Anything else: escalate.
- **The LOPO flag rule makes every option all-low or the best option <= 1.0**: intended outcome (Decision 7 all_low mode), not a bug.
- **fingerprint option mismatch outside parents h2048/h2316/h2088 in the Step 4 sanity check**: escalate (ordering/key bug).
- **`load_site` rejects batch `Batch_test` or needs files beyond `half/` + `harmonised/affine2/luts.npz`**: pre-authorised to have `prep_test_sites` write the additionally required file under `/data/test/...` mirroring `/heldout/...`; do not rename the batch key.
- **A pinned package fails to install on python 3.10 in the prep image**: STOP and escalate (the pins are what makes test features comparable to the labelled rows; do not silently float a version).
- **Importing a shipped script in the container fails** (e.g. a module-level path assumption under `/root`): pre-authorised to fix it inside `modal_test_prep.py` only (env var, `sys.path`, `chdir`, an extra `add_local_file` of a repo file the script reads); the script itself is not edited. If the fix would need a script edit, escalate.
- **`build_clean` OOM-kills a worker** (32 GiB for up to 6 full-res sites): pre-authorised to raise `memory` to 65536 and/or cap workers at 3; log it.
- **Rehearsal half-cache or LUT check fails beyond the pre-authorised tolerance, or the parent key differs**: STOP and escalate (Modal preprocessing is not reproducing the held-out pipeline; test predictions would not be trustworthy).
- **Rehearsal feature check fails (> 0.05 SD on some feature)**: STOP and escalate with the per-feature diffs; the likely cause is the numeric-stack difference of Decision 15, and the dispatcher decides whether to recompute the 3 held-out KPI rows on Modal (they are labelled training rows in r3).
- **A test site id parses as a number** (all digits, or digits-`e`-digits; `build_harmonised._sites` reads the manifest without `dtype=str`): escalate before running.
- **A test site's KPI run fails** (`run_kpis.main` returns non-zero, no fill values by design): `prep_test_sites` raises, nothing is pulled back; STOP, the dispatcher decides (the fingerprint refuses incomplete features).
- **`Volume.batch_upload` / `put_file` behaves differently in the installed modal client (1.6.0 verified to have `Volume.batch_upload`)**: pre-authorised to upload with `modal volume put --force pmdb-data data_test/Batch_test /test/raw/Batch_test` in `scripts/score_test_sites.sh` before `modal_test_prep.py::main` instead, keeping the remote site-set assertion.
- **New test images have a different detector set (e.g. SE instead of ETD)**: no change needed (parent key includes detector; patch uses BSE+Inlens only).
- **Modal failures**: read logs and Modal docs before relaunching (memory rule).

## 6. Done criteria

- `outputs/heldout_labels.csv`, `outputs/within_parent/{contrasts.csv,summary.csv,within_parent.png}`, `outputs/menu/{menu_evaluation.json,menu_predictions.csv (34 rows),selection.json}` exist and are committed.
- `modal_test_prep.py` + `scripts/score_test_sites.sh` rehearsed end-to-end on Modal with a copied held-out site (fn0mhxef): half cache and LUT equal, parent key equal, features within 0.05 SD; then local and volume `/test` cleaned up.
- Nothing from `data/` or `data_heldout/` uploaded except the rehearsal copy of fn0mhxef via `data_test/`; `/half`, `/harmonised`, `/heldout` on `pmdb-data` untouched.
- `outputs/patch_mil/*` unchanged; `/out/full/distances.npz` untouched.
- docs/patch_mil.md has the Feedback round 1 sections; AGENTS.md updated.
- `pytest -q -m "not data"` and `pytest -q -m data` pass; committed, not pushed.

## 7. Revision log

### Round 1 (2026-10-04) — trigger: user decision relayed by dispatcher ("ALL test-day compute on Modal")

1. *All test-day compute on Modal, including per-site CPU preprocessing* — accepted: Decision 11 rewritten (replaces the local-preprocessing exception); new Decisions 14 (reuse scripts, ship reference artefacts) and 15 (package pins); Step 5 rewritten as 5a `modal_test_prep.py` / 5b script / 5c rehearsal.
2. *Upload only the new raw test TIFFs, plus the labelled-reference artefacts the build scripts need, identified exactly* — accepted: entrypoint `main` uploads only `data_test/Batch_test` TIFFs to `pmdb-data:/test/raw/Batch_test/`. Reference artefacts identified by reading the scripts: `cache/harmonised/affine2/reference.json` (LUT fit; `anchors.csv` is only needed to rebuild the reference, which we load instead), `outputs/clean/targets.json` (`--targets`), `outputs/clean/summary.csv` (`--labelled-summary`), `docs/kpis/kpi_catalogue.csv` (`pmdb.kpis.CATALOGUE_PATH`). Mechanism differs from the suggestion: they are shipped with `add_local_file` into the image rather than uploaded to the volume, so they are versioned with the code and cannot go stale; the effect (available to the container, never labelled raw data) is the same.
3. *Modal functions in an image-included module, calling pmdb functions; ship scripts whose logic is not in pmdb* — accepted: `modal_test_prep.py` (own app/image; functions defined in the entry module). `pmdb.harmonise` and `pmdb.fingerprint` are called directly; `build_cache.build_cache`, `build_harmonised._collect_anchors/_fit_all`, `build_clean.main` (flatten/material_flags logic lives only there) and `run_kpis.main` (table/provenance logic lives only there) are imported from scripts shipped with `add_local_file`. Reused `modal_app.py`'s pins and catalogue placement; did not reuse `modal_app.run_site` itself because it returns site-level KPIs only, while the fingerprint needs `curves.csv` + `tile_kpis.csv`.
4. *Pull back small CSV/JSON outputs into cache_test/, outputs/clean_test/, outputs/test/kpis/* — accepted: the Decision 10 table lists every pulled file; half npz, clean TIFFs, previews and the HTML QC report stay on the volume. Fingerprint features are also computed in the container (`outputs/test/features.csv`), and Step 4's `test` mode now reads that file instead of rebuilding features locally. Step 1's parents.py missing-summary message now names `modal run modal_test_prep.py::main`.
5. *Never upload data/ or other raw files* — accepted: Out of scope + Done criteria; remote assertion that `/test/raw` holds exactly the expected sites.
6. *Rehearsal mandatory, compare against held-out artefacts with stated tolerance, then clean up* — accepted: Step 5c + `check` entrypoint; tolerances: half cache exact (pre-authorised <= 1 DN on < 0.1% of pixels), LUT exact, parent key exact, features <= 0.05 x labelled SD per feature. Finding while grounding: held-out KPIs were computed with a newer local stack than the labelled KPIs (which match the Modal pins), so the feature check also measures that.
7. *One command, wrapped in scripts/score_test_sites.sh* — accepted: `modal run modal_test_prep.py::main` then `modal run modal_patch_mil.py --mode test`. `--sites` dropped: the entrypoint takes every site in `data_test/Batch_test/`, and the container asserts the volume holds exactly those, which also catches rehearsal leftovers.
8. *Keep all other r3 decisions; note runtime/cost* — accepted: Decisions 1-9, 12, 13 unchanged; the obsolete labelled-LUT guard and the `modal volume put cache_test/...` lines are gone because labelled LUTs are no longer refitted and the prep writes straight to the volume. Runtime/cost estimate at the end of Step 5 (~25-35 min wall, well under $2).
