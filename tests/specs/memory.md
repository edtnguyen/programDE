Step 1 complete:
- Inspected current CRT sampler outputs and consumers.
- Bernoulli sampler outputs (indptr, indices) from `crt_index_sampler_fast_numba`.
- Downstream CRT uses these via `_sample_crt_indices` -> `crt_pvals_for_gene`/`crt_betas_for_gene`.
- Integration points: `run_one_gene_union_crt`, `compute_gene_null_pvals`, `compute_guide_set_null_pvals`, `ntc_groups.py`, `diagnostics.py`, `global_null_diagnostics.py`, and `_sample_crt_indices` in `pipeline_helpers.py`.

Step 2 complete:
- Added `samplers.py` with `bernoulli_index_sampler` and `stratified_permutation_sampler` returning (indptr, indices).
- `prepare_crt_inputs` now stores `covar_df_raw` and `batch_raw` (via `batch_key`) in `CRTInputs`.

Step 3 complete:
- Threaded `resampling_method`/`resampling_kwargs` through CRT runners and NTC helpers.
- `_sample_crt_indices` now dispatches between Bernoulli and stratified-permutation samplers.
- README updated with stratified-permutation usage and batch-key note.

Step 4 complete:
- Added stratified-permutation sampler tests (stratum counts, reproducibility, pipeline uniformity, default equivalence).
- Updated NTC-parallel test for new kwargs.
- Ran full pytest suite: 71 passed (cache warnings only).

S-CRT tests Step 1 complete:
- Added `tests/test_strata.py` to validate non-degenerate strata and fallback when p_hat collapses.

S-CRT tests Step 2 complete:
- Added within-stratum uniformity check to `tests/test_stratified_perm_sampler.py`.

S-CRT tests Step 3 complete:
- Added integration checks in `tests/test_end_to_end_stratified_perm.py` for batch usage and bin counts from propensity.

S-CRT tests Step 4 complete:
- Added global-null stratified-permutation calibration test and oracle-vs-fit comparison in `tests/test_end_to_end_stratified_perm.py`.

S-CRT tests Step 5 complete:
- Added burden-binning test stub marked xfail when `burden_bin` is not implemented.

S-CRT tests Step 7 complete:
- Added order-independence determinism test in `tests/test_end_to_end_stratified_perm.py`.

S-CRT tests Step 8 complete:
- Added new pytest files `tests/test_strata.py` and `tests/test_end_to_end_stratified_perm.py` (plus expanded sampler tests).
- Ran pytest: 79 passed, 1 xfailed (burden-bin stub), cache warnings only.

Burden-bin CRT implementation:
- Added `compute_bins` helper and burden-bin support in `stratified_permutation_sampler`.
- `_sample_crt_indices` now accepts burden kwargs and pulls raw burden column from covariate DataFrame.
- Extended sampler and strata tests for burden counts/strata; updated end-to-end burden improvement test (xfail if no improvement).
- Updated README snippet with burden-bin options.
- Ran pytest: 82 passed, cache warnings only.

Burden helper doc:
- Added `compute_guide_burden` helper with docstring and exported it.
- Added README snippet showing how to compute/store burden covariate.
- Added `add_burden_covariate` wrapper for adata + updated README snippet to use it.

NTC empirical null:
- Added `src/sceptre/ntc_null.py` with NTC pseudo-gene sampling, unit summaries, bin matching, empirical p-values, and batch-meta combining.
- Threaded `null_method`/`null_kwargs` into `run_one_gene_union_crt` and `run_all_genes_union_crt`.
- Added tests in `tests/test_ntc_empirical_null.py` for denom correctness, empirical matching, global null uniformity, power sanity, and default CRT compatibility.
- README updated with NTC empirical null usage snippet.
- Made burden-binning improvement test deterministic (no xfail) using a controlled synthetic scenario.
- Ran pytest: 87 passed, cache warnings only.

NTC empirical null QQ (step 1):
- Added `src/sceptre/ntc_qq.py` with expected quantiles, QQ coords, bootstrap envelope, and plotting helpers.

NTC empirical null QQ (step 2):
- Added NTC cross-fit helpers (`split_ntc_units`, cross-fit p-value functions) and wired cross-fit outputs into `run_ntc_empirical_null`.
- Added `qq_crossfit` flag to `run_all_genes_union_crt` and exposed NTC cross-fit outputs in the returned dict.
- Exported new NTC QQ and cross-fit helpers via `src/sceptre/__init__.py`.

NTC empirical null QQ (step 3):
- Added `tests/test_ntc_empirical_qq.py` (QQ utilities + cross-fit calibration under global null).
- Added README section for NTC empirical-null QQ plots (cross-fit usage and notes).

NTC empirical null pbar matching:
- Added optional pbar matching (`mean(logit(p_hat))`) in NTC empirical-null binning and cross-fit helpers.
- Implemented pbar computation per unit (real + NTC, per batch if meta) when `matching.use_pbar=True`.

S-CRT U-test Step 1 complete:
- Added test_stat/test_stat_kwargs plumbing across pipeline, null-pvals helpers, and NTC group helpers.
- Added rank caching + U-test stats (rank-biserial) with sparse rank-sum nulls in pipeline_helpers.
- Extended CRTInputs to store raw usage + rank metadata for utest.

S-CRT U-test Step 2 complete:
- Added U-test unit tests (rank-biserial vs Mann–Whitney U, sparse rank-sum correctness, OLS compatibility).
- Added end-to-end stratified-permutation null uniformity test for utest (including NTC groups) and reproducibility checks.
- Updated README with S-CRT U-test usage snippet and stat output notes.

S-CRT U-test Step 3 complete:
- Ran full pytest suite via sc_dl env: 97 passed (warnings only: pandas categorical dtype deprecation, sklearn logistic warnings, and a RuntimeWarning for rank-biserial divide-by-zero in a utest reproducibility run).

S-CRT U-test Step 3 update:
- Re-ran full pytest after masking divide-by-zero in rank-biserial; 97 passed (warnings only: pandas categorical dtype deprecation, sklearn logistic warnings).

