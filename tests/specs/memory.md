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
