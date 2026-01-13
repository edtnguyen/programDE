# Codex spec: Add stratified / permutation CRT sampler (keep current Bernoulli index sampler intact)

Context: This repo implements a SCEPTRE-style **union CRT** for testing target gene → cNMF program usage (CLR) effects. The existing CRT uses a propensity model and an efficient Bernoulli “index sampler” to generate null resamples. We want to add an alternative **stratified/permutation resampler** that is more robust under high MOI / dependence, while leaving the existing sampler unchanged and still the default. See the current method overview + API in README. :contentReference[oaicite:0]{index=0}

---

## 0) Design goals

1. **Backward compatible**: default behavior unchanged; existing APIs and outputs remain valid.
2. **Drop-in integration**: new sampler selectable via a new `resampling_method` argument in the same places `B`, `n_jobs`, and skew-normal args are set (e.g., `run_all_genes_union_crt`, NTC grouped CRT helpers).
3. **Same downstream computation**: the null betas (`beta_null`) must still be produced using the same OLS-summary machinery (no repeated lstsq).
4. **Deterministic under parallelism**: resamples reproducible given `seed0` + `gene_id` (or unit index), independent of job scheduling.
5. **Uses batch labels from covariates**: user stores `batch` in `adata.obsm["covar"]["batch"]`. The stratified-permutation sampler should optionally stratify by `(batch, propensity_bin)`.

---

## 1) Public API changes (minimal)

### 1.1 `run_all_genes_union_crt` (and any gene-level driver)
Add optional args:

- `resampling_method: str = "bernoulli_index"`
  - allowed: `"bernoulli_index"` (current default), `"stratified_perm"`
- `resampling_kwargs: dict | None = None`
  - for stratified sampler:
    - `n_bins: int = 20` (propensity quantile bins)
    - `stratify_by_batch: bool = True`
    - `batch_key: str = "batch"` (column in covar DF before encoding)
    - `min_stratum_size: int = 2` (drop tiny strata)
- Thread/parallel args remain unchanged.

All outputs (`pvals_df`, optional `pvals_raw_df`, `pvals_skew_df`, `betas_df`, etc.) should remain identical in schema and meaning.

### 1.2 NTC-group helper functions
Any function that runs CRT over NTC groups should accept and pass through the same sampler options:
- `crt_pvals_for_ntc_groups_ensemble`
- `crt_pvals_for_ntc_groups_ensemble_skew`
- `compute_ntc_group_null_pvals_parallel`
(and any other wrapper that calls the per-unit CRT runner)

Add these args (same defaults):
- `resampling_method="bernoulli_index"`
- `resampling_kwargs=None`

---

## 2) Inputs plumbing: ensure batch labels are accessible

`prepare_crt_inputs` currently one-hot encodes / z-scores covariates and adds intercept (per README). :contentReference[oaicite:1]{index=1}  
The stratified-permutation sampler needs raw batch labels.

Modify `prepare_crt_inputs` to ALSO store:
- `inputs["covar_df_raw"]` as the original covariate DataFrame (before encoding), if the input `covar` is a DataFrame.
- `inputs["batch_raw"]` = `covar_df_raw[batch_key].to_numpy()` if present, else `None`.

Rules:
- Do not change existing `inputs["C"]` (encoded/z-scored design matrix).
- If `covar` was an ndarray, set `covar_df_raw=None`, `batch_raw=None`.
- Document this in code comments only (README update optional).

---

## 3) New sampler implementation

### 3.1 Create a new module (preferred) or extend existing sampler code
Add `src/sceptre/samplers.py` (or similar) containing:

- `bernoulli_index_sampler(...)`  **(existing behavior; may be moved/wrapped, but must remain identical)**
- `stratified_permutation_sampler(...)` **(new)**

Both must return the SAME “null design representation” that downstream code already consumes to compute `beta_null` efficiently.

**Action for Codex:** Inspect current CRT null computation in `src/sceptre/crt.py` and identify the exact expected sampler output type:
- Option A: `treated_resamples_per_cell: list[np.ndarray]` (per cell, which resample indices b are treated)
- Option B: `(row_idx, col_idx)` edge list for treated cell/resample pairs
- Option C: sparse `X_tilde` matrix (N×B) with nnz at treated pairs

Implement stratified sampler to produce exactly the same type as the Bernoulli sampler returns today, so `crt.py` can remain mostly unchanged.

### 3.2 Stratified-permutation CRT logic (definition)
Given:
- observed union indicator `x_obs ∈ {0,1}^N`
- fitted propensities `p_hat` from logistic model
- optional `batch_raw` labels

Build strata as:
1) Bin `p_hat` into `n_bins` quantile bins:
   - edges = quantiles of `p_hat` at `linspace(0,1,n_bins+1)`
   - use `np.unique(edges)`; if edges collapse, fall back to 1 stratum
   - `bin_id[i] = searchsorted(edges, p_hat[i], side="right") - 1` clipped to `[0, n_bins-1]`

2) If `stratify_by_batch=True` and `batch_raw is not None`:
   - factorize batch to integer labels (stable ordering)
   - `stratum_id[i] = batch_id[i] * n_bins + bin_id[i]`
   else:
   - `stratum_id[i] = bin_id[i]`

3) Collect cell indices per stratum:
   - drop strata with size < `min_stratum_size` by merging them into a “misc” stratum OR simply keep them but note that if `m_s` is 0 or size is 1 it does nothing.
   - Must be deterministic given `seed`.

For each stratum `s`:
- `idx_s = { i : stratum_id[i]=s }`
- `m_s = sum_{i in idx_s} x_obs[i]`  (observed treated count in that stratum)

For each resample `b=1..B`, generate `x_tilde^(b)` by:
- in each stratum `s`, choose exactly `m_s` indices uniformly **without replacement** from `idx_s`
- set those chosen cells treated in resample b, others control

This ensures, for every resample:
- total treated count matches observed: `sum_i x_tilde_i^(b) = sum_i x_obs[i]`
- treated counts match within each stratum: `sum_{i in s} x_tilde_i^(b) = m_s`

### 3.3 Efficient construction (implementation notes)
Use one of these representations (match what existing code expects):

**If existing code uses per-cell treated resample lists** (S_i):
- Avoid building `list-of-empty-arrays` by incremental appends if too slow.
- Recommended:
  - build `treated_idx_by_resample: list[np.ndarray]` length B
  - then invert to per-cell lists only if required by downstream C++/Python
  - finally convert each to `np.ndarray(dtype=int32)`.

**If existing code accepts an edge list** `(rows, cols)`:
- For each resample b, concatenate chosen treated indices across strata into `treated_idx`
- append to `rows.extend(treated_idx)` and `cols.extend([b]*len(treated_idx))`
- at the end, `rows=np.asarray(rows,int32)`, `cols=np.asarray(cols,int32)`

**If existing code accepts a sparse matrix X_tilde**:
- build COO from `(rows, cols)` then convert to CSR/CSC as needed

Performance target:
- Complexity proportional to `B * treated_count` (not `B*N`)
- This is typically OK because union prevalence per gene is small.

Determinism:
- Use `rng = np.random.default_rng(seed)` inside the sampler.
- When parallelizing genes/units, seed must be `seed0 + stable_unit_index * LARGE_PRIME` (or similar) so order does not matter.

---

## 4) Integrate sampler choice into CRT per-unit runner

In `src/sceptre/crt.py` (or wherever the per-unit CRT is computed):
- Locate the function that currently:
  1) computes union x
  2) fits propensity p_hat
  3) calls Bernoulli index sampler to generate null
  4) computes `beta_null`
  5) computes empirical p-values and optional skew-normal calibration

Add a branch:
- if `resampling_method == "bernoulli_index"`: call existing sampler (no change)
- if `resampling_method == "stratified_perm"`:
  - require `x_obs`, `p_hat`, `B`
  - load `batch_raw` from `inputs["batch_raw"]` if `stratify_by_batch` and available
  - call `stratified_permutation_sampler(x_obs, p_hat, B, batch_raw, n_bins, seed)`
  - feed its output into the same null-beta computation path

Important: preserve all other logic:
- empirical p-value formula stays the same
- skew-normal calibration stays the same and still optional

---

## 5) Add tests (pytest) without breaking existing ones

Add a new test file, e.g. `tests/test_stratified_perm_sampler.py`.

### 5.1 Unit tests for the sampler
Given random `x_obs`, random `p_hat`, random `batch_raw`:

- `test_stratified_perm_preserves_stratum_counts`
  - build strata (same helper as sampler)
  - for each resample b, verify within each stratum:
    - `sum_{i in s} x_tilde^(b)[i] == m_s`
  - verify total treated count equals observed for each b

- `test_stratified_perm_reproducible`
  - same inputs + seed => identical outputs
  - different seed => outputs differ

### 5.2 End-to-end CRT null uniformity
Using `make_sceptre_style_synth(..., effect_size=0)` from README’s synthetic section. :contentReference[oaicite:2]{index=2}
- Run a small pipeline (N ~ 5k, K ~ 10, genes ~ 30, B ~ 255) with:
  - `resampling_method="bernoulli_index"` and `"stratified_perm"`
- For each method:
  - check pooled p-values roughly uniform (KS test or QQ slope sanity)
  - check NTC grouped pooled p-values roughly uniform (if the NTC grouping helper exists)

### 5.3 Compatibility test
- Ensure calling `run_all_genes_union_crt` without specifying `resampling_method` produces byte-identical (or within tiny tolerance) outputs to the previous version.

---

## 6) Update docs / examples (minimal)

In README “Usage” section, add a short example:

```python
out = run_all_genes_union_crt(
    inputs=inputs,
    B=1023,
    n_jobs=16,
    resampling_method="stratified_perm",
    resampling_kwargs=dict(n_bins=20, stratify_by_batch=True, batch_key="batch"),
    calibrate_skew_normal=False,  # recommend validating raw first
    return_raw_pvals=True,
)
````

Also note:

* batch labels must exist in `adata.obsm["covar"]["batch"]` (raw DataFrame) for batch stratification.
* If covar was provided as ndarray, batch stratification is unavailable and sampler falls back to propensity-only bins.

---

## 7) Implementation checklist for Codex

1. Find existing Bernoulli sampler output type + downstream consumer. Preserve it.
2. Implement `stratified_permutation_sampler` returning the same type.
3. Thread sampler options through:

   * `run_all_genes_union_crt`
   * per-gene/per-unit CRT runner(s)
   * NTC grouped CRT wrappers
4. Modify `prepare_crt_inputs` to store `covar_df_raw` and `batch_raw` safely.
5. Add tests described above.
6. Ensure deterministic seeds across parallel jobs.

Deliverables:

* New/modified code in `src/sceptre/` implementing sampler choice
* New tests passing
* README example snippet updated (optional but recommended)

