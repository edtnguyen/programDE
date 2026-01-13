# Codex spec: Implement Option 3 — CRT with U-test statistic + stratified-permutation resampling (backward compatible)

We already have a working SCEPTRE-style CRT pipeline with:
- Bernoulli “index sampler” and stratified-permutation sampler (both return `(indptr, indices)`),
- `_sample_crt_indices` dispatching between samplers,
- gene-level and guide-set CRT runners (`crt_pvals_for_gene`, `run_one_gene_union_crt`, `compute_gene_null_pvals`, `compute_guide_set_null_pvals`),
- NTC grouped ensemble helpers,
- tests for strata + sampler correctness + end-to-end null uniformity, and burden-bin stratification (implemented) that did not help in real data. :contentReference[oaicite:0]{index=0}

Next: add a **new test statistic** option that mimics the user’s historically successful approach:
- statistic = Mann–Whitney U / Wilcoxon rank-sum (computed as rank-sum of treated group),
- null resampling = stratified-permutation CRT (batch × propensity bins; burden bins already supported),
- p-values via empirical CRT counting (two-sided around the null center).

This must be **backward compatible**:
- default remains OLS-on-CLR statistic + current p-value pipeline,
- existing outputs and tests must remain unchanged for default.

---

## 0) High-level behavior

For each unit (gene union indicator, or NTC pseudo-gene as guide-set unit) and each program k:

**Observed statistic**
- `x_obs[i] ∈ {0,1}` = union indicator for unit
- `Y[i,k]` = program usage outcome (CLR or raw usage; keep existing default)
- Define ranks `R[:,k] = rankdata(Y[:,k], method="average")` over all cells (computed once globally)
- `rank_sum_obs[k] = Σ_{i: x_obs[i]=1} R[i,k]`
- Let `n1 = Σ x_obs`, `n0 = N - n1`
- `U_obs[k] = rank_sum_obs[k] - n1*(n1+1)/2`
- Define a signed, comparable effect size (recommended):
  - `rbc_obs[k] = 2*U_obs[k]/(n1*n0) - 1`  (rank-biserial correlation, in [-1,1])
- Use `T_obs[k] = rbc_obs[k]` as the test statistic.

**Null resamples (CRT)**
- Fit propensity `p_hat = P(x=1|C)` as already implemented.
- Sample `x_tilde^(b)` using existing `resampling_method`:
  - primary use case: `resampling_method="stratified_perm"` (batch × p-bins, optionally × burden bins)
  - must also support `"bernoulli_index"` (kept for completeness / debugging)
- For each resample b, compute `T_null[b,k]` analogously using treated ranks.

**Empirical two-sided p-value**
- `p[k] = (1 + #{b: |T_null[b,k]| >= |T_obs[k]|}) / (B+1)`

Optional skew-normal calibration:
- Keep existing path working, but DO NOT change default behavior.
- Use z-scores derived from `T_null` (per program) if enabled:
  - `z_obs = (T_obs - mean(T_null))/std(T_null)`
  - Fit skew-normal to `z_null` and compute calibrated two-sided p.
- NOTE: skew calibration has been problematic; do not change defaults (leave disabled unless user asks).

---

## 1) API changes (backward compatible)

Add a new parameter to the same public entrypoints that currently accept CRT config:

### 1.1 Gene-level CRT driver(s)
Functions like:
- `run_one_gene_union_crt`
- `crt_pvals_for_gene`
- `compute_gene_null_pvals`
- `run_all_genes_union_crt`
(and any wrappers calling them)

Add:
- `test_stat: str = "ols"` (default unchanged)
  - allowed values: `"ols"`, `"utest"` (or `"wilcoxon"`)
- `test_stat_kwargs: dict | None = None`
  - for utest: `{"rank_method": "average", "use": "clr"|"usage", "rank_dtype": "float32"}`

Behavior:
- If `test_stat="ols"`, everything works exactly as now.
- If `test_stat="utest"`, the pipeline produces **p-values per program** with the U-test statistic (and returns effect sizes as rank-biserial correlation).

### 1.2 Output schema changes (minimal + explicit)
To avoid breaking downstream code that expects `beta_hat` etc, do:

- Always return `pvals_df` exactly as before.
- For effect sizes:
  - If `test_stat="ols"`: keep existing `betas_df` (beta coefficients).
  - If `test_stat="utest"`: return a new `stats_df` (or re-use `betas_df` but add metadata).
Recommended:
- Add `out["stat_name"]` = `"beta"` or `"rank_biserial"`
- Add `out["stats_df"]` for utest, **in addition** to existing keys.
- Do not remove/rename existing keys.

---

## 2) Compute ranks once globally (do not recompute per gene)

### 2.1 Extend `CRTInputs` to optionally hold ranks
In `prepare_crt_inputs` or in `run_all_genes_union_crt` (preferred), add:

- If `test_stat="utest"`:
  - compute ranks `R` for the chosen outcome matrix `Y` (CLR or usage) **once**:
    - `R.shape = (N, K)`, store `float32` to reduce memory
  - attach to inputs: `inputs.R = R`
  - Also store `inputs.rank_info = {"rank_method": ..., "ties": ...}` (optional)

Implementation note:
- Use `scipy.stats.rankdata` per column (K ≤ ~70 is fine).
- Avoid copying big matrices unnecessarily:
  - If Y is stored as `np.ndarray`, compute ranks into a new array.
  - If Y is backed by memmap or anndata, read columns carefully.

### 2.2 Choice of outcome for ranks
Support:
- `use="clr"`: ranks are computed from CLR-transformed usage `Y_clr` (existing default outcome for OLS)
- `use="usage"`: ranks from raw usage `U` (after eps + renorm if that’s your convention)

This is controlled by `test_stat_kwargs.get("use", "clr")`.

---

## 3) Efficient computation of rank sums for null resamples

We already sample null assignments in the index form `(indptr, indices)` where each row i lists resample indices b for which cell i is treated. :contentReference[oaicite:1]{index=1}

We need an efficient way to compute:
- `rank_sum_null[b,k] = Σ_{i treated in resample b} R[i,k]`

### 3.1 Build sparse indicator `X_tilde` and use sparse-dense matmul (recommended v1)
Given `(indptr, indices)`:
- `X_tilde` is an N×B CSR sparse matrix with ones at (i,b) if treated.
- Construct:
  - `data = np.ones_like(indices, dtype=np.float32)`
  - `X = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N, B))`
  - `X.sort_indices()` (optional)
- Then:
  - `rank_sum_null = (X.T @ R).astype(np.float64)`  # shape (B, K)
  - `n1b = np.asarray(X.sum(axis=0)).ravel()`       # shape (B,)

Observed:
- `rank_sum_obs = R[x_obs_bool].sum(axis=0)`  # shape (K,)
- `n1 = x_obs.sum()`

This is usually fast enough because nnz is ~B * treated_count, not N*B.

### 3.2 Optional speed upgrade (v2): numba kernel
If matmul becomes a bottleneck, implement a numba kernel similar to existing OLS accumulation:
- Iterate over rows i, for each treated resample b in indices slice:
  - `rank_sum_null[b, :] += R[i, :]` (in blocks)
But start with sparse matmul first (simpler, correct, maintainable).

---

## 4) Compute U-test statistic from rank sums

Implement a helper:

`compute_rank_biserial(rank_sum, n1, N)` where:
- rank_sum is either:
  - `(K,)` for observed, or
  - `(B, K)` for null

Formula:
- `U = rank_sum - n1*(n1+1)/2`
- `n0 = N - n1`
- `rbc = 2*U/(n1*n0) - 1`
Guard:
- if `n1==0 or n0==0`: return NaNs for that unit (and p-values = 1)

For null, `n1` may be vector `n1b` (Bernoulli sampler) or constant (stratified perm). Support both.

Define:
- `T_obs = rbc_obs`
- `T_null = rbc_null`

Then reuse the existing empirical p-value counting code (two-sided abs compare).

---

## 5) Integrate into per-unit CRT runner(s)

### 5.1 Where to branch
Locate `crt_pvals_for_gene` / `crt_betas_for_gene` (and guide-set analogs). :contentReference[oaicite:2]{index=2}

Current flow (OLS):
1) build x_obs (union)
2) fit propensity p_hat
3) `_sample_crt_indices(...)` → `(indptr, indices)`
4) compute beta_null, beta_obs via OLS summaries
5) compute p-values

New flow:
- Add branch `if test_stat=="utest":`
  - require `inputs.R` exists (or compute ranks lazily once and cache)
  - sample indices as usual (resampling method unchanged)
  - compute `T_obs, T_null` via rank sums
  - compute p-values
  - return `stats_df` (rbc_obs) instead of betas (but keep pvals schema)

### 5.2 NTC grouped units
Guide-set CRT already exists (e.g. `compute_guide_set_null_pvals`, `ntc_groups.py`). :contentReference[oaicite:3]{index=3}
Thread through the same args:
- `test_stat`, `test_stat_kwargs`
so NTC pseudo-genes can be evaluated in the same mode.

---

## 6) Tests to add (pytest)

### 6.1 Unit test: utest statistic equals naive computation for observed
- Generate random Y (N=500, K=5), random x_obs with n1 between 20 and 200.
- Compute rbc_obs via:
  - new helper using rank_sum
  - reference: use `scipy.stats.mannwhitneyu` per k (two-sided) but extract U and compute rbc
- Assert close (tolerance 1e-6).

### 6.2 Unit test: rank-sum via sparse matmul matches naive for null
- Build a small `(indptr, indices)` with B=50 and random treated pairs.
- Construct X and compute `X.T @ R`.
- Naive loop over treated pairs to accumulate rank sums.
- Assert exact/close.

### 6.3 End-to-end global null uniformity (utest + stratified perm)
Using `make_sceptre_style_synth(effect_size=0)`:
- N=5000, K=20, B=255 (keep fast)
- Run pipeline with:
  - `test_stat="utest"`, `resampling_method="stratified_perm"`, `stratify_by_batch=True`
- Pool p-values across genes×program and check rough uniformity (loose quantile checks).
- Also check NTC pseudo-genes pooled p-values uniform.

### 6.4 Compatibility test
- Run a small synthetic run with `test_stat="ols"` (default) and assert outputs identical to pre-change (or to a stored reference) to ensure backward compatibility.

### 6.5 Determinism under parallel order
- Reuse existing determinism test infra.
- Ensure utest mode produces identical p-values when gene order is permuted (given same `seed0` and unit-id-based seeding).

---

## 7) Documentation updates (minimal)
Add a README snippet:

```python
out = run_all_genes_union_crt(
    inputs=inputs,
    B=1023,
    resampling_method="stratified_perm",
    resampling_kwargs=dict(n_bins=20, stratify_by_batch=True, batch_key="batch"),
    test_stat="utest",
    test_stat_kwargs=dict(use="clr", rank_method="average"),
    return_raw_pvals=True,
)
````

And a note:

* utest uses rank-biserial correlation as effect size; p-values are permutation/CRT empirical.

---

## 8) Implementation checklist

1. Add `test_stat` + `test_stat_kwargs` threading through all runners (gene + guide-set + NTC ensembles).
2. Add rank precompute once (store `inputs.R`).
3. Implement utest statistic from rank sums:

   * observed rank sum
   * null rank sums via sparse indicator transpose-multiply
4. Reuse existing p-value counting + null QQ helpers unchanged.
5. Add tests described above.
6. Ensure default OLS path untouched and passes all existing tests. 

```
```
