# Codex spec: Add burden-bin stratification to `stratified_perm` resampling (keep old behavior intact)

Context: We already added a `resampling_method="stratified_perm"` sampler that stratifies by `(batch, propensity_bin)` where `propensity_bin` is quantile binning of `p_hat = P(x=1|C)`. In high-MOI data, batch×p_hat still may not capture slot-competition / guide-burden dependence. We want an optional **burden-bin** dimension in the stratification: strata = `(batch, propensity_bin, burden_bin)`.

This must be **backward compatible**, with burden-binning off by default.

---

## 0) Desired API changes (minimal)

Wherever `resampling_method="stratified_perm"` is accepted (gene CRT runner, NTC-group runner, wrappers), extend `resampling_kwargs` to include:

- `burden_key: str | None = None`
  - If provided, use this covariate column as the burden scalar per cell.
  - Default: `None` (no burden stratification).
- `n_burden_bins: int = 8`
- `burden_bin_method: str = "quantile"`  (allowed: `"quantile"`, `"uniform"`)
- `burden_clip_quantiles: tuple[float,float] = (0.0, 1.0)` (optional; default no clip)
- `min_stratum_size: int = 2` (already exists; must still apply)

Batch source:
- batch labels are stored in `adata.obsm["covar"]["batch"]` (raw, before one-hot). We already pass `batch_raw` into the sampler if `stratify_by_batch=True`.

Burden source:
- burden values are in `adata.obsm["covar"][burden_key]` as a numeric series.
- Do NOT use the encoded/z-scored `C` matrix for burden (we need raw).

---

## 1) Plumb raw covariates to the sampler (if not already)

Ensure `prepare_crt_inputs(...)` returns:
- `inputs["covar_df_raw"]`: DataFrame of covariates as originally provided (before encoding)
- `inputs["batch_raw"]`: `covar_df_raw[batch_key].to_numpy()` if available
No behavior changes for `inputs["C"]`.

If this already exists from prior changes, keep as-is.

---

## 2) Implement burden binning helper

Add a helper in the sampler module:

### 2.1 `compute_bins(values, n_bins, method)`
Inputs:
- `values: (N,) float array`
- `n_bins: int`
- `method: "quantile"|"uniform"`
- optional `clip_quantiles=(q_lo,q_hi)`:
  - clip values to quantiles before binning to avoid extreme outliers dominating uniform bins

Behavior:
- If `method="quantile"`:
  - edges = quantile(values, linspace(0,1,n_bins+1))
  - edges = unique(edges); if too few unique edges -> fallback to fewer bins (edges.size-1)
- If `method="uniform"`:
  - optionally clip first
  - edges = linspace(min,max,n_bins+1)
- Return:
  - `bin_id: int array in [0, n_bins-1]`
  - `n_effective_bins: int`
  - `edges` (for debugging)

Edge cases:
- If all values equal -> return all zeros and `n_effective_bins=1`.

---

## 3) Modify stratum construction to incorporate burden bins

Current stratum id:
- without batch: `stratum_id = p_bin`
- with batch: `stratum_id = batch_id * n_p_bins + p_bin`

New stratum id with burden:
- compute `b_bin` from `burden_values` if `burden_key is not None`
- define `nP = n_effective_p_bins`, `nB = n_effective_burden_bins`
- if batch stratification enabled:
  - `stratum_id = batch_id * (nP*nB) + p_bin * nB + b_bin`
- else:
  - `stratum_id = p_bin * nB + b_bin`

Important:
- Use the **effective** number of bins after `unique(edges)` collapse, not the requested bins, to keep ids compact.
- Keep deterministic factorization order for batch labels (e.g., `pd.factorize(batch_raw, sort=True)` or stable mapping).

Also return diagnostics:
- `diag["n_effective_p_bins"]`, `diag["n_effective_burden_bins"]`, `diag["n_strata"]`
This is optional but strongly recommended; if you already have diagnostics dict, add fields.

---

## 4) Sampler behavior unchanged except stratum ids

Sampling step remains:
For each stratum s:
- `m_s = sum_{i in s} x_obs[i]`
- for each resample b, choose `m_s` indices from stratum without replacement.

This ensures exact treated counts per stratum (now including burden).

---

## 5) Integration points

### 5.1 Per-unit CRT runner
When `resampling_method="stratified_perm"`:
- gather `batch_raw` from `inputs` if `stratify_by_batch=True`
- gather `burden_values` from `inputs["covar_df_raw"][burden_key]` if `burden_key not None`
  - validate numeric dtype and no NaNs (or impute; prefer raising error with helpful msg)

Call sampler with:
- `x_obs`, `p_hat`, `B`
- `batch_raw` (optional)
- `burden_values` (optional)
- `n_bins`, `n_burden_bins`, `burden_bin_method`, `burden_clip_quantiles`, `min_stratum_size`
- `seed`

Do not alter any downstream beta/pvalue code.

### 5.2 Wrappers
- Thread `resampling_kwargs` through all wrappers (gene loop, NTC ensemble).
- Update any CLI/config defaults (if present) to keep `burden_key=None`.

---

## 6) Defaults recommendation (document in code comments)

For high MOI:
- `n_bins=20` for propensity
- `n_burden_bins=8`
- `burden_key="log1p_non_ntc_guides_per_cell"` OR `"log1p_grna_n_nonzero_assign"` (if present)
- `stratify_by_batch=True`
- `burden_bin_method="quantile"`

Do not set these as defaults globally; only examples/docs.

---

## 7) Tests to add/extend

### 7.1 Sampler correctness: treated counts preserved in 3D strata
Extend existing sampler test:
- Build random `x_obs`, `p_hat`, `batch`, `burden` (continuous)
- Construct strata ids via the sampler helper
- For each resample b:
  - for each stratum id:
    - `treated_count_resample == treated_count_obs`
- Also check global treated count preserved.

### 7.2 “Burden bins actually increase strata count”
- With batch and non-constant p_hat and burden:
  - `n_strata_with_burden > n_strata_without_burden`
(allow equality if bins collapse due to ties, but in test data it should be strictly >)

### 7.3 Reproducibility
- Same inputs/seed -> identical outputs
- Different seed -> different outputs

### 7.4 End-to-end: burden helps in a constructed misspecification sim (optional xfail if too heavy)
- Create a synthetic assignment where x correlates strongly with burden within batch even after p_hat bins.
- Expect NTC calibration improves when burden bins are included (loose check on q01 closeness to 0.01).

---

## 8) Deliverables
- Code changes implementing burden-bin stratification in the stratified sampler.
- Updated docstring / README snippet showing how to enable:
  ```python
  resampling_method="stratified_perm",
  resampling_kwargs=dict(
      n_bins=20,
      stratify_by_batch=True,
      batch_key="batch",
      burden_key="log1p_non_ntc_guides_per_cell",
      n_burden_bins=8,
      burden_bin_method="quantile",
  )
````

* New/updated tests passing.


