# Codex spec: Implement NTC empirical-null p-values for CLR-OLS via matching on (batch, n1, denom d)

We want a **principled alternative** to CRT resampling: use **NTC pseudo-genes as the empirical null distribution** for CLR-OLS test statistics, while keeping the current CRT pipeline intact and unchanged by default.

This must be backward compatible with the current codebase described in `memory.md`:
- samplers return `(indptr, indices)` and are dispatched by `_sample_crt_indices`
- OLS-on-CLR is the default test statistic
- NTC pseudo-gene grouping utilities already exist in `ntc_groups.py`
- `prepare_crt_inputs` stores `covar_df_raw` and `batch_raw` in `CRTInputs`
- existing functions include: `run_one_gene_union_crt`, `compute_gene_null_pvals`, `compute_guide_set_null_pvals`, `ntc_groups.py`, `diagnostics.py`, `global_null_diagnostics.py`, `pipeline_helpers.py` and `_sample_crt_indices`

We will add an **NTC empirical-null mode** that:
- computes the same CLR-OLS betas as today for real genes,
- computes CLR-OLS betas for many NTC pseudo-genes (guide-sets),
- matches each real gene to NTC pseudo-genes using low-dimensional summaries that govern the OLS null:
  - `n1`: treated count (sum of union indicator)
  - `d`: OLS denominator `x_resid^T x_resid` where `x_resid = M_C x`
  - (optionally) `pbar`: mean(logit(p_hat)) as a propensity summary
  - **batch handling**: default is *batch-stratified* (per-batch p-values + combine)

Then p-values are empirical vs matched NTC distribution:
- two-sided: `p = (1 + #{r: |T_ntc[r]| >= |T_obs|}) / (R_bin + 1)`


---

## 0) New public API (backward compatible)

Add a new argument to the main entrypoints that currently run tests:

### 0.1 Main gene driver(s)
In `run_all_genes_union_crt` (and any wrappers that return `pvals_df`):
- add `null_method: str = "crt"`
  - allowed: `"crt"` (default unchanged), `"ntc_empirical"`
- add `null_kwargs: dict | None = None` for `"ntc_empirical"`:
  - `ntc_labels: list[str] = ["SAFE", "non-targeting", "NTC"]` (match your repo conventions)
  - `guides_per_unit: int = 6`
  - `n_ntc_units: int = 5000` (sample-with-replacement pseudo-genes)
  - `matching: dict = {...}` (see Section 3)
  - `batch_mode: str = "meta"` (default) OR `"pooled"`
  - `combine_method: str = "fisher"` (default) OR `"stouffer"`
  - `min_treated: int = 10` and `min_control: int = 10` per batch for inclusion

No existing callsites change unless user explicitly sets `null_method="ntc_empirical"`.

### 0.2 Per-unit runner(s)
Where a single gene or guide-set unit is evaluated, add `null_method` passthrough, but keep default `"crt"`.

Outputs:
- Keep existing output keys when `null_method="crt"`.
- For `"ntc_empirical"`, still return:
  - `pvals_df` (same shape/index/columns)
  - `betas_df` (same as OLS)
  - add `pvals_df_ntc` (optional) or store into `out["pvals_df"]` as the primary p-values.
- Add metadata:
  - `out["null_method"] = "ntc_empirical"`
  - `out["ntc_matching_info"] = {bins, counts, fallbacks}`


---

## 1) What statistic do we use? (CLR-OLS unchanged)

We will use the **existing CLR-OLS** implementation for betas:
- For each gene g and program k: `beta_obs[g,k]` computed exactly as today.

We additionally need, per unit, the **OLS denominator**
- `d_g = x_resid^T x_resid = x^T M_C x`

This denominator is critical for matching because it controls the scale of the OLS coefficient under the null.

### 1.1 Add “denominator return” to the OLS helper
Find the function that currently computes `beta_k` from precomputed OLS summaries (no repeated lstsq). Modify it to optionally return `denom`:
- `beta, denom = ols_beta_from_x(..., return_denom=True)`
- `denom` is scalar per unit (or per program if your implementation differs; target scalar per unit).

Do NOT change existing default behavior.

Also compute `n1 = x.sum()` per unit (already cheap).

Optional (for later): `pbar = mean(logit(p_hat))` per unit, but only needed if matching still fails.

---

## 2) Generate many NTC pseudo-genes (sample with replacement)

We need enough NTC null units; with only ~119 NTC guides, partitions/ensembles are too few.
Implement a new helper (or extend existing `ntc_groups.py`):

`sample_ntc_pseudogenes_with_replacement(guide_names, guide2gene, ntc_labels, guides_per_unit, n_units, seed, matching_by_guide_prevalence=True)`

Behavior:
- identify NTC guide indices where `guide2gene[guide] in ntc_labels`
- for each pseudo-gene r:
  - sample `guides_per_unit` distinct NTC guides without replacement **within the pseudo-gene**
  - allow reuse across pseudo-genes (sampling with replacement at the pseudo-gene level)
- Optional: reuse your existing “matched-by-guide-frequency bins” logic from `ntc_groups.py` if available; if not, plain uniform is ok for v1 because we will match on `n1` and `d`.

Return:
- list of guide-index arrays: `ntc_units = [np.ndarray[int]] * n_units`

---

## 3) Matching strategy: (batch, n1, d) bins (optionally pbar)

We need to match real-gene units to NTC units so the null distribution of `beta` is comparable.

### 3.1 Compute unit signatures
For each **real gene** unit g and each **NTC pseudo-gene** unit r (and per batch if using meta mode):
- compute `n1_unit` = treated count (sum x)
- compute `d_unit` = `x^T M_C x` (denom)
- optional `pbar_unit` = mean(logit(p_hat)) or mean(p_hat) over all cells (only if enabled)

Store signatures in DataFrames:
- `sig_real`: index = real genes, columns `["n1","d", ...]`
- `sig_ntc`: index = ntc_unit_id, columns `["n1","d", ...]`

### 3.2 Bin-based matching (fast + stable)
Define bins using **real genes** (per batch if meta):
- `n1_bins`: quantile bins on `n1` (or use integer bins if small)
  - default: `n_n1_bins=10`
- `d_bins`: quantile bins on `log(d)` (denom is skewed)
  - default: `n_d_bins=10`
- optional `pbar_bins`: quantile bins on `pbar` if enabled
  - default: off

Assign each unit to a bin key tuple:
- `(n1_bin, d_bin)` (or `(n1_bin, d_bin, pbar_bin)`)

### 3.3 Fallback when a bin has too few NTC units
For each real unit bin key:
- if `count_ntc_in_bin >= min_ntc_per_bin` (e.g. 50): use it
- else fallback:
  1) expand to neighboring bins (Manhattan radius 1, then 2) until enough NTC units found
  2) if still insufficient, use all NTC units within the batch (meta) or global (pooled)
Record fallback stats for diagnostics.

---

## 4) Batch handling: default “meta” mode

User observed “within batch calibrated; pooled across batches not”.
So implement default:

### 4.1 `batch_mode="meta"` (recommended)
For each batch b:
1) subset cells in batch b
2) build design matrix `C_b` from covariates **excluding batch** (since constant within batch)
   - keep intercept
   - keep QC/burden covariates
3) compute CLR outcome `Y_b` as usual
4) compute `beta_obs[g,k,b]` and denom `d_obs[g,b]` for each real gene g
5) compute `beta_ntc[r,k,b]` and denom `d_ntc[r,b]` for each NTC unit r
6) compute empirical p-values per batch:
   - match bins within this batch using (n1,d)
   - `p_{gk,b} = (1 + #{r in matched_bin: |beta_ntc[r,k,b]| >= |beta_obs[g,k,b]|}) / (R_bin + 1)`

Then combine p-values across batches for each (g,k):
- only include batches where `n1>=min_treated` and `n0>=min_control`
- combine:
  - Fisher: `X = -2 Σ log p_b`, `p = 1 - chi2_cdf(X, df=2m)`
  - or Stouffer: `z = Σ w_b Φ^{-1}(1-p_b) / sqrt(Σ w_b^2)` (two-sided needs care)
Default: Fisher (simple, robust).

Also compute a combined effect size:
- return the pooled-all-cells OLS beta (existing) OR a weighted average across batches:
  - `beta_meta = Σ w_b beta_b / Σ w_b` with weights `w_b = d_obs[g,b]` (denom ≈ information)
This is optional; keep existing betas_df from pooled model unless requested.

### 4.2 `batch_mode="pooled"` (optional)
Compute everything on all cells with batch included in `C`.
Match on (n1, d) globally (no per-batch splits).
This mode is simpler but usually less calibrated; keep for debugging.

---

## 5) Core computation: empirical p-values vs NTC null

Implement:
`empirical_pvals_vs_ntc(beta_obs, beta_ntc, match_keys_obs, match_keys_ntc, min_ntc_per_bin, two_sided=True)`

Inputs:
- `beta_obs`: (G, K) float
- `beta_ntc`: (R, K) float
- `match_keys_obs`: length G integer bin ids (or tuple ids mapped to int)
- `match_keys_ntc`: length R integer bin ids
- `two_sided`: compare abs values

Algorithm (fast):
- For each bin id:
  - collect ntc indices in bin
  - for each program k:
    - sort `A_ntc = abs(beta_ntc[idx,k])`
    - for each gene g in bin:
      - `count_ge = len(A_ntc) - searchsorted(A_ntc, abs(beta_obs[g,k]), side="left")`
      - `p = (1 + count_ge) / (len(A_ntc) + 1)`
- Return `(G,K)` pval matrix.

Record bin sizes and fallback usage.

---

## 6) Integration points

### 6.1 New module
Create `src/sceptre/ntc_null.py` (or similar) containing:
- NTC pseudo-gene sampler (with replacement)
- signature computation (n1, denom d, optional pbar)
- binning + matching
- empirical p-values
- batch-meta combining

### 6.2 Hook into main pipeline
In `run_all_genes_union_crt`:
- if `null_method=="crt"`: run current path unchanged
- if `null_method=="ntc_empirical"`:
  1) determine ntc labels, sample NTC units
  2) compute real-gene betas + denom (reuse existing pooled OLS code)
  3) run per-batch subpipeline if `batch_mode=="meta"`
     - reuse existing functions but allow passing a cell mask / subset indices
     - or implement a lightweight “subset CRTInputs” builder that slices `Y`, `C`, `covar_df_raw`, etc.
  4) compute p-values via NTC matching
  5) return out dict with `pvals_df` populated from NTC empirical null

Ensure NTC generation + per-batch computations share seeds deterministically:
- `seed_ntc = seed0 + 17`
- `seed_batch = seed0 + 1000 * batch_id`
- `seed_match = seed0 + 99991`

---

## 7) Tests (pytest)

### 7.1 Unit test: denom correctness
- For random `C` and random binary `x`, verify `d = x^T M_C x` matches a reference computation:
  - compute `x_resid = x - C @ lstsq(C, x)`
  - check `d ≈ x_resid^T x_resid`

### 7.2 Unit test: empirical pvals vs NTC matches brute force
- Small synthetic:
  - G=5, R=50, K=3
  - assign random bins
  - compute empirical pvals via function
  - brute force loop compare counts
  - assert exact match

### 7.3 End-to-end: global null uniformity
Using `make_sceptre_style_synth(effect_size=0)`:
- include multiple batches
- run `null_method="ntc_empirical"`, `batch_mode="meta"`, `n_ntc_units=2000`, `min_ntc_per_bin=50`
- pool p-values across genes×program and check quantiles near uniform (loose bounds).
- also verify that NTC-vs-NTC “self p-values” (optional sanity) are roughly uniform.

### 7.4 Power sanity
Using `effect_size>0` with sparse program effects:
- verify that a nontrivial fraction of true causal gene×program tests have smaller p-values than null expectation.

### 7.5 Backward compatibility
- Existing CRT tests must still pass unchanged
- A new test ensures default call (`null_method` omitted) returns identical outputs to previous version.

---

## 8) Documentation
Update README to add a section:

**NTC empirical null (CLR-OLS)**
- explain matching on `n1` and `x^T M_C x`
- show example usage:

```python
out = run_all_genes_union_crt(
    inputs=inputs,
    null_method="ntc_empirical",
    null_kwargs=dict(
        ntc_labels=["SAFE","non-targeting"],
        guides_per_unit=6,
        n_ntc_units=5000,
        batch_mode="meta",
        combine_method="fisher",
        matching=dict(n_n1_bins=10, n_d_bins=10, min_ntc_per_bin=50),
    ),
)
````

Mention: this method assumes NTC pseudo-genes represent the null after matching.

---

## 9) Deliverables

* `src/sceptre/ntc_null.py` implementing the full pipeline
* Minimal modifications to existing functions to optionally return `denom`
* Thread `null_method` / `null_kwargs` through main entrypoints
* New pytest tests passing
* README updated with usage snippet


