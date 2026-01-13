# Codex spec: Add tests to validate stratified/permutation CRT and diagnose “no improvement” cases

Goal: We implemented a new resampling method `resampling_method="stratified_perm"` (strata = batch × propensity-bin, optionally × burden-bin). In practice, stratified CRT “didn’t help”, so we need **tests + diagnostics** that confirm:
1) stratification is non-degenerate (bins/strata not collapsed),
2) the sampler enforces exact treated counts per stratum for every resample,
3) resampling is deterministic under parallelization,
4) adding burden binning changes the null in the expected direction on synthetic settings where Bernoulli CRT is misspecified,
5) optional: U-test statistic path works and agrees with within-batch permutation baseline.

These are *tests* (pytest), not new features.

---

## 0) Repository assumptions

- Existing CRT pipeline already has:
  - per-unit union indicator computation `x` from guide sets,
  - propensity fit returning `p_hat`,
  - existing Bernoulli sampler,
  - new `stratified_perm` sampler,
  - OLS-based beta computation (fast summaries),
  - synthetic generator `make_sceptre_style_synth(...)` (effect_size=0 and effect_size>0 modes),
  - NTC grouping helpers (6-guide pseudo-genes + ensembles).

If function names differ, codex should locate the correct implementations and adapt imports accordingly.

---

## 1) Unit tests: stratification construction sanity

### 1.1 `test_strata_non_degenerate_on_random_p`
Create random `p_hat` with continuous values, random batch labels with 3–6 batches.

- Call `make_strata(p_hat, batch, n_bins=20)`
- Assert:
  - `n_unique_bins >= 10` (or at least > 3)
  - `n_strata >= n_batches * 5` (weak lower bound)
  - median stratum size > 10 for N=5000
  - fraction of strata with size < 2 is 0 (if min_stratum_size=2 filtering is enabled)

Also test with ties:
- set `p_hat = np.round(p_hat, 2)` to induce ties
- strata should collapse somewhat but still produce >1 stratum.

### 1.2 `test_strata_fallback_when_p_collapses`
Set `p_hat` constant for all cells. Expect:
- only 1 propensity bin (or 1 bin per batch if batch stratification is on)
- no crash; outputs are valid and deterministic.

---

## 2) Unit tests: stratified-permutation sampler correctness

### 2.1 `test_stratified_perm_preserves_counts_per_stratum`
Setup:
- N=2000
- random `batch` (3 batches)
- random `p_hat` continuous
- build `x_obs` by sampling Bernoulli with a nontrivial dependence on batch (to get heterogeneous counts)

Steps:
1) Build strata (same helper used by sampler) with `(batch, p_bin)` and `n_bins=10`.
2) Compute observed per-stratum treated counts:
   - `m_s = sum_{i in stratum s} x_obs[i]`
3) Generate resamples using sampler for B=200:
   - Ensure sampler returns a representation that can be converted to `x_tilde^(b)` OR directly provides treated indices per b.
4) For every resample b and every stratum s:
   - `sum_{i in s} x_tilde^(b)[i] == m_s`
5) Also verify global treated count preserved:
   - `sum_i x_tilde^(b)[i] == sum_i x_obs[i]`

### 2.2 `test_stratified_perm_is_uniform_within_stratum`
This is a statistical test to ensure sampling is not biased:
- Pick one stratum with size >= 100 and treated count m between 10 and 90.
- Over B=1000 resamples, count how often each cell is selected as treated within that stratum.
- Expected selection probability is m/|s|. Use a chi-square or max deviation check:
  - `max_abs(freq - m/|s|) < tol` where `tol` ~ 0.05 for B=1000 (loose, avoid flaky tests).
- Seed fixed to be deterministic.

### 2.3 `test_stratified_perm_reproducible_seed`
Given same inputs and same seed:
- sampler output identical (edge list identical or treated-index lists identical).
Given different seeds:
- output differs.

---

## 3) Integration tests: “is stratification actually used?”

These tests are to catch cases where `stratified_perm` accidentally degenerates to something equivalent to Bernoulli because bins collapse or batch is not passed.

### 3.1 `test_stratified_perm_uses_batch_from_covar_df`
Construct a tiny AnnData-like object (or use actual AnnData if repo tests already do) where:
- `covar_df_raw["batch"]` exists and has >1 value
- call CRT runner with `resampling_method="stratified_perm"` and `stratify_by_batch=True`
- assert the internal `batch_raw` is not None and has the right length and unique values

Implementation note: expose a debug return or use monkeypatch to intercept the call to `make_strata(...)` and assert `batch` was provided. Do not change production code behavior; only add minimal instrumentation if the repo already supports debug.

### 3.2 `test_stratified_perm_bin_count_logged_or_returned`
Add a small helper that returns diagnostic info without changing behavior:
- `diagnostics["n_strata"]`, `diagnostics["n_unique_bins"]`, `diagnostics["stratum_size_quantiles"]`
Then assert:
- `n_strata > n_batches` when p has variation.

If adding diagnostics is too invasive, test by reproducing binning logic in test and compare with sampler’s `stratum_id` output if available.

---

## 4) End-to-end calibration tests under global null (effect_size=0)

### 4.1 `test_global_null_uniform_pvals_stratified_perm`
Use `make_sceptre_style_synth(effect_size=0, confound_strength=0)`:
- N=5000, K=20, genes=30, guides_per_gene=6, NTC guides >= 15%
- Run CRT with:
  - `resampling_method="stratified_perm"` with `(batch, p_bin)` and B=255 (small for test runtime)
- Collect pooled p-values across gene×program for real genes and NTC pseudo-genes
- Assert:
  - KS test p-value > 1e-3 (loose) OR
  - empirical quantiles close to uniform:
    - `q50` in [0.45, 0.55]
    - `q10` in [0.07, 0.13]
    - `q01` in [0.005, 0.02]
Use loose bounds to avoid flaky tests.

### 4.2 `test_oracle_const_propensity_same_as_fit_under_easy_null`
On the same synthetic null, compare:
- `propensity_mode="oracle_const"` vs `propensity_mode="fit"` (if supported)
- Expect pooled p-value distributions to be similar (e.g., median absolute difference between sorted pvals < 0.02).

---

## 5) Stress test: Bernoulli CRT misspecification scenario where stratification should help

Create a synthetic “slot competition / burden” setting:
- Construct guide assignments where total guides per cell is fixed (e.g., exactly MOI=5) and guide occurrences are not independent.
- Make `x_obs` (union) correlated with burden or with another latent factor not fully captured by a simple logistic.

Expectation:
- Bernoulli CRT can show conservative/anti-conservative NTC p-values
- Stratified-permutation with `(batch, p_bin, burden_bin)` should improve calibration.

### 5.1 `test_burden_binning_improves_ntc_uniformity`
Setup:
- Build `burden = total_guides_per_cell` or `log1p_non_ntc_guides_per_cell` as an extra covariate.
- Run:
  1) stratified-perm with `(batch, p_bin)` only
  2) stratified-perm with `(batch, p_bin, burden_bin=8)`
- Compare NTC pooled p-values:
  - require that the version with burden has quantiles closer to uniform (e.g., |q01-0.01| smaller).
Keep bounds loose.

Implementation detail:
- Expose `burden_bin` option in `resampling_kwargs` (if already implemented).
- If burden-bin option doesn’t exist yet, skip this test for now OR mark xfail until implemented.

---

## 6) Optional tests for U-test statistic path (if implemented)

### 6.1 `test_utest_permutation_matches_within_batch_permutation_baseline`
For a single gene and a single program:
- Compute p-value using:
  - (A) repo’s `stratified_perm` resampling + U-statistic
  - (B) direct within-batch permutation of x (same number of permutations)
- Assert the two p-values are within, say, 0.05.

This validates that “hybrid option 3” reproduces the behavior that looked good previously.

---

## 7) Determinism under parallelization

### 7.1 `test_sampler_seed_independent_of_job_order`
Simulate two “units” (gene A and gene B) and run:
- in order A then B, record first 10 treated indices for resample 0
- in order B then A, record the same
- with the same `seed0` and stable unit indices, the unit-specific resamples must match exactly.

This catches accidental use of a global RNG that depends on iteration order.

---

## 8) Deliverables
- Add new pytest files:
  - `tests/test_strata.py`
  - `tests/test_stratified_perm_sampler.py`
  - `tests/test_end_to_end_stratified_perm.py`
  - (optional) `tests/test_utest_hybrid.py`
- Ensure tests run in < ~60s locally by using small B (e.g., 127/255) and small gene counts.
- Avoid flaky tests: use deterministic seeds, loose statistical thresholds, and small-sample robust checks.

```
