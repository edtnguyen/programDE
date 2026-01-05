```md
# SPEC: QQ-plot / p-value sanity tests (cell-level CRT on synthetic data)

## Goal
Add automated tests to detect when the **observed p-values** are incorrectly conservative/anti-conservative on synthetic null data, and to catch common implementation mistakes that cause QQ plots to deviate (wrong expected quantiles, plotting q-values, forced p=1 filters, propensity misfit, etc.).

These tests should fail loudly if:
- the NTC/observed p-values are not approximately Uniform(0,1) under a known-null synthetic dataset
- the QQ expected axis is computed incorrectly (wrong m, reused grid)
- the “observed” values are actually BH-adjusted/q-values
- the pipeline is forcing many p-values to 1.0 (skips)
- propensity fitting creates systematic distortion (oracle constant p should fix)

## Scope
- cell-level CRT pipeline
- synthetic dataset(s)
- tests implemented in `pytest`

## Required New Test Files
- `tests/test_qq_sanity.py`
- `tests/test_propensity_oracle.py`
- `tests/test_plot_expected_axis.py`

## Assumptions / Required Helper APIs (Codex should implement if missing)
If these helpers don’t exist, create minimal versions under `src/yourpkg/diagnostics.py`:

1) `qq_expected_grid(pvals: np.ndarray) -> np.ndarray`
Returns expected quantiles for QQ:
  exp_i = (i - 0.5)/m for i=1..m, m=len(pvals)

2) `is_bh_adjusted_like(pvals: np.ndarray) -> bool`
Heuristic detector: many 1.0’s, step-like distribution, mean >> 0.5, etc.

3) pipeline entrypoints (already exist or stub):
- `run_pipeline(adata, genes, B, seed, skew=False, ...) -> dict`
must return raw p-values per gene/program (NOT BH) as a DataFrame or array.
If you already have:
- `out["pvals_raw_df"]` (raw CRT pvals)
- `out["pvals_df"]` (skew-calibrated)
use those.

4) ability to override propensity:
- either `run_pipeline(..., propensity_mode="fit"|"oracle")`
- or allow passing `p_override` per gene:
  `run_pipeline(..., propensity_override_fn: Callable[[x,C,gene], p])`

If pipeline cannot be easily changed, implement a small wrapper in tests that:
- computes x for a gene
- sets p = x.mean() constant
- runs CRT sampling + p-value computation for a small set of genes/programs.

## Synthetic Data Requirements
Use a **null** synthetic generator where:
- guide assignment is independent of covariates (true propensity constant)
- usage depends on covariates only (no gene effects)
- includes NTC guides mapped to a pseudo gene “NTC” with effect 0 (optional)
If your current generator does not have NTC, create a small “NTC gene set” by selecting random genes and treating them as null for the test.

Recommended sizes for tests:
- N ~ 5000, K ~ 20, genes ~ 30, guides/gene ~ 6, B ~ 199 or 399
Keep runtimes reasonable in CI.

## Tests

### Test 1: Observed raw p-values are approximately Uniform under null
File: `tests/test_qq_sanity.py`

Steps:
1) Generate null synthetic adata.
2) Run pipeline for a subset of genes (e.g., 10 genes) and programs (all K or subset 10).
3) Extract **raw observed** p-values (not skew, not BH).
4) Assertions:
   - values are within (0,1] and finite
   - proportion of p==1.0 is small (e.g., < 10%) unless explicitly expected
   - mean ~ 0.5 within tolerance (e.g., |mean-0.5| < 0.08)
   - KS test vs Uniform(0,1) not extreme (loose threshold due to dependence):
       kstest(pvals, "uniform").pvalue > 1e-4
   - empirical type I error near alpha:
       at alpha=0.05: |reject_rate - 0.05| < 3*sqrt(alpha*(1-alpha)/m)

Note: do NOT expect perfect uniform; use loose thresholds.

### Test 2: Fail if BH/q-values are accidentally plotted/returned as “raw”
File: `tests/test_qq_sanity.py`

Steps:
1) Take the pipeline raw pvals (p).
2) Compute BH-adjusted q (using statsmodels or a simple BH).
3) Assert that the pipeline raw pvals are not identical to BH:
   - `np.mean(np.isclose(p, q)) < 0.05`
4) Add heuristic:
   - raw pvals should not have an excessive mass at 1.0 (BH often does)

If pipeline’s “raw” is actually BH, this test fails.

### Test 3: Expected QQ axis computed per-curve (m must match)
File: `tests/test_plot_expected_axis.py`

Goal: prevent bug where expected grid is computed from all pvals and reused for NTC subset.

Implementation:
- Implement `qq_expected_grid(pvals)`:
  exp = (arange(1,m+1)-0.5)/m
- Test that:
  - `qq_expected_grid(all_pvals)` length == len(all_pvals)
  - `qq_expected_grid(ntc_pvals)` length == len(ntc_pvals)
- Simulate bug:
  - compute exp_all and use for subset
  - assert mismatch in lengths triggers error in plotting function
Add guard in plotting util:
- if `len(expected) != len(pvals_sorted)`: raise ValueError
Write a test that expects this ValueError.

### Test 4: Forced p=1.0 / skip logic is not dominating
File: `tests/test_qq_sanity.py`

Steps:
- Count how many tests are skipped or set to p=1 due to low treated count, etc.
- On the null synthetic generator, this should not dominate.
Assertions:
- fraction(p==1.0) < 0.2 (or chosen threshold)
- log warning emitted if above threshold (optional)

### Test 5: Oracle propensity reduces conservativeness (diagnostic)
File: `tests/test_propensity_oracle.py`

Because in null synthetic data the true propensity is constant:
- fitted propensity should be close to constant and CRT should calibrate.

Test:
1) Run pipeline in normal mode (fit propensity) -> get pvals_fit.
2) Run pipeline with oracle propensity:
   - for each gene, set p_i = mean(x) constant -> pvals_oracle.
3) Compare calibration metrics:
   - KS pvalue vs uniform should be >= for oracle mode (or at least not much worse)
   - mean deviation |mean-0.5| should be smaller for oracle or within tolerance.
Fail condition:
- if oracle is much better than fit (e.g., KS improves by > 0.2 and fit fails),
  then propensity fitting is likely overfitting/misconfigured; fail with message:
  “Propensity fit is distorting null; increase regularization or cross-fit.”

### Test 6: CRT-null p-values are uniform (leave-one-out)
File: `tests/test_qq_sanity.py`

For 1–2 representative tests (one gene × one program):
1) Extract null stats `T_null` from CRT internals (or add debug API).
2) Compute null pvals via leave-one-out helper:
   `p_null = crt_null_pvals_from_null_stats_fast(T_null, two_sided=True)`
3) KS test against Uniform:
   - `kstest(p_null, "uniform").pvalue > 1e-3` (should pass easily)

This ensures the null-pval helper + plotting input is correct.

## Output / Debug on Failure
When any test fails, print:
- count / fraction of p==1.0
- min/median/max of pvals
- number of tests m
- treated count summary (min/median/max) for the gene set
- (for propensity oracle test) std of fitted p_i vs oracle constant

## Acceptance Criteria
- All tests pass on the null synthetic dataset.
- The plotting function throws an error if expected grid length mismatches curve pvals.
- Oracle propensity mode does not drastically outperform fit mode on null synthetic data.
- CRT-null pvals (leave-one-out) are approximately uniform.

## Notes / Guardrails
- Use loose thresholds because p-values across programs/genes are dependent.
- Keep B small for CI speed (199/399). Add `RUN_SLOW=1` tests for B=1023 locally.
- Do not use skew-normal calibration in these core null tests unless separately holdout-fitting.
```
