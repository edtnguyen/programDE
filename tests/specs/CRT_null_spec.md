# SPEC: CRT-null p-values + QQ null curve (cell-level CRT)

## Goal
Fix the QQ plot “null” curve by generating **CRT-null p-values** correctly. The pipeline currently outputs only **observed** p-values (per gene/program), but not p-values that are themselves drawn under the CRT null. We will add a helper to compute leave-one-out CRT-null p-values from a vector of null test statistics and wire it into the QQ plotting/diagnostics path.

## Background / Definitions
For one test (one gene × one program), CRT produces null test statistics:
- `T_null = [T^(1), ..., T^(B)]` (e.g., beta under each resample)

A CRT-null p-value for each null draw (leave-one-out) is:
- two-sided:
  p_null[b] = (1 + #{b' != b : |T_null[b']| >= |T_null[b]|}) / B
- one-sided (right tail):
  p_null[b] = (1 + #{b' != b : T_null[b'] >= T_null[b]}) / B

These p-values should be approximately Uniform(0,1) if the null is implemented correctly.

## Required Changes

### 1) Add fast helper: `crt_null_pvals_from_null_stats_fast`
Implement an O(B log B) function that computes leave-one-out CRT-null p-values from a 1D array of null statistics.

**File**
- `src/yourpkg/diagnostics.py` (or `src/yourpkg/crt_null.py`)

**Signature**
```python
def crt_null_pvals_from_null_stats_fast(T_null: np.ndarray, two_sided: bool = True) -> np.ndarray:
    """
    Input:  T_null shape (B,)
    Output: p_null shape (B,), in [1/B, 1]
    Definition (two-sided): p_b = (1 + #{b'!=b: |T_{b'}| >= |T_b|}) / B
    """
````

**Implementation constraints**

* No Python loop over b.
* Must handle ties deterministically.
* Must match naive leave-one-out definition exactly.

**Algorithm (tie-safe)**
Let `A = abs(T_null)` if two_sided else `A = T_null`.
Sort ascending: `A_sorted`.
For each element `A[i]`, compute:

* `count_lt[i] = searchsorted(A_sorted, A[i], side="left")`
  Then:
* `p[i] = (B - count_lt[i]) / B`
  Finally:
* clip to `[1/B, 1]` for safety.

Rationale: `count_others_ge = B - count_lt - 1` (exclude self) and `p = (1 + count_others_ge)/B = (B - count_lt)/B`. Using `side="left"` correctly treats ties as “>=”.

### 2) Add optional matrix helper (nice-to-have)

Compute null pvals for multiple programs for one gene.

**Signature**

```python
def crt_null_pvals_from_null_stats_matrix(beta_null_bk: np.ndarray, two_sided: bool = True) -> np.ndarray:
    """
    beta_null_bk shape (B, K) -> p_null_bk shape (B, K)
    Implement by looping over K (K is small ~70), calling the 1D helper each time.
    """
```

### 3) Wire into QQ plot code

Update QQ plotting utility to accept *either* p-values or null statistics.

**File**

* wherever `qq_plot_mock.png` is generated (e.g., `scripts/qq_plot.py` or `src/yourpkg/plotting.py`)

**Change**
Add parameters:

* `null_pvals: Optional[np.ndarray] = None`
* `null_stats: Optional[np.ndarray] = None`
* `null_two_sided: bool = True`

Logic:

* if `null_pvals is None and null_stats is not None`:

  * compute `null_pvals = crt_null_pvals_from_null_stats_fast(null_stats, two_sided=null_two_sided)`
* plot dashed “null” curve from `null_pvals` (not from observed pvals)

### 4) (Optional) Add a lightweight diagnostics hook to collect null stats

Because the main pipeline does not store `beta_null`, add a minimal debug/diagnostics path:

**Option A (recommended minimal)**
Expose a function that runs CRT for a single (gene, program) and returns `beta_null` so scripts can build QQ null curves from a small set of tests.

**Signature**

```python
def crt_null_stats_for_test(...)-> np.ndarray:
    """Return beta_null shape (B,) for one selected gene x one program."""
```

**Option B (collector)**
Allow `crt_pvals_for_gene(...)` to accept a callback:

* `null_collector(gene, beta_null_bk)` (called only when provided)
  to sample a small subset of tests for diagnostics without bloating memory.

## Tests (pytest)

### Unit tests for null pvals helper

**File**

* `tests/test_null_pvals.py`

**Test 1: matches naive (no ties)**

* generate random `T_null` (float), B=200
* naive loop:
  p_naive[b] = (1 + sum(|T[-b]| >= |T[b]|)) / B
* fast helper output equals naive (exact match or atol 1e-15)

**Test 2: matches naive with ties**

* construct `T_null` with duplicates (e.g., round or inject repeated values)
* verify fast matches naive

**Test 3: bounds**

* ensure output in `[1/B, 1]`

### Smoke test for QQ null curve generation

**File**

* `tests/test_qq_null_curve.py`
* generate a `T_null` from `np.random.normal(size=B)`
* compute p_null via helper
* KS test against Uniform(0,1) should not be extreme (loose threshold; this is stochastic)

  * e.g., `kstest(p_null, "uniform").pvalue > 1e-3` for B=5000

## Acceptance Criteria

* QQ null curve generated from `null_pvals` is near y=x on synthetic normal nulls.
* `crt_null_pvals_from_null_stats_fast` matches naive leave-one-out p-values (including ties).
* Plotting code never uses observed p-values as the null curve by mistake.
* Documentation comment in QQ plot script clarifies: dashed null curve = CRT-null p-values (leave-one-out).

## Notes / Pitfalls

* Do not forget the factor-of-2 for two-sided *calibrated* p-values; this spec is about CRT-null p-values built from null stats (two-sided handled via `abs()`).
* If later adding skew-normal calibration for null diagnostics, use a train/holdout split of null draws; do not fit and evaluate on the same draws.



