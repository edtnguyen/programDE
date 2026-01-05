### A. Data plumbing / invariants (AnnData IO)

* **Shapes + alignment**

  * `adata.obsm["covar"]` is `(N, p)` and row-aligned with `adata.X` / `adata.obs`.
  * `adata.obsm["cnmf_usage"]` (or wherever you store it) is `(N, K)` and row-aligned.
  * `adata["guide_assignment"]` is row-aligned with cells (or clearly documented if not).
* **Mapping integrity**

  * `adata.uns["guide2gene"]` contains only guides that exist in `guide_assignment`.
  * Each gene has ≥1 guide; genes with 0 guides are skipped with a clear reason.
* **Binary per-guide**

  * Values in `guide_assignment` are 0/1 (or bool). If counts exist, you explicitly binarize and test that.

### B. CLR transform tests (cell-level)

1. **Row-sum zero**

```python
Y = clr_transform(U, eps=1e-6)
assert np.allclose(Y.sum(axis=1), 0.0, atol=1e-6)
```

2. **Pairwise ratio identity**
   For any programs `a,b`:

```python
lhs = Y[:, a] - Y[:, b]
rhs = np.log(Ueps[:, a]) - np.log(Ueps[:, b])
assert np.allclose(lhs, rhs, atol=1e-6)
```

(where `Ueps` is the floored+renormalized usage)

3. **Finite values**

```python
assert np.isfinite(Y).all()
```

### C. Union indicator correctness (guide→gene)

* For a small synthetic matrix, verify union matches explicit OR:

```python
x_union = union_indicator(guide_assignment, guides_for_gene)
x_or = np.zeros(N, dtype=np.int8)
for g in guides_for_gene:
    x_or |= guide_assignment[:, g].astype(np.int8)
assert np.array_equal(x_union, x_or)
```

* **Edge cases**

  * If all guides absent → `x_union.sum()==0` and gene is skipped.
  * If all cells have some guide → `x_union.sum()==N` handled (propensity saturates; CRT degenerates → skip or clip).

### D. Propensity model sanity (logistic)

1. **Intercept-only sanity**
   If `C` is just intercept, then `p_i` should be constant and near `x.mean()`:

```python
p = fit_propensity(C_intercept_only, x)
assert np.std(p) < 1e-8
assert abs(p.mean() - x.mean()) < 1e-3
```

2. **Bounds / clipping**

```python
assert np.all(p > 0) and np.all(p < 1)   # or within [pmin,1-pmin] if you clip
```

3. **Permutation invariance**
   Permute rows of `C` and `x` together → `p` permutes the same way.

4. **Separation robustness**
   Construct a case where `x` is perfectly predicted by one covariate; ensure your code does not crash (regularization/clipping), and emits warning or proceeds deterministically.

### E. CRT sampler correctness (fast sampler vs naive Bernoulli)

Use small sizes so the naive reference is feasible: e.g. `N=500`, `B=2000`.

1. **Marginal inclusion probability**
   Generate random `p_i ~ Uniform(0.001, 0.05)`. For each cell `i`, compare empirical inclusion rate across resamples:

```python
Xnaive = (rng.random((B,N)) < p[None,:]).astype(np.uint8)
indptr, indices = fast_sampler(p, B, seed=0)
Xfast = np.zeros((B,N), dtype=np.uint8)
for b in range(B):
    Xfast[b, indices[indptr[b]:indptr[b+1]]] = 1

mean_abs_fast = np.mean(np.abs(Xfast.mean(axis=0) - p))
mean_abs_naive = np.mean(np.abs(Xnaive.mean(axis=0) - Xfast.mean(axis=0)))
assert mean_abs_fast < 1e-2
assert mean_abs_naive < 1e-2
```

2. **Total treated per resample distribution**
   Check mean/var of `Xfast.sum(axis=1)` matches `sum(p)` and `sum(p*(1-p))` approximately:

```python
s = Xfast.sum(axis=1)
assert abs(s.mean() - p.sum()) < 0.02 * p.sum()
assert abs(s.var() - (p*(1-p)).sum()) < 0.1 * (p*(1-p)).sum()
```

3. **No duplicate cells within a resample**
   For each `b`, verify uniqueness of indices slice:

```python
idx = indices[indptr[b]:indptr[b+1]]
assert idx.size == np.unique(idx).size
```

4. **Determinism**
   Same seed → identical `(indptr, indices)`. Different seed → differs for at least one `b`.

### F. OLS coefficient computation correctness (summary-stat implementation)

This is a must-have: compare your “fast beta” vs `np.linalg.lstsq` on random data.

For random `C (N×p)`, binary `x`, and `Y (N×K)`:

```python
beta_fast = beta_from_summaries(x, C, Y)          # your implementation, returns (K,)
beta_ref  = []
X = np.column_stack([x, C])                      # make sure C already includes intercept OR not consistently
for k in range(K):
    coef, *_ = np.linalg.lstsq(X, Y[:,k], rcond=None)
    beta_ref.append(coef[0])
beta_ref = np.array(beta_ref)

assert np.allclose(beta_fast, beta_ref, atol=1e-7, rtol=1e-6)
```

Also add:

* **Rank-deficient C**: either raise a clear error or apply ridge and test that behavior explicitly.

### G. CRT p-value logic tests

1. **Exact-count test**
   With a tiny fixed example:

* `beta_obs = 2.0`
* `beta_null = [0.1, 2.0, -3.0, 1.9]` (B=4)
  Two-sided count `|null| >= |obs|` is 2 (`2.0`, `-3.0`) →
  `p = (1+2)/(4+1)=0.6`

```python
assert crt_pvalue(beta_obs, beta_null) == 0.6
```

2. **Monotonicity**
   If you make `|beta_obs|` larger, p-value should not decrease spuriously:

```python
p1 = crt_pvalue(1.0, beta_null)
p2 = crt_pvalue(2.0, beta_null)
assert p2 <= p1
```

### H. End-to-end validity: null calibration (Type I error)

Simulate a null dataset where `x` depends on `C`, but `Y` depends only on `C` (not `x`).

* Generate `C` (with intercept + some covariates)
* Generate `p = sigmoid(C @ theta)`
* Sample `x ~ Bernoulli(p)`
* Generate `Y = C @ Gamma + noise` (K programs worth), then optionally re-CLR it if you simulate in usage-space

Run pipeline for many genes/programs (or repeated runs) and check:

* p-values are ~Uniform(0,1) (KS test not significant, or QQ plot slope ~1)
* false positive rate at 0.05 is ~5% within binomial error:

```python
alpha = 0.05
fp = (pvals < alpha).mean()
assert abs(fp - alpha) < 3*np.sqrt(alpha*(1-alpha)/len(pvals))
```

### I. End-to-end power test (one true effect)

Simulate with one program having a true effect:

* pick `k*`
* set `Y[:,k*] += beta_true * x`
  Pipeline should produce:
* smaller p-values for `k*` than other programs
* correct sign of `beta_obs[k*]`

### J. Skew-normal calibration tests (if enabled)

1. **Default is two-sided**
   Verify your calibration function returns:
   `2 * min(F(z_obs), 1 - F(z_obs))` by default.

2. **Normal special case**
   If you feed null z-scores drawn from `N(0,1)`:

* fitted skew parameter should be near 0 (or at least calibrated p-values ~ uniform)
* compare calibrated p-values to standard normal p-values for a few z’s.

3. **Fallback works**
   Force degenerate input (all z’s equal, or sd≈0) → must fall back to empirical p-values (and not crash).

### K. Parallel / batching / reproducibility tests

* **Gene-order invariance**
  Running genes in different orders yields identical outputs given fixed per-gene seeds.
* **Joblib/multiprocessing determinism**
  In parallel, p-values should match serial run (within tolerance) when seeds are derived deterministically from gene id.
* **No shared-state RNG**
  Ensure Numba RNG usage is seeded per call; add a test that two consecutive calls with same seed match.

### L. Performance regression (practical)

Pick a fixed benchmark (e.g., `N=100k, B=1023, K=70, p=30, G=50`) and assert:

* runtime < some threshold on your dev machine/CI
* memory doesn’t blow up (e.g., no dense `B×N` arrays)

