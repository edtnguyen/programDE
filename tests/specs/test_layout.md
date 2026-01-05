Here’s a concrete `pytest` layout + a synthetic AnnData generator that matches your schema:

* `adata.obsm["covar"]` → covariate matrix
* `adata.uns["guide2gene"]` → `dict[str,str]`
* `adata.obsm["guide_assignment"]` → sparse `(N, n_guides)` binary matrix (recommended; easiest to keep aligned with cells)

You can copy these files as-is and then implement the functions they import in `src/yourpkg/...`.

---

## Suggested repo layout

```
.
├── pyproject.toml
├── src/
│   └── yourpkg/
│       ├── __init__.py
│       ├── preprocess.py      # CLR, flooring
│       ├── design.py          # union indicator from guide_assignment + guide2gene
│       ├── propensity.py      # logistic fit p(x=1|C)
│       ├── sampler.py         # fast CRT sampler (numba)
│       ├── ols.py             # beta via summaries
│       ├── crt.py             # p-values
│       └── calibrate.py       # skew-normal (optional)
└── tests/
    ├── conftest.py
    ├── utils_synth.py
    ├── test_preprocess.py
    ├── test_design.py
    ├── test_propensity.py
    ├── test_sampler.py
    ├── test_ols.py
    ├── test_crt.py
    └── test_integration_null.py
```

---

## `tests/utils_synth.py` (synthetic AnnData generator)

```python
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

def _softmax_rows(X: np.ndarray) -> np.ndarray:
    X = X - X.max(axis=1, keepdims=True)
    E = np.exp(X)
    return E / E.sum(axis=1, keepdims=True)

def make_synthetic_adata(
    N: int = 5000,
    K: int = 20,
    n_genes: int = 30,
    guides_per_gene: int = 6,
    moi_mean: float = 5.5,
    n_cov: int = 8,
    seed: int = 0,
    add_true_effect: bool = False,
    true_gene_idx: int = 0,
    true_prog_idx: int = 0,
    true_beta: float = 0.5,
):
    """
    Creates AnnData with:
      - obsm["covar"] : (N, p) float64 covariates WITH intercept in col0
      - obsm["usage"] : (N, K) compositional program usage (rows sum to 1)
      - obsm["guide_assignment"] : (N, n_guides) sparse binary
      - uns["guide2gene"] : dict guide_name -> gene_name
      - uns["gene2guides"] : dict gene_name -> [guide_names] (handy for tests)

    add_true_effect:
      If True, injects a gene->program effect into usage by shifting logits of one program
      for cells with x_union=1 for that gene.
    """
    rng = np.random.default_rng(seed)

    # -------------------------
    # Covariates (include intercept)
    # -------------------------
    # Basic covariates: batch (0/1), library size, etc.
    batch = rng.integers(0, 2, size=N).astype(np.float64)
    libsize = rng.lognormal(mean=10.0, sigma=0.3, size=N).astype(np.float64)
    depth = np.log1p(libsize)
    covs = [batch, depth]

    # Add extra nuisance covariates
    for _ in range(max(0, n_cov - 2)):
        covs.append(rng.normal(0, 1, size=N).astype(np.float64))

    C_no_intercept = np.column_stack(covs)  # (N, n_cov)
    intercept = np.ones((N, 1), dtype=np.float64)
    C = np.column_stack([intercept, C_no_intercept])  # (N, p)
    p = C.shape[1]

    # -------------------------
    # Guides and guide2gene mapping
    # -------------------------
    gene_names = [f"GENE{g:04d}" for g in range(n_genes)]
    guide_names = []
    guide2gene = {}
    gene2guides = {}

    for g, gene in enumerate(gene_names):
        gs = []
        for j in range(guides_per_gene):
            guide = f"g{g:04d}_{j:02d}"
            guide_names.append(guide)
            guide2gene[guide] = gene
            gs.append(guide)
        gene2guides[gene] = gs

    n_guides = len(guide_names)

    # -------------------------
    # Guide assignment matrix (binary, sparse), with MOI ~ Poisson(moi_mean)
    # -------------------------
    # Sample number of guides per cell, then choose those guides uniformly.
    # This is not a realistic guide assignment model; it’s a stress test for plumbing.
    moI = rng.poisson(moi_mean, size=N)
    moI = np.clip(moI, 0, min(50, n_guides))  # avoid pathological large draws

    rows = []
    cols = []
    for i in range(N):
        m = int(moI[i])
        if m == 0:
            continue
        chosen = rng.choice(n_guides, size=m, replace=False)
        rows.extend([i] * m)
        cols.extend(chosen.tolist())

    data = np.ones(len(rows), dtype=np.int8)
    Gmat = sp.csr_matrix((data, (np.array(rows), np.array(cols))), shape=(N, n_guides), dtype=np.int8)
    Gmat.eliminate_zeros()

    # -------------------------
    # Usage model (compositional)
    # -------------------------
    # Generate logits as linear function of covariates, then softmax -> composition.
    W = rng.normal(0, 0.3, size=(p, K))
    logits = C @ W + rng.normal(0, 0.2, size=(N, K))

    # Optional true effect: increase one program’s logit in treated cells
    if add_true_effect:
        true_gene = gene_names[true_gene_idx]
        guide_idx = np.array([guide_names.index(g) for g in gene2guides[true_gene]], dtype=np.int32)
        # union indicator
        x_union = (Gmat[:, guide_idx].sum(axis=1).A1 > 0).astype(np.float64)
        logits[:, true_prog_idx] += true_beta * x_union

    usage = _softmax_rows(logits).astype(np.float64)

    # sprinkle near-zeros: push a fraction of entries down and renormalize
    # (mimics cNMF having lots of tiny usages)
    mask = rng.random(size=usage.shape) < 0.15
    usage[mask] *= 1e-6
    usage /= usage.sum(axis=1, keepdims=True)

    # -------------------------
    # Build AnnData
    # -------------------------
    adata = ad.AnnData(X=sp.csr_matrix((N, 1), dtype=np.float32))
    adata.obs = pd.DataFrame(index=[f"cell{i:06d}" for i in range(N)])
    adata.obsm["covar"] = C
    adata.obsm["usage"] = usage
    adata.obsm["guide_assignment"] = Gmat
    adata.uns["guide2gene"] = guide2gene
    adata.uns["gene2guides"] = gene2guides
    adata.uns["guide_names"] = guide_names
    adata.uns["gene_names"] = gene_names

    return adata
```

---

## `tests/conftest.py` (fixtures)

```python
import pytest
from .utils_synth import make_synthetic_adata

@pytest.fixture(scope="session")
def adata_small():
    return make_synthetic_adata(N=800, K=10, n_genes=10, guides_per_gene=6, seed=1)

@pytest.fixture(scope="session")
def adata_medium():
    return make_synthetic_adata(N=5000, K=20, n_genes=30, guides_per_gene=6, seed=2)

@pytest.fixture(scope="session")
def adata_with_effect():
    return make_synthetic_adata(
        N=4000, K=15, n_genes=20, guides_per_gene=6, seed=3,
        add_true_effect=True, true_gene_idx=0, true_prog_idx=2, true_beta=0.8
    )
```

---

## Unit tests (expecting your package API)

### `tests/test_preprocess.py`

```python
import numpy as np

from yourpkg.preprocess import clr_transform  # you implement

def test_clr_rowsum_zero(adata_small):
    U = adata_small.obsm["usage"]
    Y = clr_transform(U, eps=1e-6)
    assert np.all(np.isfinite(Y))
    assert np.allclose(Y.sum(axis=1), 0.0, atol=1e-6)

def test_clr_pairwise_ratio_identity(adata_small):
    U = adata_small.obsm["usage"]
    eps = 1e-6
    Y, Ueps = clr_transform(U, eps=eps, return_u_eps=True)  # optional convenience
    a, b = 1, 3
    lhs = Y[:, a] - Y[:, b]
    rhs = np.log(Ueps[:, a]) - np.log(Ueps[:, b])
    assert np.allclose(lhs, rhs, atol=1e-6)
```

### `tests/test_design.py`

```python
import numpy as np

from yourpkg.design import union_indicator  # you implement

def test_union_indicator_matches_or(adata_small):
    G = adata_small.obsm["guide_assignment"]
    gene2guides = adata_small.uns["gene2guides"]
    guide_names = adata_small.uns["guide_names"]

    gene = list(gene2guides.keys())[0]
    guides = gene2guides[gene]
    guide_idx = [guide_names.index(g) for g in guides]

    x_union = union_indicator(G, guide_idx)

    # explicit OR
    x_or = (G[:, guide_idx].sum(axis=1).A1 > 0).astype(np.int8)
    assert np.array_equal(x_union.astype(np.int8), x_or)
```

### `tests/test_propensity.py`

```python
import numpy as np

from yourpkg.propensity import fit_propensity  # you implement

def test_propensity_intercept_only_constant(adata_small):
    x = np.random.default_rng(0).integers(0, 2, size=adata_small.n_obs).astype(np.int8)
    C = np.ones((adata_small.n_obs, 1), dtype=np.float64)  # intercept-only
    p = fit_propensity(C, x)
    assert np.std(p) < 1e-8
    assert abs(p.mean() - x.mean()) < 1e-2
    assert np.all((p > 0) & (p < 1))
```

### `tests/test_sampler.py`

```python
import numpy as np

from yourpkg.sampler import crt_index_sampler_fast_csc  # you implement (numba), returns (indptr, indices)

def _dense_from_csc(indptr, indices, B, N):
    X = np.zeros((B, N), dtype=np.uint8)
    for b in range(B):
        idx = indices[indptr[b]:indptr[b+1]]
        X[b, idx] = 1
    return X

def test_sampler_matches_naive_marginals():
    rng = np.random.default_rng(0)
    N = 400
    B = 1500
    p = rng.uniform(0.001, 0.05, size=N).astype(np.float64)

    # naive
    Xnaive = (rng.random((B, N)) < p[None, :]).astype(np.uint8)

    # fast
    indptr, indices = crt_index_sampler_fast_csc(p, B, seed=123)
    Xfast = _dense_from_csc(indptr, indices, B, N)

    # per-cell inclusion rates
    err = np.max(np.abs(Xfast.mean(axis=0) - p))
    assert err < 6e-3

    # total treated per resample mean ~ sum p
    assert abs(Xfast.sum(axis=1).mean() - p.sum()) < 0.03 * p.sum()

def test_no_duplicates_within_resample():
    rng = np.random.default_rng(1)
    N, B = 200, 800
    p = rng.uniform(0.01, 0.2, size=N).astype(np.float64)
    indptr, indices = crt_index_sampler_fast_csc(p, B, seed=7)
    for b in range(B):
        idx = indices[indptr[b]:indptr[b+1]]
        assert idx.size == np.unique(idx).size
```

### `tests/test_ols.py`

```python
import numpy as np

from yourpkg.ols import beta_from_summaries  # you implement

def test_beta_matches_lstsq(adata_small):
    rng = np.random.default_rng(0)
    N = adata_small.n_obs
    K = adata_small.obsm["usage"].shape[1]

    # build synthetic Y directly for this unit test
    Y = rng.normal(0, 1, size=(N, K)).astype(np.float64)

    # C includes intercept
    C = adata_small.obsm["covar"].astype(np.float64)
    x = rng.integers(0, 2, size=N).astype(np.int8)

    beta_fast = beta_from_summaries(x, C, Y)  # (K,)

    # reference
    X = np.column_stack([x.astype(np.float64), C])
    beta_ref = np.zeros(K, dtype=np.float64)
    for k in range(K):
        coef, *_ = np.linalg.lstsq(X, Y[:, k], rcond=None)
        beta_ref[k] = coef[0]

    assert np.allclose(beta_fast, beta_ref, atol=1e-7, rtol=1e-6)
```

### `tests/test_crt.py`

```python
import numpy as np

from yourpkg.crt import crt_empirical_pvalue  # you implement

def test_crt_pvalue_handcount():
    beta_obs = 2.0
    beta_null = np.array([0.1, 2.0, -3.0, 1.9])
    p = crt_empirical_pvalue(beta_obs, beta_null)
    assert abs(p - 0.6) < 1e-12
```

---

## Integration sanity (null / power)

### `tests/test_integration_null.py` (mark as slow)

```python
import os
import numpy as np
import pytest

from yourpkg.preprocess import clr_transform
from yourpkg.design import union_indicator
from yourpkg.propensity import fit_propensity
from yourpkg.sampler import crt_index_sampler_fast_csc
from yourpkg.ols import beta_from_summaries
from yourpkg.crt import crt_empirical_pvalue

SLOW = os.environ.get("RUN_SLOW", "0") == "1"

@pytest.mark.skipif(not SLOW, reason="set RUN_SLOW=1 to run")
def test_type1_error_approximately_uniform(adata_medium):
    rng = np.random.default_rng(0)
    G = adata_medium.obsm["guide_assignment"]
    C = adata_medium.obsm["covar"].astype(np.float64)
    U = adata_medium.obsm["usage"].astype(np.float64)
    Y = clr_transform(U, eps=1e-6)

    guide_names = adata_medium.uns["guide_names"]
    gene2guides = adata_medium.uns["gene2guides"]
    genes = list(gene2guides.keys())[:8]  # keep small

    B = 399
    pvals = []

    for gene in genes:
        guide_idx = [guide_names.index(g) for g in gene2guides[gene]]
        x = union_indicator(G, guide_idx).astype(np.int8)

        # if too few treated, skip
        if x.sum() < 20 or x.sum() > (len(x) - 20):
            continue

        p = fit_propensity(C, x)
        indptr, indices = crt_index_sampler_fast_csc(p, B, seed=hash(gene) % (2**31 - 1))

        beta_obs = beta_from_summaries(x, C, Y)  # (K,)

        # compute null betas (K,) for each b, then pvals for a subset of programs
        # (you'll likely have a faster vectorized/numba path; this is correctness-only)
        for k in range(min(5, Y.shape[1])):
            beta_null = np.empty(B, dtype=np.float64)
            for b in range(B):
                xt = np.zeros_like(x)
                idx = indices[indptr[b]:indptr[b+1]]
                xt[idx] = 1
                beta_null[b] = beta_from_summaries(xt, C, Y[:, [k]])[0]
            pvals.append(crt_empirical_pvalue(beta_obs[k], beta_null))

    pvals = np.array(pvals)
    assert pvals.size > 20
    # crude uniformity check: mean should be ~0.5
    assert abs(pvals.mean() - 0.5) < 0.1
```

---

## Notes that prevent “false failures”

* Seed policy: use deterministic per-gene seeds (e.g., `seed = hash(gene) mod 2^31-1`) so parallel runs match serial.
* In integration tests, skip genes with extremely low/high treated count (CRT degenerates).
* Keep “naive vs fast sampler” tests small enough to build dense `(B,N)` matrices.


