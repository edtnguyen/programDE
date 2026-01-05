import numpy as np

from src.sceptre.crt import crt_index_sampler_fast_numba


def _build_dense_from_indices(indptr, indices, B, N):
    X = np.zeros((B, N), dtype=np.uint8)
    for b in range(B):
        X[b, indices[indptr[b] : indptr[b + 1]]] = 1
    return X


def test_crt_sampler_marginal_inclusion():
    rng = np.random.default_rng(0)
    N = 200
    B = 500
    p = rng.uniform(0.001, 0.05, size=N).astype(np.float64)
    Xnaive = (rng.random((B, N)) < p[None, :]).astype(np.uint8)

    indptr, indices = crt_index_sampler_fast_numba(p, B, seed=1)
    Xfast = _build_dense_from_indices(indptr, indices, B, N)

    mean_abs_fast = np.mean(np.abs(Xfast.mean(axis=0) - p))
    mean_abs_naive = np.mean(np.abs(Xnaive.mean(axis=0) - Xfast.mean(axis=0)))
    assert mean_abs_fast < 1e-2
    assert mean_abs_naive < 1e-2


def test_crt_sampler_total_treated_distribution():
    rng = np.random.default_rng(1)
    N = 150
    B = 400
    p = rng.uniform(0.01, 0.08, size=N).astype(np.float64)
    indptr, indices = crt_index_sampler_fast_numba(p, B, seed=2)
    Xfast = _build_dense_from_indices(indptr, indices, B, N)
    s = Xfast.sum(axis=1)

    mean_target = p.sum()
    var_target = (p * (1.0 - p)).sum()
    assert abs(s.mean() - mean_target) < 0.05 * mean_target
    assert abs(s.var() - var_target) < 0.2 * var_target


def test_crt_sampler_no_duplicates_per_resample():
    rng = np.random.default_rng(2)
    N = 120
    B = 200
    p = rng.uniform(0.01, 0.05, size=N).astype(np.float64)
    indptr, indices = crt_index_sampler_fast_numba(p, B, seed=3)
    for b in range(B):
        idx = indices[indptr[b] : indptr[b + 1]]
        assert idx.size == np.unique(idx).size


def test_crt_sampler_determinism():
    rng = np.random.default_rng(3)
    N = 80
    B = 120
    p = rng.uniform(0.01, 0.07, size=N).astype(np.float64)

    indptr1, indices1 = crt_index_sampler_fast_numba(p, B, seed=4)
    indptr2, indices2 = crt_index_sampler_fast_numba(p, B, seed=4)
    assert np.array_equal(indptr1, indptr2)
    assert np.array_equal(indices1, indices2)

    indptr3, indices3 = crt_index_sampler_fast_numba(p, B, seed=5)
    assert not (np.array_equal(indptr1, indptr3) and np.array_equal(indices1, indices3))
