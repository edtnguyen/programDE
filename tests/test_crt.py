import numpy as np

from src.sceptre.crt import (
    crt_betas_for_gene,
    crt_index_sampler_fast_numba,
    crt_pvals_for_gene,
)


def test_crt_index_sampler_shapes():
    p = np.full(10, 0.2, dtype=np.float64)
    B = 17
    indptr, idx = crt_index_sampler_fast_numba(p, B, 123)
    assert indptr.shape[0] == B + 1
    assert idx.shape[0] == indptr[-1]


def test_crt_pvals_and_betas_shapes():
    rng = np.random.default_rng(0)
    N = 20
    B = 15
    p = np.full(N, 0.3, dtype=np.float64)
    indptr, idx = crt_index_sampler_fast_numba(p, B, 123)

    C = rng.normal(size=(N, 3))
    Y = rng.normal(size=(N, 4))
    A = np.linalg.inv(C.T @ C)
    CTY = C.T @ Y
    obs_idx = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    pvals, betas = crt_pvals_for_gene(indptr, idx, C, Y, A, CTY, obs_idx, B)
    assert pvals.shape == (4,)
    assert betas.shape == (4,)
    assert np.all(pvals >= 0.0)
    assert np.all(pvals <= 1.0)

    beta_obs, beta_null = crt_betas_for_gene(indptr, idx, C, Y, A, CTY, obs_idx, B)
    assert beta_obs.shape == (4,)
    assert beta_null.shape == (B, 4)
