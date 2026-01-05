import time

import numpy as np
import pytest

from src.sceptre.crt import crt_index_sampler_fast_numba, crt_pvals_for_gene


@pytest.mark.performance
def test_crt_performance_smoke():
    rng = np.random.default_rng(7)
    N = 400
    p = 4
    K = 5
    B = 63

    C = rng.normal(size=(N, p))
    Y = rng.normal(size=(N, K))
    CtC = C.T @ C
    A = np.linalg.inv(CtC)
    CTY = C.T @ Y
    obs_idx = np.flatnonzero(rng.random(N) < 0.2).astype(np.int32)
    p_prop = rng.uniform(0.05, 0.2, size=N).astype(np.float64)

    crt_index_sampler_fast_numba(p_prop, 5, seed=0)
    crt_pvals_for_gene(
        np.array([0, 0], dtype=np.int64),
        np.empty(0, dtype=np.int32),
        C,
        Y,
        A,
        CTY,
        obs_idx,
        1,
    )

    indptr, indices = crt_index_sampler_fast_numba(p_prop, B, seed=1)
    start = time.perf_counter()
    crt_pvals_for_gene(indptr, indices, C, Y, A, CTY, obs_idx, B)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
