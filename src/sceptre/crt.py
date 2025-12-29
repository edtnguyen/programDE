"""
Numba-accelerated CRT resampling and coefficient computation.
"""

import numba as nb
import numpy as np


@nb.njit(inline="always")
def _sample_unique_ints(B: int, m: int, out: np.ndarray, start: int) -> None:
    """
    B: number of resample
    Use JIT compiler from numba library to compile this function into machine code before
    running to make it run faster.
    Fill `out[start:start+m]` with unique integers in [0, B): Randomly draw m unique resample IDs in the
    range [0,B) without replacement.
    """
    for t in range(m):
        while True:
            r = np.random.randint(0, B)
            dup = False
            for s in range(t):
                if out[start + s] == r:
                    dup = True
                    break
            if not dup:
                out[start + t] = r
                break


@nb.njit
def crt_index_sampler_fast_numba(p: np.ndarray, B: int, seed: int):
    """
    For each cell j first draw M_j ~ Binomial(B, p_j), then choose M_j resample IDs without replacement.
    Returns resample-wise index lists in CSC-like (indptr, indices) format.
    """
    np.random.seed(seed)
    N = p.shape[0]

    M = np.empty(N, dtype=np.int32)
    offsets = np.empty(N + 1, dtype=np.int64)
    offsets[0] = 0
    total = 0
    for j in range(N):
        mj = np.random.binomial(B, p[j])
        M[j] = mj
        total += mj
        offsets[j + 1] = total

    choices = np.empty(total, dtype=np.int32)
    counts = np.zeros(B, dtype=np.int32)

    for j in range(N):
        mj = M[j]
        if mj == 0:
            continue
        start = offsets[j]
        _sample_unique_ints(B, mj, choices, start)
        for t in range(mj):
            b = choices[start + t]
            counts[b] += 1

    indptr = np.empty(B + 1, dtype=np.int64)
    indptr[0] = 0
    run = 0
    for b in range(B):
        run += counts[b]
        indptr[b + 1] = run

    indices = np.empty(total, dtype=np.int32)
    write = np.empty(B, dtype=np.int64)
    for b in range(B):
        write[b] = indptr[b]

    for j in range(N):
        mj = M[j]
        if mj == 0:
            continue
        start = offsets[j]
        for t in range(mj):
            b = choices[start + t]
            pos = write[b]
            indices[pos] = j
            write[b] = pos + 1

    return indptr, indices


@nb.njit
def _beta_from_summaries(
    n1: int,
    v: np.ndarray,
    sY: np.ndarray,
    A: np.ndarray,
    CTY: np.ndarray,
) -> np.ndarray:
    """
    Closed-form OLS coefficient for x ~ N1 and covariates summarized by v, sY.
    """
    p = v.shape[0]
    K = sY.shape[0]

    tmp = np.zeros(p, dtype=np.float64)
    for i in range(p):
        acc = 0.0
        for j in range(p):
            acc += v[j] * A[j, i]
        tmp[i] = acc

    den = float(n1)
    for i in range(p):
        den -= tmp[i] * v[i]
    if den <= 1e-12:
        return np.zeros(K, dtype=np.float64)

    beta = np.empty(K, dtype=np.float64)
    for k in range(K):
        acc = 0.0
        for i in range(p):
            acc += tmp[i] * CTY[i, k]
        beta[k] = (sY[k] - acc) / den
    return beta


@nb.njit
def crt_pvals_for_gene(
    indptr: np.ndarray,
    indices: np.ndarray,
    C: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    CTY: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
):
    """
    Compute CRT p-values and observed betas for one gene.
    """
    N, p = C.shape
    K = Y.shape[1]

    n1_obs = obs_idx.shape[0]
    v_obs = np.zeros(p, dtype=np.float64)
    sY_obs = np.zeros(K, dtype=np.float64)
    for t in range(n1_obs):
        i = obs_idx[t]
        for j in range(p):
            v_obs[j] += C[i, j]
        for k in range(K):
            sY_obs[k] += Y[i, k]

    beta_obs = _beta_from_summaries(n1_obs, v_obs, sY_obs, A, CTY)
    abs_obs = np.abs(beta_obs)

    ge = np.zeros(K, dtype=np.int32)

    for b in range(B):
        start = indptr[b]
        end = indptr[b + 1]
        n1 = end - start

        v = np.zeros(p, dtype=np.float64)
        sY = np.zeros(K, dtype=np.float64)
        for pos in range(start, end):
            i = indices[pos]
            for j in range(p):
                v[j] += C[i, j]
            for k in range(K):
                sY[k] += Y[i, k]

        beta = _beta_from_summaries(n1, v, sY, A, CTY)
        for k in range(K):
            if abs(beta[k]) >= abs_obs[k]:
                ge[k] += 1

    pvals = np.empty(K, dtype=np.float64)
    for k in range(K):
        pvals[k] = (1.0 + ge[k]) / (B + 1.0)

    return pvals, beta_obs
