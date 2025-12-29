import numpy as np
import numba as nb
from typing import List, Tuple, Any


@nb.njit(inline="always")
def _sample_unique_ints(B: int, m: int, out: np.ndarray, start: int) -> None:
    """
    Fill `out[start:start+m]` with unique integers in [0, B).
    """
    if m > 0:
        out[start : start + m] = np.random.choice(B, m, replace=False)


@nb.njit
def crt_index_sampler_fast_numba(p: np.ndarray, B: int, seed: int):
    """
    For each cell j: M_j ~ Binomial(B, p_j), then choose M_j resample IDs without replacement.
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