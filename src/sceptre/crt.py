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

    """
    offset[j] = starting index in the array that stores resemple IDs of cell j

"""
    cell_offsets = np.empty(N + 1, dtype=np.int64)
    cell_offsets[0] = 0
    total = 0
    for j in range(N):
        mj = np.random.binomial(B, p[j])
        M[j] = mj
        total += mj
        cell_offsets[j + 1] = total

"""
    choices array contains the resample IDs for all cells
    choices[ offsets[j] : offsets[j+1] ] = resample IDs for cell j
"""

    resample_ids = np.empty(total, dtype=np.int32)

    # count = number of cells in each resample ID 
    counts = np.zeros(B, dtype=np.int32)

    for j in range(N):
        mj = M[j]
        if mj == 0:
            continue
        start = cell_offsets[j]
        _sample_unique_ints(B, mj, resample_ids, start)
        for t in range(mj):
            b = resample_ids[start + t]
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
        start = cell_offsets[j]
        for t in range(mj):
            b = resample_ids[start + t]
            pos = write[b]
            indices[pos] = j
            write[b] = pos + 1

    return indptr, indices