import numba as nb
import numpy as np


@nb.njit(inline="always")
def _sample_unique_ints(B: int, m: int, out: np.ndarray, start: int) -> None:
    """
    Fill `out[start:start+m]` with unique integers in [0, B): randomly draw m resemble IDs from B
    """
    if m > 0:
        out[start : start + m] = np.random.choice(B, m, replace=False)


@nb.njit
def crt_index_sampler_fast_numba(p: np.ndarray, B: int, seed: int):
    """
    For each cell j: M_j ~ Binomial(B, p_j), then choose M_j resample IDs without replacement.
    Returns resample-wise index lists in CSC-like (indptr, indices) format.
    cell_resample_idprt[j] = starting index in the array that stores resample IDs of cell j.
    cell_resample_ids array contains the resample IDs for all cells
    cell_resample_ids[ cell_resample_idprt[j] : cell_resample_idprt[j+1] ] : resample IDs for cell j
    counts: number of cells in each resample ID
    """
    np.random.seed(seed)
    N = p.shape[0]
    M = np.empty(N, dtype=np.int32)

    cell_resample_idprt = np.empty(N + 1, dtype=np.int64)
    cell_resample_idprt[0] = 0
    total = 0
    for j in range(N):
        mj = np.random.binomial(B, p[j])
        M[j] = mj
        total += mj
        cell_resample_idprt[j + 1] = total

    cell_resample_ids = np.empty(total, dtype=np.int32)
    counts = np.zeros(B, dtype=np.int32)

    for j in range(N):
        mj = M[j]
        if mj == 0:
            continue
        start = cell_resample_idprt[j]
        _sample_unique_ints(B, mj, cell_resample_ids, start)
        for t in range(mj):
            b = cell_resample_ids[start + t]
            counts[b] += 1

    indptr = np.empty(B + 1, dtype=np.int64)
    indptr[0] = 0
    ncells = 0
    for b in range(B):
        ncells += counts[b]
        indptr[b + 1] = ncells

    indices = np.empty(total, dtype=np.int32)
    write = np.empty(B, dtype=np.int64)
    for b in range(B):
        write[b] = indptr[b]

    """
    write[b]: next write position in indices array for resample b
    indices array will store, for each resample b, the list of cell indices j that are included in resample b
    every time we find that cell j is included in resample b, we write j into indices[ write[b] ], and increment write[b] by 1
    indicating the next available position in indices for resample b
    """

    for j in range(N):
        mj = M[j]
        if mj == 0:
            continue
        start = cell_resample_idprt[j]
        for t in range(mj):
            b = cell_resample_ids[start + t]
            pos = write[b]
            indices[pos] = j
            write[b] = pos + 1

    return indptr, indices
