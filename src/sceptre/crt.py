import numpy as np
import numba as nb


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

    cell_resample_idprt = np.empty(N + 1, dtype=np.int64)
    cell_resample_idprt[0] = 0
    for j in range(N):
        cell_resample_idprt[j + 1] = (
            cell_resample_idprt[j] + np.random.binomial(B, p[j])
        )

    total = cell_resample_idprt[N]
    cell_resample_ids = np.empty(total, dtype=np.int32)
    counts = np.zeros(B, dtype=np.int32)
    for j in range(N):
        start, end = cell_resample_idprt[j], cell_resample_idprt[j + 1]
        m = end - start
        if m == 0:
            continue
        _sample_unique_ints(B, m, cell_resample_ids, start)
        for i in range(start, end):
            b = cell_resample_ids[i]
            counts[b] += 1

    indptr = np.empty(B + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = np.cumsum(counts)

    indices = np.empty(total, dtype=np.int32)
    write_pos = indptr[:-1].copy()

    for j in range(N):
        start, end = cell_resample_idprt[j], cell_resample_idprt[j + 1]
        for i in range(start, end):
            b = cell_resample_ids[i]
            pos = write_pos[b]
            indices[pos] = j
            write_pos[b] = pos + 1

    return indptr, indices
