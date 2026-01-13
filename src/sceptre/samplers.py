"""
CRT resampling samplers.
"""

from typing import Optional, Tuple

import numpy as np

from .crt import crt_index_sampler_fast_numba


def bernoulli_index_sampler(
    p: np.ndarray, B: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper around the existing Bernoulli index sampler.
    Returns (indptr, indices) in CSC-like format.
    """
    return crt_index_sampler_fast_numba(p, B, seed)


def compute_bins(
    values: np.ndarray,
    n_bins: int,
    method: str = "quantile",
    clip_quantiles: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, int, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if n_bins <= 1:
        return np.zeros(values.shape[0], dtype=np.int32), 1, np.array([])

    vals = values
    q_lo, q_hi = clip_quantiles
    if q_lo < 0.0 or q_hi > 1.0 or q_lo > q_hi:
        raise ValueError("clip_quantiles must be within [0,1] and q_lo <= q_hi.")

    if q_lo > 0.0 or q_hi < 1.0:
        lo = np.quantile(vals, q_lo)
        hi = np.quantile(vals, q_hi)
        vals = np.clip(vals, lo, hi)

    if method == "quantile":
        edges = np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1))
    elif method == "uniform":
        edges = np.linspace(np.min(vals), np.max(vals), n_bins + 1)
    else:
        raise ValueError("method must be 'quantile' or 'uniform'.")

    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int32), 1, edges

    n_bins_eff = edges.size - 1
    bin_id = np.searchsorted(edges, vals, side="right") - 1
    bin_id = np.clip(bin_id, 0, n_bins_eff - 1).astype(np.int32)
    return bin_id, n_bins_eff, edges


def _propensity_bins(p_hat: np.ndarray, n_bins: int) -> Tuple[np.ndarray, int]:
    bin_id, n_bins_eff, _ = compute_bins(p_hat, n_bins, method="quantile")
    return bin_id, n_bins_eff


def stratified_permutation_sampler(
    x_obs: np.ndarray,
    p_hat: np.ndarray,
    B: int,
    seed: int,
    batch_raw: Optional[np.ndarray] = None,
    n_bins: int = 20,
    stratify_by_batch: bool = True,
    min_stratum_size: int = 2,
    burden_values: Optional[np.ndarray] = None,
    n_burden_bins: int = 8,
    burden_bin_method: str = "quantile",
    burden_clip_quantiles: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified permutation CRT sampler.
    Returns (indptr, indices) in the same format as bernoulli_index_sampler.
    """
    if B <= 0:
        raise ValueError("B must be positive.")

    x_obs = np.asarray(x_obs, dtype=np.int8)
    p_hat = np.asarray(p_hat, dtype=np.float64)
    if x_obs.ndim != 1 or p_hat.ndim != 1:
        raise ValueError("x_obs and p_hat must be 1D arrays.")
    if x_obs.shape[0] != p_hat.shape[0]:
        raise ValueError("x_obs and p_hat must have the same length.")

    if not np.all(np.isfinite(p_hat)):
        raise ValueError("p_hat contains non-finite values.")

    bin_id, n_bins_eff = _propensity_bins(p_hat, n_bins)

    burden_bin = None
    n_burden_bins_eff = 1
    if burden_values is not None:
        burden_values = np.asarray(burden_values, dtype=np.float64)
        if burden_values.ndim != 1:
            raise ValueError("burden_values must be a 1D array.")
        if burden_values.shape[0] != p_hat.shape[0]:
            raise ValueError("burden_values must have the same length as p_hat.")
        if not np.all(np.isfinite(burden_values)):
            raise ValueError("burden_values contains non-finite values.")
        burden_bin, n_burden_bins_eff, _ = compute_bins(
            burden_values,
            n_burden_bins,
            method=burden_bin_method,
            clip_quantiles=burden_clip_quantiles,
        )
    else:
        burden_bin = np.zeros_like(bin_id, dtype=np.int32)

    if stratify_by_batch and batch_raw is not None:
        import pandas as pd

        batch_id, _ = pd.factorize(batch_raw, sort=False)
        batch_id = batch_id.astype(np.int64, copy=False)
        stratum_id = batch_id * (n_bins_eff * n_burden_bins_eff) + bin_id * n_burden_bins_eff + burden_bin
    else:
        stratum_id = (bin_id * n_burden_bins_eff + burden_bin).astype(np.int64, copy=False)

    if min_stratum_size is not None and min_stratum_size > 1:
        unique, counts = np.unique(stratum_id, return_counts=True)
        small = unique[counts < min_stratum_size]
        if small.size > 0:
            misc_id = int(stratum_id.max()) + 1 if stratum_id.size else 0
            stratum_id = stratum_id.copy()
            mask = np.isin(stratum_id, small)
            stratum_id[mask] = misc_id

    x_obs_int = x_obs.astype(np.int32, copy=False)
    total_treated = int(x_obs_int.sum())
    if total_treated == 0:
        indptr = np.zeros(B + 1, dtype=np.int64)
        return indptr, np.empty(0, dtype=np.int32)

    strata = []
    for sid in np.unique(stratum_id):
        idx_s = np.nonzero(stratum_id == sid)[0].astype(np.int32, copy=False)
        m_s = int(x_obs_int[idx_s].sum())
        strata.append((idx_s, m_s))

    rng = np.random.default_rng(seed)
    indptr = (np.arange(B + 1, dtype=np.int64) * total_treated)
    indices = np.empty(total_treated * B, dtype=np.int32)

    for b in range(B):
        pos = b * total_treated
        for idx_s, m_s in strata:
            if m_s == 0:
                continue
            if m_s == idx_s.size:
                chosen = idx_s
            else:
                chosen = rng.choice(idx_s, size=m_s, replace=False)
            indices[pos : pos + m_s] = chosen
            pos += m_s

    return indptr, indices
