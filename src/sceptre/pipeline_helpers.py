"""
Internal helpers for CRT pipeline orchestration.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from .adata_utils import union_obs_idx_from_cols
from .crt import crt_betas_for_gene, crt_pvals_for_gene
from .samplers import bernoulli_index_sampler, stratified_permutation_sampler
from .skew_normal import compute_empirical_p_value, fit_and_evaluate_skew_normal


def _extract_probabilities(output) -> np.ndarray:
    """
    Normalize propensity model outputs into probability vector.
    """
    if isinstance(output, tuple) or isinstance(output, list):
        return np.asarray(output[0], dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _gene_obs_idx(inputs: Any, gene: str) -> np.ndarray:
    if gene not in inputs.gene_to_cols:
        raise KeyError(f"Gene `{gene}` not present in gene_to_cols mapping.")
    return union_obs_idx_from_cols(inputs.G, inputs.gene_to_cols[gene])


def _gene_seed(gene: str, base_seed: int) -> int:
    return (hash(gene) ^ base_seed) & 0xFFFFFFFF


def _union_indicator(n_cells: int, obs_idx: np.ndarray) -> np.ndarray:
    y01 = np.zeros(n_cells, dtype=np.int8)
    y01[obs_idx] = 1
    return y01


def _fit_propensity(
    inputs: Any,
    obs_idx: np.ndarray,
    propensity_model: Callable,
) -> np.ndarray:
    y01 = _union_indicator(inputs.C.shape[0], obs_idx)
    return _extract_probabilities(propensity_model(inputs.C, y01))


def _sample_crt_indices(
    p: np.ndarray,
    B: int,
    seed: int,
    *,
    resampling_method: str = "bernoulli_index",
    resampling_kwargs: Optional[Dict[str, Any]] = None,
    obs_idx: Optional[np.ndarray] = None,
    inputs: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if resampling_method == "bernoulli_index":
        return bernoulli_index_sampler(p, B, seed)

    if resampling_method != "stratified_perm":
        raise ValueError(
            "resampling_method must be 'bernoulli_index' or 'stratified_perm'."
        )
    if obs_idx is None:
        raise ValueError("obs_idx is required for stratified_perm resampling.")

    kwargs = dict(resampling_kwargs or {})
    n_bins = int(kwargs.get("n_bins", 20))
    stratify_by_batch = bool(kwargs.get("stratify_by_batch", True))
    batch_key = kwargs.get("batch_key", "batch")
    min_stratum_size = kwargs.get("min_stratum_size", 2)
    burden_key = kwargs.get("burden_key", None)
    n_burden_bins = int(kwargs.get("n_burden_bins", 8))
    burden_bin_method = kwargs.get("burden_bin_method", "quantile")
    burden_clip_quantiles = kwargs.get("burden_clip_quantiles", (0.0, 1.0))

    batch_raw = None
    burden_values = None
    covar_df_raw = getattr(inputs, "covar_df_raw", None) if inputs is not None else None

    if stratify_by_batch and inputs is not None:
        if covar_df_raw is not None and batch_key in covar_df_raw.columns:
            batch_raw = covar_df_raw[batch_key].to_numpy()
        else:
            batch_raw = getattr(inputs, "batch_raw", None)

    if burden_key is not None:
        if covar_df_raw is None or burden_key not in covar_df_raw.columns:
            raise ValueError(
                f"burden_key '{burden_key}' not found in covariate DataFrame."
            )
        burden_series = pd.to_numeric(covar_df_raw[burden_key], errors="coerce")
        burden_values = burden_series.to_numpy()

    if burden_values is not None and not np.all(np.isfinite(burden_values)):
        raise ValueError("burden_values contains NaNs or non-finite values.")

    x_obs = _union_indicator(p.shape[0], obs_idx)
    return stratified_permutation_sampler(
        x_obs=x_obs,
        p_hat=p,
        B=B,
        seed=seed,
        batch_raw=batch_raw,
        n_bins=n_bins,
        stratify_by_batch=stratify_by_batch,
        min_stratum_size=min_stratum_size,
        burden_values=burden_values,
        n_burden_bins=n_burden_bins,
        burden_bin_method=burden_bin_method,
        burden_clip_quantiles=burden_clip_quantiles,
    )


def _normalize_test_stat(test_stat: Optional[str]) -> str:
    if test_stat is None:
        return "ols"
    stat = str(test_stat).lower()
    if stat in ("ols", "beta"):
        return "ols"
    if stat in ("utest", "wilcoxon", "mannwhitney", "mannwhitneyu"):
        return "utest"
    raise ValueError("test_stat must be 'ols' or 'utest'.")


def _usage_for_ranks(U: np.ndarray, eps_quantile: float) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    eps = np.quantile(U, eps_quantile)
    if not np.isfinite(eps) or eps <= 0.0:
        eps = np.finfo(np.float64).tiny
    U2 = np.maximum(U, eps)
    row_sums = U2.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return U2 / row_sums


def _ensure_rank_matrix(
    inputs: Any,
    use: str = "clr",
    rank_method: str = "average",
    rank_dtype: str = "float32",
) -> np.ndarray:
    info = getattr(inputs, "rank_info", None) or {}
    cached = getattr(inputs, "R", None)
    if (
        cached is not None
        and info.get("use") == use
        and info.get("rank_method") == rank_method
        and info.get("rank_dtype") == rank_dtype
    ):
        return cached

    if use == "clr":
        Y = inputs.Y
    elif use == "usage":
        U = getattr(inputs, "U", None)
        if U is None:
            raise ValueError("inputs.U is required for use='usage' in utest.")
        eps_q = float(getattr(inputs, "usage_eps_quantile", 1e-4))
        Y = _usage_for_ranks(U, eps_q)
    else:
        raise ValueError("test_stat_kwargs['use'] must be 'clr' or 'usage'.")

    if not np.all(np.isfinite(Y)):
        raise ValueError("Non-finite values found in outcome matrix for ranking.")

    N, K = Y.shape
    R = np.empty((N, K), dtype=rank_dtype)
    for k in range(K):
        R[:, k] = rankdata(Y[:, k], method=rank_method)

    inputs.R = R
    inputs.rank_info = {
        "use": use,
        "rank_method": rank_method,
        "rank_dtype": rank_dtype,
    }
    return R


def _rank_sums_from_indices(
    indptr: np.ndarray,
    indices: np.ndarray,
    R: np.ndarray,
    N: int,
    B: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.ones(indices.shape[0], dtype=np.float64)
    X = sp.csc_matrix((data, indices, indptr), shape=(N, B))
    rank_sum_null = (X.T @ R).astype(np.float64)
    n1b = (indptr[1:] - indptr[:-1]).astype(np.int32)
    return rank_sum_null, n1b


def _rank_biserial_from_rank_sum(
    rank_sum: np.ndarray,
    n1: np.ndarray,
    N: int,
) -> np.ndarray:
    n1_arr = np.asarray(n1, dtype=np.float64)
    if rank_sum.ndim == 2 and n1_arr.ndim == 1:
        n1_arr = n1_arr[:, None]
    n0_arr = N - n1_arr
    denom = n1_arr * n0_arr
    U = rank_sum - n1_arr * (n1_arr + 1.0) / 2.0
    rbc = np.full_like(U, np.nan, dtype=np.float64)
    np.divide(2.0 * U, denom, out=rbc, where=denom > 0.0)
    rbc = rbc - 1.0
    return rbc


def _utest_stats_from_indices(
    inputs: Any,
    indptr: np.ndarray,
    indices: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
    test_stat_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    kwargs = dict(test_stat_kwargs or {})
    use = kwargs.get("use", "clr")
    rank_method = kwargs.get("rank_method", "average")
    rank_dtype = kwargs.get("rank_dtype", "float32")

    R = _ensure_rank_matrix(inputs, use=use, rank_method=rank_method, rank_dtype=rank_dtype)
    N = R.shape[0]

    n1_obs = obs_idx.size
    if n1_obs > 0:
        rank_sum_obs = np.asarray(R[obs_idx].sum(axis=0), dtype=np.float64)
    else:
        rank_sum_obs = np.zeros(R.shape[1], dtype=np.float64)
    rbc_obs = _rank_biserial_from_rank_sum(rank_sum_obs, n1_obs, N)

    rank_sum_null, n1b = _rank_sums_from_indices(indptr, indices, R, N, B)
    rbc_null = _rank_biserial_from_rank_sum(rank_sum_null, n1b, N)
    return rbc_obs, rbc_null


def _empirical_crt(
    inputs: Any,
    indptr: np.ndarray,
    idx: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return crt_pvals_for_gene(
        indptr,
        idx,
        inputs.C,
        inputs.Y,
        inputs.A,
        inputs.CTY,
        obs_idx.astype(np.int32),
        B,
    )


def _empirical_utest(
    inputs: Any,
    indptr: np.ndarray,
    idx: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
    test_stat_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rbc_obs, rbc_null = _utest_stats_from_indices(
        inputs, indptr, idx, obs_idx, B, test_stat_kwargs
    )
    if not np.all(np.isfinite(rbc_obs)):
        return np.ones(rbc_obs.shape[0], dtype=np.float64), rbc_obs
    pvals = _raw_pvals_from_stats(rbc_obs, rbc_null)
    return pvals, rbc_obs


def _utest_skew_calibrated_crt(
    inputs: Any,
    indptr: np.ndarray,
    idx: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
    side_code: int,
    test_stat_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rbc_obs, rbc_null = _utest_stats_from_indices(
        inputs, indptr, idx, obs_idx, B, test_stat_kwargs
    )
    pvals_sn, skew_params = _compute_skew_normal_pvals(
        rbc_obs, rbc_null, side_code
    )
    pvals_raw = _raw_pvals_from_stats(rbc_obs, rbc_null)
    return pvals_sn, rbc_obs, skew_params, pvals_raw


def _compute_skew_normal_pvals(
    beta_obs: np.ndarray,
    beta_null: np.ndarray,
    side_code: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit skew-normal to null statistics per program and compute calibrated p-values.
    """
    K = beta_obs.shape[0]
    pvals_sn = np.empty(K, dtype=np.float64)
    params = np.empty((K, 3), dtype=np.float64)

    for k in range(K):
        null_k = beta_null[:, k]
        raw_k = (1.0 + np.sum(np.abs(null_k) >= abs(beta_obs[k]))) / (
            null_k.shape[0] + 1.0
        )
        mu = null_k.mean()
        sd = null_k.std()
        if not np.isfinite(sd) or sd <= 0.0:
            pvals_sn[k] = raw_k
            params[k, :] = np.nan
            continue

        z_null = (null_k - mu) / sd
        z_obs = (beta_obs[k] - mu) / sd

        sn_result = fit_and_evaluate_skew_normal(z_obs, z_null, side_code)
        params_k = sn_result[:3]
        params[k, :] = params_k if np.all(np.isfinite(params_k)) else np.nan
        p_val = sn_result[3]
        if p_val < 0.0:
            p_val = compute_empirical_p_value(z_null, z_obs, side_code)
        if not np.isfinite(p_val):
            p_val = raw_k
        pvals_sn[k] = p_val

    return pvals_sn, params


def _raw_pvals_from_stats(stat_obs: np.ndarray, stat_null: np.ndarray) -> np.ndarray:
    abs_obs = np.abs(stat_obs)
    ge = np.sum(np.abs(stat_null) >= abs_obs, axis=0)
    B = stat_null.shape[0]
    return (1.0 + ge) / (B + 1.0)


def _raw_pvals_from_betas(beta_obs: np.ndarray, beta_null: np.ndarray) -> np.ndarray:
    return _raw_pvals_from_stats(beta_obs, beta_null)


def _skew_calibrated_crt(
    inputs: Any,
    indptr: np.ndarray,
    idx: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
    side_code: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta_obs, beta_null = crt_betas_for_gene(
        indptr,
        idx,
        inputs.C,
        inputs.Y,
        inputs.A,
        inputs.CTY,
        obs_idx.astype(np.int32),
        B,
    )
    pvals_sn, skew_params = _compute_skew_normal_pvals(beta_obs, beta_null, side_code)
    pvals_raw = _raw_pvals_from_stats(beta_obs, beta_null)
    return pvals_sn, beta_obs, skew_params, pvals_raw


def _stack_gene_results(
    results: List[Any],
    gene_list: List[str],
    program_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    pval_mat = np.vstack([r.pvals for r in results])
    beta_mat = np.vstack([r.betas for r in results])
    treated = np.array([r.n_treated for r in results], dtype=int)

    pvals_df = pd.DataFrame(pval_mat, index=gene_list, columns=program_names)
    betas_df = pd.DataFrame(beta_mat, index=gene_list, columns=program_names)
    treated_df = pd.Series(treated, index=gene_list, name="n_union_positive_cells")
    return pvals_df, betas_df, treated_df


def _stack_skew_outputs(
    results: List[Any],
    gene_list: List[str],
    program_names: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    pvals_sn_mat = np.vstack([r.pvals_sn for r in results])
    skew_params = np.stack([r.skew_params for r in results])
    pvals_skew_df = pd.DataFrame(
        pvals_sn_mat, index=gene_list, columns=program_names
    )
    return pvals_skew_df, skew_params


def _stack_raw_outputs(
    results: List[Any],
    gene_list: List[str],
    program_names: List[str],
) -> pd.DataFrame:
    pvals_raw_mat = np.vstack([r.pvals_raw for r in results])
    return pd.DataFrame(pvals_raw_mat, index=gene_list, columns=program_names)
