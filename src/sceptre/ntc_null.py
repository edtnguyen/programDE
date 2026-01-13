"""
NTC empirical-null p-values for CLR-OLS via matching on (n1, denom d).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .adata_utils import union_obs_idx_from_cols


@dataclass
class NTCNullResult:
    pvals: np.ndarray
    betas: np.ndarray
    n_treated: np.ndarray
    matching_info: Dict[str, Any]
    ntc_crossfit: Optional[Dict[str, Any]] = None


def split_ntc_units(
    ntc_unit_ids: np.ndarray,
    frac_A: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministically split NTC unit ids into A/B for cross-fit diagnostics.
    """
    ids = np.asarray(ntc_unit_ids, dtype=np.int64)
    if ids.size == 0:
        raise ValueError("ntc_unit_ids must contain at least one element.")
    if not (0.0 < frac_A < 1.0):
        raise ValueError("frac_A must be in (0, 1).")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids.size)
    cut = int(np.round(frac_A * ids.size))
    cut = np.clip(cut, 1, ids.size - 1)
    ids_A = ids[perm[:cut]]
    ids_B = ids[perm[cut:]]
    return ids_A, ids_B


def sample_ntc_pseudogenes_with_replacement(
    guide_names: Sequence[str],
    guide2gene: Mapping[str, str],
    ntc_labels: Sequence[str],
    guides_per_unit: int,
    n_units: int,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    ntc_set = set(ntc_labels)
    ntc_cols = [
        i for i, g in enumerate(guide_names) if guide2gene.get(g) in ntc_set
    ]
    if len(ntc_cols) < guides_per_unit:
        raise ValueError(
            "Not enough NTC guides to sample pseudo-genes. "
            f"Need {guides_per_unit}, have {len(ntc_cols)}."
        )
    ntc_cols = np.asarray(ntc_cols, dtype=np.int32)
    out: List[np.ndarray] = []
    for _ in range(int(n_units)):
        cols = rng.choice(ntc_cols, size=guides_per_unit, replace=False)
        out.append(np.sort(cols.astype(np.int32)))
    return out


def _summaries_for_obs_idx(
    C: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    CTY: np.ndarray,
    obs_idx: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    n1 = int(obs_idx.size)
    if n1 == 0:
        return np.zeros(Y.shape[1], dtype=np.float64), 0, 0.0
    v = C[obs_idx].sum(axis=0)
    sY = Y[obs_idx].sum(axis=0)
    tmp = A @ v
    denom = float(n1 - v @ tmp)
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.zeros(Y.shape[1], dtype=np.float64), n1, denom
    acc = tmp @ CTY
    beta = (sY - acc) / denom
    return beta.astype(np.float64, copy=False), n1, denom


def _compute_unit_stats(
    C: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    CTY: np.ndarray,
    obs_idx_list: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_units = len(obs_idx_list)
    K = Y.shape[1]
    betas = np.zeros((n_units, K), dtype=np.float64)
    n1 = np.zeros(n_units, dtype=np.int32)
    denom = np.zeros(n_units, dtype=np.float64)
    for i, obs_idx in enumerate(obs_idx_list):
        beta, n1_i, d = _summaries_for_obs_idx(C, Y, A, CTY, obs_idx)
        betas[i, :] = beta
        n1[i] = n1_i
        denom[i] = d
    return betas, n1, denom


def _make_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1:
        return np.array([])
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.array([])
    return edges


def _assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(values.shape[0], dtype=np.int32)
    bin_id = np.searchsorted(edges, values, side="right") - 1
    bin_id = np.clip(bin_id, 0, edges.size - 2)
    return bin_id.astype(np.int32)


def _match_ntc_bins(
    obs_bins: np.ndarray,
    ntc_bins: np.ndarray,
    n_bins_n1: int,
    n_bins_d: int,
    min_ntc_per_bin: int,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[str, Any]]:
    info: Dict[str, Any] = {"fallback_counts": 0, "bin_sizes": {}}
    obs_keys = np.unique(obs_bins, axis=0)
    match_map: Dict[Tuple[int, int], np.ndarray] = {}

    max_radius = max(n_bins_n1, n_bins_d)
    for key in obs_keys:
        key_tuple = (int(key[0]), int(key[1]))
        mask = (ntc_bins[:, 0] == key_tuple[0]) & (ntc_bins[:, 1] == key_tuple[1])
        idx = np.nonzero(mask)[0]
        if idx.size >= min_ntc_per_bin:
            match_map[key_tuple] = idx
            info["bin_sizes"][key_tuple] = int(idx.size)
            continue
        found = False
        for r in range(1, max_radius + 1):
            mask = (
                (np.abs(ntc_bins[:, 0] - key_tuple[0]) + np.abs(ntc_bins[:, 1] - key_tuple[1]))
                <= r
            )
            idx = np.nonzero(mask)[0]
            if idx.size >= min_ntc_per_bin:
                match_map[key_tuple] = idx
                info["bin_sizes"][key_tuple] = int(idx.size)
                info["fallback_counts"] += 1
                found = True
                break
        if not found:
            match_map[key_tuple] = np.arange(ntc_bins.shape[0], dtype=np.int32)
            info["bin_sizes"][key_tuple] = int(ntc_bins.shape[0])
            info["fallback_counts"] += 1
    return match_map, info


def _bin_edges_from_obs(
    n1_obs: np.ndarray,
    d_obs: np.ndarray,
    n_n1_bins: int,
    n_d_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n1_edges = _make_bin_edges(n1_obs.astype(np.float64), n_n1_bins)
    d_vals = np.log(np.maximum(d_obs.astype(np.float64), 1e-12))
    d_edges = _make_bin_edges(d_vals, n_d_bins)
    return n1_edges, d_edges


def _empirical_pvals_from_edges(
    beta_obs: np.ndarray,
    beta_ntc: np.ndarray,
    n1_obs: np.ndarray,
    d_obs: np.ndarray,
    n1_ntc: np.ndarray,
    d_ntc: np.ndarray,
    n1_edges: np.ndarray,
    d_edges: np.ndarray,
    min_ntc_per_bin: int,
    two_sided: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    d_vals_obs = np.log(np.maximum(d_obs.astype(np.float64), 1e-12))
    d_vals_ntc = np.log(np.maximum(d_ntc.astype(np.float64), 1e-12))

    obs_bins = np.column_stack(
        [
            _assign_bins(n1_obs.astype(np.float64), n1_edges),
            _assign_bins(d_vals_obs, d_edges),
        ]
    )
    ntc_bins = np.column_stack(
        [
            _assign_bins(n1_ntc.astype(np.float64), n1_edges),
            _assign_bins(d_vals_ntc, d_edges),
        ]
    )

    n_bins_n1_eff = max(1, n1_edges.size - 1)
    n_bins_d_eff = max(1, d_edges.size - 1)
    match_map, info = _match_ntc_bins(
        obs_bins,
        ntc_bins,
        n_bins_n1_eff,
        n_bins_d_eff,
        min_ntc_per_bin,
    )
    info.update(
        {
            "n_bins_n1": n_bins_n1_eff,
            "n_bins_d": n_bins_d_eff,
            "min_ntc_per_bin": int(min_ntc_per_bin),
        }
    )

    G, K = beta_obs.shape
    pvals = np.ones((G, K), dtype=np.float64)
    abs_beta_obs = np.abs(beta_obs) if two_sided else beta_obs

    obs_key_list = [tuple(row) for row in obs_bins.tolist()]
    obs_groups: Dict[Tuple[int, int], List[int]] = {}
    for i, key in enumerate(obs_key_list):
        obs_groups.setdefault(key, []).append(i)

    for key, obs_idx in obs_groups.items():
        ntc_idx = match_map.get(key)
        if ntc_idx is None or ntc_idx.size == 0:
            continue
        beta_ntc_bin = beta_ntc[ntc_idx, :]
        if two_sided:
            beta_ntc_bin = np.abs(beta_ntc_bin)
        for k in range(K):
            vals = np.sort(beta_ntc_bin[:, k])
            for g in obs_idx:
                count_ge = vals.size - np.searchsorted(
                    vals, abs_beta_obs[g, k], side="left"
                )
                pvals[g, k] = (1.0 + count_ge) / (vals.size + 1.0)

    return pvals, info


def empirical_pvals_vs_ntc(
    beta_obs: np.ndarray,
    beta_ntc: np.ndarray,
    n1_obs: np.ndarray,
    d_obs: np.ndarray,
    n1_ntc: np.ndarray,
    d_ntc: np.ndarray,
    *,
    n_n1_bins: int = 10,
    n_d_bins: int = 10,
    min_ntc_per_bin: int = 50,
    two_sided: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n1_edges, d_edges = _bin_edges_from_obs(n1_obs, d_obs, n_n1_bins, n_d_bins)
    return _empirical_pvals_from_edges(
        beta_obs,
        beta_ntc,
        n1_obs,
        d_obs,
        n1_ntc,
        d_ntc,
        n1_edges,
        d_edges,
        min_ntc_per_bin,
        two_sided,
    )


def compute_ntc_empirical_pvals_crossfit(
    beta_real: np.ndarray,
    sig_real: pd.DataFrame,
    beta_ntcA: np.ndarray,
    sig_ntcA: pd.DataFrame,
    beta_ntcB: np.ndarray,
    sig_ntcB: pd.DataFrame,
    matching_cfg: Dict[str, Any],
    two_sided: bool = True,
) -> Dict[str, Any]:
    """
    Compute cross-fit empirical p-values for real units and NTC_B vs NTC_A.
    """
    n_n1_bins = int(matching_cfg.get("n_n1_bins", 10))
    n_d_bins = int(matching_cfg.get("n_d_bins", 10))
    min_ntc_per_bin = int(matching_cfg.get("min_ntc_per_bin", 50))

    n1_real = sig_real["n1"].to_numpy(dtype=np.float64, copy=False)
    d_real = sig_real["d"].to_numpy(dtype=np.float64, copy=False)
    n1_ntcA = sig_ntcA["n1"].to_numpy(dtype=np.float64, copy=False)
    d_ntcA = sig_ntcA["d"].to_numpy(dtype=np.float64, copy=False)
    n1_ntcB = sig_ntcB["n1"].to_numpy(dtype=np.float64, copy=False)
    d_ntcB = sig_ntcB["d"].to_numpy(dtype=np.float64, copy=False)

    n1_edges, d_edges = _bin_edges_from_obs(n1_real, d_real, n_n1_bins, n_d_bins)

    p_real_vs_A, info_real = _empirical_pvals_from_edges(
        beta_real,
        beta_ntcA,
        n1_real,
        d_real,
        n1_ntcA,
        d_ntcA,
        n1_edges,
        d_edges,
        min_ntc_per_bin,
        two_sided,
    )
    p_ntcB_vs_A, info_ntcB = _empirical_pvals_from_edges(
        beta_ntcB,
        beta_ntcA,
        n1_ntcB,
        d_ntcB,
        n1_ntcA,
        d_ntcA,
        n1_edges,
        d_edges,
        min_ntc_per_bin,
        two_sided,
    )

    return {
        "p_real_vs_A": p_real_vs_A,
        "p_ntcB_vs_A": p_ntcB_vs_A,
        "matching_info": {"real_vs_A": info_real, "ntcB_vs_A": info_ntcB},
        "n1_edges": n1_edges,
        "d_edges": d_edges,
    }


def compute_ntc_empirical_pvals_gene_agg_crossfit(
    beta_real: np.ndarray,
    sig_real: pd.DataFrame,
    beta_ntcA: np.ndarray,
    sig_ntcA: pd.DataFrame,
    beta_ntcB: np.ndarray,
    sig_ntcB: pd.DataFrame,
    matching_cfg: Dict[str, Any],
    n1_edges: Optional[np.ndarray] = None,
    d_edges: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Cross-fit empirical p-values using max-|beta| aggregated per unit.
    """
    n_n1_bins = int(matching_cfg.get("n_n1_bins", 10))
    n_d_bins = int(matching_cfg.get("n_d_bins", 10))
    min_ntc_per_bin = int(matching_cfg.get("min_ntc_per_bin", 50))

    n1_real = sig_real["n1"].to_numpy(dtype=np.float64, copy=False)
    d_real = sig_real["d"].to_numpy(dtype=np.float64, copy=False)
    n1_ntcA = sig_ntcA["n1"].to_numpy(dtype=np.float64, copy=False)
    d_ntcA = sig_ntcA["d"].to_numpy(dtype=np.float64, copy=False)
    n1_ntcB = sig_ntcB["n1"].to_numpy(dtype=np.float64, copy=False)
    d_ntcB = sig_ntcB["d"].to_numpy(dtype=np.float64, copy=False)

    if n1_edges is None or d_edges is None:
        n1_edges, d_edges = _bin_edges_from_obs(
            n1_real, d_real, n_n1_bins, n_d_bins
        )

    t_real = np.max(np.abs(beta_real), axis=1)[:, None]
    t_ntcA = np.max(np.abs(beta_ntcA), axis=1)[:, None]
    t_ntcB = np.max(np.abs(beta_ntcB), axis=1)[:, None]

    p_real_vs_A, info_real = _empirical_pvals_from_edges(
        t_real,
        t_ntcA,
        n1_real,
        d_real,
        n1_ntcA,
        d_ntcA,
        n1_edges,
        d_edges,
        min_ntc_per_bin,
        two_sided=False,
    )
    p_ntcB_vs_A, info_ntcB = _empirical_pvals_from_edges(
        t_ntcB,
        t_ntcA,
        n1_ntcB,
        d_ntcB,
        n1_ntcA,
        d_ntcA,
        n1_edges,
        d_edges,
        min_ntc_per_bin,
        two_sided=False,
    )

    return {
        "p_real_gene_vs_A": p_real_vs_A[:, 0],
        "p_ntcB_gene_vs_A": p_ntcB_vs_A[:, 0],
        "matching_info": {"real_vs_A": info_real, "ntcB_vs_A": info_ntcB},
    }


def _subset_matrix_for_batch(
    C: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Cb = C[mask]
    Yb = Y[mask]
    if Cb.shape[1] == 0:
        raise ValueError("Covariate matrix has zero columns after subsetting.")
    std = Cb.std(axis=0)
    keep = (std > 0.0)
    keep[0] = True
    Cb = Cb[:, keep]
    return Cb, Yb, keep


def _combine_pvals_fisher(pvals: np.ndarray, counts: np.ndarray) -> np.ndarray:
    stat = -2.0 * np.sum(np.log(np.clip(pvals, 1e-300, 1.0)), axis=0)
    df = 2.0 * counts
    out = np.ones_like(stat, dtype=np.float64)
    mask = counts > 0
    out[mask] = chi2.sf(stat[mask], df[mask])
    return out


def _combine_pvals_stouffer(pvals: np.ndarray, counts: np.ndarray) -> np.ndarray:
    z = norm.isf(np.clip(pvals, 1e-15, 1.0) / 2.0)
    z_sum = np.sum(z, axis=0)
    denom = np.sqrt(np.maximum(counts, 1.0))
    z_comb = z_sum / denom
    out = np.ones_like(z_sum, dtype=np.float64)
    mask = counts > 0
    out[mask] = 2.0 * norm.sf(z_comb[mask])
    return out


def run_ntc_empirical_null(
    inputs: Any,
    genes: Sequence[str],
    *,
    base_seed: int,
    null_kwargs: Optional[Dict[str, Any]] = None,
    qq_crossfit: bool = False,
) -> NTCNullResult:
    kwargs = dict(null_kwargs or {})
    ntc_labels = kwargs.get("ntc_labels", ["SAFE", "non-targeting", "NTC"])
    guides_per_unit = int(kwargs.get("guides_per_unit", 6))
    n_ntc_units = int(kwargs.get("n_ntc_units", 5000))
    matching = dict(kwargs.get("matching", {}))
    batch_mode = kwargs.get("batch_mode", "meta")
    combine_method = kwargs.get("combine_method", "fisher")
    min_treated = int(kwargs.get("min_treated", 10))
    min_control = int(kwargs.get("min_control", 10))
    n_n1_bins = int(matching.get("n_n1_bins", 10))
    n_d_bins = int(matching.get("n_d_bins", 10))
    min_ntc_per_bin = int(matching.get("min_ntc_per_bin", 50))
    qq_frac_A = float(kwargs.get("qq_crossfit_frac_A", 0.5))
    qq_seed = int(kwargs.get("qq_crossfit_seed", int(base_seed) + 91))

    if combine_method not in ("fisher", "stouffer"):
        raise ValueError("combine_method must be 'fisher' or 'stouffer'.")
    if batch_mode not in ("meta", "pooled"):
        raise ValueError("batch_mode must be 'meta' or 'pooled'.")

    seed_ntc = int(base_seed) + 17
    ntc_units = sample_ntc_pseudogenes_with_replacement(
        inputs.guide_names,
        inputs.guide2gene,
        ntc_labels,
        guides_per_unit,
        n_ntc_units,
        seed_ntc,
    )

    obs_idx_real = [union_obs_idx_from_cols(inputs.G, inputs.gene_to_cols[g]) for g in genes]
    obs_idx_ntc = [union_obs_idx_from_cols(inputs.G, cols) for cols in ntc_units]
    ntc_unit_ids = [f"NTC_unit_{i:06d}" for i in range(n_ntc_units)]

    ntc_A_idx = None
    ntc_B_idx = None
    ntc_A_labels: List[str] = []
    ntc_B_labels: List[str] = []
    if qq_crossfit:
        if n_ntc_units < 2:
            raise ValueError("qq_crossfit requires at least two NTC units.")
        ntc_ids = np.arange(n_ntc_units, dtype=np.int64)
        ntc_A_idx, ntc_B_idx = split_ntc_units(
            ntc_ids, frac_A=qq_frac_A, seed=qq_seed
        )
        if ntc_A_idx.size == 0 or ntc_B_idx.size == 0:
            raise ValueError("qq_crossfit split produced an empty A or B set.")
        ntc_A_labels = [ntc_unit_ids[i] for i in ntc_A_idx]
        ntc_B_labels = [ntc_unit_ids[i] for i in ntc_B_idx]

    beta_pooled, n1_pooled, d_pooled = _compute_unit_stats(
        inputs.C, inputs.Y, inputs.A, inputs.CTY, obs_idx_real
    )

    if batch_mode == "pooled":
        beta_ntc, n1_ntc, d_ntc = _compute_unit_stats(
            inputs.C, inputs.Y, inputs.A, inputs.CTY, obs_idx_ntc
        )
        pvals, match_info = empirical_pvals_vs_ntc(
            beta_pooled,
            beta_ntc,
            n1_pooled,
            d_pooled,
            n1_ntc,
            d_ntc,
            n_n1_bins=n_n1_bins,
            n_d_bins=n_d_bins,
            min_ntc_per_bin=min_ntc_per_bin,
            two_sided=True,
        )
        ntc_crossfit = None
        if qq_crossfit and ntc_A_idx is not None and ntc_B_idx is not None:
            beta_ntcA = beta_ntc[ntc_A_idx]
            n1_ntcA = n1_ntc[ntc_A_idx]
            d_ntcA = d_ntc[ntc_A_idx]
            beta_ntcB = beta_ntc[ntc_B_idx]
            n1_ntcB = n1_ntc[ntc_B_idx]
            d_ntcB = d_ntc[ntc_B_idx]

            sig_real = pd.DataFrame({"n1": n1_pooled, "d": d_pooled}, index=genes)
            sig_ntcA = pd.DataFrame(
                {"n1": n1_ntcA, "d": d_ntcA}, index=ntc_A_labels
            )
            sig_ntcB = pd.DataFrame(
                {"n1": n1_ntcB, "d": d_ntcB}, index=ntc_B_labels
            )
            cross_prog = compute_ntc_empirical_pvals_crossfit(
                beta_pooled,
                sig_real,
                beta_ntcA,
                sig_ntcA,
                beta_ntcB,
                sig_ntcB,
                matching,
                two_sided=True,
            )
            cross_gene = compute_ntc_empirical_pvals_gene_agg_crossfit(
                beta_pooled,
                sig_real,
                beta_ntcA,
                sig_ntcA,
                beta_ntcB,
                sig_ntcB,
                matching,
                n1_edges=cross_prog["n1_edges"],
                d_edges=cross_prog["d_edges"],
            )

            ntc_crossfit = {
                "batches": [],
                "p_real_vs_A_by_batch": {},
                "p_ntcB_vs_A_by_batch": {},
                "p_real_gene_vs_A_by_batch": {},
                "p_ntcB_gene_vs_A_by_batch": {},
                "meta_p_real_vs_A": pd.DataFrame(
                    cross_prog["p_real_vs_A"],
                    index=genes,
                    columns=inputs.program_names,
                ),
                "meta_p_ntcB_vs_A": pd.DataFrame(
                    cross_prog["p_ntcB_vs_A"],
                    index=ntc_B_labels,
                    columns=inputs.program_names,
                ),
                "meta_p_real_gene_vs_A": pd.Series(
                    cross_gene["p_real_gene_vs_A"], index=genes
                ),
                "meta_p_ntcB_gene_vs_A": pd.Series(
                    cross_gene["p_ntcB_gene_vs_A"], index=ntc_B_labels
                ),
                "matching_diagnostics": {
                    "batch_mode": "pooled",
                    "ntc_A_count": int(len(ntc_A_labels)),
                    "ntc_B_count": int(len(ntc_B_labels)),
                    "real_vs_A": cross_prog["matching_info"]["real_vs_A"],
                    "ntcB_vs_A": cross_prog["matching_info"]["ntcB_vs_A"],
                },
            }
        return NTCNullResult(
            pvals=pvals,
            betas=beta_pooled,
            n_treated=n1_pooled,
            matching_info=match_info,
            ntc_crossfit=ntc_crossfit,
        )

    covar_df = inputs.covar_df_raw
    if covar_df is None:
        raise ValueError("covar_df_raw is required for batch_mode='meta'.")
    if "batch" not in covar_df.columns:
        raise ValueError("batch column missing in covariate DataFrame.")
    batch_raw = covar_df["batch"].to_numpy()
    batch_labels = pd.unique(batch_raw)

    G = len(genes)
    K = inputs.Y.shape[1]
    pvals_batches = []
    counts_batches = []
    match_infos = []
    cf_real_batches = []
    cf_ntcB_batches = []
    cf_real_gene_batches = []
    cf_ntcB_gene_batches = []
    cf_counts_real_batches = []
    cf_counts_ntcB_batches = []
    cf_counts_real_gene_batches = []
    cf_counts_ntcB_gene_batches = []
    cf_real_by_batch: Dict[Any, pd.DataFrame] = {}
    cf_ntcB_by_batch: Dict[Any, pd.DataFrame] = {}
    cf_real_gene_by_batch: Dict[Any, pd.Series] = {}
    cf_ntcB_gene_by_batch: Dict[Any, pd.Series] = {}
    cf_match_info_by_batch: Dict[Any, Any] = {}

    for b in batch_labels:
        mask = batch_raw == b
        if mask.sum() == 0:
            continue
        Cb, Yb, _ = _subset_matrix_for_batch(inputs.C, inputs.Y, mask)
        try:
            Ab = np.linalg.inv(Cb.T @ Cb)
        except np.linalg.LinAlgError:
            Ab = np.linalg.pinv(Cb.T @ Cb)
        CTYb = Cb.T @ Yb

        mask_idx = np.nonzero(mask)[0]
        global_to_local = -np.ones(mask.shape[0], dtype=np.int64)
        global_to_local[mask_idx] = np.arange(mask_idx.size, dtype=np.int64)

        obs_idx_real_b = []
        n1_real_b = np.zeros(G, dtype=np.int32)
        n0_real_b = np.zeros(G, dtype=np.int32)
        for idx in obs_idx_real:
            local = global_to_local[idx]
            local = local[local >= 0].astype(np.int32, copy=False)
            obs_idx_real_b.append(local)
        for i, local in enumerate(obs_idx_real_b):
            n1_real_b[i] = local.size
            n0_real_b[i] = mask_idx.size - n1_real_b[i]

        keep_mask = (n1_real_b >= min_treated) & (n0_real_b >= min_control)
        if not np.any(keep_mask):
            continue

        beta_real_b, n1_real_b, d_real_b = _compute_unit_stats(
            Cb, Yb, Ab, CTYb, obs_idx_real_b
        )
        beta_ntc_b, n1_ntc_b, d_ntc_b = _compute_unit_stats(
            Cb, Yb, Ab, CTYb, [global_to_local[idx][global_to_local[idx] >= 0].astype(np.int32, copy=False) for idx in obs_idx_ntc]
        )
        pvals_b, match_info = empirical_pvals_vs_ntc(
            beta_real_b,
            beta_ntc_b,
            n1_real_b,
            d_real_b,
            n1_ntc_b,
            d_ntc_b,
            n_n1_bins=n_n1_bins,
            n_d_bins=n_d_bins,
            min_ntc_per_bin=min_ntc_per_bin,
            two_sided=True,
        )
        pvals_b[~keep_mask, :] = 1.0
        counts = keep_mask.astype(np.int32)[:, None] * np.ones((1, K), dtype=np.int32)
        pvals_batches.append(pvals_b)
        counts_batches.append(counts)
        match_infos.append(match_info)

        if qq_crossfit and ntc_A_idx is not None and ntc_B_idx is not None:
            beta_ntcA_b = beta_ntc_b[ntc_A_idx]
            n1_ntcA_b = n1_ntc_b[ntc_A_idx]
            d_ntcA_b = d_ntc_b[ntc_A_idx]
            beta_ntcB_b = beta_ntc_b[ntc_B_idx]
            n1_ntcB_b = n1_ntc_b[ntc_B_idx]
            d_ntcB_b = d_ntc_b[ntc_B_idx]

            sig_real_b = pd.DataFrame({"n1": n1_real_b, "d": d_real_b}, index=genes)
            sig_ntcA_b = pd.DataFrame(
                {"n1": n1_ntcA_b, "d": d_ntcA_b}, index=ntc_A_labels
            )
            sig_ntcB_b = pd.DataFrame(
                {"n1": n1_ntcB_b, "d": d_ntcB_b}, index=ntc_B_labels
            )
            cross_prog = compute_ntc_empirical_pvals_crossfit(
                beta_real_b,
                sig_real_b,
                beta_ntcA_b,
                sig_ntcA_b,
                beta_ntcB_b,
                sig_ntcB_b,
                matching,
                two_sided=True,
            )
            p_real_vs_A_b = cross_prog["p_real_vs_A"]
            p_ntcB_vs_A_b = cross_prog["p_ntcB_vs_A"]

            p_real_vs_A_b[~keep_mask, :] = 1.0
            n0_ntcB_b = mask_idx.size - n1_ntcB_b
            keep_ntcB = (n1_ntcB_b >= min_treated) & (n0_ntcB_b >= min_control)
            p_ntcB_vs_A_b[~keep_ntcB, :] = 1.0

            cf_real_by_batch[b] = pd.DataFrame(
                p_real_vs_A_b, index=genes, columns=inputs.program_names
            )
            cf_ntcB_by_batch[b] = pd.DataFrame(
                p_ntcB_vs_A_b, index=ntc_B_labels, columns=inputs.program_names
            )
            cf_real_batches.append(p_real_vs_A_b)
            cf_ntcB_batches.append(p_ntcB_vs_A_b)
            cf_counts_real_batches.append(
                keep_mask.astype(np.int32)[:, None]
                * np.ones((1, K), dtype=np.int32)
            )
            cf_counts_ntcB_batches.append(
                keep_ntcB.astype(np.int32)[:, None]
                * np.ones((1, K), dtype=np.int32)
            )

            cross_gene = compute_ntc_empirical_pvals_gene_agg_crossfit(
                beta_real_b,
                sig_real_b,
                beta_ntcA_b,
                sig_ntcA_b,
                beta_ntcB_b,
                sig_ntcB_b,
                matching,
                n1_edges=cross_prog["n1_edges"],
                d_edges=cross_prog["d_edges"],
            )
            p_real_gene_b = cross_gene["p_real_gene_vs_A"]
            p_ntcB_gene_b = cross_gene["p_ntcB_gene_vs_A"]
            p_real_gene_b[~keep_mask] = 1.0
            p_ntcB_gene_b[~keep_ntcB] = 1.0

            cf_real_gene_by_batch[b] = pd.Series(p_real_gene_b, index=genes)
            cf_ntcB_gene_by_batch[b] = pd.Series(p_ntcB_gene_b, index=ntc_B_labels)
            cf_real_gene_batches.append(p_real_gene_b)
            cf_ntcB_gene_batches.append(p_ntcB_gene_b)
            cf_counts_real_gene_batches.append(keep_mask.astype(np.int32))
            cf_counts_ntcB_gene_batches.append(keep_ntcB.astype(np.int32))
            cf_match_info_by_batch[b] = {
                "program_level": cross_prog["matching_info"],
                "gene_level": cross_gene["matching_info"],
            }

    if not pvals_batches:
        return NTCNullResult(
            pvals=np.ones((G, K), dtype=np.float64),
            betas=beta_pooled,
            n_treated=n1_pooled,
            matching_info={"reason": "no_batches_passed"},
        )

    pvals_stack = np.stack(pvals_batches, axis=0)
    counts_stack = np.stack(counts_batches, axis=0)
    counts = counts_stack.sum(axis=0)

    if combine_method == "fisher":
        p_comb = _combine_pvals_fisher(pvals_stack, counts)
    else:
        p_comb = _combine_pvals_stouffer(pvals_stack, counts)

    ntc_crossfit = None
    if qq_crossfit and cf_real_batches and cf_ntcB_batches:
        cf_real_stack = np.stack(cf_real_batches, axis=0)
        cf_ntcB_stack = np.stack(cf_ntcB_batches, axis=0)
        cf_counts_real = np.stack(cf_counts_real_batches, axis=0).sum(axis=0)
        cf_counts_ntcB = np.stack(cf_counts_ntcB_batches, axis=0).sum(axis=0)

        cf_real_gene_stack = np.stack(cf_real_gene_batches, axis=0)
        cf_ntcB_gene_stack = np.stack(cf_ntcB_gene_batches, axis=0)
        cf_counts_real_gene = np.stack(cf_counts_real_gene_batches, axis=0).sum(axis=0)
        cf_counts_ntcB_gene = np.stack(cf_counts_ntcB_gene_batches, axis=0).sum(axis=0)

        if combine_method == "fisher":
            meta_real = _combine_pvals_fisher(cf_real_stack, cf_counts_real)
            meta_ntcB = _combine_pvals_fisher(cf_ntcB_stack, cf_counts_ntcB)
            meta_real_gene = _combine_pvals_fisher(
                cf_real_gene_stack, cf_counts_real_gene
            )
            meta_ntcB_gene = _combine_pvals_fisher(
                cf_ntcB_gene_stack, cf_counts_ntcB_gene
            )
        else:
            meta_real = _combine_pvals_stouffer(cf_real_stack, cf_counts_real)
            meta_ntcB = _combine_pvals_stouffer(cf_ntcB_stack, cf_counts_ntcB)
            meta_real_gene = _combine_pvals_stouffer(
                cf_real_gene_stack, cf_counts_real_gene
            )
            meta_ntcB_gene = _combine_pvals_stouffer(
                cf_ntcB_gene_stack, cf_counts_ntcB_gene
            )

        ntc_crossfit = {
            "batches": list(cf_real_by_batch.keys()),
            "p_real_vs_A_by_batch": cf_real_by_batch,
            "p_ntcB_vs_A_by_batch": cf_ntcB_by_batch,
            "p_real_gene_vs_A_by_batch": cf_real_gene_by_batch,
            "p_ntcB_gene_vs_A_by_batch": cf_ntcB_gene_by_batch,
            "meta_p_real_vs_A": pd.DataFrame(
                meta_real, index=genes, columns=inputs.program_names
            ),
            "meta_p_ntcB_vs_A": pd.DataFrame(
                meta_ntcB, index=ntc_B_labels, columns=inputs.program_names
            ),
            "meta_p_real_gene_vs_A": pd.Series(meta_real_gene, index=genes),
            "meta_p_ntcB_gene_vs_A": pd.Series(meta_ntcB_gene, index=ntc_B_labels),
            "matching_diagnostics": {
                "batch_mode": "meta",
                "combine_method": combine_method,
                "ntc_A_count": int(len(ntc_A_labels)),
                "ntc_B_count": int(len(ntc_B_labels)),
                "per_batch": cf_match_info_by_batch,
            },
        }

    return NTCNullResult(
        pvals=p_comb,
        betas=beta_pooled,
        n_treated=n1_pooled,
        matching_info={
            "batch_mode": "meta",
            "combine_method": combine_method,
            "batch_count": int(len(pvals_batches)),
            "batch_matching_info": match_infos,
        },
        ntc_crossfit=ntc_crossfit,
    )


def compute_ols_denom_reference(C: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(C, x, rcond=None)
    resid = x - C @ coef
    return float(resid.T @ resid)


def compute_ols_denom(C: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    CtC = C.T @ C
    try:
        A = np.linalg.inv(CtC)
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(CtC)
    v = C.T @ x
    n1 = float(np.sum(x))
    return float(n1 - v.T @ (A @ v))
