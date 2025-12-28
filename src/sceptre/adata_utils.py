"""
Utilities for extracting matrices from AnnData and preparing covariates/outcomes.
"""

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

THREAD_ENV_VARS: Tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def limit_threading(n_threads: int = 1) -> None:
    """
    Clamp BLAS/OMP thread counts to keep CRT resampling predictable.
    """
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n_threads))


def get_from_adata_any(adata: Any, key: str) -> Any:
    """
    Try common AnnData containers, then fall back to item access.
    """
    if hasattr(adata, "obsm") and key in adata.obsm:
        return adata.obsm[key]
    if hasattr(adata, "layers") and key in adata.layers:
        return adata.layers[key]
    if hasattr(adata, "obsp") and key in adata.obsp:
        return adata.obsp[key]
    if hasattr(adata, "uns") and key in adata.uns:
        return adata.uns[key]
    if hasattr(adata, "obs") and key in adata.obs:
        return adata.obs[key]
    try:
        return adata[key]
    except Exception as exc:  # pylint: disable=broad-except
        raise KeyError(f"Could not find `{key}` in adata.* containers.") from exc


def to_csc_matrix(G: Any) -> Tuple[sp.csc_matrix, Optional[List[str]]]:
    """
    Cast guide assignment to CSC. Returns matrix and optional column names.
    The CSC format is particularly efficient for the column-wise operations
    often used in statistical genetics and single-cell analysis.
    """
    if isinstance(G, pd.DataFrame):
        guide_names = list(G.columns)
        return sp.csc_matrix(G.to_numpy()), guide_names
    if sp.issparse(G):
        return G.tocsc(), None
    arr = np.asarray(G)
    return sp.csc_matrix(arr), None


def get_covar_matrix(
    adata: Any,
    covar_key: str = "covar",
    add_intercept: bool = True,
    standardize: bool = True,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Fetch covariates, optionally z-score columns and prepend intercept.
    """
    C = get_from_adata_any(adata, covar_key)
    if isinstance(C, pd.DataFrame):
        covar_cols = list(C.columns)
        C = C.to_numpy()
    else:
        covar_cols = None
        C = np.asarray(C)
    C = C.astype(np.float64, copy=False)

    if standardize:
        mu = C.mean(axis=0)
        sd = C.std(axis=0)
        sd[sd == 0.0] = 1.0
        C = (C - mu) / sd

    if add_intercept:
        C = np.column_stack([np.ones((C.shape[0], 1), dtype=np.float64), C])

    return C, covar_cols


def clr_from_usage(U: Any, eps_quantile: float = 1e-4) -> np.ndarray:
    """
    Compute centered log-ratio from usage matrix with flooring and renormalization.
    """
    U = np.asarray(U, dtype=np.float64)
    eps = np.quantile(U, eps_quantile)
    U2 = np.maximum(U, eps)
    U2 /= U2.sum(axis=1, keepdims=True)
    logU = np.log(U2)
    return logU - logU.mean(axis=1, keepdims=True)


def build_gene_to_cols(
    guide_names: Sequence[str],
    guide2gene: Mapping[str, str],
) -> Dict[str, List[int]]:
    """
    Map each gene to the list of guide column indexes targeting it.
    """
    guide_to_col = {g: i for i, g in enumerate(guide_names)}
    gene_to_cols: Dict[str, List[int]] = {}
    for guide, gene in guide2gene.items():
        j = guide_to_col.get(guide)
        if j is None:
            continue
        gene_to_cols.setdefault(gene, []).append(j)
    return {g: sorted(cols) for g, cols in gene_to_cols.items()}


def union_obs_idx_from_cols(G_csc: sp.csc_matrix, cols: Iterable[int]) -> np.ndarray:
    """
    Collect treated cell indices for a set of guide columns (union indicator).
    """
    indptr = G_csc.indptr
    indices = G_csc.indices
    chunks: List[np.ndarray] = []
    for j in cols:
        chunks.append(indices[indptr[j] : indptr[j + 1]])
    if not chunks:
        return np.empty(0, dtype=np.int32)
    rows = np.concatenate(chunks).astype(np.int32, copy=False)
    return np.unique(rows)


def get_program_names(adata: Any, n_programs: int) -> List[str]:
    """
    Resolve program names if stored in adata.uns, otherwise auto-number.
    """
    if hasattr(adata, "uns") and isinstance(getattr(adata, "uns", None), dict):
        uns = getattr(adata, "uns")
        if "program_names" in uns:
            return list(uns["program_names"])
    return [f"program_{k}" for k in range(n_programs)]
