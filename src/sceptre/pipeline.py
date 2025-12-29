"""
High-level pipeline to run SCEPTRE-like CRT across genes and programs.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from .adata_utils import (
    build_gene_to_cols,
    clr_from_usage,
    get_covar_matrix,
    get_from_adata_any,
    get_program_names,
    limit_threading,
    to_csc_matrix,
    union_obs_idx_from_cols,
)
from .crt import crt_index_sampler_fast_numba, crt_pvals_for_gene
from .propensity import fit_propensity_logistic


@dataclass
class CRTInputs:
    C: np.ndarray
    Y: np.ndarray
    A: np.ndarray
    CTY: np.ndarray
    G: sp.csc_matrix
    guide_names: List[str]
    guide2gene: Dict[str, str]
    gene_to_cols: Dict[str, List[int]]
    program_names: List[str]
    covar_cols: Optional[List[str]] = None


@dataclass
class CRTGeneResult:
    gene: str
    pvals: np.ndarray
    betas: np.ndarray
    n_treated: int


def prepare_crt_inputs(
    adata: Any,
    usage_key: str = "cnmf_usage",
    covar_key: str = "covar",
    guide_assignment_key: str = "guide_assignment",
    guide_names_key: str = "guide_names",
    guide2gene_key: str = "guide2gene",
    eps_quantile: float = 1e-4,
    add_intercept: bool = True,
    standardize: bool = True,
    clamp_threads: bool = True,
) -> CRTInputs:
    """
    Load matrices from AnnData, build CLR usage, and precompute regression pieces.
    adata: AnnData-like object with required data
    usage_key: key in adata to extract usage matrix
    covar_key: key in adata to extract covariate matrix
    guide_assignment_key: key in adata to extract guide assignment matrix
    guide_names_key: key in adata.uns to extract guide column names if not in DataFrame
    guide2gene_key: key in adata to extract guide-to-gene mapping
    eps_quantile: quantile for flooring small values in usage before CLR
    add_intercept: whether to add intercept column to covariate matrix
    standardize: whether to z-score covariate columns
    clamp_threads: whether to limit threading for numerical libraries
    Returns:
        CRTInputs dataclass with all required inputs for CRT
    """
    if clamp_threads:
        limit_threading()

    C, covar_cols = get_covar_matrix(
        adata,
        covar_key=covar_key,
        add_intercept=add_intercept,
        standardize=standardize,
    )

    U = get_from_adata_any(adata, usage_key)
    if isinstance(U, pd.DataFrame):
        U = U.to_numpy()
    Y = clr_from_usage(U, eps_quantile=eps_quantile)

    G_raw = get_from_adata_any(adata, guide_assignment_key)
    G, guide_names = to_csc_matrix(G_raw)
    if guide_names is None:
        if hasattr(adata, "uns") and guide_names_key in getattr(adata, "uns", {}):
            guide_names = list(getattr(adata, "uns")[guide_names_key])
        else:
            raise ValueError(
                "Guide column names are required; provide them via a DataFrame or adata.uns."
            )

    guide2gene_raw = get_from_adata_any(adata, guide2gene_key)
    guide2gene = dict(guide2gene_raw)
    gene_to_cols = build_gene_to_cols(guide_names, guide2gene)
    if not gene_to_cols:
        raise ValueError("No guides mapped to genes; check guide2gene and guide names.")

    if C.shape[0] != Y.shape[0] or C.shape[0] != G.shape[0]:
        raise ValueError(
            f"Cell counts mismatch: C {C.shape[0]}, Y {Y.shape[0]}, G {G.shape[0]}"
        )

    """
        Precompute regression pieces: A = (C^T C)^{-1}, CTY = C^T Y.
        CtC: p x p matrix, C^T C
        A: inverse of CtC
        CTY: p x K matrix, C^T Y
    """
    CtC = C.T @ C
    A = np.linalg.inv(CtC)
    CTY = C.T @ Y

    program_names = get_program_names(adata, Y.shape[1])

    return CRTInputs(
        C=C.astype(np.float64, copy=False),
        Y=Y.astype(np.float64, copy=False),
        A=A.astype(np.float64, copy=False),
        CTY=CTY.astype(np.float64, copy=False),
        G=G,
        guide_names=list(guide_names),
        guide2gene=guide2gene,
        gene_to_cols=gene_to_cols,
        program_names=program_names,
        covar_cols=covar_cols,
    )


def _extract_probabilities(output) -> np.ndarray:
    """
    Normalize propensity model outputs into probability vector.
    """
    if isinstance(output, tuple) or isinstance(output, list):
        return np.asarray(output[0], dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def run_one_gene_union_crt(
    gene: str,
    inputs: CRTInputs,
    B: int = 1023,
    base_seed: int = 123,
    propensity_model: Callable = fit_propensity_logistic,
) -> CRTGeneResult:
    """
    Run union CRT for a single gene and return p-values and betas across programs.
    gene: target gene name
    inputs: CRTInputs dataclass with all required inputs
    B: number of resamples for CRT
    base_seed: base random seed for reproducibility
    propensity_model: function to fit propensity scores given C and y01
    y01: binary union indicator for the gene
    Returns:
        CRTGeneResult dataclass with all results for the gene
    """
    if gene not in inputs.gene_to_cols:
        raise KeyError(f"Gene `{gene}` not present in gene_to_cols mapping.")
    obs_idx = union_obs_idx_from_cols(inputs.G, inputs.gene_to_cols[gene])

    """
    Handle edge cases with no treated cells or all treated cells.
    In these cases, a meaningful comparison is impossible. 
    The function returns a trivial result (p-values of 1.0, effect sizes of 0) without performing the test.
    """
    if obs_idx.size == 0 or obs_idx.size == inputs.C.shape[0]:
        K = inputs.Y.shape[1]
        return CRTGeneResult(
            gene=gene,
            pvals=np.ones(K, dtype=np.float64),
            betas=np.zeros(K, dtype=np.float64),
            n_treated=int(obs_idx.size),
        )

    y01 = np.zeros(inputs.C.shape[0], dtype=np.int8)
    y01[obs_idx] = 1

    p = _extract_probabilities(propensity_model(inputs.C, y01))
    seed = (hash(gene) ^ base_seed) & 0xFFFFFFFF

    indptr, idx = crt_index_sampler_fast_numba(p, B, seed)

    pvals, beta_obs = crt_pvals_for_gene(
        indptr,
        idx,
        inputs.C,
        inputs.Y,
        inputs.A,
        inputs.CTY,
        obs_idx.astype(np.int32),
        B,
    )
    return CRTGeneResult(
        gene=gene,
        pvals=pvals,
        betas=beta_obs,
        n_treated=int(obs_idx.size),
    )


def run_all_genes_union_crt(
    inputs: CRTInputs,
    genes: Optional[Iterable[str]] = None,
    B: int = 1023,
    n_jobs: int = 8,
    base_seed: int = 123,
    propensity_model: Callable = fit_propensity_logistic,
    backend: str = "loky",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[CRTGeneResult]]:
    """
    Run union CRT across all genes and return DataFrames for p-values and betas.
    inputs: CRTInputs dataclass with all required inputs
    genes: optional list of genes to test; if None, test all genes in inputs.gene_to_cols
    B: number of resamples for CRT
    n_jobs: number of parallel jobs to use
    base_seed: base random seed for reproducibility
    propensity_model: function to fit propensity scores given C and y01
    backend: joblib parallelization backend
    Returns:
        pvals_df: DataFrame of CRT p-values (genes x programs)
        betas_df: DataFrame of CRT effect sizes (genes x programs)
        treated_df: Series of number of treated cells per gene
        results: list of CRTGeneResult dataclasses for all genes
    """
    gene_list = sorted(inputs.gene_to_cols.keys()) if genes is None else list(genes)

    results: List[CRTGeneResult] = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(run_one_gene_union_crt)(
            gene,
            inputs,
            B=B,
            base_seed=base_seed,
            propensity_model=propensity_model,
        )
        for gene in gene_list
    )

    pval_mat = np.vstack([r.pvals for r in results])
    beta_mat = np.vstack([r.betas for r in results])
    treated = np.array([r.n_treated for r in results], dtype=int)

    pvals_df = pd.DataFrame(pval_mat, index=gene_list, columns=inputs.program_names)
    betas_df = pd.DataFrame(beta_mat, index=gene_list, columns=inputs.program_names)
    treated_df = pd.Series(treated, index=gene_list, name="n_union_positive_cells")

    return pvals_df, betas_df, treated_df, results


def store_results_in_adata(
    adata: Any,
    pvals_df: pd.DataFrame,
    betas_df: pd.DataFrame,
    treated_df: pd.Series,
    prefix: str = "crt_union_gene_program",
) -> None:
    """
    Save CRT outputs into adata.uns with a consistent prefix.
    adata: AnnData-like object to store results in
    pvals_df: DataFrame of CRT p-values (genes x programs)
    betas_df: DataFrame of CRT effect sizes (genes x programs)
    treated_df: Series of number of treated cells per gene
    prefix: prefix for keys in adata.uns to store results
    """

    if not hasattr(adata, "uns"):
        raise AttributeError("AnnData-like object missing `.uns` to store results.")
    adata.uns[f"{prefix}_pvals"] = pvals_df
    adata.uns[f"{prefix}_betas"] = betas_df
    adata.uns[f"{prefix}_n_positive"] = treated_df
