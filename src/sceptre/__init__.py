"""
SCEPTRE-like conditional randomization testing utilities.

This package exposes helpers to:
- pull matrices out of AnnData objects
- build centered log-ratio usage matrices and covariates
- fit guide-union propensities
- run the fast CRT resampling with numba
- orchestrate gene-by-program testing and storing results
"""

from .adata_utils import (
    THREAD_ENV_VARS,
    clr_from_usage,
    get_covar_matrix,
    get_from_adata_any,
    get_program_names,
    limit_threading,
    to_csc_matrix,
    union_obs_idx_from_cols,
)
from .crt import crt_betas_for_gene, crt_index_sampler_fast_numba, crt_pvals_for_gene
from .pipeline import (
    CRTGeneResult,
    CRTInputs,
    prepare_crt_inputs,
    run_all_genes_union_crt,
    run_one_gene_union_crt,
    store_results_in_adata,
)
from .propensity import fit_propensity_logistic
from .skew_normal import (
    check_for_outliers,
    check_sn_tail,
    compute_empirical_p_value,
    fit_and_evaluate_skew_normal,
    fit_skew_normal,
)

__all__ = [
    "THREAD_ENV_VARS",
    "clr_from_usage",
    "get_covar_matrix",
    "get_from_adata_any",
    "get_program_names",
    "limit_threading",
    "to_csc_matrix",
    "union_obs_idx_from_cols",
    "crt_index_sampler_fast_numba",
    "crt_pvals_for_gene",
    "crt_betas_for_gene",
    "CRTGeneResult",
    "CRTInputs",
    "prepare_crt_inputs",
    "run_all_genes_union_crt",
    "run_one_gene_union_crt",
    "store_results_in_adata",
    "fit_propensity_logistic",
    "fit_skew_normal",
    "compute_empirical_p_value",
    "check_sn_tail",
    "check_for_outliers",
    "fit_and_evaluate_skew_normal",
]
