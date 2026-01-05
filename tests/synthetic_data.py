from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class MockAdata:
    def __init__(self):
        self.obsm = {}
        self.layers = {}
        self.obsp = {}
        self.uns = {}
        self.obs = {}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SyntheticTruth:
    C_numeric: np.ndarray
    Y_clr: np.ndarray
    usage: np.ndarray
    x_by_gene: Dict[str, np.ndarray]
    gene_names: List[str]
    effect_gene: Optional[str]
    effect_program: Optional[int]
    effect_size: float


def make_synthetic_adata(
    rng: np.random.Generator,
    n_cells: int = 120,
    n_programs: int = 5,
    n_genes: int = 3,
    guides_per_gene: int = 2,
    n_covariates: int = 3,
    *,
    effect_gene: Optional[str] = None,
    effect_program: int = 0,
    effect_size: float = 0.0,
    include_categorical: bool = False,
    categorical_levels: int = 3,
    propensity_mode: str = "covariate",
    propensity_range: Tuple[float, float] = (0.1, 0.3),
    ntc_genes: Sequence[str] = ("non-targeting", "safe-targeting"),
    return_truth: bool = False,
) -> Tuple[MockAdata, Optional[SyntheticTruth]]:
    """
    Build a lightweight AnnData-like object for tests.
    Returns (adata, truth) if return_truth, else (adata, None).
    """
    adata = MockAdata()

    C_numeric = rng.normal(size=(n_cells, n_covariates))
    if include_categorical:
        covar_df = pd.DataFrame(
            C_numeric, columns=[f"covar_{i}" for i in range(n_covariates)]
        )
        covar_df["batch"] = pd.Categorical(
            rng.integers(0, categorical_levels, size=n_cells)
        )
        covar = covar_df
    else:
        covar = C_numeric

    base_genes = [f"gene_{i}" for i in range(n_genes)]
    gene_names = list(base_genes)
    for ntc in ntc_genes:
        if ntc not in gene_names:
            gene_names.append(ntc)

    guide_names: List[str] = []
    guide2gene: Dict[str, str] = {}
    gene_to_cols: Dict[str, List[int]] = {}
    for gene in gene_names:
        cols = []
        for j in range(guides_per_gene):
            gname = f"{gene}_g{j}"
            guide2gene[gname] = gene
            guide_names.append(gname)
            cols.append(len(guide_names) - 1)
        gene_to_cols[gene] = cols

    x_by_gene: Dict[str, np.ndarray] = {}
    if propensity_mode not in ("covariate", "constant"):
        raise ValueError("propensity_mode must be 'covariate' or 'constant'.")

    for gene in gene_names:
        if propensity_mode == "covariate":
            theta = rng.normal(scale=0.5, size=n_covariates)
            p = _sigmoid(C_numeric @ theta)
            p = np.clip(p, 0.05, 0.95)
        else:
            p_val = rng.uniform(*propensity_range)
            p = np.full(n_cells, p_val, dtype=np.float64)
        x_by_gene[gene] = rng.binomial(1, p, size=n_cells).astype(np.int8)

    n_guides = len(guide_names)
    G = np.zeros((n_cells, n_guides), dtype=np.int8)
    for gene, cols in gene_to_cols.items():
        x = x_by_gene[gene]
        active = np.flatnonzero(x)
        if active.size == 0:
            continue
        choices = rng.integers(0, len(cols), size=active.size)
        for idx, sel in zip(active, choices):
            G[idx, cols[sel]] = 1

    gamma = rng.normal(scale=0.2, size=(n_covariates, n_programs))
    Y = C_numeric @ gamma + rng.normal(scale=0.1, size=(n_cells, n_programs))
    if effect_gene is not None and effect_gene in x_by_gene:
        Y[:, effect_program] += effect_size * x_by_gene[effect_gene]
    Y = Y - Y.mean(axis=1, keepdims=True)
    U = np.exp(Y)
    U /= U.sum(axis=1, keepdims=True)

    adata.obsm["covar"] = covar
    adata.obsm["cnmf_usage"] = U
    adata.obsm["guide_assignment"] = G
    adata.uns["guide_names"] = guide_names
    adata.uns["guide2gene"] = guide2gene
    adata.uns["program_names"] = [f"program_{k}" for k in range(n_programs)]

    truth = SyntheticTruth(
        C_numeric=C_numeric,
        Y_clr=Y,
        usage=U,
        x_by_gene=x_by_gene,
        gene_names=gene_names,
        effect_gene=effect_gene,
        effect_program=effect_program,
        effect_size=effect_size,
    )
    return adata, truth if return_truth else None
