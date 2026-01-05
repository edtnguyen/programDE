from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


class MockAdata:
    def __init__(self):
        self.obsm = {}
        self.layers = {}
        self.obsp = {}
        self.uns = {}
        self.obs = {}

def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


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
    ntc_frac_guides: float = 0.15,
    moi_mean: float = 5.5,
    frac_causal_genes: float = 0.0,
    n_effect_programs: int = 3,
    confound_strength: Optional[float] = None,
    eps_near_zero: float = 1e-6,
    sprinkle_near_zero_frac: float = 0.15,
    return_truth: bool = False,
) -> Tuple[MockAdata, Optional[SyntheticTruth]]:
    """
    Build a lightweight AnnData-like object for tests.
    Returns (adata, truth) if return_truth, else (adata, None).
    """
    adata = MockAdata()
    if n_genes <= 0 or n_programs <= 0 or n_cells <= 0:
        raise ValueError("n_cells, n_programs, and n_genes must be positive.")
    if guides_per_gene <= 0:
        raise ValueError("guides_per_gene must be positive.")
    if propensity_mode not in ("covariate", "constant"):
        raise ValueError("propensity_mode must be 'covariate' or 'constant'.")
    if not 0.0 <= ntc_frac_guides < 1.0:
        raise ValueError("ntc_frac_guides must be in [0, 1).")
    if moi_mean < 0.0:
        raise ValueError("moi_mean must be non-negative.")

    n_covariates = max(1, int(n_covariates))
    covs = []
    if n_covariates >= 1:
        batch = rng.integers(0, 2, size=n_cells).astype(np.float64)
        covs.append(batch)
    if n_covariates >= 2:
        libsize = rng.lognormal(mean=10.0, sigma=0.35, size=n_cells).astype(
            np.float64
        )
        log_depth = np.log1p(libsize)
        covs.append(log_depth)
    while len(covs) < n_covariates:
        covs.append(rng.normal(0, 1, size=n_cells).astype(np.float64))
    C_numeric = np.column_stack(covs)

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

    target_gene_names = [f"gene_{i}" for i in range(n_genes)]
    ntc_gene_names = list(dict.fromkeys(ntc_genes))
    gene_names = target_gene_names + [g for g in ntc_gene_names if g not in target_gene_names]

    guide_names: List[str] = []
    guide2gene: Dict[str, str] = {}
    gene2guides: Dict[str, List[str]] = {}

    for gene in target_gene_names:
        cols = []
        for j in range(guides_per_gene):
            gname = f"{gene}_g{j + 1:02d}"
            guide2gene[gname] = gene
            guide_names.append(gname)
            cols.append(len(guide_names) - 1)
        gene2guides[gene] = [guide_names[i] for i in cols]

    n_target_guides = n_genes * guides_per_gene
    if ntc_frac_guides == 0.0:
        n_ntc_guides = 0
    else:
        n_ntc_guides = int(
            np.round((ntc_frac_guides / (1.0 - ntc_frac_guides)) * n_target_guides)
        )
        if ntc_gene_names:
            n_ntc_guides = max(n_ntc_guides, len(ntc_gene_names))

    ntc_guide_idx = np.array([], dtype=np.int32)
    if n_ntc_guides > 0 and ntc_gene_names:
        base = n_ntc_guides // len(ntc_gene_names)
        rem = n_ntc_guides % len(ntc_gene_names)
        ntc_guides = []
        for i, gene in enumerate(ntc_gene_names):
            count = base + (1 if i < rem else 0)
            guides = []
            for j in range(count):
                gname = f"{gene}_g{j + 1:04d}"
                guide2gene[gname] = gene
                guide_names.append(gname)
                guides.append(gname)
            gene2guides[gene] = guides
            ntc_guides.extend(guides)
        ntc_guide_idx = np.arange(n_target_guides, n_target_guides + n_ntc_guides, dtype=np.int32)
    else:
        for gene in ntc_gene_names:
            gene2guides.setdefault(gene, [])

    n_guides = len(guide_names)
    guide_to_col = {g: i for i, g in enumerate(guide_names)}
    max_guides = min(50, n_guides)
    moi_draws = rng.poisson(moi_mean, size=n_cells)
    moi_draws = np.clip(moi_draws, 0, max_guides).astype(np.int32)

    if confound_strength is None:
        confound_strength = 0.5 if propensity_mode == "covariate" else 0.0
    gene_base_weights = rng.lognormal(mean=0.0, sigma=0.6, size=n_genes).astype(
        np.float64
    )
    if effect_gene is not None and effect_size != 0.0:
        if effect_gene in target_gene_names:
            effect_idx = target_gene_names.index(effect_gene)
            gene_base_weights[effect_idx] *= 2.0
    if confound_strength > 0:
        tilt = rng.normal(0.0, confound_strength, size=n_genes).astype(np.float64)
        gene_w_batch0 = gene_base_weights * np.exp(-tilt)
        gene_w_batch1 = gene_base_weights * np.exp(+tilt)
    else:
        gene_w_batch0 = gene_base_weights
        gene_w_batch1 = gene_base_weights
    gene_p_batch0 = gene_w_batch0 / gene_w_batch0.sum()
    gene_p_batch1 = gene_w_batch1 / gene_w_batch1.sum()

    G = np.zeros((n_cells, n_guides), dtype=np.int8)
    for i in range(n_cells):
        mi = int(moi_draws[i])
        if mi == 0:
            continue
        is_ntc = (
            rng.random(mi) < ntc_frac_guides if ntc_guide_idx.size > 0 else np.zeros(mi, dtype=bool)
        )
        n_ntc = int(is_ntc.sum())
        n_tgt = mi - n_ntc

        chosen_cols: List[int] = []
        if n_ntc > 0 and ntc_guide_idx.size > 0:
            replace = n_ntc > ntc_guide_idx.size
            ntc_choices = rng.choice(ntc_guide_idx, size=n_ntc, replace=replace)
            chosen_cols.extend(ntc_choices.tolist())

        if n_tgt > 0 and n_genes > 0:
            gene_p = gene_p_batch1 if C_numeric[i, 0] >= 0.5 else gene_p_batch0
            gene_idx = rng.choice(n_genes, size=n_tgt, replace=True, p=gene_p)
            within = rng.integers(0, guides_per_gene, size=n_tgt)
            guide_idx = gene_idx * guides_per_gene + within
            chosen_cols.extend(guide_idx.tolist())

        if not chosen_cols:
            continue
        chosen_unique = np.unique(np.asarray(chosen_cols, dtype=np.int32))
        G[i, chosen_unique] = 1

    W = rng.normal(0, 0.25, size=(C_numeric.shape[1], n_programs))
    logits = C_numeric @ W + rng.normal(0, 0.3, size=(n_cells, n_programs))

    causal_genes: List[str] = []
    gene_effects: Dict[str, np.ndarray] = {}
    if effect_gene is not None and effect_size != 0.0:
        if effect_gene in target_gene_names:
            eff = np.zeros(n_programs, dtype=np.float64)
            eff[effect_program] = effect_size
            gene_effects[effect_gene] = eff
            causal_genes = [effect_gene]
    elif effect_size != 0.0 and frac_causal_genes > 0.0:
        n_causal = max(1, int(np.round(frac_causal_genes * n_genes)))
        causal_idx = rng.choice(n_genes, size=n_causal, replace=False)
        causal_genes = [target_gene_names[i] for i in causal_idx]
        for gene in causal_genes:
            eff = np.zeros(n_programs, dtype=np.float64)
            n_eff = min(n_effect_programs, n_programs)
            progs = rng.choice(n_programs, size=n_eff, replace=False)
            signs = rng.choice([-1.0, 1.0], size=n_eff)
            eff[progs] = signs * effect_size
            gene_effects[gene] = eff

    for gene in causal_genes:
        guides = gene2guides.get(gene, [])
        if not guides:
            continue
        cols = [guide_to_col[g] for g in guides]
        x_union = (G[:, cols].sum(axis=1) > 0).astype(np.float64)
        logits += x_union[:, None] * gene_effects[gene][None, :]

    usage = _softmax_rows(logits).astype(np.float64)
    if sprinkle_near_zero_frac > 0:
        mask = rng.random(size=usage.shape) < sprinkle_near_zero_frac
        usage[mask] *= eps_near_zero
        usage /= usage.sum(axis=1, keepdims=True)

    adata.obsm["covar"] = covar
    adata.obsm["cnmf_usage"] = usage
    adata.obsm["usage"] = usage
    adata.obsm["guide_assignment"] = G
    adata.uns["guide_names"] = guide_names
    adata.uns["guide2gene"] = guide2gene
    adata.uns["program_names"] = [f"program_{k}" for k in range(n_programs)]
    adata.uns["gene2guides"] = gene2guides
    adata.uns["target_gene_names"] = target_gene_names
    adata.uns["ntc_gene_names"] = ntc_gene_names
    adata.uns["causal_genes"] = causal_genes
    adata.uns["gene_effects"] = {g: gene_effects[g].tolist() for g in causal_genes}
    adata.uns["synth_params"] = dict(
        N=n_cells,
        K=n_programs,
        n_target_genes=n_genes,
        guides_per_gene=guides_per_gene,
        ntc_frac_guides=ntc_frac_guides,
        moi_mean=moi_mean,
        frac_causal_genes=frac_causal_genes,
        n_effect_programs=n_effect_programs,
        effect_size=effect_size,
        confound_strength=confound_strength,
    )

    if return_truth:
        x_by_gene: Dict[str, np.ndarray] = {}
        for gene, guides in gene2guides.items():
            cols = [guide_to_col[g] for g in guides]
            if cols:
                x_by_gene[gene] = (G[:, cols].sum(axis=1) > 0).astype(np.int8)
            else:
                x_by_gene[gene] = np.zeros(n_cells, dtype=np.int8)
    else:
        x_by_gene = {}

    usage_eps = np.clip(usage, eps_near_zero, 1.0)
    log_usage = np.log(usage_eps)
    Y_clr = log_usage - log_usage.mean(axis=1, keepdims=True)

    truth = SyntheticTruth(
        C_numeric=C_numeric,
        Y_clr=Y_clr,
        usage=usage,
        x_by_gene=x_by_gene,
        gene_names=gene_names,
        effect_gene=effect_gene,
        effect_program=effect_program,
        effect_size=effect_size,
    )
    return adata, truth if return_truth else None


def make_sceptre_style_synth(
    N: int = 50_000,
    K: int = 70,
    n_target_genes: int = 300,
    guides_per_gene: int = 6,
    ntc_frac_guides: float = 0.15,
    moi_mean: float = 5.5,
    frac_causal_genes: float = 0.10,
    n_effect_programs: int = 3,
    effect_size: float = 0.6,
    n_cov_extra: int = 6,
    confound_strength: float = 0.0,
    eps_near_zero: float = 1e-6,
    sprinkle_near_zero_frac: float = 0.15,
    seed: int = 0,
):
    """
    AnnData generator matching the sceptre-style synthetic spec.
    See tests/specs/synthetic_generator.md for full details.
    """
    rng = np.random.default_rng(seed)

    if guides_per_gene != 6:
        raise ValueError("This generator assumes guides_per_gene=6.")
    if not 0.0 <= ntc_frac_guides < 1.0:
        raise ValueError("ntc_frac_guides must be in [0, 1).")

    n_target_guides = n_target_genes * guides_per_gene
    n_ntc_guides = int(
        np.round((ntc_frac_guides / (1.0 - ntc_frac_guides)) * n_target_guides)
    )
    n_guides = n_target_guides + n_ntc_guides

    target_gene_names = [f"GENE{g:05d}" for g in range(n_target_genes)]
    ntc_gene_name = "NTC"

    guide_names: List[str] = []
    guide2gene: Dict[str, str] = {}
    gene2guides: Dict[str, List[str]] = {}

    for gene in target_gene_names:
        guides = []
        for j in range(guides_per_gene):
            gname = f"{gene}_g{j + 1:02d}"
            guide_names.append(gname)
            guide2gene[gname] = gene
            guides.append(gname)
        gene2guides[gene] = guides

    ntc_guides: List[str] = []
    for j in range(n_ntc_guides):
        gname = f"NTC_g{j + 1:04d}"
        guide_names.append(gname)
        guide2gene[gname] = ntc_gene_name
        ntc_guides.append(gname)
    gene2guides[ntc_gene_name] = ntc_guides

    ntc_guide_idx = np.arange(n_target_guides, n_guides, dtype=np.int32)
    target_gene_to_guide_idx = {
        gene: np.arange(
            i * guides_per_gene,
            (i + 1) * guides_per_gene,
            dtype=np.int32,
        )
        for i, gene in enumerate(target_gene_names)
    }

    batch = rng.integers(0, 2, size=N).astype(np.float64)
    libsize = rng.lognormal(mean=10.0, sigma=0.35, size=N).astype(np.float64)
    log_depth = np.log1p(libsize)

    covs = [batch, log_depth]
    for _ in range(max(0, n_cov_extra)):
        covs.append(rng.normal(0, 1, size=N).astype(np.float64))

    C_no_intercept = np.column_stack(covs)
    intercept = np.ones((N, 1), dtype=np.float64)
    C = np.column_stack([intercept, C_no_intercept])

    W = rng.normal(0, 0.25, size=(C.shape[1], K))
    logits = C @ W + rng.normal(0, 0.3, size=(N, K))

    n_causal = max(1, int(np.round(frac_causal_genes * n_target_genes)))
    causal_gene_idx = rng.choice(n_target_genes, size=n_causal, replace=False)
    causal_genes = [target_gene_names[i] for i in causal_gene_idx]

    gene_effects: Dict[str, np.ndarray] = {}
    for gene in causal_genes:
        eff = np.zeros(K, dtype=np.float64)
        progs = rng.choice(K, size=min(n_effect_programs, K), replace=False)
        signs = rng.choice([-1.0, 1.0], size=progs.size)
        eff[progs] = signs * effect_size
        gene_effects[gene] = eff

    gene_base_weights = rng.lognormal(
        mean=0.0, sigma=0.6, size=n_target_genes
    ).astype(np.float64)
    if confound_strength > 0:
        tilt = rng.normal(0, confound_strength, size=n_target_genes).astype(
            np.float64
        )
        gene_w_batch0 = gene_base_weights * np.exp(-tilt)
        gene_w_batch1 = gene_base_weights * np.exp(+tilt)
    else:
        gene_w_batch0 = gene_base_weights
        gene_w_batch1 = gene_base_weights

    gene_p_batch0 = gene_w_batch0 / gene_w_batch0.sum()
    gene_p_batch1 = gene_w_batch1 / gene_w_batch1.sum()

    m = rng.poisson(moi_mean, size=N)
    m = np.clip(m, 0, min(50, n_guides)).astype(np.int32)

    rows: List[int] = []
    cols: List[int] = []
    x_causal = np.zeros((N, len(causal_genes)), dtype=np.int8)
    causal_gene_to_j = {g: j for j, g in enumerate(causal_genes)}

    for i in range(N):
        mi = int(m[i])
        if mi == 0:
            continue

        is_ntc = rng.random(mi) < ntc_frac_guides
        n_ntc = int(is_ntc.sum())
        n_tgt = mi - n_ntc

        chosen_cols: List[int] = []

        if n_ntc > 0:
            replace = n_ntc > ntc_guide_idx.size
            ntc_choices = rng.choice(
                ntc_guide_idx, size=n_ntc, replace=replace
            )
            chosen_cols.extend(ntc_choices.tolist())

        if n_tgt > 0:
            gene_p = gene_p_batch1 if batch[i] >= 0.5 else gene_p_batch0
            gene_idx = rng.choice(
                n_target_genes, size=n_tgt, replace=True, p=gene_p
            )
            within = rng.integers(0, guides_per_gene, size=n_tgt)
            guide_idx = gene_idx * guides_per_gene + within
            chosen_cols.extend(guide_idx.tolist())

            if causal_genes:
                for gidx in np.unique(gene_idx):
                    gene_name = target_gene_names[int(gidx)]
                    j = causal_gene_to_j.get(gene_name)
                    if j is not None:
                        x_causal[i, j] = 1

        if not chosen_cols:
            continue
        chosen_unique = np.unique(np.asarray(chosen_cols, dtype=np.int32))
        rows.extend([i] * chosen_unique.size)
        cols.extend(chosen_unique.tolist())

    data = np.ones(len(rows), dtype=np.int8)
    G = sp.csr_matrix(
        (data, (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(N, n_guides),
        dtype=np.int8,
    )
    G.eliminate_zeros()

    for j, gene in enumerate(causal_genes):
        logits += x_causal[:, [j]].astype(np.float64) * gene_effects[gene][None, :]

    usage = _softmax_rows(logits).astype(np.float64)
    if sprinkle_near_zero_frac > 0:
        mask = rng.random(size=usage.shape) < sprinkle_near_zero_frac
        usage[mask] *= eps_near_zero
        usage /= usage.sum(axis=1, keepdims=True)

    adata = ad.AnnData(X=sp.csr_matrix((N, 1), dtype=np.float32))
    adata.obs = pd.DataFrame(index=[f"cell{i:07d}" for i in range(N)])
    adata.obsm["covar"] = C
    adata.obsm["usage"] = usage
    adata.obsm["guide_assignment"] = G
    adata.uns["guide2gene"] = guide2gene
    adata.uns["gene2guides"] = gene2guides
    adata.uns["guide_names"] = guide_names
    adata.uns["target_gene_names"] = target_gene_names
    adata.uns["ntc_gene_name"] = ntc_gene_name
    adata.uns["causal_genes"] = causal_genes
    adata.uns["gene_effects"] = {
        g: gene_effects[g].tolist() for g in causal_genes
    }
    adata.uns["synth_params"] = dict(
        N=N,
        K=K,
        n_target_genes=n_target_genes,
        guides_per_gene=guides_per_gene,
        n_ntc_guides=n_ntc_guides,
        ntc_frac_guides=ntc_frac_guides,
        moi_mean=moi_mean,
        frac_causal_genes=frac_causal_genes,
        n_effect_programs=n_effect_programs,
        effect_size=effect_size,
        confound_strength=confound_strength,
        seed=seed,
    )
    return adata
