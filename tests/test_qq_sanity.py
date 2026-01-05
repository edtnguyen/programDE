import numpy as np
from scipy.stats import kstest

from src.sceptre.diagnostics import crt_null_pvals_from_null_stats_fast, is_bh_adjusted_like
from src.sceptre.pipeline import prepare_crt_inputs, run_all_genes_union_crt
from tests.synthetic_data import make_synthetic_adata


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    q = np.empty_like(p)
    q[order] = p[order] * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    return np.clip(q, 0.0, 1.0)


def _collect_raw_pvals(out):
    pvals = out["pvals_df"].to_numpy().ravel()
    pvals = pvals[np.isfinite(pvals)]
    return pvals


def test_raw_pvals_uniform_under_null():
    rng = np.random.default_rng(0)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=600,
        n_programs=8,
        n_genes=6,
        guides_per_gene=3,
        n_covariates=4,
        effect_gene=None,
        effect_size=0.0,
        propensity_mode="constant",
        propensity_range=(0.1, 0.3),
    )
    inputs = prepare_crt_inputs(adata)
    genes = [g for g in inputs.gene_to_cols if g.startswith("gene_")]
    out = run_all_genes_union_crt(
        inputs=inputs,
        genes=genes,
        B=79,
        n_jobs=1,
        calibrate_skew_normal=False,
    )
    pvals = _collect_raw_pvals(out)
    assert np.all(np.isfinite(pvals))
    assert np.all(pvals > 0.0)
    assert np.all(pvals <= 1.0)

    frac_ones = np.mean(pvals >= 1.0 - 1e-12)
    assert frac_ones < 0.2

    mean_val = np.mean(pvals)
    assert abs(mean_val - 0.5) < 0.08

    stat, pval = kstest(pvals, "uniform")
    assert pval > 1e-4

    alpha = 0.05
    reject_rate = np.mean(pvals < alpha)
    tol = 3.0 * np.sqrt(alpha * (1.0 - alpha) / pvals.size)
    assert abs(reject_rate - alpha) < tol + 0.02


def test_raw_pvals_not_bh_adjusted():
    rng = np.random.default_rng(1)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=500,
        n_programs=6,
        n_genes=5,
        guides_per_gene=3,
        n_covariates=3,
        effect_gene=None,
        effect_size=0.0,
        propensity_mode="constant",
        propensity_range=(0.1, 0.3),
    )
    inputs = prepare_crt_inputs(adata)
    genes = [g for g in inputs.gene_to_cols if g.startswith("gene_")]
    out = run_all_genes_union_crt(
        inputs=inputs,
        genes=genes,
        B=79,
        n_jobs=1,
        calibrate_skew_normal=False,
    )
    pvals = _collect_raw_pvals(out)
    qvals = _bh_adjust(pvals)
    # Discrete CRT p-values can coincide with BH-adjusted values at the extremes.
    assert np.mean(np.isclose(pvals, qvals)) < 0.1
    assert not is_bh_adjusted_like(pvals)


def test_null_pvals_uniform_for_single_test():
    rng = np.random.default_rng(2)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=400,
        n_programs=4,
        n_genes=4,
        guides_per_gene=3,
        n_covariates=3,
        effect_gene=None,
        effect_size=0.0,
        propensity_mode="constant",
        propensity_range=(0.1, 0.3),
    )
    inputs = prepare_crt_inputs(adata)
    gene = [g for g in inputs.gene_to_cols if g.startswith("gene_")][0]
    from src.sceptre import crt_null_stats_for_test

    null_stats = crt_null_stats_for_test(
        gene=gene,
        program_index=0,
        inputs=inputs,
        B=199,
    )
    p_null = crt_null_pvals_from_null_stats_fast(null_stats, two_sided=True)
    stat, pval = kstest(p_null, "uniform")
    assert pval > 1e-3
