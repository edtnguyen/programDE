import numpy as np
from scipy.stats import kstest

from src.sceptre.pipeline import prepare_crt_inputs, run_all_genes_union_crt
from tests.synthetic_data import make_synthetic_adata


def test_null_calibration_type_i_error():
    rng = np.random.default_rng(10)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=180,
        n_programs=4,
        n_genes=4,
        guides_per_gene=2,
        n_covariates=3,
        effect_gene=None,
        effect_size=0.0,
    )
    inputs = prepare_crt_inputs(adata)
    out = run_all_genes_union_crt(inputs, B=63, n_jobs=1, calibrate_skew_normal=False)
    pvals = out["pvals_df"].to_numpy().ravel()

    alpha = 0.05
    fp = (pvals < alpha).mean()
    tol = 3.0 * np.sqrt(alpha * (1.0 - alpha) / pvals.size)
    assert abs(fp - alpha) < tol + 0.02

    stat, pval = kstest(pvals, "uniform")
    assert pval > 1e-3


def test_power_one_true_effect():
    rng = np.random.default_rng(11)
    effect_gene = "gene_0"
    effect_program = 0
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=200,
        n_programs=5,
        n_genes=4,
        guides_per_gene=2,
        n_covariates=3,
        effect_gene=effect_gene,
        effect_program=effect_program,
        effect_size=2.0,
    )
    inputs = prepare_crt_inputs(adata)
    out = run_all_genes_union_crt(
        inputs,
        genes=[effect_gene],
        B=127,
        n_jobs=1,
        calibrate_skew_normal=False,
    )

    pvals = out["pvals_df"].loc[effect_gene].to_numpy()
    betas = out["betas_df"].loc[effect_gene].to_numpy()
    assert betas[effect_program] > 0.0

    other_pvals = np.delete(pvals, effect_program)
    assert pvals[effect_program] <= np.median(other_pvals)
    assert pvals[effect_program] < 0.2
