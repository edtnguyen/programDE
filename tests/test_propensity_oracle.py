import numpy as np
from scipy.stats import kstest

from src.sceptre.pipeline import prepare_crt_inputs, run_all_genes_union_crt
from tests.synthetic_data import make_synthetic_adata


def _oracle_propensity(C: np.ndarray, y01: np.ndarray):
    p = np.full(C.shape[0], y01.mean(), dtype=np.float64)
    return p


def _collect_raw_pvals(out):
    pvals = out["pvals_df"].to_numpy().ravel()
    return pvals[np.isfinite(pvals)]


def test_oracle_propensity_not_worse_than_fit():
    rng = np.random.default_rng(3)
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

    out_fit = run_all_genes_union_crt(
        inputs=inputs,
        genes=genes,
        B=79,
        n_jobs=1,
        calibrate_skew_normal=False,
    )
    out_oracle = run_all_genes_union_crt(
        inputs=inputs,
        genes=genes,
        B=79,
        n_jobs=1,
        calibrate_skew_normal=False,
        propensity_model=_oracle_propensity,
    )

    p_fit = _collect_raw_pvals(out_fit)
    p_oracle = _collect_raw_pvals(out_oracle)

    _, p_fit_ks = kstest(p_fit, "uniform")
    _, p_oracle_ks = kstest(p_oracle, "uniform")
    assert p_oracle_ks >= p_fit_ks - 0.2

    mean_fit = np.mean(p_fit)
    mean_oracle = np.mean(p_oracle)
    assert abs(mean_oracle - 0.5) <= abs(mean_fit - 0.5) + 0.05
