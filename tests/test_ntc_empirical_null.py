import numpy as np
import pandas as pd
from scipy.stats import kstest

from src.sceptre.ntc_null import (
    compute_ols_denom,
    compute_ols_denom_reference,
    empirical_pvals_vs_ntc,
)
from src.sceptre.pipeline import prepare_crt_inputs, run_all_genes_union_crt
from tests.synthetic_data import make_sceptre_style_synth


def _covar_df_from_synth(adata):
    covar_mat = np.asarray(adata.obsm["covar"])
    cols = ["batch", "log_depth"] + [
        f"covar_{i}" for i in range(max(0, covar_mat.shape[1] - 3))
    ]
    return pd.DataFrame(covar_mat[:, 1:], columns=cols, index=adata.obs_names)


def test_ols_denom_matches_reference():
    rng = np.random.default_rng(0)
    n = 200
    p = 5
    C = rng.normal(size=(n, p))
    C[:, 0] = 1.0
    x = (rng.random(n) < 0.2).astype(np.float64)
    d_ref = compute_ols_denom_reference(C, x)
    d_fast = compute_ols_denom(C, x)
    assert np.isfinite(d_ref)
    assert np.isfinite(d_fast)
    assert np.allclose(d_ref, d_fast, rtol=1e-6, atol=1e-6)


def test_empirical_pvals_vs_ntc_matches_bruteforce():
    rng = np.random.default_rng(1)
    G, R, K = 5, 40, 3
    beta_obs = rng.normal(size=(G, K))
    beta_ntc = rng.normal(size=(R, K))
    n1_obs = rng.integers(5, 20, size=G)
    d_obs = rng.random(G) + 0.1
    n1_ntc = rng.integers(5, 20, size=R)
    d_ntc = rng.random(R) + 0.1

    pvals, _ = empirical_pvals_vs_ntc(
        beta_obs,
        beta_ntc,
        n1_obs,
        d_obs,
        n1_ntc,
        d_ntc,
        n_n1_bins=1,
        n_d_bins=1,
        min_ntc_per_bin=1,
        two_sided=True,
    )

    brute = np.zeros_like(pvals)
    abs_ntc = np.abs(beta_ntc)
    for g in range(G):
        for k in range(K):
            count_ge = np.sum(abs_ntc[:, k] >= abs(beta_obs[g, k]))
            brute[g, k] = (1.0 + count_ge) / (R + 1.0)

    assert np.allclose(pvals, brute)


def test_global_null_uniform_pvals_ntc_empirical():
    adata = make_sceptre_style_synth(
        N=4000,
        K=12,
        n_target_genes=25,
        guides_per_gene=6,
        ntc_frac_guides=0.15,
        frac_causal_genes=0.10,
        n_effect_programs=3,
        effect_size=0.0,
        confound_strength=0.0,
        seed=2,
    )
    adata.obsm["covar"] = _covar_df_from_synth(adata)
    inputs = prepare_crt_inputs(adata=adata, usage_key="usage")
    out = run_all_genes_union_crt(
        inputs=inputs,
        null_method="ntc_empirical",
        null_kwargs=dict(
            ntc_labels=["NTC"],
            guides_per_unit=6,
            n_ntc_units=800,
            batch_mode="meta",
            combine_method="fisher",
            matching=dict(n_n1_bins=8, n_d_bins=8, min_ntc_per_bin=30),
        ),
    )
    pvals = out["pvals_df"].to_numpy().ravel()
    pvals = pvals[np.isfinite(pvals)]
    q01, q10, q50 = np.quantile(pvals, [0.01, 0.10, 0.50])
    assert 0.001 <= q01 <= 0.03
    assert 0.02 <= q10 <= 0.22
    assert 0.40 <= q50 <= 0.60
    _, pval = kstest(pvals, "uniform")
    assert pval > 1e-6


def test_power_sanity_ntc_empirical():
    adata = make_sceptre_style_synth(
        N=4000,
        K=12,
        n_target_genes=25,
        guides_per_gene=6,
        ntc_frac_guides=0.15,
        frac_causal_genes=0.20,
        n_effect_programs=3,
        effect_size=0.8,
        confound_strength=0.0,
        seed=3,
    )
    adata.obsm["covar"] = _covar_df_from_synth(adata)
    inputs = prepare_crt_inputs(adata=adata, usage_key="usage")
    out = run_all_genes_union_crt(
        inputs=inputs,
        null_method="ntc_empirical",
        null_kwargs=dict(
            ntc_labels=["NTC"],
            guides_per_unit=6,
            n_ntc_units=800,
            batch_mode="meta",
            combine_method="fisher",
            matching=dict(n_n1_bins=8, n_d_bins=8, min_ntc_per_bin=30),
        ),
    )
    pvals = out["pvals_df"]
    causal_genes = set(adata.uns.get("causal_genes", []))
    if not causal_genes:
        assert False, "Synthetic generator did not mark any causal genes."
    causal = pvals.loc[pvals.index.intersection(causal_genes)]
    frac_small = np.mean(causal.to_numpy().ravel() < 0.1)
    assert frac_small > 0.05


def test_default_null_method_matches_crt():
    adata = make_sceptre_style_synth(
        N=1500,
        K=8,
        n_target_genes=12,
        guides_per_gene=6,
        ntc_frac_guides=0.15,
        frac_causal_genes=0.10,
        n_effect_programs=2,
        effect_size=0.0,
        confound_strength=0.0,
        seed=4,
    )
    adata.obsm["covar"] = _covar_df_from_synth(adata)
    inputs = prepare_crt_inputs(adata=adata, usage_key="usage")
    out_default = run_all_genes_union_crt(inputs=inputs, B=63, n_jobs=1)
    out_crt = run_all_genes_union_crt(
        inputs=inputs, B=63, n_jobs=1, null_method="crt"
    )
    assert np.allclose(out_default["pvals_df"], out_crt["pvals_df"])
