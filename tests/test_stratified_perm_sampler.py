import numpy as np
import pandas as pd
from scipy.stats import kstest

from src.sceptre.pipeline import prepare_crt_inputs, run_all_genes_union_crt
from src.sceptre.samplers import _propensity_bins, stratified_permutation_sampler
from tests.synthetic_data import make_synthetic_adata


def _stratum_ids(p_hat, batch_raw, n_bins, stratify_by_batch):
    bin_id, n_bins_eff = _propensity_bins(p_hat, n_bins)
    if stratify_by_batch and batch_raw is not None:
        batch_id, _ = pd.factorize(batch_raw, sort=False)
        return batch_id.astype(np.int64) * n_bins_eff + bin_id
    return bin_id.astype(np.int64)


def test_stratified_perm_preserves_stratum_counts():
    rng = np.random.default_rng(0)
    n_cells = 200
    B = 25
    x_obs = (rng.random(n_cells) < 0.2).astype(np.int8)
    p_hat = rng.random(n_cells)
    batch_raw = rng.integers(0, 3, size=n_cells)

    indptr, indices = stratified_permutation_sampler(
        x_obs=x_obs,
        p_hat=p_hat,
        B=B,
        seed=7,
        batch_raw=batch_raw,
        n_bins=5,
        stratify_by_batch=True,
        min_stratum_size=1,
    )

    stratum_id = _stratum_ids(p_hat, batch_raw, n_bins=5, stratify_by_batch=True)
    strata = {sid: np.nonzero(stratum_id == sid)[0] for sid in np.unique(stratum_id)}
    m_s = {sid: int(x_obs[idx].sum()) for sid, idx in strata.items()}
    total_treated = int(x_obs.sum())

    for b in range(B):
        sel = indices[indptr[b] : indptr[b + 1]]
        x_tilde = np.zeros(n_cells, dtype=np.int8)
        x_tilde[sel] = 1
        assert int(x_tilde.sum()) == total_treated
        for sid, idx in strata.items():
            assert int(x_tilde[idx].sum()) == m_s[sid]


def test_stratified_perm_reproducible():
    rng = np.random.default_rng(1)
    n_cells = 150
    x_obs = (rng.random(n_cells) < 0.25).astype(np.int8)
    p_hat = rng.random(n_cells)
    batch_raw = rng.integers(0, 4, size=n_cells)

    out1 = stratified_permutation_sampler(
        x_obs=x_obs,
        p_hat=p_hat,
        B=19,
        seed=11,
        batch_raw=batch_raw,
        n_bins=4,
        stratify_by_batch=True,
        min_stratum_size=1,
    )
    out2 = stratified_permutation_sampler(
        x_obs=x_obs,
        p_hat=p_hat,
        B=19,
        seed=11,
        batch_raw=batch_raw,
        n_bins=4,
        stratify_by_batch=True,
        min_stratum_size=1,
    )
    out3 = stratified_permutation_sampler(
        x_obs=x_obs,
        p_hat=p_hat,
        B=19,
        seed=12,
        batch_raw=batch_raw,
        n_bins=4,
        stratify_by_batch=True,
        min_stratum_size=1,
    )

    assert np.array_equal(out1[0], out2[0])
    assert np.array_equal(out1[1], out2[1])
    assert not np.array_equal(out1[1], out3[1])


def test_stratified_perm_pipeline_uniform_under_null():
    rng = np.random.default_rng(2)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=600,
        n_programs=6,
        n_genes=6,
        guides_per_gene=3,
        n_covariates=4,
        include_categorical=True,
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
        resampling_method="stratified_perm",
        resampling_kwargs=dict(n_bins=10, stratify_by_batch=True, batch_key="batch"),
        calibrate_skew_normal=False,
    )
    pvals = out["pvals_df"].to_numpy().ravel()
    pvals = pvals[np.isfinite(pvals)]

    stat, pval = kstest(pvals, "uniform")
    assert pval > 1e-4


def test_bernoulli_default_matches_explicit():
    rng = np.random.default_rng(3)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=300,
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
    genes = [g for g in inputs.gene_to_cols if g.startswith("gene_")]
    out_default = run_all_genes_union_crt(
        inputs=inputs,
        genes=genes,
        B=31,
        n_jobs=1,
        calibrate_skew_normal=False,
    )
    out_explicit = run_all_genes_union_crt(
        inputs=inputs,
        genes=genes,
        B=31,
        n_jobs=1,
        resampling_method="bernoulli_index",
        calibrate_skew_normal=False,
    )
    assert np.allclose(out_default["pvals_df"], out_explicit["pvals_df"])
