import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from src.sceptre.adata_utils import union_obs_idx_from_cols
from src.sceptre.pipeline import prepare_crt_inputs, run_all_genes_union_crt, run_one_gene_union_crt
from src.sceptre.pipeline_helpers import _fit_propensity, _gene_seed, _sample_crt_indices
from src.sceptre.propensity import fit_propensity_logistic
from src.sceptre.samplers import _propensity_bins, stratified_permutation_sampler
from tests.synthetic_data import make_sceptre_style_synth, make_synthetic_adata


def test_stratified_perm_uses_batch_from_covar_df(monkeypatch):
    rng = np.random.default_rng(10)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=300,
        n_programs=4,
        n_genes=3,
        guides_per_gene=3,
        n_covariates=3,
        include_categorical=True,
        effect_gene=None,
        effect_size=0.0,
        propensity_mode="constant",
        propensity_range=(0.1, 0.3),
    )
    inputs = prepare_crt_inputs(adata)
    gene = [g for g in inputs.gene_to_cols if g.startswith("gene_")][0]

    import src.sceptre.pipeline_helpers as ph

    captured = {}
    original = ph.stratified_permutation_sampler

    def _wrapped(*args, **kwargs):
        captured["batch_raw"] = kwargs.get("batch_raw")
        return original(*args, **kwargs)

    monkeypatch.setattr(ph, "stratified_permutation_sampler", _wrapped)

    _ = run_one_gene_union_crt(
        gene=gene,
        inputs=inputs,
        B=31,
        base_seed=7,
        resampling_method="stratified_perm",
        resampling_kwargs=dict(
            n_bins=5,
            stratify_by_batch=True,
            batch_key="batch",
            min_stratum_size=1,
        ),
        calibrate_skew_normal=False,
    )

    batch_raw = captured.get("batch_raw")
    assert batch_raw is not None
    assert batch_raw.shape[0] == inputs.C.shape[0]
    assert np.unique(batch_raw).size > 1


def test_stratified_perm_bin_count_from_propensity():
    rng = np.random.default_rng(11)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=400,
        n_programs=4,
        n_genes=3,
        guides_per_gene=3,
        n_covariates=3,
        include_categorical=True,
        effect_gene=None,
        effect_size=0.0,
        propensity_mode="covariate",
        propensity_range=(0.1, 0.3),
    )
    inputs = prepare_crt_inputs(adata)
    gene = [g for g in inputs.gene_to_cols if g.startswith("gene_")][0]
    obs_idx = union_obs_idx_from_cols(inputs.G, inputs.gene_to_cols[gene])
    p_hat = _fit_propensity(inputs, obs_idx, fit_propensity_logistic)

    covar_df = inputs.covar_df_raw
    assert covar_df is not None and "batch" in covar_df.columns
    batch_raw = covar_df["batch"].to_numpy()

    bin_id, n_bins_eff = _propensity_bins(p_hat, n_bins=10)
    batch_id, _ = pd.factorize(batch_raw, sort=False)
    stratum_id = batch_id.astype(np.int64) * n_bins_eff + bin_id

    assert np.unique(bin_id).size > 1
    assert np.unique(stratum_id).size > np.unique(batch_id).size


def test_global_null_uniform_pvals_stratified_perm():
    adata = make_sceptre_style_synth(
        N=5000,
        K=20,
        n_target_genes=30,
        guides_per_gene=6,
        ntc_frac_guides=0.15,
        frac_causal_genes=0.10,
        n_effect_programs=3,
        effect_size=0.0,
        confound_strength=0.0,
        seed=2,
    )
    covar_mat = np.asarray(adata.obsm["covar"])
    cols = ["batch", "log_depth"] + [
        f"covar_{i}" for i in range(max(0, covar_mat.shape[1] - 3))
    ]
    adata.obsm["covar"] = pd.DataFrame(
        covar_mat[:, 1:], columns=cols, index=adata.obs_names
    )

    inputs = prepare_crt_inputs(adata=adata, usage_key="usage")
    out = run_all_genes_union_crt(
        inputs=inputs,
        B=127,
        n_jobs=1,
        resampling_method="stratified_perm",
        resampling_kwargs=dict(n_bins=20, stratify_by_batch=True, batch_key="batch"),
        calibrate_skew_normal=False,
    )
    pvals = out["pvals_df"].to_numpy().ravel()
    pvals = pvals[np.isfinite(pvals)]

    q01, q10, q50 = np.quantile(pvals, [0.01, 0.10, 0.50])
    assert 0.005 <= q01 <= 0.03
    assert 0.06 <= q10 <= 0.16
    assert 0.43 <= q50 <= 0.57

    _, pval = kstest(pvals, "uniform")
    assert pval > 1e-3


def test_oracle_propensity_close_to_fit_under_null():
    rng = np.random.default_rng(12)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=500,
        n_programs=6,
        n_genes=5,
        guides_per_gene=3,
        n_covariates=3,
        include_categorical=True,
        effect_gene=None,
        effect_size=0.0,
        propensity_mode="constant",
        propensity_range=(0.1, 0.3),
    )
    inputs = prepare_crt_inputs(adata)
    genes = [g for g in inputs.gene_to_cols if g.startswith("gene_")]

    def _oracle_propensity(C: np.ndarray, y01: np.ndarray) -> np.ndarray:
        return np.full(C.shape[0], y01.mean(), dtype=np.float64)

    out_fit = run_all_genes_union_crt(
        inputs=inputs,
        genes=genes,
        B=63,
        n_jobs=1,
        resampling_method="stratified_perm",
        resampling_kwargs=dict(n_bins=10, stratify_by_batch=True, batch_key="batch"),
        calibrate_skew_normal=False,
    )
    out_oracle = run_all_genes_union_crt(
        inputs=inputs,
        genes=genes,
        B=63,
        n_jobs=1,
        resampling_method="stratified_perm",
        resampling_kwargs=dict(n_bins=10, stratify_by_batch=True, batch_key="batch"),
        calibrate_skew_normal=False,
        propensity_model=_oracle_propensity,
    )

    p_fit = np.sort(out_fit["pvals_df"].to_numpy().ravel())
    p_oracle = np.sort(out_oracle["pvals_df"].to_numpy().ravel())
    med_diff = np.median(np.abs(p_fit - p_oracle))
    assert med_diff < 0.06


def test_burden_binning_improves_ntc_uniformity():
    rng = np.random.default_rng(14)
    n_cells = 800
    B = 127

    batch = rng.integers(0, 2, size=n_cells)
    burden = rng.normal(0, 1, size=n_cells)
    x_prob = 1.0 / (1.0 + np.exp(-2.0 * burden))
    x_obs = (rng.random(n_cells) < x_prob).astype(np.int32)

    Y = (1.5 * burden + 0.2 * rng.normal(0, 1, size=n_cells)).reshape(-1, 1)
    C = np.column_stack([np.ones(n_cells), batch.astype(np.float64)])
    A = np.linalg.inv(C.T @ C)
    CTY = C.T @ Y

    obs_idx = np.nonzero(x_obs)[0].astype(np.int32)
    p_hat = np.full(n_cells, x_obs.mean(), dtype=np.float64)

    from src.sceptre.crt import crt_betas_for_gene
    from src.sceptre.pipeline_helpers import _raw_pvals_from_betas

    indptr_no, idx_no = stratified_permutation_sampler(
        x_obs=x_obs,
        p_hat=p_hat,
        B=B,
        seed=7,
        batch_raw=batch,
        n_bins=1,
        stratify_by_batch=True,
        min_stratum_size=1,
    )
    beta_obs_no, beta_null_no = crt_betas_for_gene(
        indptr_no,
        idx_no,
        C,
        Y,
        A,
        CTY,
        obs_idx,
        B,
    )
    p_no = _raw_pvals_from_betas(beta_obs_no, beta_null_no)[0]

    indptr_b, idx_b = stratified_permutation_sampler(
        x_obs=x_obs,
        p_hat=p_hat,
        B=B,
        seed=7,
        batch_raw=batch,
        n_bins=1,
        stratify_by_batch=True,
        min_stratum_size=1,
        burden_values=burden,
        n_burden_bins=5,
        burden_bin_method="quantile",
    )
    beta_obs_b, beta_null_b = crt_betas_for_gene(
        indptr_b,
        idx_b,
        C,
        Y,
        A,
        CTY,
        obs_idx,
        B,
    )
    p_b = _raw_pvals_from_betas(beta_obs_b, beta_null_b)[0]

    assert p_b >= p_no


def test_sampler_seed_independent_of_job_order():
    rng = np.random.default_rng(13)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=400,
        n_programs=4,
        n_genes=4,
        guides_per_gene=3,
        n_covariates=3,
        include_categorical=True,
        effect_gene=None,
        effect_size=0.0,
        propensity_mode="constant",
        propensity_range=(0.1, 0.3),
    )
    inputs = prepare_crt_inputs(adata)
    genes = [g for g in inputs.gene_to_cols if g.startswith("gene_")][:2]

    def _run(order):
        out = {}
        for gene in order:
            obs_idx = union_obs_idx_from_cols(inputs.G, inputs.gene_to_cols[gene])
            p_hat = _fit_propensity(inputs, obs_idx, fit_propensity_logistic)
            seed = _gene_seed(gene, 101)
            indptr, idx = _sample_crt_indices(
                p_hat,
                31,
                seed,
                resampling_method="stratified_perm",
                resampling_kwargs=dict(
                    n_bins=8,
                    stratify_by_batch=True,
                    batch_key="batch",
                    min_stratum_size=1,
                ),
                obs_idx=obs_idx,
                inputs=inputs,
            )
            out[gene] = (indptr.copy(), idx.copy())
        return out

    out1 = _run(genes)
    out2 = _run(list(reversed(genes)))

    for gene in genes:
        indptr1, idx1 = out1[gene]
        indptr2, idx2 = out2[gene]
        assert np.array_equal(indptr1, indptr2)
        assert np.array_equal(idx1, idx2)
