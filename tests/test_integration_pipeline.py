from src.sceptre import (
    compute_gene_null_pvals,
    limit_threading,
    prepare_crt_inputs,
    run_all_genes_union_crt,
)
from src.visualization import qq_plot_ntc_pvals


def test_full_pipeline_integration(mock_adata):
    limit_threading()
    inputs = prepare_crt_inputs(mock_adata)
    out = run_all_genes_union_crt(
        inputs,
        B=31,
        n_jobs=1,
        calibrate_skew_normal=True,
        return_raw_pvals=True,
        return_skew_normal=True,
    )

    assert "pvals_df" in out
    assert "betas_df" in out
    assert "treated_df" in out
    assert "pvals_raw_df" in out
    assert "pvals_skew_df" in out
    assert "skew_params" in out

    assert out["pvals_df"].shape == out["betas_df"].shape
    assert out["pvals_raw_df"].shape == out["pvals_df"].shape
    assert out["pvals_skew_df"].shape == out["pvals_df"].shape
    assert out["skew_params"].shape[:2] == out["pvals_df"].shape
    assert out["skew_params"].shape[2] == 3

    null_pvals = compute_gene_null_pvals("non-targeting", inputs, B=31).ravel()
    ax = qq_plot_ntc_pvals(
        pvals_raw_df=out["pvals_raw_df"],
        guide2gene=mock_adata.uns["guide2gene"],
        ntc_genes=["non-targeting", "safe-targeting"],
        pvals_skew_df=out["pvals_df"],
        null_pvals=null_pvals,
        show_ref_line=True,
        show_conf_band=True,
    )
    assert ax is not None
