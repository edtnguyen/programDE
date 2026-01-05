import numpy as np

from src.sceptre import prepare_crt_inputs, run_all_genes_union_crt, run_one_gene_union_crt
from tests.synthetic_data import make_synthetic_adata


def test_prepare_crt_inputs_shapes(mock_adata):
    inputs = prepare_crt_inputs(mock_adata)
    assert inputs.C.shape[0] == inputs.Y.shape[0]
    assert inputs.C.shape[0] == inputs.G.shape[0]
    assert inputs.Y.shape[1] == len(inputs.program_names)


def test_run_all_genes_union_crt_raw(mock_adata):
    inputs = prepare_crt_inputs(mock_adata)
    out = run_all_genes_union_crt(inputs, B=15, n_jobs=1, calibrate_skew_normal=False)
    assert "pvals_df" in out
    assert "betas_df" in out
    assert "treated_df" in out
    assert "results" in out
    assert out["pvals_df"].shape[0] == len(out["results"])


def test_run_all_genes_union_crt_skew_outputs(mock_adata):
    inputs = prepare_crt_inputs(mock_adata)
    out = run_all_genes_union_crt(
        inputs,
        B=15,
        n_jobs=1,
        calibrate_skew_normal=True,
        return_raw_pvals=True,
        return_skew_normal=True,
    )
    assert "pvals_raw_df" in out
    assert "pvals_skew_df" in out
    assert "skew_params" in out
    assert np.allclose(out["pvals_df"], out["pvals_skew_df"])


def test_union_crt_trivial_no_treated():
    rng = np.random.default_rng(7)
    adata, _ = make_synthetic_adata(
        rng, n_cells=50, n_programs=3, n_genes=2, guides_per_gene=2
    )
    guide_names = adata.uns["guide_names"]
    cols = [i for i, g in enumerate(guide_names) if g.startswith("gene_0_")]
    adata.obsm["guide_assignment"][:, cols] = 0

    inputs = prepare_crt_inputs(adata)
    result = run_one_gene_union_crt("gene_0", inputs, B=31)
    assert np.all(result.pvals == 1.0)
    assert np.all(result.betas == 0.0)


def test_union_crt_trivial_all_treated():
    rng = np.random.default_rng(8)
    adata, _ = make_synthetic_adata(
        rng, n_cells=50, n_programs=3, n_genes=2, guides_per_gene=2
    )
    guide_names = adata.uns["guide_names"]
    cols = [i for i, g in enumerate(guide_names) if g.startswith("gene_0_")]
    adata.obsm["guide_assignment"][:, cols] = 1

    inputs = prepare_crt_inputs(adata)
    result = run_one_gene_union_crt("gene_0", inputs, B=31)
    assert np.all(result.pvals == 1.0)
    assert np.all(result.betas == 0.0)
