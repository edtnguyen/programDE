import numpy as np

from src.sceptre.pipeline import prepare_crt_inputs, run_all_genes_union_crt
from tests.synthetic_data import make_synthetic_adata


def test_gene_order_invariance():
    rng = np.random.default_rng(5)
    adata, _ = make_synthetic_adata(
        rng, n_cells=80, n_programs=4, n_genes=3, guides_per_gene=2
    )
    inputs = prepare_crt_inputs(adata)
    genes = list(inputs.gene_to_cols.keys())
    order2 = list(reversed(genes))

    out1 = run_all_genes_union_crt(inputs, genes=genes, B=31, n_jobs=1)
    out2 = run_all_genes_union_crt(inputs, genes=order2, B=31, n_jobs=1)

    p1 = out1["pvals_df"].loc[genes]
    p2 = out2["pvals_df"].reindex(genes)
    b1 = out1["betas_df"].loc[genes]
    b2 = out2["betas_df"].reindex(genes)
    assert np.allclose(p1.values, p2.values)
    assert np.allclose(b1.values, b2.values)


def test_parallel_determinism():
    rng = np.random.default_rng(6)
    adata, _ = make_synthetic_adata(
        rng, n_cells=90, n_programs=4, n_genes=3, guides_per_gene=2
    )
    inputs = prepare_crt_inputs(adata)

    out_serial = run_all_genes_union_crt(inputs, B=31, n_jobs=1)
    out_parallel = run_all_genes_union_crt(
        inputs, B=31, n_jobs=2, backend="threading"
    )

    assert np.allclose(out_serial["pvals_df"], out_parallel["pvals_df"])
    assert np.allclose(out_serial["betas_df"], out_parallel["betas_df"])


def test_gene_order_invariance_utest():
    rng = np.random.default_rng(7)
    adata, _ = make_synthetic_adata(
        rng, n_cells=80, n_programs=4, n_genes=3, guides_per_gene=2
    )
    inputs = prepare_crt_inputs(adata)
    genes = list(inputs.gene_to_cols.keys())
    order2 = list(reversed(genes))

    out1 = run_all_genes_union_crt(
        inputs, genes=genes, B=31, n_jobs=1, test_stat="utest"
    )
    out2 = run_all_genes_union_crt(
        inputs, genes=order2, B=31, n_jobs=1, test_stat="utest"
    )

    p1 = out1["pvals_df"].loc[genes]
    p2 = out2["pvals_df"].reindex(genes)
    b1 = out1["betas_df"].loc[genes]
    b2 = out2["betas_df"].reindex(genes)
    assert np.allclose(p1.values, p2.values)
    assert np.allclose(b1.values, b2.values)


def test_parallel_determinism_utest():
    rng = np.random.default_rng(8)
    adata, _ = make_synthetic_adata(
        rng, n_cells=90, n_programs=4, n_genes=3, guides_per_gene=2
    )
    inputs = prepare_crt_inputs(adata)

    out_serial = run_all_genes_union_crt(
        inputs, B=31, n_jobs=1, test_stat="utest"
    )
    out_parallel = run_all_genes_union_crt(
        inputs, B=31, n_jobs=2, backend="threading", test_stat="utest"
    )

    assert np.allclose(out_serial["pvals_df"], out_parallel["pvals_df"])
    assert np.allclose(out_serial["betas_df"], out_parallel["betas_df"])
