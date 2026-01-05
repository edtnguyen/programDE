import numpy as np
import pandas as pd
import pytest

from src.sceptre.adata_utils import (
    build_gene_to_cols,
    clr_from_usage,
    encode_categorical_covariates,
    get_covar_matrix,
    get_program_names,
    union_obs_idx_from_cols,
)


def test_encode_categorical_covariates_numeric_threshold():
    df = pd.DataFrame(
        {
            "batch": [0, 1, 2, 1],
            "donor": ["a", "b", "a", "c"],
            "flag": [True, False, True, False],
            "numeric": [0.1, 0.2, 0.3, 0.4],
        }
    )
    out = encode_categorical_covariates(
        df, drop_first=True, numeric_as_category_threshold=3
    )

    assert "numeric" in out.columns
    assert "batch" not in out.columns
    assert any(col.startswith("batch_") for col in out.columns)
    assert any(col.startswith("donor_") for col in out.columns)
    assert any(col.startswith("flag_") for col in out.columns)


def test_get_covar_matrix_encodes_object_array():
    class MockAdata:
        def __init__(self):
            self.obsm = {}
            self.layers = {}
            self.obsp = {}
            self.uns = {}
            self.obs = {}

    adata = MockAdata()
    adata.obsm["covar"] = np.array([["a", "x"], ["b", "y"]], dtype=object)
    C, cols = get_covar_matrix(adata, one_hot_encode=True, drop_first=False)
    assert C.dtype == np.float64
    assert cols is not None
    assert C.shape[0] == 2
    assert C.shape[1] >= 2


def test_get_covar_matrix_rejects_object_without_encoding():
    class MockAdata:
        def __init__(self):
            self.obsm = {}
            self.layers = {}
            self.obsp = {}
            self.uns = {}
            self.obs = {}

    adata = MockAdata()
    adata.obsm["covar"] = np.array([["a", "x"], ["b", "y"]], dtype=object)
    with pytest.raises(ValueError):
        get_covar_matrix(adata, one_hot_encode=False)


def test_get_covar_matrix_intercept_and_standardize():
    class MockAdata:
        def __init__(self):
            self.obsm = {}
            self.layers = {}
            self.obsp = {}
            self.uns = {}
            self.obs = {}

    adata = MockAdata()
    adata.obsm["covar"] = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    C, cols = get_covar_matrix(
        adata,
        add_intercept=True,
        standardize=True,
        numeric_as_category_threshold=None,
    )
    assert cols is not None
    assert C.shape[1] == 2
    assert np.allclose(C[:, 0], 1.0)


def test_clr_from_usage_shapes():
    U = np.array([[0.2, 0.8], [0.5, 0.5]])
    Y = clr_from_usage(U, eps_quantile=0.01)
    assert Y.shape == U.shape
    assert np.all(np.isfinite(Y))


def test_clr_row_sum_zero_and_ratio_identity():
    rng = np.random.default_rng(0)
    U = rng.dirichlet(alpha=[1.0, 1.0, 1.0], size=10)
    eps_q = 1e-4
    Y = clr_from_usage(U, eps_quantile=eps_q)
    assert np.allclose(Y.sum(axis=1), 0.0, atol=1e-6)

    eps = np.quantile(U, eps_q)
    Ueps = np.maximum(U, eps)
    Ueps /= Ueps.sum(axis=1, keepdims=True)
    lhs = Y[:, 0] - Y[:, 1]
    rhs = np.log(Ueps[:, 0]) - np.log(Ueps[:, 1])
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_build_gene_to_cols_and_union_obs():
    guide_names = ["g1", "g2", "g3"]
    guide2gene = {"g1": "A", "g2": "A", "g3": "B"}
    gene_to_cols = build_gene_to_cols(guide_names, guide2gene)
    assert set(gene_to_cols["A"]) == {0, 1}

    import scipy.sparse as sp

    G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    G_csc = sp.csc_matrix(G)
    obs_idx = union_obs_idx_from_cols(G_csc, gene_to_cols["A"])
    assert set(obs_idx.tolist()) == {0, 1}

    x_union = np.zeros(G.shape[0], dtype=np.int8)
    x_union[obs_idx] = 1
    x_or = np.zeros(G.shape[0], dtype=np.int8)
    for col in gene_to_cols["A"]:
        x_or |= G[:, col].astype(np.int8)
    assert np.array_equal(x_union, x_or)


def test_union_obs_idx_empty_cols():
    import scipy.sparse as sp

    G = sp.csc_matrix(np.zeros((3, 2), dtype=np.int8))
    obs_idx = union_obs_idx_from_cols(G, [])
    assert obs_idx.size == 0


def test_union_obs_idx_all_zero_cols():
    import scipy.sparse as sp

    G = sp.csc_matrix(np.zeros((4, 3), dtype=np.int8))
    obs_idx = union_obs_idx_from_cols(G, [0, 2])
    assert obs_idx.size == 0


def test_get_program_names(mock_adata):
    names = get_program_names(mock_adata, 5)
    assert names == mock_adata.uns["program_names"]
