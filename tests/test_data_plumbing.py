import numpy as np
import pytest

from src.sceptre.adata_utils import build_gene_to_cols
from src.sceptre.pipeline import prepare_crt_inputs
from tests.synthetic_data import make_synthetic_adata


def test_prepare_crt_inputs_alignment_error():
    rng = np.random.default_rng(0)
    adata, truth = make_synthetic_adata(rng, n_cells=20, n_programs=4, return_truth=True)
    assert truth is not None
    adata.obsm["cnmf_usage"] = np.vstack([truth.usage, truth.usage[:1]])
    with pytest.raises(ValueError):
        prepare_crt_inputs(adata)


def test_prepare_crt_inputs_alignment_error_guide_assignment():
    rng = np.random.default_rng(1)
    adata, _ = make_synthetic_adata(rng, n_cells=20, n_programs=4)
    adata.obsm["guide_assignment"] = adata.obsm["guide_assignment"][:10, :]
    with pytest.raises(ValueError):
        prepare_crt_inputs(adata)


def test_build_gene_to_cols_skips_missing_guides():
    guide_names = ["g0", "g1"]
    guide2gene = {"g0": "A", "g1": "A", "g2": "B"}
    gene_to_cols = build_gene_to_cols(guide_names, guide2gene)
    assert "A" in gene_to_cols
    assert "B" not in gene_to_cols


def test_guide_assignment_is_binary():
    rng = np.random.default_rng(1)
    adata, _ = make_synthetic_adata(rng, n_cells=15, n_programs=3)
    G = adata.obsm["guide_assignment"]
    assert np.all(np.isin(G, [0, 1]))
