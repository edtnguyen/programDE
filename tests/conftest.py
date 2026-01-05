import os
import sys

import numpy as np
import pytest

from .synthetic_data import make_synthetic_adata


def pytest_configure():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mpl_cache"))

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


@pytest.fixture
def mock_adata():
    rng = np.random.default_rng(0)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=40,
        n_programs=5,
        n_genes=2,
        guides_per_gene=2,
        n_covariates=3,
    )
    return adata


@pytest.fixture
def synthetic_adata_truth():
    rng = np.random.default_rng(1)
    return make_synthetic_adata(
        rng,
        n_cells=160,
        n_programs=6,
        n_genes=4,
        guides_per_gene=2,
        n_covariates=4,
        return_truth=True,
    )
