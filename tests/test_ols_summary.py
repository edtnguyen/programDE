import numpy as np
import pytest

from src.sceptre.crt import _beta_from_summaries
from src.sceptre.pipeline import prepare_crt_inputs
from tests.synthetic_data import MockAdata


def test_beta_from_summaries_matches_lstsq():
    rng = np.random.default_rng(0)
    N = 80
    p = 3
    K = 4

    C = rng.normal(size=(N, p - 1))
    C = np.column_stack([np.ones(N), C])
    x = rng.integers(0, 2, size=N).astype(np.float64)
    Y = rng.normal(size=(N, K))

    CtC = C.T @ C
    A = np.linalg.inv(CtC)
    CTY = C.T @ Y
    v = C.T @ x
    sY = Y.T @ x

    beta_fast = _beta_from_summaries(int(x.sum()), v, sY, A, CTY)

    X = np.column_stack([x, C])
    beta_ref = []
    for k in range(K):
        coef, *_ = np.linalg.lstsq(X, Y[:, k], rcond=None)
        beta_ref.append(coef[0])
    beta_ref = np.array(beta_ref)

    assert np.allclose(beta_fast, beta_ref, atol=1e-7, rtol=1e-6)


def test_prepare_crt_inputs_rank_deficient_raises():
    adata = MockAdata()
    C = np.column_stack([np.ones(10), np.arange(10), np.arange(10)])
    adata.obsm["covar"] = C
    adata.obsm["cnmf_usage"] = np.tile([0.5, 0.5], (10, 1))
    adata.obsm["guide_assignment"] = np.zeros((10, 2), dtype=np.int8)
    adata.uns["guide_names"] = ["g0", "g1"]
    adata.uns["guide2gene"] = {"g0": "A", "g1": "B"}
    adata.uns["program_names"] = ["p0", "p1"]

    with pytest.raises(np.linalg.LinAlgError):
        prepare_crt_inputs(adata)
