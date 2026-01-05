import numpy as np
import pytest

from src.sceptre.crt import compute_null_pvals_from_null_stats
from src.sceptre.pipeline_helpers import _raw_pvals_from_betas


def test_crt_pvalue_exact_count():
    beta_obs = np.array([2.0])
    beta_null = np.array([[0.1], [2.0], [-3.0], [1.9]])
    p = _raw_pvals_from_betas(beta_obs, beta_null)[0]
    assert np.isclose(p, 0.6)


def test_crt_pvalue_monotonicity():
    beta_null = np.array([[0.1], [2.0], [-3.0], [1.9]])
    p1 = _raw_pvals_from_betas(np.array([1.0]), beta_null)[0]
    p2 = _raw_pvals_from_betas(np.array([2.0]), beta_null)[0]
    assert p2 <= p1


def test_compute_null_pvals_from_null_stats_shapes():
    rng = np.random.default_rng(0)
    null = rng.normal(size=(50, 3))
    pvals = compute_null_pvals_from_null_stats(null, side_code=0)
    assert pvals.shape == null.shape
    assert np.all((pvals > 0.0) & (pvals <= 1.0))

    pvals_1d = compute_null_pvals_from_null_stats(null[:, 0], side_code=1)
    assert pvals_1d.shape == (null.shape[0],)


def test_compute_null_pvals_from_null_stats_invalid_side():
    with pytest.raises(ValueError):
        compute_null_pvals_from_null_stats(np.array([0.0, 1.0]), side_code=2)
