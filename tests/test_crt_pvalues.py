import numpy as np

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
