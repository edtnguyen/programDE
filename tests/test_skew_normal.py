import numpy as np
from scipy.stats import norm, skewnorm

from src.sceptre.pipeline_helpers import _compute_skew_normal_pvals
from src.sceptre.skew_normal import (
    compute_empirical_p_value,
    fit_and_evaluate_skew_normal,
    fit_skew_normal,
)


def test_fit_skew_normal_params():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    params = fit_skew_normal(z)
    assert params.shape == (5,)
    assert np.all(np.isfinite(params))


def test_fit_and_evaluate_skew_normal_returns_p():
    rng = np.random.default_rng(0)
    null = rng.normal(size=200)
    out = fit_and_evaluate_skew_normal(0.5, null, 0)
    assert out.shape == (4,)
    assert out[3] <= 1.0
    assert out[3] >= -1.0


def test_compute_empirical_p_value_two_sided():
    null = np.array([-1.0, 0.0, 1.0])
    p = compute_empirical_p_value(null, 0.5, 0)
    assert 0.0 < p <= 1.0


def test_fit_skew_normal_alpha_near_zero_for_normal():
    rng = np.random.default_rng(1)
    z = rng.normal(size=1000)
    params = fit_skew_normal(z)
    assert abs(params[2]) < 1.0


def test_fit_and_evaluate_two_sided_matches_tail():
    rng = np.random.default_rng(2)
    null = rng.normal(size=600)
    z_obs = 1.0
    out = fit_and_evaluate_skew_normal(z_obs, null, 0)
    assert out[3] > 0.0

    xi, omega, alpha = out[:3]
    dist = skewnorm(alpha, loc=xi, scale=omega)
    median = np.median(null)
    tail = dist.sf(z_obs) if z_obs >= median else dist.cdf(z_obs)
    expected = 2.0 * tail
    assert np.isclose(out[3], expected, rtol=1e-5, atol=1e-6)


def test_skew_normal_fallback_on_degenerate_null():
    beta_obs = np.array([0.0])
    beta_null = np.zeros((50, 1), dtype=np.float64)
    pvals, params = _compute_skew_normal_pvals(beta_obs, beta_null, side_code=0)
    assert np.allclose(pvals, 1.0)
    assert np.all(np.isnan(params))


def test_skew_normal_pvals_close_to_normal_for_normal_null():
    rng = np.random.default_rng(4)
    null = rng.normal(size=1000)
    for z_obs in (-1.0, 0.0, 1.5):
        out = fit_and_evaluate_skew_normal(z_obs, null, 0)
        p_sn = out[3]
        p_ref = 2.0 * norm.sf(abs(z_obs))
        assert np.isclose(p_sn, p_ref, rtol=0.2, atol=0.05)
