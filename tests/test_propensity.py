import numpy as np

from src.sceptre.propensity import fit_propensity_logistic


def test_fit_propensity_logistic_bounds():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(30, 3))
    y = rng.integers(0, 2, size=30)
    p, model = fit_propensity_logistic(C, y, max_iter=100, n_jobs=1)
    assert p.shape == (30,)
    assert np.all(p > 0.0)
    assert np.all(p < 1.0)
    assert hasattr(model, "coef_")


def test_fit_propensity_intercept_only():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=50)
    C = np.ones((50, 1))
    p, _ = fit_propensity_logistic(C, y, max_iter=200, n_jobs=1)
    assert np.std(p) < 1e-8
    assert abs(p.mean() - y.mean()) < 1e-2


def test_fit_propensity_permutation_invariance():
    rng = np.random.default_rng(2)
    C = rng.normal(size=(40, 3))
    y = rng.integers(0, 2, size=40)
    p1, _ = fit_propensity_logistic(C, y, max_iter=200, n_jobs=1)

    perm = rng.permutation(C.shape[0])
    p2, _ = fit_propensity_logistic(C[perm], y[perm], max_iter=200, n_jobs=1)
    inv = np.argsort(perm)
    assert np.allclose(p1, p2[inv], atol=1e-6)


def test_fit_propensity_separation_robustness():
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    y = (x > 0.0).astype(np.int8)
    C = np.column_stack([x, rng.normal(size=60)])
    p, _ = fit_propensity_logistic(C, y, max_iter=200, n_jobs=1)
    assert np.all(np.isfinite(p))
    assert np.all(p > 0.0)
    assert np.all(p < 1.0)
