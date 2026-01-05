import numpy as np

from src.sceptre.diagnostics import crt_null_pvals_from_null_stats_fast


def _naive_null_pvals(T_null: np.ndarray, two_sided: bool) -> np.ndarray:
    B = T_null.size
    out = np.empty(B, dtype=np.float64)
    for b in range(B):
        if two_sided:
            val = abs(T_null[b])
            others = np.abs(T_null[np.arange(B) != b])
            count = np.sum(others >= val)
        else:
            val = T_null[b]
            others = T_null[np.arange(B) != b]
            count = np.sum(others >= val)
        out[b] = (1.0 + count) / B
    return out


def test_null_pvals_fast_matches_naive_no_ties():
    rng = np.random.default_rng(0)
    T_null = rng.normal(size=200)
    p_fast = crt_null_pvals_from_null_stats_fast(T_null, two_sided=True)
    p_naive = _naive_null_pvals(T_null, two_sided=True)
    assert np.allclose(p_fast, p_naive, atol=1e-12)


def test_null_pvals_fast_matches_naive_with_ties():
    rng = np.random.default_rng(1)
    T_null = rng.normal(size=200)
    T_null = np.round(T_null, 1)
    p_fast = crt_null_pvals_from_null_stats_fast(T_null, two_sided=True)
    p_naive = _naive_null_pvals(T_null, two_sided=True)
    assert np.allclose(p_fast, p_naive, atol=1e-12)


def test_null_pvals_bounds():
    rng = np.random.default_rng(2)
    T_null = rng.normal(size=150)
    p_fast = crt_null_pvals_from_null_stats_fast(T_null, two_sided=False)
    assert np.all(p_fast >= 1.0 / T_null.size)
    assert np.all(p_fast <= 1.0)
