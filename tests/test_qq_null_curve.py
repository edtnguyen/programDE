import numpy as np
from scipy.stats import kstest

from src.sceptre.diagnostics import crt_null_pvals_from_null_stats_fast


def test_qq_null_curve_uniformity():
    rng = np.random.default_rng(3)
    T_null = rng.normal(size=5000)
    p_null = crt_null_pvals_from_null_stats_fast(T_null, two_sided=True)
    stat, pval = kstest(p_null, "uniform")
    assert pval > 1e-3
