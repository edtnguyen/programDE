import numpy as np
import pandas as pd
import pytest

from src.sceptre.diagnostics import qq_expected_grid
from src.visualization import qq_plot_ntc_pvals


def test_qq_expected_grid_lengths():
    p_all = np.random.uniform(size=50)
    p_sub = np.random.uniform(size=10)
    exp_all = qq_expected_grid(p_all)
    exp_sub = qq_expected_grid(p_sub)
    assert exp_all.size == p_all.size
    assert exp_sub.size == p_sub.size


def test_plot_expected_grid_length_mismatch_raises():
    pvals_df = pd.DataFrame(
        np.random.uniform(size=(2, 5)),
        index=["non-targeting", "safe-targeting"],
        columns=[f"program_{i}" for i in range(5)],
    )
    guide2gene = {"g0": "non-targeting", "g1": "safe-targeting"}
    null_pvals = np.random.uniform(size=20)
    expected_grid = qq_expected_grid(np.random.uniform(size=30))

    with pytest.raises(ValueError):
        qq_plot_ntc_pvals(
            pvals_raw_df=pvals_df,
            guide2gene=guide2gene,
            ntc_genes=["non-targeting", "safe-targeting"],
            null_pvals=null_pvals,
            expected_grid_raw=expected_grid,
            show_ref_line=True,
            show_conf_band=False,
        )
