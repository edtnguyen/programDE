import numpy as np
import pandas as pd

from src.visualization import qq_plot_ntc_pvals


def test_qq_plot_ntc_pvals_raw_only():
    pvals_df = pd.DataFrame(
        np.random.uniform(size=(2, 5)),
        index=["non-targeting", "safe-targeting"],
        columns=[f"program_{i}" for i in range(5)],
    )
    guide2gene = {"g0": "non-targeting", "g1": "safe-targeting"}
    ax = qq_plot_ntc_pvals(
        pvals_raw_df=pvals_df,
        guide2gene=guide2gene,
        ntc_genes=["non-targeting", "safe-targeting"],
        pvals_skew_df=None,
        null_pvals=np.random.uniform(size=20),
        show_ref_line=True,
        show_conf_band=False,
    )
    assert ax is not None


def test_qq_plot_ntc_pvals_with_null_pvals():
    rng = np.random.default_rng(0)
    pvals_df = pd.DataFrame(
        rng.uniform(size=(2, 4)),
        index=["non-targeting", "safe-targeting"],
        columns=[f"program_{i}" for i in range(4)],
    )
    guide2gene = {"g0": "non-targeting", "g1": "safe-targeting"}
    null_pvals = rng.uniform(size=50)
    ax = qq_plot_ntc_pvals(
        pvals_raw_df=pvals_df,
        guide2gene=guide2gene,
        ntc_genes=["non-targeting", "safe-targeting"],
        null_pvals=null_pvals,
        show_ref_line=True,
        show_conf_band=False,
    )
    assert ax is not None


def test_qq_plot_ntc_pvals_with_null_stats():
    rng = np.random.default_rng(2)
    pvals_df = pd.DataFrame(
        rng.uniform(size=(2, 3)),
        index=["non-targeting", "safe-targeting"],
        columns=[f"program_{i}" for i in range(3)],
    )
    guide2gene = {"g0": "non-targeting", "g1": "safe-targeting"}
    null_stats = rng.normal(size=200)
    ax = qq_plot_ntc_pvals(
        pvals_raw_df=pvals_df,
        guide2gene=guide2gene,
        ntc_genes=["non-targeting", "safe-targeting"],
        null_stats=null_stats,
        null_two_sided=True,
        show_ref_line=True,
        show_conf_band=False,
    )
    assert ax is not None
