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
        show_ref_line=True,
        show_conf_band=False,
    )
    assert ax is not None
