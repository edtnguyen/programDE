"""
QQ plot helpers for CRT p-value calibration checks.
"""

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def qq_plot_non_targeting_pvals(
    pvals_df: pd.DataFrame,
    guide2gene: Mapping[str, str],
    non_targeting_genes: Iterable[str],
    *,
    ax=None,
    seed: Optional[int] = 0,
    title: Optional[str] = None,
    label_non_targeting: str = "non-targeting",
    label_null: str = "null",
    color_non_targeting: str = "#1f77b4",
    color_null: str = "#7f7f7f",
    show_ref_line: bool = True,
    show_conf_band: bool = True,
    conf_alpha: float = 0.05,
    conf_color: str = "#d9d9d9",
):
    """
    QQ plot comparing non-targeting p-values to a null Uniform(0,1) reference.
    """
    non_targeting = list(dict.fromkeys(non_targeting_genes))
    if len(non_targeting) == 0:
        raise ValueError("non_targeting_genes must contain at least one gene name.")

    gene_names = set(guide2gene.values())
    missing = [g for g in non_targeting if g not in gene_names]
    if missing:
        raise ValueError(
            "non_targeting_genes not found in guide2gene values: "
            + ", ".join(sorted(missing))
        )

    missing_idx = [g for g in non_targeting if g not in pvals_df.index]
    if missing_idx:
        raise ValueError(
            "non_targeting_genes not found in pvals_df index: "
            + ", ".join(sorted(missing_idx))
        )

    pvals = pvals_df.loc[non_targeting].to_numpy().ravel()
    pvals = pvals[np.isfinite(pvals)]
    if pvals.size == 0:
        raise ValueError("No finite p-values available for non-targeting genes.")

    pvals = np.clip(pvals, 1e-300, 1.0)
    m = pvals.size

    expected = (np.arange(1, m + 1) - 0.5) / m
    x = -np.log10(expected)

    y_non_target = -np.log10(np.sort(pvals))

    rng = np.random.default_rng(seed)
    null_pvals = rng.uniform(size=m)
    y_null = -np.log10(np.sort(null_pvals))

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6, 5))

    if show_conf_band:
        from scipy.stats import beta

        i = np.arange(1, m + 1)
        lower = beta.ppf(conf_alpha / 2.0, i, m - i + 1)
        upper = beta.ppf(1.0 - conf_alpha / 2.0, i, m - i + 1)
        lower = -np.log10(np.clip(lower, 1e-300, 1.0))
        upper = -np.log10(np.clip(upper, 1e-300, 1.0))
        ax.fill_between(x, lower, upper, color=conf_color, alpha=0.5, label="95% CI")

    if show_ref_line:
        xmin, xmax = float(x.min()), float(x.max())
        ax.plot([xmin, xmax], [xmin, xmax], color="#333333", linewidth=1.0, label="y=x")

    ax.plot(x, y_non_target, label=label_non_targeting, color=color_non_targeting)
    ax.plot(x, y_null, label=label_null, color=color_null, linestyle="--")
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return ax
