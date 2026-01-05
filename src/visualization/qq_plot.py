"""
QQ plot helpers for CRT p-value calibration checks.
"""

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def qq_plot_ntc_pvals(
    pvals_raw_df: Optional[pd.DataFrame],
    guide2gene: Mapping[str, str],
    ntc_genes: Iterable[str],
    *,
    pvals_skew_df: Optional[pd.DataFrame] = None,
    null_pvals: Sequence[float],
    ax=None,
    title: Optional[str] = None,
    label_ntc_raw: str = "NTC (raw)",
    label_skew: str = "NTC (skew)",
    label_null: str = "null",
    color_ntc_raw: str = "#1f77b4",
    color_skew: str = "#ff7f0e",
    color_null: str = "#7f7f7f",
    show_ref_line: bool = True,
    show_conf_band: bool = True,
    conf_alpha: float = 0.05,
    conf_color: str = "#d9d9d9",
):
    """
    QQ plot comparing NTC (negative-control) p-values to a CRT-null reference.
    If pvals_skew_df is provided, plots both raw and skew-calibrated curves.
    """
    if pvals_raw_df is None:
        raise ValueError("pvals_raw_df is required.")
    if null_pvals is None:
        raise ValueError("null_pvals is required.")
    ntc = list(dict.fromkeys(ntc_genes))
    if len(ntc) == 0:
        raise ValueError("ntc_genes must contain at least one gene name.")

    gene_names = set(guide2gene.values())
    missing = [g for g in ntc if g not in gene_names]
    if missing:
        raise ValueError(
            "ntc_genes not found in guide2gene values: " + ", ".join(sorted(missing))
        )

    def _extract_pvals(df: pd.DataFrame, label: str) -> np.ndarray:
        missing_idx = [g for g in ntc if g not in df.index]
        if missing_idx:
            raise ValueError(
                f"ntc_genes not found in {label} index: "
                + ", ".join(sorted(missing_idx))
            )
        pvals = df.loc[ntc].to_numpy().ravel()
        pvals = pvals[np.isfinite(pvals)]
        if pvals.size == 0:
            raise ValueError(f"No finite p-values available for {label}.")
        return np.clip(pvals, 1e-300, 1.0)

    def _qq_data(pvals: np.ndarray):
        m = pvals.size
        expected = (np.arange(1, m + 1) - 0.5) / m
        x = -np.log10(expected)
        y = -np.log10(np.sort(pvals))
        return x, y, m

    def _normalize_null_pvals(pvals: Sequence[float]) -> np.ndarray:
        arr = np.asarray(pvals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("null_pvals contains no finite values.")
        return np.clip(arr, 1e-300, 1.0)

    pvals_raw = _extract_pvals(pvals_raw_df, "pvals_raw_df")
    x_raw, y_raw, m_raw = _qq_data(pvals_raw)

    if pvals_skew_df is not None:
        pvals_skew = _extract_pvals(pvals_skew_df, "pvals_skew_df")
        x_skew, y_skew, m_skew = _qq_data(pvals_skew)
    else:
        x_skew = y_skew = None
        m_skew = 0

    null_arr = _normalize_null_pvals(null_pvals)
    x_null, y_null, m_null = _qq_data(null_arr)

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6, 5))

    if show_conf_band:
        from scipy.stats import beta

        i = np.arange(1, m_null + 1)
        lower = beta.ppf(conf_alpha / 2.0, i, m_null - i + 1)
        upper = beta.ppf(1.0 - conf_alpha / 2.0, i, m_null - i + 1)
        lower = -np.log10(np.clip(lower, 1e-300, 1.0))
        upper = -np.log10(np.clip(upper, 1e-300, 1.0))
        expected = (np.arange(1, m_null + 1) - 0.5) / m_null
        x_band = -np.log10(expected)
        ax.fill_between(
            x_band, lower, upper, color=conf_color, alpha=0.5, label="95% CI"
        )

    if show_ref_line:
        xmin = float(np.min(x_raw))
        xmax = float(np.max(x_raw))
        if x_skew is not None:
            xmin = min(xmin, float(np.min(x_skew)))
            xmax = max(xmax, float(np.max(x_skew)))
        ax.plot([xmin, xmax], [xmin, xmax], color="#333333", linewidth=1.0, label="y=x")

    ax.plot(x_raw, y_raw, label=label_ntc_raw, color=color_ntc_raw)
    if x_skew is not None:
        ax.plot(x_skew, y_skew, label=label_skew, color=color_skew)
    ax.plot(x_null, y_null, label=label_null, color=color_null, linestyle="--")
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return ax
