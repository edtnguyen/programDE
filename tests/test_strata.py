import numpy as np
import pandas as pd

from src.sceptre.samplers import _propensity_bins, compute_bins


def _make_stratum_id(
    p_hat: np.ndarray,
    batch_raw: np.ndarray,
    *,
    n_bins: int,
    stratify_by_batch: bool,
    min_stratum_size: int = 2,
):
    bin_id, n_bins_eff = _propensity_bins(p_hat, n_bins)
    if stratify_by_batch and batch_raw is not None:
        batch_id, _ = pd.factorize(batch_raw, sort=False)
        stratum_id = batch_id.astype(np.int64, copy=False) * n_bins_eff + bin_id
    else:
        stratum_id = bin_id.astype(np.int64, copy=False)

    if min_stratum_size is not None and min_stratum_size > 1:
        unique, counts = np.unique(stratum_id, return_counts=True)
        small = unique[counts < min_stratum_size]
        if small.size > 0:
            misc_id = int(stratum_id.max()) + 1 if stratum_id.size else 0
            stratum_id = stratum_id.copy()
            mask = np.isin(stratum_id, small)
            stratum_id[mask] = misc_id
    return stratum_id, bin_id, n_bins_eff


def test_strata_non_degenerate_on_random_p():
    rng = np.random.default_rng(0)
    n_cells = 5000
    p_hat = rng.random(n_cells)
    batch_raw = rng.integers(0, 5, size=n_cells)

    stratum_id, bin_id, _ = _make_stratum_id(
        p_hat,
        batch_raw,
        n_bins=20,
        stratify_by_batch=True,
        min_stratum_size=2,
    )
    n_unique_bins = np.unique(bin_id).size
    n_strata = np.unique(stratum_id).size
    n_batches = np.unique(batch_raw).size
    sizes = np.bincount(stratum_id)

    assert n_unique_bins >= 10
    assert n_strata >= n_batches * 5
    assert np.median(sizes) > 10
    assert np.mean(sizes < 2) == 0.0

    p_tied = np.round(p_hat, 2)
    stratum_tied, _, _ = _make_stratum_id(
        p_tied,
        batch_raw,
        n_bins=20,
        stratify_by_batch=True,
        min_stratum_size=2,
    )
    assert np.unique(stratum_tied).size > 1


def test_strata_fallback_when_p_collapses():
    rng = np.random.default_rng(1)
    n_cells = 2000
    p_hat = np.full(n_cells, 0.3, dtype=np.float64)
    batch_raw = rng.integers(0, 3, size=n_cells)

    stratum_id, bin_id, n_bins_eff = _make_stratum_id(
        p_hat,
        batch_raw,
        n_bins=20,
        stratify_by_batch=True,
        min_stratum_size=2,
    )
    assert n_bins_eff == 1
    assert np.unique(bin_id).size == 1
    assert np.unique(stratum_id).size == np.unique(batch_raw).size

    stratum_id2, _, _ = _make_stratum_id(
        p_hat,
        batch_raw,
        n_bins=20,
        stratify_by_batch=True,
        min_stratum_size=2,
    )
    assert np.array_equal(stratum_id, stratum_id2)


def test_strata_burden_increases_strata_count():
    rng = np.random.default_rng(2)
    n_cells = 4000
    p_hat = rng.random(n_cells)
    batch_raw = rng.integers(0, 4, size=n_cells)
    burden = rng.normal(0, 1, size=n_cells)

    stratum_no_burden, _, _ = _make_stratum_id(
        p_hat,
        batch_raw,
        n_bins=12,
        stratify_by_batch=True,
        min_stratum_size=2,
    )
    n_strata_no = np.unique(stratum_no_burden).size

    burden_bin, n_burden_bins_eff, _ = compute_bins(
        burden, n_bins=6, method="quantile"
    )
    bin_id, n_bins_eff = _propensity_bins(p_hat, 12)
    batch_id, _ = pd.factorize(batch_raw, sort=False)
    stratum_with_burden = (
        batch_id.astype(np.int64) * (n_bins_eff * n_burden_bins_eff)
        + bin_id * n_burden_bins_eff
        + burden_bin
    )
    n_strata_with = np.unique(stratum_with_burden).size

    assert n_strata_with > n_strata_no
