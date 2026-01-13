"""
Parallel helpers for NTC-group CRT diagnostics.
"""

from typing import List, Mapping, Sequence

import numpy as np

compute_guide_set_null_pvals = None


def _resolve_compute_null_fn():
    fn = globals().get("compute_guide_set_null_pvals")
    if callable(fn):
        return fn
    from .pipeline import compute_guide_set_null_pvals as _compute

    return _compute


def compute_ntc_group_null_pvals_parallel(
    inputs,
    ntc_groups_ens: Sequence[Mapping[str, Sequence[str]]],
    B: int,
    base_seed: int = 123,
    n_jobs: int = 8,
    backend: str = "threading",
) -> np.ndarray:
    """
    Compute concatenated CRT-null p-values for all NTC groups across ensembles.
    Uses joblib for parallelization across guide sets.
    """
    guide_to_col = {g: i for i, g in enumerate(inputs.guide_names)}
    guide_sets: List[List[int]] = []
    for groups in ntc_groups_ens:
        for guides in groups.values():
            cols = [guide_to_col[g] for g in guides if g in guide_to_col]
            if cols:
                guide_sets.append(cols)

    if not guide_sets:
        raise ValueError("No NTC guide sets available for null p-value computation.")

    compute_fn = _resolve_compute_null_fn()

    def _null_for_cols(cols: List[int]) -> np.ndarray:
        return compute_fn(
            guide_idx=cols,
            inputs=inputs,
            B=B,
            base_seed=base_seed,
        ).ravel()

    if n_jobs == 1:
        null_list = [_null_for_cols(cols) for cols in guide_sets]
    else:
        from joblib import Parallel, delayed

        null_list = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_null_for_cols)(cols) for cols in guide_sets
        )

    return np.concatenate(null_list)
