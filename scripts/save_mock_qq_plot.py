#!/usr/bin/env python3
"""
Generate a mock CRT run and save a QQ plot PNG for visual inspection.
"""

import argparse
import os
import sys
import tempfile


def _set_runtime_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    mpl_dir = os.path.join(tempfile.gettempdir(), "matplotlib")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a mock CRT QQ plot to disk."
    )
    parser.add_argument(
        "--out",
        default=os.path.join("reports", "qq_plot_mock.png"),
        help="Output PNG path (default: reports/qq_plot_mock.png).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    _set_runtime_env()
    args = _parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import numpy as np

    from src.sceptre import (
        build_ntc_group_inputs,
        crt_pvals_for_ntc_groups_ensemble,
        crt_pvals_for_ntc_groups_ensemble_skew,
        limit_threading,
        make_ntc_groups_ensemble,
        prepare_crt_inputs,
        run_all_genes_union_crt,
    )
    from src.visualization import qq_plot_ntc_pvals
    from tests.synthetic_data import make_synthetic_adata

    limit_threading()
    rng = np.random.default_rng(args.seed)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=300,
        n_programs=6,
        n_genes=4,
        guides_per_gene=12,
        n_covariates=4,
    )

    inputs = prepare_crt_inputs(adata)
    out = run_all_genes_union_crt(
        inputs,
        B=63,
        n_jobs=1,
        calibrate_skew_normal=True,
        return_raw_pvals=True,
        return_skew_normal=True,
    )

    from src.sceptre import compute_guide_set_null_pvals

    ntc_genes = ["non-targeting", "safe-targeting"]
    ntc_guides, guide_freq, guide_to_bin, real_sigs = build_ntc_group_inputs(
        inputs,
        ntc_label=ntc_genes,
        group_size=6,
        n_bins=6,
    )
    ntc_groups_ens = make_ntc_groups_ensemble(
        ntc_guides=ntc_guides,
        ntc_freq=guide_freq,
        real_gene_bin_sigs=real_sigs,
        guide_to_bin=guide_to_bin,
        n_ensemble=5,
        seed0=11,
        group_size=6,
    )
    ntc_group_pvals_ens = crt_pvals_for_ntc_groups_ensemble(
        inputs=inputs,
        ntc_groups_ens=ntc_groups_ens,
        B=63,
        seed0=23,
    )
    ntc_group_pvals_skew_ens = crt_pvals_for_ntc_groups_ensemble_skew(
        inputs=inputs,
        ntc_groups_ens=ntc_groups_ens,
        B=63,
        seed0=23,
    )

    guide_to_col = {g: i for i, g in enumerate(inputs.guide_names)}
    null_list = []
    for groups in ntc_groups_ens:
        for guides in groups.values():
            cols = [guide_to_col[g] for g in guides if g in guide_to_col]
            if not cols:
                continue
            null_list.append(
                compute_guide_set_null_pvals(
                    guide_idx=cols, inputs=inputs, B=63
                ).ravel()
            )
    null_pvals = np.concatenate(null_list)
    ax = qq_plot_ntc_pvals(
        pvals_raw_df=out["pvals_raw_df"],
        guide2gene=adata.uns["guide2gene"],
        ntc_genes=ntc_genes,
        pvals_skew_df=out["pvals_df"],
        null_pvals=null_pvals,
        ntc_group_pvals_ens=ntc_group_pvals_ens,
        ntc_group_pvals_skew_ens=ntc_group_pvals_skew_ens,
        title="Mock NTC QQ plot (raw vs skew)",
        show_ref_line=True,
        show_conf_band=True,
    )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    print(f"Saved QQ plot to {out_path}")


if __name__ == "__main__":
    main()
