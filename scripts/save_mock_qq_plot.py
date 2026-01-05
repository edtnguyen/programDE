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

    from src.sceptre import prepare_crt_inputs, run_all_genes_union_crt, limit_threading
    from src.visualization import qq_plot_ntc_pvals
    from tests.synthetic_data import make_synthetic_adata

    limit_threading()
    rng = np.random.default_rng(args.seed)
    adata, _ = make_synthetic_adata(
        rng,
        n_cells=200,
        n_programs=6,
        n_genes=4,
        guides_per_gene=2,
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

    from src.sceptre import compute_gene_null_pvals, crt_null_stats_for_test

    null_pvals = compute_gene_null_pvals("non-targeting", inputs, B=63).ravel()
    null_stats = crt_null_stats_for_test(
        "non-targeting", 0, inputs, B=63
    )
    ax = qq_plot_ntc_pvals(
        pvals_raw_df=out["pvals_raw_df"],
        guide2gene=adata.uns["guide2gene"],
        ntc_genes=["non-targeting", "safe-targeting"],
        pvals_skew_df=out["pvals_df"],
        null_pvals=null_pvals,
        null_stats=null_stats,
        show_null_skew=True,
        null_skew_samples=500,
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
