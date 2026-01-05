#!/usr/bin/env python3
"""
Minimal in-memory smoke test for the CRT + skew-normal pipeline.
"""

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


def _build_mock_adata(rng, n_cells=40, n_programs=5, n_guides=6):
    class MockAdata:
        def __init__(self):
            self.obsm = {}
            self.layers = {}
            self.obsp = {}
            self.uns = {}
            self.obs = {}

    adata = MockAdata()

    adata.obsm["covar"] = rng.normal(size=(n_cells, 3))
    adata.obsm["cnmf_usage"] = rng.dirichlet(
        alpha=[1.0] * n_programs, size=n_cells
    )

    guide_mat = (rng.random((n_cells, n_guides)) < 0.25).astype("int8")
    adata.obsm["guide_assignment"] = guide_mat

    guide_names = [f"g{j}" for j in range(n_guides)]
    adata.uns["guide_names"] = guide_names
    adata.uns["guide2gene"] = {
        "g0": "geneA",
        "g1": "geneA",
        "g2": "geneB",
        "g3": "geneB",
        "g4": "non-targeting",
        "g5": "safe-targeting",
    }
    adata.uns["program_names"] = [f"program_{k}" for k in range(n_programs)]
    return adata


def main() -> None:
    _set_runtime_env()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import numpy as np

    from src.sceptre import prepare_crt_inputs, run_all_genes_union_crt, limit_threading
    from src.visualization import qq_plot_ntc_pvals

    limit_threading()
    rng = np.random.default_rng(0)
    adata = _build_mock_adata(rng)

    inputs = prepare_crt_inputs(adata)
    out = run_all_genes_union_crt(
        inputs,
        B=31,
        n_jobs=1,
        calibrate_skew_normal=True,
        return_raw_pvals=True,
        return_skew_normal=True,
    )

    from src.sceptre import compute_gene_null_pvals, crt_null_stats_for_test

    null_pvals = compute_gene_null_pvals("non-targeting", inputs, B=31).ravel()
    null_stats = crt_null_stats_for_test(
        "non-targeting", 0, inputs, B=31
    )
    ax = qq_plot_ntc_pvals(
        pvals_raw_df=out["pvals_raw_df"],
        guide2gene=adata.uns["guide2gene"],
        ntc_genes=["non-targeting", "safe-targeting"],
        pvals_skew_df=out["pvals_df"],
        null_pvals=null_pvals,
        null_stats=null_stats,
        show_null_skew=True,
        null_skew_samples=200,
        show_ref_line=True,
        show_conf_band=True,
    )
    print("Mock CRT run OK. QQ plot axis:", type(ax))


if __name__ == "__main__":
    main()
