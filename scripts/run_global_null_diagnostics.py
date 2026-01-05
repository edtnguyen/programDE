#!/usr/bin/env python3
"""
Run global-null diagnostics and save summary outputs.
"""

import argparse
import logging
import os
import sys


def _set_runtime_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run global-null diagnostics for sceptre CRT pipeline."
    )
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n-target-genes", type=int, default=50)
    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frac-causal-genes", type=float, default=0.2)
    parser.add_argument("--effect-size", type=float, default=0.0)
    parser.add_argument("--n-ensemble", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--n-null-groups",
        type=int,
        default=10,
        help="Number of NTC groups to use for CRT-null p-values (0 = all).",
    )
    parser.add_argument(
        "--out-csv", default="diagnostics_units.csv", help="Diagnostics CSV output."
    )
    parser.add_argument(
        "--out-summary",
        default="diagnostics_summary.txt",
        help="Diagnostics summary output.",
    )
    parser.add_argument(
        "--plot-fit",
        default=os.path.join("reports", "qq_plot_global_null_fit.png"),
        help="QQ plot path for fitted propensity.",
    )
    parser.add_argument(
        "--plot-oracle",
        default=os.path.join("reports", "qq_plot_global_null_oracle.png"),
        help="QQ plot path for oracle propensity.",
    )
    return parser.parse_args()


def main() -> None:
    _set_runtime_env()
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import numpy as np
    import pandas as pd

    from tests.synthetic_data import make_sceptre_style_synth
    from src.sceptre import (
        build_ntc_group_inputs,
        compute_guide_set_null_pvals,
        crt_pvals_for_ntc_groups_ensemble,
        crt_pvals_for_ntc_groups_ensemble_skew,
        limit_threading,
        make_ntc_groups_ensemble,
        prepare_crt_inputs,
        run_all_genes_union_crt,
    )
    from src.sceptre.global_null_diagnostics import (
        collect_gene_diagnostics,
        collect_ntc_group_diagnostics,
        oracle_propensity_const,
        summarize_diagnostics,
        write_summary,
    )
    from src.visualization import qq_plot_ntc_pvals

    limit_threading()
    os.makedirs("reports", exist_ok=True)
    mpl_dir = os.path.join("reports", "tmp", "mpl")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)

    adata = make_sceptre_style_synth(
        N=args.N,
        K=args.K,
        n_target_genes=args.n_target_genes,
        frac_causal_genes=args.frac_causal_genes,
        effect_size=args.effect_size,
        seed=args.seed,
    )
    inputs = prepare_crt_inputs(
        adata=adata,
        usage_key="usage",
        add_intercept=False,
        standardize=False,
    )

    out = run_all_genes_union_crt(
        inputs=inputs,
        B=args.B,
        n_jobs=1,
        calibrate_skew_normal=True,
        return_raw_pvals=True,
        return_skew_normal=True,
    )

    ntc_labels = ["NTC"]
    ntc_guides, guide_freq, guide_to_bin, real_sigs = build_ntc_group_inputs(
        inputs=inputs,
        ntc_label=ntc_labels,
        group_size=args.group_size,
        n_bins=args.n_bins,
    )
    ntc_groups_ens = make_ntc_groups_ensemble(
        ntc_guides=ntc_guides,
        ntc_freq=guide_freq,
        real_gene_bin_sigs=real_sigs,
        guide_to_bin=guide_to_bin,
        n_ensemble=args.n_ensemble,
        seed0=args.seed,
        group_size=args.group_size,
    )

    ntc_group_pvals_fit = crt_pvals_for_ntc_groups_ensemble(
        inputs=inputs,
        ntc_groups_ens=ntc_groups_ens,
        B=args.B,
        seed0=args.seed,
    )
    ntc_group_pvals_skew_fit = crt_pvals_for_ntc_groups_ensemble_skew(
        inputs=inputs,
        ntc_groups_ens=ntc_groups_ens,
        B=args.B,
        seed0=args.seed,
    )
    ntc_group_pvals_oracle = crt_pvals_for_ntc_groups_ensemble(
        inputs=inputs,
        ntc_groups_ens=ntc_groups_ens,
        B=args.B,
        seed0=args.seed,
        propensity_model=oracle_propensity_const,
    )
    ntc_group_pvals_skew_oracle = crt_pvals_for_ntc_groups_ensemble_skew(
        inputs=inputs,
        ntc_groups_ens=ntc_groups_ens,
        B=args.B,
        seed0=args.seed,
        propensity_model=oracle_propensity_const,
    )

    guide_to_col = {g: i for i, g in enumerate(inputs.guide_names)}
    all_groups = []
    for groups in ntc_groups_ens:
        all_groups.extend(list(groups.values()))
    if args.n_null_groups is not None and args.n_null_groups > 0:
        rng = np.random.default_rng(args.seed)
        if args.n_null_groups < len(all_groups):
            keep = rng.choice(
                len(all_groups), size=args.n_null_groups, replace=False
            )
            all_groups = [all_groups[i] for i in keep]

    null_list = []
    for guides in all_groups:
        cols = [guide_to_col[g] for g in guides if g in guide_to_col]
        if not cols:
            continue
        null_list.append(
            compute_guide_set_null_pvals(
                guide_idx=cols, inputs=inputs, B=args.B
            ).ravel()
        )
    null_pvals = np.concatenate(null_list)

    ax_fit = qq_plot_ntc_pvals(
        pvals_raw_df=out["pvals_raw_df"],
        guide2gene=adata.uns["guide2gene"],
        ntc_genes=ntc_labels,
        pvals_skew_df=out["pvals_df"],
        null_pvals=null_pvals,
        ntc_group_pvals_ens=ntc_group_pvals_fit,
        ntc_group_pvals_skew_ens=ntc_group_pvals_skew_fit,
        show_ntc_ensemble_band=True,
        show_all_pvals=True,
        title="Global null: NTC grouped (fit propensity)",
    )
    ax_fit.figure.tight_layout()
    ax_fit.figure.savefig(args.plot_fit, dpi=150)

    ax_oracle = qq_plot_ntc_pvals(
        pvals_raw_df=out["pvals_raw_df"],
        guide2gene=adata.uns["guide2gene"],
        ntc_genes=ntc_labels,
        pvals_skew_df=out["pvals_df"],
        null_pvals=null_pvals,
        ntc_group_pvals_ens=ntc_group_pvals_oracle,
        ntc_group_pvals_skew_ens=ntc_group_pvals_skew_oracle,
        show_ntc_ensemble_band=True,
        show_all_pvals=True,
        title="Global null: NTC grouped (oracle propensity)",
    )
    ax_oracle.figure.tight_layout()
    ax_oracle.figure.savefig(args.plot_oracle, dpi=150)

    gene_list = [g for g in inputs.gene_to_cols.keys() if g != "NTC"]
    diagnostics = []
    diagnostics.extend(
        collect_gene_diagnostics(
            inputs=inputs,
            genes=gene_list,
            B=args.B,
            base_seed=args.seed,
        )
    )
    diagnostics.extend(
        collect_ntc_group_diagnostics(
            inputs=inputs,
            ntc_groups=ntc_groups_ens[0],
            B=args.B,
            base_seed=args.seed,
        )
    )

    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(args.out_csv, index=False)

    summary = summarize_diagnostics(
        diagnostics_df=diagnostics_df,
        B=args.B,
        base_seed=args.seed,
    )
    write_summary(summary, args.out_summary)

    sample_ntc = diagnostics_df[diagnostics_df["unit_type"] == "ntc_group"]
    if not sample_ntc.empty:
        sample_ntc = sample_ntc.sample(
            n=min(5, sample_ntc.shape[0]), random_state=args.seed
        )
        print(
            "NTC debug (k=0): p_raw, p_skew, z_obs, null_z_mean, null_z_sd, null_z_skew"
        )
        for _, row in sample_ntc.iterrows():
            print(
                f"{row['unit_id']}: "
                f"p_raw={row['p_raw_k0']:.4g} "
                f"p_skew={row['p_skew_k0']:.4g} "
                f"z_obs={row['z_obs_k0']:.3f} "
                f"null_z_mean={row['null_z_mean_k0']:.3g} "
                f"null_z_sd={row['null_z_sd_k0']:.3g} "
                f"null_z_skew={row['null_z_skew_k0']:.3g}"
            )

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_summary}")
    print(f"Saved {args.plot_fit}")
    print(f"Saved {args.plot_oracle}")


if __name__ == "__main__":
    main()
