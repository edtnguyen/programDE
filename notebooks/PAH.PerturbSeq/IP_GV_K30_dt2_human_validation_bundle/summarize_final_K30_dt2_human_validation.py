#!/usr/bin/env python3
"""Summarize final K30/dt2.0 validation in six human PAEC donors.

This is program-level external validation. It does not test whether each mouse
target gene is causal in human PAH. Experimentally targeted genes have already
been removed from program gene sets and the human differential-expression
ranking by pah_external_validation.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


CONTROL_TARGET_LABELS = {
    "non-targeting",
    "non_targeting",
    "nontargeting",
    "ntc",
    "control",
    "safe-targeting",
    "safe_targeting",
    "safe targeting",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--crt-ip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-label", default="K30_dt2_0_n100")
    parser.add_argument("--storey-fdr", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_required(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", **kwargs)
    if frame.empty:
        raise ValueError(f"Table is empty: {path}")
    return frame


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    normalized = series.astype(str).str.strip().str.lower()
    output = pd.Series(pd.NA, index=series.index, dtype="boolean")
    output.loc[normalized.isin({"true", "1", "yes"})] = True
    output.loc[normalized.isin({"false", "0", "no"})] = False
    return output


def normalize_program(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.replace(r"\.0$", "", regex=True)
    return values


def storey_qvalues(
    pvalues: np.ndarray,
    *,
    seed: int,
    lambdas: np.ndarray | None = None,
    n_bootstrap: int = 2000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Storey q values using bootstrap lambda selection."""
    p = np.clip(np.asarray(pvalues, dtype=float), 0.0, 1.0)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("pvalues must be a non-empty vector.")
    if not np.isfinite(p).all():
        raise ValueError("pvalues contain NaN or infinite values.")

    if lambdas is None:
        lambdas = np.arange(0.0, 0.95, 0.05)

    m = len(p)
    pi0_grid = np.array(
        [(p >= lam).mean() / (1.0 - lam) for lam in lambdas],
        dtype=float,
    )
    minimum_pi0 = min(float(pi0_grid.min()), 1.0)

    rng = np.random.default_rng(seed)
    mse = np.zeros(len(lambdas), dtype=float)
    for _ in range(n_bootstrap):
        sampled = rng.choice(p, size=m, replace=True)
        estimates = np.array(
            [
                (sampled >= lam).mean() / (1.0 - lam)
                for lam in lambdas
            ],
            dtype=float,
        )
        mse += (estimates - minimum_pi0) ** 2
    mse /= n_bootstrap

    selected = int(np.argmin(mse))
    pi0 = min(float(pi0_grid[selected]), 1.0)

    order = np.argsort(p)
    ranked = p[order]
    q_sorted = pi0 * m * ranked / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q, {
        "pi0": pi0,
        "selected_lambda": float(lambdas[selected]),
        "n_tests": int(m),
        "n_bootstrap": int(n_bootstrap),
    }


def validation_support_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    for column in [
        "hedges_g",
        "exact_permutation_p",
        "score_BH_FDR",
        "NES",
        "NOM_p_value",
        "GSEA_FDR",
        "human_expression_coverage",
    ]:
        if column in result:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    if "complete_separation" in result:
        result["complete_separation"] = parse_bool(
            result["complete_separation"]
        )
    else:
        result["complete_separation"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

    if "direction_concordant" in result:
        result["human_score_GSEA_direction_concordant"] = parse_bool(
            result["direction_concordant"]
        )
    else:
        result["human_score_GSEA_direction_concordant"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

    result["GSEA_FDR_lt_0.25"] = (
        result.get("GSEA_FDR", pd.Series(np.nan, index=result.index))
        < 0.25
    )
    result["GSEA_FDR_lt_0.05"] = (
        result.get("GSEA_FDR", pd.Series(np.nan, index=result.index))
        < 0.05
    )
    result["large_donor_effect_abs_g_ge_1"] = (
        result.get("hedges_g", pd.Series(np.nan, index=result.index))
        .abs()
        .ge(1.0)
    )
    result["human_program_support"] = (
        result["GSEA_FDR_lt_0.25"]
        | (
            result["large_donor_effect_abs_g_ge_1"]
            & result["complete_separation"].fillna(False)
        )
    )
    return result


def save_bar_plot(
    frame: pd.DataFrame,
    *,
    value: str,
    title: str,
    xlabel: str,
    output: Path,
    label_column: str = "program",
) -> None:
    plot = frame.dropna(subset=[value]).copy()
    plot = plot.sort_values(value)
    if plot.empty:
        return

    height = max(6.0, 0.24 * len(plot))
    fig, ax = plt.subplots(figsize=(9, height))
    positions = np.arange(len(plot))
    ax.barh(positions, plot[value])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(plot[label_column].astype(str))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Program")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validation_dir = args.validation_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = validation_dir / "external_validation_summary.tsv"
    mapping_path = (
        validation_dir / "program_mouse_to_human_mapping.tsv.gz"
    )
    coverage_path = validation_dir / "ortholog_coverage.tsv"
    score_path = validation_dir / "donor_program_scores_long.tsv"
    effects_path = validation_dir / "program_donor_score_effects.tsv"
    gsea_path = validation_dir / "program_PAH_preranked_GSEA.tsv"
    overlap_path = validation_dir / "program_PAH_DE_overlap.tsv"
    targets_removed_path = (
        validation_dir / "experimentally_targeted_genes_removed.tsv"
    )
    metadata_path = validation_dir / "model_validation_metadata.json"

    required = [
        summary_path,
        mapping_path,
        coverage_path,
        score_path,
        effects_path,
        gsea_path,
        overlap_path,
        targets_removed_path,
        metadata_path,
        args.crt_ip,
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            "Missing final-model validation inputs:\n"
            + "\n".join(missing)
        )

    final_summary = read_required(summary_path)
    final_summary["program"] = normalize_program(
        final_summary["program"]
    )
    final_summary = validation_support_flags(final_summary)

    coverage = read_required(coverage_path)
    coverage["program"] = normalize_program(coverage["program"])

    mapping = read_required(mapping_path, compression="infer")
    mapping["program"] = normalize_program(mapping["program"])

    donor_scores = read_required(score_path)
    donor_scores["program"] = normalize_program(
        donor_scores["program"]
    )

    donor_effects = read_required(effects_path)
    donor_effects["program"] = normalize_program(
        donor_effects["program"]
    )

    gsea = read_required(gsea_path)
    gsea["program"] = normalize_program(gsea["program"])

    overlap = read_required(overlap_path)
    overlap["program"] = normalize_program(overlap["program"])

    removed_targets = read_required(targets_removed_path)
    validation_metadata = json.loads(metadata_path.read_text())

    # Primary mouse CRT and Storey q<0.10.
    crt = read_required(args.crt_ip)
    crt["target_name"] = crt["target_name"].astype(str)
    crt["program_name"] = normalize_program(crt["program_name"])

    non_control = ~crt["target_name"].str.strip().str.lower().isin(
        CONTROL_TARGET_LABELS
    )
    crt_primary = crt.loc[non_control].copy()

    if "p_skew" not in crt_primary:
        raise KeyError("IP CRT table lacks p_skew.")
    crt_primary["p_skew"] = pd.to_numeric(
        crt_primary["p_skew"],
        errors="coerce",
    )
    if crt_primary["p_skew"].isna().any():
        raise ValueError("IP CRT p_skew contains missing/non-numeric values.")

    storey_q, storey_metadata = storey_qvalues(
        crt_primary["p_skew"].to_numpy(),
        seed=args.seed,
    )
    crt_primary["Storey_q_IP"] = storey_q
    crt_primary["IP_Storey_q_lt_threshold"] = (
        crt_primary["Storey_q_IP"] < args.storey_fdr
    )

    significant_ip = crt_primary.loc[
        crt_primary["IP_Storey_q_lt_threshold"]
    ].copy()

    focused = significant_ip.merge(
        final_summary,
        left_on="program_name",
        right_on="program",
        how="left",
        validate="many_to_one",
        suffixes=("_mouse", "_human"),
    )

    # Program-level human support does not prove that the target association
    # exists in humans. It establishes relevance of the mouse program to an
    # independent human PAH endothelial dataset.
    focused["mouse_beta_vs_human_NES_same_sign"] = (
        np.sign(pd.to_numeric(focused["beta_clr"], errors="coerce"))
        == np.sign(pd.to_numeric(focused["NES"], errors="coerce"))
    )
    focused["mouse_beta_vs_human_donor_g_same_sign"] = (
        np.sign(pd.to_numeric(focused["beta_clr"], errors="coerce"))
        == np.sign(pd.to_numeric(focused["hedges_g"], errors="coerce"))
    )

    focused = focused.sort_values(
        ["Storey_q_IP", "target_name", "program_name"]
    )

    # Tables.
    final_summary.to_csv(
        output_dir / "K30_dt2_human_validation_all_programs.tsv",
        sep="\t",
        index=False,
    )
    crt_primary.to_csv(
        output_dir / "IP_CRT_with_Storey_q.tsv",
        sep="\t",
        index=False,
    )
    focused.to_csv(
        output_dir / "IP_q10_target_program_human_support.tsv",
        sep="\t",
        index=False,
    )

    # Workbook.
    workbook_path = (
        output_dir / "K30_dt2_human_validation_review.xlsx"
    )
    if workbook_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{workbook_path} exists. Pass --overwrite."
        )

    summary_rows = [
        {
            "metric": "Model",
            "value": args.model_label,
            "interpretation": "Final 100-replicate K30, dt=2.0 model.",
        },
        {
            "metric": "Human dataset",
            "value": "GSE185479",
            "interpretation": (
                "Cultured PAECs: 3 healthy and 3 PAH donors."
            ),
        },
        {
            "metric": "Evaluable programs",
            "value": int(
                parse_bool(final_summary["evaluable"])
                .fillna(False)
                .sum()
            ),
            "interpretation": (
                "Programs with at least the configured number of "
                "mapped/expressed one-to-one orthologs."
            ),
        },
        {
            "metric": "Programs with GSEA FDR <0.25",
            "value": int(final_summary["GSEA_FDR_lt_0.25"].sum()),
            "interpretation": (
                "Common exploratory threshold for preranked GSEA."
            ),
        },
        {
            "metric": "Programs with GSEA FDR <0.05",
            "value": int(final_summary["GSEA_FDR_lt_0.05"].sum()),
            "interpretation": "Stricter human transcriptomic support.",
        },
        {
            "metric": "Programs with large, separated donor effect",
            "value": int(
                (
                    final_summary[
                        "large_donor_effect_abs_g_ge_1"
                    ]
                    & final_summary[
                        "complete_separation"
                    ].fillna(False)
                ).sum()
            ),
            "interpretation": (
                "|Hedges g|>=1 plus complete PAH/healthy donor "
                "separation; exact p-values are coarse with six donors."
            ),
        },
        {
            "metric": "IP target-program pairs at Storey q<0.10",
            "value": int(len(significant_ip)),
            "interpretation": (
                "Primary mouse target-program associations."
            ),
        },
        {
            "metric": "IP q<0.10 pairs whose program has human support",
            "value": int(
                focused["human_program_support"].fillna(False).sum()
            ),
            "interpretation": (
                "Program-level human support only; not target-specific "
                "human causal validation."
            ),
        },
        {
            "metric": "Storey pi0 in IP CRT",
            "value": storey_metadata["pi0"],
            "interpretation": (
                "Estimated null proportion across non-control IP "
                "target-program tests."
            ),
        },
    ]
    summary_table = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
    ) as writer:
        summary_table.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )
        focused.to_excel(
            writer,
            sheet_name="IP_q10_human_support",
            index=False,
        )
        final_summary.to_excel(
            writer,
            sheet_name="All_programs",
            index=False,
        )
        crt_primary.to_excel(
            writer,
            sheet_name="IP_CRT_StoreyQ",
            index=False,
        )
        donor_scores.to_excel(
            writer,
            sheet_name="Donor_scores",
            index=False,
        )
        donor_effects.to_excel(
            writer,
            sheet_name="Donor_effects",
            index=False,
        )
        gsea.to_excel(
            writer,
            sheet_name="Preranked_GSEA",
            index=False,
        )
        overlap.to_excel(
            writer,
            sheet_name="DE_overlap",
            index=False,
        )
        coverage.to_excel(
            writer,
            sheet_name="Ortholog_coverage",
            index=False,
        )
        mapping.to_excel(
            writer,
            sheet_name="Mouse_human_mapping",
            index=False,
        )
        removed_targets.to_excel(
            writer,
            sheet_name="Removed_targets",
            index=False,
        )
        pd.DataFrame(
            [
                {"field": key, "value": json.dumps(value)}
                for key, value in validation_metadata.items()
            ]
            + [
                {
                    "field": "created_utc",
                    "value": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "field": "IP_Storey_metadata",
                    "value": json.dumps(storey_metadata),
                },
            ]
        ).to_excel(
            writer,
            sheet_name="Metadata",
            index=False,
        )

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = cell.font.copy(bold=True)
            for column in sheet.columns:
                width = 10
                for cell in list(column)[:2000]:
                    if cell.value is not None:
                        width = max(
                            width,
                            min(len(str(cell.value)) + 2, 55),
                        )
                sheet.column_dimensions[
                    column[0].column_letter
                ].width = width

    # Plots: all programs.
    save_bar_plot(
        final_summary,
        value="hedges_g",
        title=(
            "Final K30/dt2.0 programs in human PAH PAEC donors"
        ),
        xlabel="Hedges g: PAH minus healthy donor program score",
        output=output_dir / "human_PAH_donor_effects_all_programs",
    )
    save_bar_plot(
        final_summary,
        value="NES",
        title=(
            "Final K30/dt2.0 programs in human PAH DE ranking"
        ),
        xlabel="Preranked GSEA NES",
        output=output_dir / "human_PAH_GSEA_all_programs",
    )

    # Focused target-program pairs.
    focused_plot = focused.copy()
    focused_plot["pair"] = (
        focused_plot["target_name"].astype(str)
        + " → P"
        + focused_plot["program_name"].astype(str)
    )

    if not focused_plot.empty:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13, max(4.5, 0.65 * len(focused_plot))),
        )
        y = np.arange(len(focused_plot))

        axes[0].barh(y, focused_plot["hedges_g"])
        axes[0].axvline(0, color="black", linewidth=1)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(focused_plot["pair"])
        axes[0].set_xlabel("Human donor-score Hedges g")
        axes[0].set_title("Program activity in PAH versus healthy donors")

        axes[1].barh(y, focused_plot["NES"])
        axes[1].axvline(0, color="black", linewidth=1)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(focused_plot["pair"])
        axes[1].set_xlabel("Human PAH GSEA NES")
        axes[1].set_title("Program enrichment in human PAH DE")

        fig.suptitle(
            "Human support for IP_GFP Storey q<0.10 target-program pairs"
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / "IP_q10_target_program_human_support.pdf"
        )
        fig.savefig(
            output_dir / "IP_q10_target_program_human_support.png",
            dpi=200,
        )
        plt.close(fig)

    # Combined review PDF.
    review_images = [
        validation_dir / "donor_program_scores.png",
        validation_dir / "human_PAH_effects.png",
        validation_dir / "donor_score_vs_GSEA.png",
        validation_dir / "ortholog_coverage.png",
        output_dir / "human_PAH_donor_effects_all_programs.png",
        output_dir / "human_PAH_GSEA_all_programs.png",
        output_dir / "IP_q10_target_program_human_support.png",
    ]
    with PdfPages(
        output_dir / "K30_dt2_human_validation_review.pdf"
    ) as pdf:
        for image_path in review_images:
            if not image_path.is_file():
                continue
            image = plt.imread(image_path)
            fig, axis = plt.subplots(figsize=(11, 8.5))
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(image_path.stem)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    notes = f"""Final K30, density-threshold 2.0 human PAH validation
========================================================

Dataset
-------
GSE185479: cultured pulmonary arterial endothelial cells from three healthy
and three PAH donors.

Circularity control
-------------------
All experimentally targeted genes were removed from the mouse program lists,
mapped human program sets, human PAH differential-expression ranking, and
overlap universe.

Primary interpretation
----------------------
This analysis validates cNMF programs, not individual target-gene causality in
human PAH. The program associated with a mouse perturbation is considered
human-relevant when it shows one or more of:

- preranked GSEA FDR < 0.25 in the human PAH differential-expression ranking;
- a large donor-score effect (absolute Hedges g >= 1) with complete separation
  of the three PAH and three healthy donors.

The donor-score exact test has only choose(6,3)=20 possible labelings. Its
smallest attainable exact two-sided p-value is coarse, so effect size, donor
separation, ortholog coverage, and GSEA should be considered jointly.

IP CRT
------
The IP target-program table was recalculated with Storey q values over all
non-control target-program tests.

Storey pi0: {storey_metadata['pi0']:.6g}
Selected lambda: {storey_metadata['selected_lambda']:.6g}
Storey threshold: {args.storey_fdr}
Significant pairs: {len(significant_ip)}

Failure to validate
-------------------
A mouse program can fail validation because GSE185479 is cultured PAEC rather
than intact lung tissue. Immune, platelet, epithelial, stromal, or in-vivo
vascular-niche programs may be absent from this dataset.
"""
    (output_dir / "HOW_TO_INTERPRET_HUMAN_VALIDATION.txt").write_text(
        notes
    )

    manifest = {
        "model": args.model_label,
        "validation_dir": str(validation_dir),
        "output_dir": str(output_dir),
        "IP_CRT": str(args.crt_ip),
        "IP_Storey": storey_metadata,
        "n_programs": int(len(final_summary)),
        "n_IP_q10_pairs": int(len(significant_ip)),
        "n_IP_q10_pairs_with_human_program_support": int(
            focused["human_program_support"].fillna(False).sum()
        ),
    }
    (
        output_dir / "K30_dt2_human_validation_manifest.json"
    ).write_text(json.dumps(manifest, indent=2) + "\n")

    required_outputs = [
        workbook_path,
        output_dir / "K30_dt2_human_validation_review.pdf",
        output_dir / "K30_dt2_human_validation_all_programs.tsv",
        output_dir / "IP_q10_target_program_human_support.tsv",
        output_dir / "HOW_TO_INTERPRET_HUMAN_VALIDATION.txt",
    ]
    missing_outputs = [
        str(path)
        for path in required_outputs
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_outputs:
        raise RuntimeError(
            "Missing summary outputs:\n"
            + "\n".join(missing_outputs)
        )

    print(summary_table.to_string(index=False))
    print("\nIP q<0.10 pairs with program-level human validation:")
    columns = [
        "target_name",
        "program_name",
        "beta_clr",
        "Storey_q_IP",
        "hedges_g",
        "complete_separation",
        "NES",
        "GSEA_FDR",
        "human_program_support",
    ]
    print(
        focused[
            [column for column in columns if column in focused]
        ].to_string(index=False)
    )
    print(f"\nWrote final validation review to: {output_dir}")


if __name__ == "__main__":
    main()
