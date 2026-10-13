#!/usr/bin/env python3
"""BMPR2–SOX17–program–PAH directional triangulation."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import re

import anndata as ad
import gseapy as gp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
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

CONTRASTS = OrderedDict(
    {
        "GSE70456_BMPR2_siRNA_baseline": {
            "dataset": "GSE70456",
            "contrast": "BMPR2_siRNA_baseline",
            "expression": "GSE70456_log2_expression.tsv.gz",
            "metadata": "GSE70456_metadata.tsv",
            "de_suffix": "_limma.tsv.gz",
        },
        "GSE70456_BMPR2_siRNA_PMA": {
            "dataset": "GSE70456",
            "contrast": "BMPR2_siRNA_PMA",
            "expression": "GSE70456_log2_expression.tsv.gz",
            "metadata": "GSE70456_metadata.tsv",
            "de_suffix": "_limma.tsv.gz",
        },
        "GSE157231_BMPR2_siRNA_normoxia": {
            "dataset": "GSE157231",
            "contrast": "BMPR2_siRNA_normoxia",
            "expression": "GSE157231_logCPM.tsv.gz",
            "metadata": "GSE157231_metadata.tsv",
            "de_suffix": "_edgeR.tsv.gz",
        },
        "GSE157231_BMPR2_siRNA_6h_hypoxia": {
            "dataset": "GSE157231",
            "contrast": "BMPR2_siRNA_6h_hypoxia",
            "expression": "GSE157231_logCPM.tsv.gz",
            "metadata": "GSE157231_metadata.tsv",
            "de_suffix": "_edgeR.tsv.gz",
        },
        "GSE157231_BMPR2_siRNA_24h_hypoxia": {
            "dataset": "GSE157231",
            "contrast": "BMPR2_siRNA_24h_hypoxia",
            "expression": "GSE157231_logCPM.tsv.gz",
            "metadata": "GSE157231_metadata.tsv",
            "de_suffix": "_edgeR.tsv.gz",
        },
    }
)


def read_spectra(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    if frame.shape[0] == k:
        result = frame
    elif frame.shape[1] == k:
        result = frame.transpose()
    else:
        raise ValueError(
            f"Neither spectra dimension equals K={k}: {frame.shape}"
        )
    result.index = result.index.astype(str).str.replace(r"\.0$", "", regex=True)
    result.columns = result.columns.astype(str)
    return result.apply(pd.to_numeric, errors="coerce")


def read_orthologs(path: Path) -> pd.DataFrame:
    columns = [
        "mgi_id",
        "mouse_symbol",
        "mouse_entrez",
        "hgnc_id",
        "human_symbol",
        "human_entrez",
    ]
    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        dtype=str,
        comment="#",
    ).dropna(subset=["mouse_symbol", "human_symbol"])
    table["mouse_key"] = table["mouse_symbol"].str.strip().str.upper()
    table["human_key"] = table["human_symbol"].str.strip().str.upper()
    mouse_degree = table.groupby("mouse_key")["human_key"].transform("nunique")
    human_degree = table.groupby("human_key")["mouse_key"].transform("nunique")
    return table.loc[
        (mouse_degree == 1) & (human_degree == 1)
    ].drop_duplicates(["mouse_key", "human_key"])


def targeted_genes(
    prepared_h5ad: Path,
    orthologs: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    adata = ad.read_h5ad(prepared_h5ad, backed="r")
    try:
        targets = [str(value) for value in np.asarray(adata.uns["guide_targets"])]
    finally:
        adata.file.close()
    mouse = {
        target.strip().upper()
        for target in targets
        if target.strip().lower() not in CONTROL_TARGET_LABELS
    }
    human = set(
        orthologs.loc[
            orthologs["mouse_key"].isin(mouse),
            "human_key",
        ]
    )
    human.update(mouse)
    return mouse, human


def build_mapping(
    spectra: pd.DataFrame,
    orthologs: pd.DataFrame,
    excluded_mouse: set[str],
    excluded_human: set[str],
    top_n: int,
) -> pd.DataFrame:
    ortholog_index = orthologs.set_index("mouse_key", drop=False)
    rows = []

    for program in spectra.index:
        ranked = spectra.loc[program].dropna().sort_values(ascending=False)
        non_target_rank = 0
        program_rows = []

        for mouse_gene, loading in ranked.items():
            mouse_gene = str(mouse_gene)
            mouse_key = mouse_gene.upper()
            if mouse_key in excluded_mouse:
                continue

            non_target_rank += 1
            if non_target_rank > top_n:
                break
            if mouse_key not in ortholog_index.index:
                continue

            row = ortholog_index.loc[mouse_key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            human_key = str(row["human_key"]).upper()
            if human_key in excluded_human:
                continue

            program_rows.append(
                {
                    "program": str(program),
                    "mouse_gene": mouse_gene,
                    "mouse_rank": non_target_rank,
                    "mouse_loading": float(loading),
                    "human_gene": str(row["human_symbol"]),
                    "human_gene_key": human_key,
                    "rank_weight_raw": 1.0 / math.log2(non_target_rank + 1.0),
                }
            )

        if program_rows:
            current = pd.DataFrame(program_rows).drop_duplicates(
                "human_gene_key",
                keep="first",
            )
            rows.extend(current.to_dict(orient="records"))

    return pd.DataFrame(rows)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product([-1.0, 1.0], repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def paired_effect(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    mapping: pd.DataFrame,
    contrast: str,
    min_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = metadata.loc[metadata["contrast"] == contrast].copy()
    sample_column = (
        "gsm"
        if "gsm" in metadata.columns
        else "biological_sample"
    )
    metadata["treated"] = pd.to_numeric(metadata["treated"]).astype(int)
    metadata = metadata.sort_values(["pair_id", "treated"])

    missing = set(metadata[sample_column]).difference(expression.columns)
    if missing:
        raise KeyError(f"{contrast}: missing expression samples {sorted(missing)}")

    expression = expression[metadata[sample_column]].copy()
    expression.index = expression.index.astype(str).str.upper()
    mean = expression.mean(axis=1)
    sd = expression.std(axis=1, ddof=1)
    valid = sd > 0
    z = expression.loc[valid].sub(mean.loc[valid], axis=0).div(
        sd.loc[valid],
        axis=0,
    )

    score_rows = []
    effect_rows = []

    for program, group in mapping.groupby("program", sort=True):
        group = group.loc[group["human_gene_key"].isin(z.index)].copy()
        if len(group) < min_genes:
            effect_rows.append(
                {
                    "contrast": contrast,
                    "program": str(program),
                    "n_genes": int(len(group)),
                    "evaluable": False,
                }
            )
            continue

        weights = group.set_index("human_gene_key")["rank_weight_raw"]
        weights = weights.groupby(level=0).max()
        weights = weights / weights.sum()

        scores = z.loc[weights.index].transpose().dot(weights)
        sample_to_score = scores.to_dict()

        current = metadata[
            [sample_column, "pair_id", "treated"]
        ].copy()
        current["program"] = str(program)
        current["score"] = current[sample_column].map(sample_to_score)
        score_rows.extend(current.to_dict(orient="records"))

        pivot = current.pivot(
            index="pair_id",
            columns="treated",
            values="score",
        )
        differences = (pivot[1] - pivot[0]).to_numpy(dtype=float)

        effect_rows.append(
            {
                "contrast": contrast,
                "program": str(program),
                "n_genes": int(len(weights)),
                "evaluable": True,
                "mean_paired_difference": float(differences.mean()),
                "exact_sign_flip_p": exact_sign_flip_p(differences),
                "n_pairs": int(len(differences)),
                "all_pairs_same_direction": bool(
                    np.all(differences > 0)
                    or np.all(differences < 0)
                ),
            }
        )

    effects = pd.DataFrame(effect_rows)
    valid_p = effects["exact_sign_flip_p"].notna()
    effects["BH_FDR_within_contrast"] = np.nan
    if valid_p.any():
        effects.loc[valid_p, "BH_FDR_within_contrast"] = multipletests(
            effects.loc[valid_p, "exact_sign_flip_p"],
            method="fdr_bh",
        )[1]
    return pd.DataFrame(score_rows), effects


def run_gsea(
    de: pd.DataFrame,
    mapping: pd.DataFrame,
    min_genes: int,
    permutations: int,
    threads: int,
    seed: int,
) -> pd.DataFrame:
    de["gene"] = de["gene"].astype(str).str.upper()
    statistic_column = (
        "signed_stat"
        if "signed_stat" in de.columns
        else "signed_sqrt_F"
    )
    de[statistic_column] = pd.to_numeric(
        de[statistic_column],
        errors="coerce",
    )
    ranking = (
        de[["gene", statistic_column]]
        .dropna()
        .sort_values(["gene", statistic_column], ascending=[True, False])
        .drop_duplicates("gene")
        .sort_values(statistic_column, ascending=False)
    )
    ranking_genes = set(ranking["gene"])

    gene_sets = {}
    for program, group in mapping.groupby("program"):
        genes = sorted(
            set(group["human_gene_key"]).intersection(ranking_genes)
        )
        if len(genes) >= min_genes:
            gene_sets[str(program)] = genes

    result = gp.prerank(
        rnk=ranking,
        gene_sets=gene_sets,
        min_size=min_genes,
        max_size=300,
        permutation_num=permutations,
        threads=threads,
        seed=seed,
        outdir=None,
        ascending=False,
        verbose=False,
    ).res2d.reset_index(drop=True)

    result = result.rename(
        columns={
            "Term": "program",
            "NOM p-val": "NOM_p_value",
            "FDR q-val": "GSEA_FDR",
            "Lead_genes": "leading_edge_genes",
        }
    )
    result["program"] = (
        result["program"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    for column in ["NES", "NOM_p_value", "GSEA_FDR"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def read_mouse_target(path: Path, target: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame["target_name"] = frame["target_name"].astype(str)
    frame["program_name"] = (
        frame["program_name"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    subset = frame.loc[
        frame["target_name"].str.upper() == target.upper()
    ].copy()
    if subset.empty:
        raise ValueError(f"No {target} rows found in {path}.")
    return subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--mgi-report", type=Path, required=True)
    parser.add_argument("--prepared-h5ad", type=Path, required=True)
    parser.add_argument("--gse70456-dir", type=Path, required=True)
    parser.add_argument("--gse157231-dir", type=Path, required=True)
    parser.add_argument("--mouse-crt-ip", type=Path, required=True)
    parser.add_argument("--sox17-results", type=Path, required=True)
    parser.add_argument("--human-pah-programs", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--min-genes", type=int, default=15)
    parser.add_argument("--gsea-permutations", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra = read_spectra(args.spectra, args.k)
    orthologs = read_orthologs(args.mgi_report)
    excluded_mouse, excluded_human = targeted_genes(
        args.prepared_h5ad,
        orthologs,
    )
    mapping = build_mapping(
        spectra,
        orthologs,
        excluded_mouse,
        excluded_human,
        args.top_n,
    )
    mapping.to_csv(
        output_dir / "program_mouse_to_human_mapping.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    all_scores = []
    all_effects = []
    all_gsea = []

    dataset_dirs = {
        "GSE70456": args.gse70456_dir,
        "GSE157231": args.gse157231_dir,
    }

    for contrast_label, specification in CONTRASTS.items():
        directory = dataset_dirs[specification["dataset"]]
        expression = pd.read_csv(
            directory / specification["expression"],
            sep="\t",
            index_col=0,
            compression="infer",
        )
        metadata = pd.read_csv(
            directory / specification["metadata"],
            sep="\t",
            dtype=str,
        )
        contrast = specification["contrast"]

        scores, effects = paired_effect(
            expression,
            metadata,
            mapping,
            contrast,
            args.min_genes,
        )
        scores["contrast_label"] = contrast_label
        effects["contrast_label"] = contrast_label
        all_scores.append(scores)
        all_effects.append(effects)

        de_path = directory / f"{contrast}{specification['de_suffix']}"
        de = pd.read_csv(de_path, sep="\t", compression="infer")
        gsea = run_gsea(
            de,
            mapping,
            args.min_genes,
            args.gsea_permutations,
            args.threads,
            args.seed,
        )
        gsea["contrast"] = contrast
        gsea["contrast_label"] = contrast_label
        all_gsea.append(gsea)

    scores = pd.concat(all_scores, ignore_index=True)
    effects = pd.concat(all_effects, ignore_index=True)
    gsea = pd.concat(all_gsea, ignore_index=True)

    scores.to_csv(
        output_dir / "BMPR2_program_scores_long.tsv",
        sep="\t",
        index=False,
    )
    effects.to_csv(
        output_dir / "BMPR2_program_score_effects.tsv",
        sep="\t",
        index=False,
    )
    gsea.to_csv(
        output_dir / "BMPR2_program_GSEA.tsv",
        sep="\t",
        index=False,
    )

    mouse = read_mouse_target(args.mouse_crt_ip, "BMPR2")
    mouse = mouse.rename(
        columns={
            "program_name": "program",
            "beta_clr": "mouse_IP_Bmpr2_beta_clr",
            "p_skew": "mouse_IP_Bmpr2_p_skew",
            "primary_q_BH": "mouse_IP_Bmpr2_primary_q_BH",
        }
    )

    sox17 = pd.read_csv(args.sox17_results, sep="\t")
    sox17["program"] = (
        sox17["program"].astype(str).str.replace(r"\.0$", "", regex=True)
    )

    pah = pd.read_csv(args.human_pah_programs, sep="\t")
    pah["program"] = (
        pah["program"].astype(str).str.replace(r"\.0$", "", regex=True)
    )

    program_table = pd.DataFrame(
        {"program": [str(value) for value in spectra.index]}
    ).merge(
        mouse[
            [
                "program",
                "mouse_IP_Bmpr2_beta_clr",
                "mouse_IP_Bmpr2_p_skew",
                "mouse_IP_Bmpr2_primary_q_BH",
            ]
        ],
        on="program",
        how="left",
    )

    effect_wide = effects.pivot(
        index="program",
        columns="contrast_label",
        values="mean_paired_difference",
    )
    effect_wide.columns = [
        f"BMPR2_score_effect__{column}"
        for column in effect_wide.columns
    ]
    program_table = program_table.merge(
        effect_wide.reset_index(),
        on="program",
        how="left",
    )

    gsea_wide = gsea.pivot(
        index="program",
        columns="contrast_label",
        values="NES",
    )
    gsea_wide.columns = [
        f"BMPR2_GSEA_NES__{column}"
        for column in gsea_wide.columns
    ]
    program_table = program_table.merge(
        gsea_wide.reset_index(),
        on="program",
        how="left",
    )

    sox17_columns = [
        column
        for column in [
            "program",
            "mouse_IP_Sox17_beta_clr",
            "median_human_SOX17_loss_score_effect",
            "median_human_SOX17_loss_GSEA_NES",
            "human_PAH_NES",
            "human_PAH_GSEA_FDR",
            "full_directional_chain",
            "directional_chain_score",
        ]
        if column in sox17.columns
    ]
    program_table = program_table.merge(
        sox17[sox17_columns],
        on="program",
        how="left",
    )

    pah_columns = [
        column
        for column in [
            "program",
            "hedges_g",
            "NES",
            "GSEA_FDR",
            "complete_separation",
            "leading_edge_genes",
        ]
        if column in pah.columns
    ]
    pah_subset = pah[pah_columns].rename(
        columns={
            column: f"human_PAH_{column}"
            for column in pah_columns
            if column != "program"
        }
    )
    # Avoid duplicate columns already supplied by SOX17 results.
    duplicate_columns = [
        column
        for column in pah_subset.columns
        if column != "program" and column in program_table.columns
    ]
    pah_subset = pah_subset.drop(columns=duplicate_columns)
    program_table = program_table.merge(
        pah_subset,
        on="program",
        how="left",
    )

    score_columns = [
        column
        for column in program_table.columns
        if column.startswith("BMPR2_score_effect__")
    ]
    gsea_columns = [
        column
        for column in program_table.columns
        if column.startswith("BMPR2_GSEA_NES__")
    ]

    program_table["median_human_BMPR2_loss_score_effect"] = (
        program_table[score_columns].median(axis=1)
    )
    program_table["median_human_BMPR2_loss_GSEA_NES"] = (
        program_table[gsea_columns].median(axis=1)
    )

    for column in score_columns:
        program_table[f"mouse_concordant__{column}"] = (
            np.sign(program_table[column])
            == np.sign(program_table["mouse_IP_Bmpr2_beta_clr"])
        )
    for column in gsea_columns:
        program_table[f"mouse_concordant__{column}"] = (
            np.sign(program_table[column])
            == np.sign(program_table["mouse_IP_Bmpr2_beta_clr"])
        )

    program_table["n_BMPR2_score_contrasts_mouse_concordant"] = (
        program_table[
            [
                f"mouse_concordant__{column}"
                for column in score_columns
            ]
        ].sum(axis=1)
    )
    program_table["n_BMPR2_GSEA_contrasts_mouse_concordant"] = (
        program_table[
            [
                f"mouse_concordant__{column}"
                for column in gsea_columns
            ]
        ].sum(axis=1)
    )

    program_table["BMPR2_loss_matches_SOX17_loss"] = (
        np.sign(
            program_table["median_human_BMPR2_loss_score_effect"]
        )
        == np.sign(
            program_table["median_human_SOX17_loss_score_effect"]
        )
    )
    program_table["BMPR2_loss_matches_human_PAH"] = (
        np.sign(
            program_table["median_human_BMPR2_loss_score_effect"]
        )
        == np.sign(program_table["human_PAH_NES"])
    )
    program_table["mouse_BMPR2_matches_human_PAH"] = (
        np.sign(program_table["mouse_IP_Bmpr2_beta_clr"])
        == np.sign(program_table["human_PAH_NES"])
    )

    program_table["BMPR2_SOX17_PAH_axis"] = (
        (program_table["n_BMPR2_score_contrasts_mouse_concordant"] >= 3)
        & (program_table["n_BMPR2_GSEA_contrasts_mouse_concordant"] >= 3)
        & program_table["BMPR2_loss_matches_SOX17_loss"].fillna(False)
        & program_table["BMPR2_loss_matches_human_PAH"].fillna(False)
    )
    program_table["axis_score"] = (
        program_table["n_BMPR2_score_contrasts_mouse_concordant"]
        + program_table["n_BMPR2_GSEA_contrasts_mouse_concordant"]
        + program_table["BMPR2_loss_matches_SOX17_loss"].fillna(False).astype(int)
        + program_table["BMPR2_loss_matches_human_PAH"].fillna(False).astype(int)
        + program_table["mouse_BMPR2_matches_human_PAH"].fillna(False).astype(int)
    )

    program_table = program_table.sort_values(
        [
            "BMPR2_SOX17_PAH_axis",
            "axis_score",
            "human_PAH_GSEA_FDR",
        ],
        ascending=[False, False, True],
    )
    program_table.to_csv(
        output_dir / "BMPR2_SOX17_PAH_directional_axis.tsv",
        sep="\t",
        index=False,
    )

    with pd.ExcelWriter(
        output_dir / "BMPR2_SOX17_PAH_axis_review.xlsx",
        engine="openpyxl",
    ) as writer:
        program_table.to_excel(
            writer,
            sheet_name="Directional_axis",
            index=False,
        )
        effects.to_excel(
            writer,
            sheet_name="BMPR2_score_effects",
            index=False,
        )
        gsea.to_excel(
            writer,
            sheet_name="BMPR2_GSEA",
            index=False,
        )
        scores.to_excel(
            writer,
            sheet_name="BMPR2_sample_scores",
            index=False,
        )
        mapping.to_excel(
            writer,
            sheet_name="Program_mapping",
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

    def make_heatmap(
        table: pd.DataFrame,
        value: str,
        stem: str,
        title: str,
    ) -> Path:
        matrix = table.pivot(
            index="program",
            columns="contrast_label",
            values=value,
        )
        matrix = matrix.reindex(
            index=program_table["program"].astype(str)
        )
        fig, axis = plt.subplots(
            figsize=(11, max(8, 0.28 * len(matrix)))
        )
        image = axis.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
        )
        axis.set_xticks(np.arange(len(matrix.columns)))
        axis.set_xticklabels(
            matrix.columns,
            rotation=45,
            ha="right",
        )
        axis.set_yticks(np.arange(len(matrix.index)))
        axis.set_yticklabels("P" + matrix.index.astype(str))
        axis.set_title(title)
        colorbar = fig.colorbar(image, ax=axis)
        colorbar.set_label(value)
        fig.tight_layout()
        path = output_dir / stem
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=200)
        plt.close(fig)
        matrix.to_csv(path.with_suffix(".tsv"), sep="\t")
        return path.with_suffix(".png")

    figures = [
        make_heatmap(
            effects,
            "mean_paired_difference",
            "BMPR2_program_score_effects",
            "Human BMPR2-loss effects on transferred K30 programs",
        ),
        make_heatmap(
            gsea,
            "NES",
            "BMPR2_program_GSEA",
            "Human BMPR2-loss GSEA of transferred K30 programs",
        ),
    ]

    scatter = program_table.dropna(
        subset=[
            "median_human_BMPR2_loss_score_effect",
            "median_human_SOX17_loss_score_effect",
        ]
    )
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        scatter["median_human_BMPR2_loss_score_effect"],
        scatter["median_human_SOX17_loss_score_effect"],
    )
    axis.axhline(0, linewidth=1)
    axis.axvline(0, linewidth=1)
    for row in scatter.itertuples(index=False):
        axis.annotate(
            f"P{row.program}",
            (
                row.median_human_BMPR2_loss_score_effect,
                row.median_human_SOX17_loss_score_effect,
            ),
            fontsize=7,
        )
    axis.set_xlabel("Median human BMPR2-loss program effect")
    axis.set_ylabel("Median human SOX17-loss program effect")
    axis.set_title("BMPR2–SOX17 program-axis concordance")
    fig.tight_layout()
    path = output_dir / "BMPR2_vs_SOX17_program_effects"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    figures.append(path.with_suffix(".png"))

    with PdfPages(
        output_dir / "BMPR2_SOX17_PAH_axis_review.pdf"
    ) as pdf:
        for image_path in figures:
            image = plt.imread(image_path)
            fig, axis = plt.subplots(figsize=(11, 8.5))
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(image_path.stem)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    notes = """BMPR2–SOX17–program–PAH directional triangulation
================================================

Primary human BMPR2 datasets
----------------------------
GSE70456:
    primary HPAEC siBMPR2 versus control, baseline and PMA-stimulated,
    four paired biological replicates.

GSE157231:
    HPAEC siBMPR2 versus control under normoxia, 6 h hypoxia, and
    24 h hypoxia. Technical replicates are summed before paired analysis;
    the two patients are the independent units.

Directional interpretation
--------------------------
A high-priority axis program should:

1. move in the same direction after mouse Bmpr2 perturbation and at least
   three independent human BMPR2-loss contrasts;
2. show concordant program-score and GSEA directions;
3. move similarly after human SOX17 loss, consistent with the published
   BMPR2–SOX17 signaling axis;
4. move in the same direction in independent human PAH PAECs.

All experimentally targeted genes are removed from transferred program sets.
This prevents BMPR2, SOX17, and the other target genes from directly driving
program validation.

Limitations
-----------
- GSE70456 is microarray-based and uses one HPAEC donor across experiments.
- GSE157231 has two biological donors; technical replicates are not treated as
  independent.
- Program scores and GSEA should be interpreted jointly.
- The axis flag summarizes directional consistency; it is not a formal
  mediation analysis.
"""
    (output_dir / "HOW_TO_INTERPRET.txt").write_text(notes)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "K": args.k,
        "n_axis_programs": int(
            program_table["BMPR2_SOX17_PAH_axis"].sum()
        ),
        "outputs": {
            "workbook": str(
                output_dir / "BMPR2_SOX17_PAH_axis_review.xlsx"
            ),
            "pdf": str(
                output_dir / "BMPR2_SOX17_PAH_axis_review.pdf"
            ),
            "table": str(
                output_dir / "BMPR2_SOX17_PAH_directional_axis.tsv"
            ),
        },
    }
    (output_dir / "ANALYSIS_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    required = [
        output_dir / "BMPR2_SOX17_PAH_axis_review.xlsx",
        output_dir / "BMPR2_SOX17_PAH_axis_review.pdf",
        output_dir / "BMPR2_SOX17_PAH_directional_axis.tsv",
        output_dir / "HOW_TO_INTERPRET.txt",
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "Missing BMPR2 axis outputs:\n" + "\n".join(missing)
        )

    print(program_table.head(20).to_string(index=False))
    print(f"\nWrote BMPR2–SOX17 review to {output_dir}")


if __name__ == "__main__":
    main()
