#!/usr/bin/env python3
"""
Cross-species validation of mouse cNMF programs in human PAH PAEC data.

Subcommands
-----------
prepare-gse
    Read the six processed 10x matrices from GSE185479, apply moderate QC,
    save count/log-normalized AnnData objects, and export donor pseudobulk counts.

validate-model
    Transfer one mouse cNMF model to one-to-one human orthologs, remove all
    experimentally targeted genes, score donors, run preranked GSEA against
    human PAH differential expression, and create per-model plots/tables.

summarize
    Combine all K x density-threshold validation results without automatically
    selecting K.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import gseapy as gp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import hypergeom
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


def make_unique(values: Sequence[str]) -> pd.Index:
    """Return deterministic unique names while preserving order."""
    counts: dict[str, int] = {}
    output: list[str] = []
    for raw in values:
        value = str(raw)
        n_seen = counts.get(value, 0)
        output.append(value if n_seen == 0 else f"{value}-{n_seen}")
        counts[value] = n_seen + 1
    return pd.Index(output)


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one match for {pattern!r} under {directory}; "
            f"found {len(matches)}: {[str(x) for x in matches]}"
        )
    return matches[0]


def read_10x_triplet(
    matrix_path: Path,
    features_path: Path,
    barcodes_path: Path,
    gsm: str,
    donor: str,
    condition: str,
) -> ad.AnnData:
    """Read prefixed 10x Matrix Market files supplied by GEO."""
    with gzip.open(matrix_path, "rb") as handle:
        matrix = mmread(handle).tocsr().transpose().tocsr()

    features = pd.read_csv(
        features_path,
        sep="\t",
        header=None,
        compression="infer",
        dtype=str,
    )
    barcodes = pd.read_csv(
        barcodes_path,
        sep="\t",
        header=None,
        compression="infer",
        dtype=str,
    )[0].astype(str)

    if matrix.shape[0] != len(barcodes):
        raise ValueError(
            f"{gsm}: matrix has {matrix.shape[0]} cells but "
            f"{len(barcodes)} barcodes."
        )
    if matrix.shape[1] != len(features):
        raise ValueError(
            f"{gsm}: matrix has {matrix.shape[1]} features but "
            f"{len(features)} feature rows."
        )
    if features.shape[1] < 2:
        raise ValueError(f"{gsm}: features file has fewer than two columns.")

    gene_ids = features.iloc[:, 0].astype(str)
    symbols = features.iloc[:, 1].astype(str)
    feature_types = (
        features.iloc[:, 2].astype(str)
        if features.shape[1] >= 3
        else pd.Series("Gene Expression", index=features.index)
    )

    obs_names = pd.Index([f"{gsm}:{x}" for x in barcodes], name=None)
    var_names = make_unique(gene_ids)

    obs = pd.DataFrame(index=obs_names)
    obs["barcode"] = barcodes.to_numpy()
    obs["gsm"] = gsm
    obs["donor"] = donor
    obs["condition"] = condition
    obs["dataset"] = "GSE185479"

    var = pd.DataFrame(index=var_names)
    var["gene_id"] = gene_ids.to_numpy()
    var["symbol"] = symbols.to_numpy()
    var["feature_type"] = feature_types.to_numpy()

    return ad.AnnData(X=matrix, obs=obs, var=var)


def calculate_qc(adata: ad.AnnData) -> None:
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    adata.X = adata.X.tocsr()

    total = np.asarray(adata.X.sum(axis=1)).ravel()
    n_genes = np.asarray(adata.X.getnnz(axis=1)).ravel()

    symbols = adata.var["symbol"].astype(str).str.upper()
    mt_mask = symbols.str.startswith("MT-").to_numpy()
    if mt_mask.any():
        mt_counts = np.asarray(adata.X[:, mt_mask].sum(axis=1)).ravel()
    else:
        mt_counts = np.zeros(adata.n_obs, dtype=float)

    pct_mt = np.divide(
        mt_counts * 100.0,
        total,
        out=np.zeros_like(mt_counts, dtype=float),
        where=total > 0,
    )

    adata.obs["total_counts"] = total
    adata.obs["n_genes_by_counts"] = n_genes
    adata.obs["total_counts_mt"] = mt_counts
    adata.obs["pct_counts_mt"] = pct_mt


def normalize_log1p(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    normalized = adata.copy()
    matrix = normalized.X.tocsr().astype(np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(
        target_sum,
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0,
    )
    matrix = sp.diags(scale).dot(matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    normalized.X = matrix
    normalized.uns["log1p"] = {"base": None}
    normalized.uns["normalization"] = {
        "method": "library_size_log1p",
        "target_sum": float(target_sum),
    }
    return normalized


def prepare_gse(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    sample_metadata_path = Path(args.sample_metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_metadata = pd.read_csv(sample_metadata_path, sep="\t", dtype=str)
    required_columns = {"gsm", "donor", "condition"}
    missing = required_columns.difference(sample_metadata.columns)
    if missing:
        raise KeyError(
            f"Sample metadata is missing columns: {sorted(missing)}"
        )
    if sample_metadata["gsm"].duplicated().any():
        raise ValueError("Sample metadata contains duplicated GSM accessions.")
    if sample_metadata["donor"].duplicated().any():
        raise ValueError("Sample metadata contains duplicated donor IDs.")

    expected_conditions = {"healthy", "PAH"}
    observed_conditions = set(sample_metadata["condition"])
    if observed_conditions != expected_conditions:
        raise ValueError(
            f"Expected conditions {sorted(expected_conditions)}; "
            f"found {sorted(observed_conditions)}."
        )

    sample_objects: list[ad.AnnData] = []
    reference_gene_ids: list[str] | None = None
    reference_symbols: list[str] | None = None

    for row in sample_metadata.itertuples(index=False):
        gsm = str(row.gsm)
        matrix_path = find_one(raw_dir, f"{gsm}_*_matrix.mtx.gz")
        features_path = find_one(raw_dir, f"{gsm}_*_features.tsv.gz")
        barcodes_path = find_one(raw_dir, f"{gsm}_*_barcodes.tsv.gz")

        print(f"Reading {gsm}: {matrix_path.name}", flush=True)
        sample = read_10x_triplet(
            matrix_path=matrix_path,
            features_path=features_path,
            barcodes_path=barcodes_path,
            gsm=gsm,
            donor=str(row.donor),
            condition=str(row.condition),
        )

        gene_ids = sample.var["gene_id"].astype(str).tolist()
        symbols = sample.var["symbol"].astype(str).tolist()
        if reference_gene_ids is None:
            reference_gene_ids = gene_ids
            reference_symbols = symbols
        else:
            if gene_ids != reference_gene_ids:
                raise ValueError(
                    f"{gsm}: feature IDs/order differ from the first sample."
                )
            if symbols != reference_symbols:
                raise ValueError(
                    f"{gsm}: gene symbols/order differ from the first sample."
                )

        sample_objects.append(sample)

    if not sample_objects:
        raise RuntimeError("No GSE185479 samples were loaded.")

    matrix = sp.vstack([obj.X for obj in sample_objects], format="csr")
    obs = pd.concat([obj.obs for obj in sample_objects], axis=0)
    var = sample_objects[0].var.copy()
    adata = ad.AnnData(X=matrix, obs=obs, var=var)

    if not adata.obs_names.is_unique:
        raise ValueError("Combined cell IDs are not unique.")
    if not adata.var_names.is_unique:
        raise ValueError("Combined feature IDs are not unique.")

    calculate_qc(adata)

    qc_before = (
        adata.obs.groupby(["donor", "condition"], observed=True)
        .size()
        .rename("cells_before_qc")
        .reset_index()
    )

    keep_cells = (
        (adata.obs["n_genes_by_counts"] >= args.min_genes)
        & (adata.obs["total_counts"] >= args.min_counts)
        & (adata.obs["pct_counts_mt"] <= args.max_pct_mt)
    ).to_numpy()

    adata = adata[keep_cells].copy()
    if adata.n_obs == 0:
        raise ValueError("All cells failed the configured QC filters.")

    n_cells_per_gene = np.asarray(adata.X.getnnz(axis=0)).ravel()
    keep_genes = n_cells_per_gene >= args.min_cells_per_gene
    adata = adata[:, keep_genes].copy()
    adata.var["n_cells"] = n_cells_per_gene[keep_genes]

    calculate_qc(adata)
    adata.X = adata.X.tocsr().astype(np.float32)
    adata.obs.index.name = None
    adata.var.index.name = None

    qc_after = (
        adata.obs.groupby(["donor", "condition"], observed=True)
        .size()
        .rename("cells_after_qc")
        .reset_index()
    )
    qc_summary = qc_before.merge(
        qc_after,
        on=["donor", "condition"],
        how="left",
    ).fillna({"cells_after_qc": 0})
    qc_summary["retained_fraction"] = (
        qc_summary["cells_after_qc"] / qc_summary["cells_before_qc"]
    )

    counts_h5ad = output_dir / "GSE185479_PAEC_QC_counts.h5ad"
    log_h5ad = output_dir / "GSE185479_PAEC_log1p.h5ad"
    qc_tsv = output_dir / "GSE185479_QC_summary.tsv"
    pseudobulk_path = output_dir / "GSE185479_pseudobulk_counts.tsv.gz"
    pseudobulk_metadata_path = (
        output_dir / "GSE185479_pseudobulk_metadata.tsv"
    )

    adata.uns["source"] = {
        "accession": "GSE185479",
        "description": (
            "Cultured human pulmonary arterial endothelial cells: "
            "three healthy and three PAH donors"
        ),
    }
    adata.uns["qc"] = {
        "min_genes": int(args.min_genes),
        "min_counts": int(args.min_counts),
        "max_pct_mt": float(args.max_pct_mt),
        "min_cells_per_gene": int(args.min_cells_per_gene),
    }
    adata.write_h5ad(counts_h5ad, compression="gzip")

    normalized = normalize_log1p(adata)
    normalized.write_h5ad(log_h5ad, compression="gzip")

    # Build donor pseudobulk counts, then collapse duplicate Ensembl features
    # to uppercase human symbols.
    donors = sample_metadata["donor"].astype(str).tolist()
    donor_vectors: dict[str, np.ndarray] = {}
    for donor in donors:
        mask = (adata.obs["donor"].astype(str) == donor).to_numpy()
        if not mask.any():
            raise ValueError(f"No QC-passed cells remain for donor {donor}.")
        donor_vectors[donor] = np.asarray(
            adata.X[mask].sum(axis=0)
        ).ravel()

    symbols = (
        adata.var["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan})
    )
    pseudobulk = pd.DataFrame(donor_vectors, index=symbols)
    pseudobulk = pseudobulk.loc[pseudobulk.index.notna()]
    pseudobulk = pseudobulk.groupby(level=0, sort=True).sum()
    pseudobulk = pseudobulk.loc[pseudobulk.sum(axis=1) > 0]
    pseudobulk = np.rint(pseudobulk).astype(np.int64)
    pseudobulk.index.name = "gene"
    pseudobulk.to_csv(pseudobulk_path, sep="\t", compression="gzip")

    pb_metadata = sample_metadata[["donor", "condition", "gsm"]].copy()
    pb_metadata.to_csv(
        pseudobulk_metadata_path,
        sep="\t",
        index=False,
    )

    qc_summary.to_csv(qc_tsv, sep="\t", index=False)

    report = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "counts_h5ad": str(counts_h5ad),
        "log_h5ad": str(log_h5ad),
        "pseudobulk_counts": str(pseudobulk_path),
        "pseudobulk_metadata": str(pseudobulk_metadata_path),
        "cell_counts": {
            str(k): int(v)
            for k, v in adata.obs["donor"].value_counts().to_dict().items()
        },
    }
    (output_dir / "GSE185479_preparation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


def read_one_to_one_orthologs(path: Path) -> pd.DataFrame:
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
    )
    table = table.dropna(subset=["mouse_symbol", "human_symbol"]).copy()
    table["mouse_symbol"] = table["mouse_symbol"].str.strip()
    table["human_symbol"] = (
        table["human_symbol"].str.strip().str.upper()
    )
    table["mouse_key"] = table["mouse_symbol"].str.upper()
    table["human_key"] = table["human_symbol"].str.upper()

    # The MGI report is defined as one-to-one protein-coding orthology.
    # Still remove any accidental duplicate symbol relationships.
    mouse_degree = table.groupby("mouse_key")["human_key"].transform("nunique")
    human_degree = table.groupby("human_key")["mouse_key"].transform("nunique")
    table = table.loc[(mouse_degree == 1) & (human_degree == 1)].copy()
    table = table.drop_duplicates(["mouse_key", "human_key"])
    return table


def read_spectra(path: Path, k: int) -> pd.DataFrame:
    spectra = pd.read_csv(path, sep="\t", index_col=0)
    if spectra.shape[0] == k:
        result = spectra
    elif spectra.shape[1] == k:
        result = spectra.transpose()
    else:
        raise ValueError(
            f"{path}: expected one dimension to equal K={k}; "
            f"observed shape {spectra.shape}."
        )

    result.index = result.index.astype(str)
    result.columns = result.columns.astype(str)
    result = result.apply(pd.to_numeric, errors="coerce")
    if result.isna().all(axis=None):
        raise ValueError(f"{path}: spectra contain no numeric values.")
    return result


def read_targets(
    prepared_h5ad: Path,
    orthologs: pd.DataFrame,
) -> tuple[list[str], list[str], pd.DataFrame]:
    adata = ad.read_h5ad(prepared_h5ad, backed="r")
    try:
        if "guide_targets" not in adata.uns:
            raise KeyError("Prepared H5AD lacks uns['guide_targets'].")
        raw_targets = [str(x) for x in np.asarray(adata.uns["guide_targets"])]
    finally:
        if adata.isbacked:
            adata.file.close()

    mouse_targets = sorted(
        {
            target
            for target in raw_targets
            if target.strip().lower() not in CONTROL_TARGET_LABELS
        },
        key=str.upper,
    )
    mouse_keys = {x.upper() for x in mouse_targets}

    mapping = orthologs.loc[
        orthologs["mouse_key"].isin(mouse_keys),
        [
            "mouse_symbol",
            "mouse_key",
            "human_symbol",
            "human_key",
            "mgi_id",
            "hgnc_id",
        ],
    ].copy()
    human_targets = sorted(mapping["human_key"].unique().tolist())
    return mouse_targets, human_targets, mapping


def build_program_mapping(
    spectra: pd.DataFrame,
    orthologs: pd.DataFrame,
    excluded_mouse: set[str],
    excluded_human: set[str],
    human_genes: set[str],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    ortholog_index = orthologs.set_index("mouse_key", drop=False)

    for program in spectra.index:
        values = spectra.loc[program].dropna().sort_values(ascending=False)
        candidate_rows: list[dict[str, object]] = []
        n_scanned = 0

        for mouse_gene, score in values.items():
            mouse_gene = str(mouse_gene)
            mouse_key = mouse_gene.upper()
            if mouse_key in excluded_mouse:
                continue
            n_scanned += 1
            if n_scanned > top_n:
                break
            if mouse_key not in ortholog_index.index:
                continue

            ortholog_row = ortholog_index.loc[mouse_key]
            if isinstance(ortholog_row, pd.DataFrame):
                ortholog_row = ortholog_row.iloc[0]
            human_key = str(ortholog_row["human_key"]).upper()
            if human_key in excluded_human:
                continue

            candidate_rows.append(
                {
                    "program": str(program),
                    "mouse_gene": mouse_gene,
                    "mouse_gene_key": mouse_key,
                    "human_gene": str(ortholog_row["human_symbol"]),
                    "human_gene_key": human_key,
                    "mouse_spectrum_score": float(score),
                    "mouse_rank_after_target_removal": n_scanned,
                    "present_in_human_data": human_key in human_genes,
                }
            )

        mapped = pd.DataFrame(candidate_rows)
        if mapped.empty:
            coverage_rows.append(
                {
                    "program": str(program),
                    "top_n_mouse_non_target": int(min(top_n, len(values))),
                    "n_one_to_one_human": 0,
                    "n_present_in_human_data": 0,
                    "ortholog_coverage": 0.0,
                    "human_expression_coverage": 0.0,
                }
            )
            continue

        # In case two mouse symbols unexpectedly map to the same human symbol,
        # retain the higher-ranked mouse program gene.
        mapped = mapped.sort_values(
            [
                "present_in_human_data",
                "mouse_rank_after_target_removal",
            ],
            ascending=[False, True],
        ).drop_duplicates("human_gene_key", keep="first")

        # Rank-decay weights are less sensitive than raw cNMF scores to
        # cross-species/platform scale differences.
        ranks = mapped["mouse_rank_after_target_removal"].astype(float)
        mapped["rank_weight_raw"] = 1.0 / np.log2(ranks + 1.0)

        present = mapped["present_in_human_data"].astype(bool)
        available_weight = mapped.loc[present, "rank_weight_raw"].sum()
        mapped["rank_weight"] = 0.0
        if available_weight > 0:
            mapped.loc[present, "rank_weight"] = (
                mapped.loc[present, "rank_weight_raw"] / available_weight
            )

        n_one_to_one = int(len(mapped))
        n_present = int(present.sum())
        coverage_rows.append(
            {
                "program": str(program),
                "top_n_mouse_non_target": int(min(top_n, len(values))),
                "n_one_to_one_human": n_one_to_one,
                "n_present_in_human_data": n_present,
                "ortholog_coverage": float(n_one_to_one / top_n),
                "human_expression_coverage": float(n_present / top_n),
            }
        )
        rows.extend(mapped.to_dict(orient="records"))

    mapping = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    return mapping, coverage


def exact_permutation_pvalue(
    scores: pd.Series,
    conditions: pd.Series,
) -> tuple[float, float, int]:
    sample_names = scores.index.tolist()
    labels = conditions.loc[sample_names].astype(str)
    n_pah = int((labels == "PAH").sum())
    n_total = len(sample_names)
    if n_pah == 0 or n_pah == n_total:
        return np.nan, np.nan, 0

    values = scores.to_numpy(dtype=float)
    observed_mask = (labels.to_numpy() == "PAH")
    observed = float(
        values[observed_mask].mean() - values[~observed_mask].mean()
    )

    null_stats: list[float] = []
    for indices in itertools.combinations(range(n_total), n_pah):
        mask = np.zeros(n_total, dtype=bool)
        mask[list(indices)] = True
        stat = float(values[mask].mean() - values[~mask].mean())
        null_stats.append(stat)

    null = np.asarray(null_stats)
    p_value = float(
        np.mean(np.abs(null) >= (abs(observed) - 1e-12))
    )
    return observed, p_value, int(len(null))


def hedges_g(
    case_values: np.ndarray,
    control_values: np.ndarray,
) -> float:
    case_values = np.asarray(case_values, dtype=float)
    control_values = np.asarray(control_values, dtype=float)
    n_case = len(case_values)
    n_control = len(control_values)
    if n_case < 2 or n_control < 2:
        return np.nan

    pooled_var = (
        (n_case - 1) * np.var(case_values, ddof=1)
        + (n_control - 1) * np.var(control_values, ddof=1)
    ) / (n_case + n_control - 2)
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan

    cohen_d = (
        np.mean(case_values) - np.mean(control_values)
    ) / math.sqrt(pooled_var)
    correction = 1.0 - 3.0 / (4.0 * (n_case + n_control) - 9.0)
    return float(correction * cohen_d)


def score_programs_on_donors(
    mapping: pd.DataFrame,
    coverage: pd.DataFrame,
    logcpm: pd.DataFrame,
    metadata: pd.DataFrame,
    min_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = metadata.copy()
    metadata["donor"] = metadata["donor"].astype(str)
    metadata = metadata.set_index("donor", drop=False)

    donor_order = metadata.index.tolist()
    missing_donors = set(donor_order).difference(logcpm.columns)
    if missing_donors:
        raise KeyError(
            f"logCPM matrix lacks donors: {sorted(missing_donors)}"
        )

    logcpm = logcpm[donor_order].copy()
    gene_mean = logcpm.mean(axis=1)
    gene_sd = logcpm.std(axis=1, ddof=1)
    valid_sd = gene_sd > 0
    z = logcpm.loc[valid_sd].sub(gene_mean.loc[valid_sd], axis=0)
    z = z.div(gene_sd.loc[valid_sd], axis=0)

    score_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []

    for coverage_row in coverage.itertuples(index=False):
        program = str(coverage_row.program)
        program_map = mapping.loc[
            (mapping["program"].astype(str) == program)
            & mapping["present_in_human_data"].astype(bool)
        ].copy()
        program_map = program_map.loc[
            program_map["human_gene_key"].isin(z.index)
        ].copy()

        if len(program_map) < min_genes:
            effect_rows.append(
                {
                    "program": program,
                    "n_scored_genes": int(len(program_map)),
                    "evaluable": False,
                    "PAH_minus_healthy_score": np.nan,
                    "hedges_g": np.nan,
                    "exact_permutation_p": np.nan,
                    "n_label_permutations": 0,
                    "complete_separation": False,
                }
            )
            continue

        weights = program_map.set_index("human_gene_key")[
            "rank_weight_raw"
        ].astype(float)
        weights = weights.groupby(level=0).max()
        weights = weights / weights.sum()

        genes = [gene for gene in weights.index if gene in z.index]
        weights = weights.loc[genes]
        weights = weights / weights.sum()
        scores = z.loc[genes].transpose().dot(weights)
        scores.name = program

        for donor, score in scores.items():
            score_rows.append(
                {
                    "program": program,
                    "donor": donor,
                    "condition": str(metadata.loc[donor, "condition"]),
                    "score": float(score),
                    "n_scored_genes": int(len(genes)),
                }
            )

        conditions = metadata["condition"]
        observed, exact_p, n_permutations = exact_permutation_pvalue(
            scores=scores,
            conditions=conditions,
        )
        case = scores.loc[conditions == "PAH"].to_numpy()
        control = scores.loc[conditions == "healthy"].to_numpy()
        complete_separation = bool(
            (np.min(case) > np.max(control))
            or (np.max(case) < np.min(control))
        )

        effect_rows.append(
            {
                "program": program,
                "n_scored_genes": int(len(genes)),
                "evaluable": True,
                "mean_score_PAH": float(np.mean(case)),
                "mean_score_healthy": float(np.mean(control)),
                "PAH_minus_healthy_score": float(observed),
                "hedges_g": hedges_g(case, control),
                "exact_permutation_p": float(exact_p),
                "n_label_permutations": int(n_permutations),
                "complete_separation": complete_separation,
            }
        )

    score_table = pd.DataFrame(score_rows)
    effect_table = pd.DataFrame(effect_rows)

    valid_p = effect_table["exact_permutation_p"].notna()
    effect_table["score_BH_FDR"] = np.nan
    if valid_p.any():
        effect_table.loc[valid_p, "score_BH_FDR"] = multipletests(
            effect_table.loc[valid_p, "exact_permutation_p"],
            method="fdr_bh",
        )[1]

    return score_table, effect_table


def run_gsea(
    program_sets: dict[str, list[str]],
    de_table: pd.DataFrame,
    min_size: int,
    max_size: int,
    permutations: int,
    threads: int,
    seed: int,
) -> pd.DataFrame:
    if not program_sets:
        return pd.DataFrame(
            columns=[
                "program",
                "ES",
                "NES",
                "NOM_p_value",
                "GSEA_FDR",
                "leading_edge_genes",
            ]
        )

    de = de_table.copy()
    if "signed_sqrt_F" in de.columns:
        stat = pd.to_numeric(de["signed_sqrt_F"], errors="coerce")
    elif {"logFC", "PValue"}.issubset(de.columns):
        logfc = pd.to_numeric(de["logFC"], errors="coerce")
        pvalue = pd.to_numeric(de["PValue"], errors="coerce")
        stat = np.sign(logfc) * -np.log10(pvalue.clip(lower=1e-300))
    else:
        raise KeyError(
            "DE table requires signed_sqrt_F or both logFC and PValue."
        )

    ranking = pd.DataFrame(
        {
            "gene": de["gene"].astype(str).str.upper(),
            "score": stat,
        }
    ).dropna()
    ranking = ranking.loc[np.isfinite(ranking["score"])]
    ranking = ranking.sort_values(
        ["gene", "score"],
        ascending=[True, False],
    ).drop_duplicates("gene", keep="first")
    ranking = ranking.sort_values("score", ascending=False)

    ranking_genes = set(ranking["gene"])
    filtered_sets = {
        str(program): sorted(set(genes).intersection(ranking_genes))
        for program, genes in program_sets.items()
    }
    filtered_sets = {
        program: genes
        for program, genes in filtered_sets.items()
        if min_size <= len(genes) <= max_size
    }
    if not filtered_sets:
        return pd.DataFrame(
            columns=[
                "program",
                "ES",
                "NES",
                "NOM_p_value",
                "GSEA_FDR",
                "leading_edge_genes",
            ]
        )

    result = gp.prerank(
        rnk=ranking[["gene", "score"]],
        gene_sets=filtered_sets,
        threads=max(1, int(threads)),
        min_size=int(min_size),
        max_size=int(max_size),
        permutation_num=int(permutations),
        outdir=None,
        seed=int(seed),
        ascending=False,
        verbose=False,
    ).res2d.copy()

    rename = {
        "Term": "program",
        "ES": "ES",
        "NES": "NES",
        "NOM p-val": "NOM_p_value",
        "FDR q-val": "GSEA_FDR",
        "FWER p-val": "FWER_p_value",
        "Lead_genes": "leading_edge_genes",
        "Tag %": "tag_fraction",
        "Gene %": "gene_fraction",
    }
    result = result.rename(columns=rename)
    if "program" not in result.columns:
        if result.index.name is not None:
            result = result.reset_index().rename(
                columns={result.index.name: "program"}
            )
        else:
            result = result.reset_index().rename(
                columns={"index": "program"}
            )

    for column in [
        "ES",
        "NES",
        "NOM_p_value",
        "GSEA_FDR",
        "FWER_p_value",
    ]:
        if column in result:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def hypergeometric_overlap(
    program_sets: dict[str, list[str]],
    de_table: pd.DataFrame,
) -> pd.DataFrame:
    de = de_table.copy()
    de["gene"] = de["gene"].astype(str).str.upper()
    de = de.drop_duplicates("gene")
    background = set(de["gene"])

    if "FDR" not in de or "logFC" not in de:
        raise KeyError("DE table requires FDR and logFC columns.")
    fdr = pd.to_numeric(de["FDR"], errors="coerce")
    logfc = pd.to_numeric(de["logFC"], errors="coerce")
    up = set(de.loc[(fdr < 0.05) & (logfc > 0), "gene"])
    down = set(de.loc[(fdr < 0.05) & (logfc < 0), "gene"])

    rows: list[dict[str, object]] = []
    for program, genes_raw in program_sets.items():
        genes = set(genes_raw).intersection(background)
        for direction, disease_set in [("PAH_up", up), ("PAH_down", down)]:
            overlap = genes.intersection(disease_set)
            M = len(background)
            n = len(disease_set)
            N = len(genes)
            k = len(overlap)
            p_value = (
                float(hypergeom.sf(k - 1, M, n, N))
                if M > 0 and n > 0 and N > 0 and k > 0
                else 1.0
            )
            rows.append(
                {
                    "program": str(program),
                    "direction": direction,
                    "background_genes": M,
                    "disease_DE_genes": n,
                    "program_genes": N,
                    "overlap_genes_n": k,
                    "overlap_genes": ";".join(sorted(overlap)),
                    "hypergeometric_p": p_value,
                }
            )

    output = pd.DataFrame(rows)
    output["hypergeometric_BH_FDR"] = np.nan
    for direction, indices in output.groupby("direction").groups.items():
        output.loc[indices, "hypergeometric_BH_FDR"] = multipletests(
            output.loc[indices, "hypergeometric_p"],
            method="fdr_bh",
        )[1]
    return output


def plot_model_results(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    model_title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scores.empty:
        donor_order = (
            metadata.sort_values(["condition", "donor"])["donor"]
            .astype(str)
            .tolist()
        )
        score_matrix = scores.pivot(
            index="program",
            columns="donor",
            values="score",
        )
        score_matrix = score_matrix.reindex(columns=donor_order)
        score_matrix.to_csv(
            output_dir / "donor_program_score_matrix.tsv",
            sep="\t",
        )

        fig_height = max(5.0, 0.25 * len(score_matrix.index))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        image = ax.imshow(score_matrix.to_numpy(), aspect="auto")
        ax.set_xticks(np.arange(len(score_matrix.columns)))
        ax.set_xticklabels(score_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(score_matrix.index)))
        ax.set_yticklabels(score_matrix.index)
        ax.set_xlabel("Human donor")
        ax.set_ylabel("Mouse cNMF program")
        ax.set_title(f"{model_title}: donor-level transferred program scores")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Weighted mean gene z-score")
        fig.tight_layout()
        fig.savefig(output_dir / "donor_program_scores.pdf")
        fig.savefig(
            output_dir / "donor_program_scores.png",
            dpi=200,
        )
        plt.close(fig)

    evaluable = summary.loc[summary["evaluable"].fillna(False)].copy()
    if not evaluable.empty:
        evaluable = evaluable.sort_values("hedges_g")
        fig_height = max(5.0, 0.23 * len(evaluable))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.barh(
            np.arange(len(evaluable)),
            evaluable["hedges_g"],
        )
        ax.axvline(0.0, linewidth=1)
        ax.set_yticks(np.arange(len(evaluable)))
        ax.set_yticklabels(evaluable["program"].astype(str))
        ax.set_xlabel("Hedges g: PAH minus healthy")
        ax.set_ylabel("Program")
        ax.set_title(f"{model_title}: human PAH donor-score effects")
        fig.tight_layout()
        fig.savefig(output_dir / "human_PAH_effects.pdf")
        fig.savefig(output_dir / "human_PAH_effects.png", dpi=200)
        plt.close(fig)

    if {"NES", "hedges_g"}.issubset(summary.columns):
        scatter = summary.dropna(subset=["NES", "hedges_g"])
        if not scatter.empty:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(scatter["hedges_g"], scatter["NES"])
            for row in scatter.itertuples(index=False):
                ax.annotate(
                    str(row.program),
                    (float(row.hedges_g), float(row.NES)),
                    fontsize=7,
                )
            ax.axhline(0.0, linewidth=1)
            ax.axvline(0.0, linewidth=1)
            ax.set_xlabel("Donor-score Hedges g")
            ax.set_ylabel("Preranked GSEA NES")
            ax.set_title(
                f"{model_title}: two human PAH validation summaries"
            )
            fig.tight_layout()
            fig.savefig(
                output_dir / "donor_score_vs_GSEA.pdf"
            )
            fig.savefig(
                output_dir / "donor_score_vs_GSEA.png",
                dpi=200,
            )
            plt.close(fig)

    if not coverage.empty:
        coverage_plot = coverage.sort_values(
            "human_expression_coverage"
        )
        fig_height = max(5.0, 0.23 * len(coverage_plot))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.barh(
            np.arange(len(coverage_plot)),
            coverage_plot["human_expression_coverage"],
        )
        ax.set_yticks(np.arange(len(coverage_plot)))
        ax.set_yticklabels(coverage_plot["program"].astype(str))
        ax.set_xlim(0, 1)
        ax.set_xlabel(
            "Fraction of top mouse genes with 1:1 human ortholog in dataset"
        )
        ax.set_ylabel("Program")
        ax.set_title(f"{model_title}: cross-species mapping coverage")
        fig.tight_layout()
        fig.savefig(output_dir / "ortholog_coverage.pdf")
        fig.savefig(output_dir / "ortholog_coverage.png", dpi=200)
        plt.close(fig)


def validate_model(args: argparse.Namespace) -> None:
    spectra_path = Path(args.spectra)
    ortholog_path = Path(args.orthologs)
    prepared_h5ad = Path(args.prepared_h5ad)
    logcpm_path = Path(args.human_logcpm)
    metadata_path = Path(args.human_metadata)
    de_path = Path(args.human_de)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra = read_spectra(spectra_path, args.k)
    orthologs = read_one_to_one_orthologs(ortholog_path)
    mouse_targets, human_targets, target_mapping = read_targets(
        prepared_h5ad=prepared_h5ad,
        orthologs=orthologs,
    )

    logcpm = pd.read_csv(
        logcpm_path,
        sep="\t",
        index_col=0,
        compression="infer",
    )
    logcpm.index = (
        logcpm.index.astype(str).str.strip().str.upper()
    )
    logcpm = logcpm.groupby(level=0).mean()

    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    if set(metadata["condition"]) != {"healthy", "PAH"}:
        raise ValueError(
            "Human metadata must contain healthy and PAH conditions."
        )

    de_table = pd.read_csv(
        de_path,
        sep="\t",
        compression="infer",
    )
    if "gene" not in de_table:
        de_table = de_table.rename(
            columns={de_table.columns[0]: "gene"}
        )
    de_table["gene"] = (
        de_table["gene"].astype(str).str.strip().str.upper()
    )

    excluded_mouse = {x.upper() for x in mouse_targets}
    # Use curated MGI mappings as primary exclusions. Add symbol-identity
    # fallbacks so a targeted gene is never retained solely because an
    # ortholog record is absent or uses an unexpected alias.
    excluded_human = {x.upper() for x in human_targets}
    excluded_human.update(x.upper() for x in mouse_targets)
    human_genes = set(logcpm.index)

    target_mapping = target_mapping.copy()
    if not target_mapping.empty:
        target_mapping["mapping_method"] = "MGI_one_to_one"
    mapped_mouse_keys = set(
        target_mapping.get("mouse_key", pd.Series(dtype=str)).astype(str)
    )
    fallback_rows = []
    for target in mouse_targets:
        target_key = target.upper()
        if target_key not in mapped_mouse_keys:
            fallback_rows.append(
                {
                    "mouse_symbol": target,
                    "mouse_key": target_key,
                    "human_symbol": target_key,
                    "human_key": target_key,
                    "mgi_id": pd.NA,
                    "hgnc_id": pd.NA,
                    "mapping_method": "symbol_identity_fallback",
                }
            )
    if fallback_rows:
        target_mapping = pd.concat(
            [target_mapping, pd.DataFrame(fallback_rows)],
            ignore_index=True,
        )

    mapping, coverage = build_program_mapping(
        spectra=spectra,
        orthologs=orthologs,
        excluded_mouse=excluded_mouse,
        excluded_human=excluded_human,
        human_genes=human_genes,
        top_n=args.top_n,
    )

    target_mapping.to_csv(
        output_dir / "experimentally_targeted_genes_removed.tsv",
        sep="\t",
        index=False,
    )
    coverage.to_csv(
        output_dir / "ortholog_coverage.tsv",
        sep="\t",
        index=False,
    )
    mapping.to_csv(
        output_dir / "program_mouse_to_human_mapping.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    scores, score_effects = score_programs_on_donors(
        mapping=mapping,
        coverage=coverage,
        logcpm=logcpm,
        metadata=metadata,
        min_genes=args.min_genes,
    )
    scores.to_csv(
        output_dir / "donor_program_scores_long.tsv",
        sep="\t",
        index=False,
    )
    score_effects.to_csv(
        output_dir / "program_donor_score_effects.tsv",
        sep="\t",
        index=False,
    )

    # Remove all experimentally targeted genes from the disease ranking and
    # enrichment universe as well as from the transferred program sets.
    de_table = de_table.loc[
        ~de_table["gene"].isin(excluded_human)
    ].copy()

    program_sets: dict[str, list[str]] = {}
    if not mapping.empty:
        for program, group in mapping.loc[
            mapping["present_in_human_data"].astype(bool)
        ].groupby("program", sort=False):
            genes = sorted(
                set(group["human_gene_key"]).difference(excluded_human)
            )
            if len(genes) >= args.min_genes:
                program_sets[str(program)] = genes

    gsea = run_gsea(
        program_sets=program_sets,
        de_table=de_table,
        min_size=args.min_genes,
        max_size=args.top_n,
        permutations=args.gsea_permutations,
        threads=args.threads,
        seed=args.seed,
    )
    gsea.to_csv(
        output_dir / "program_PAH_preranked_GSEA.tsv",
        sep="\t",
        index=False,
    )

    overlap = hypergeometric_overlap(
        program_sets=program_sets,
        de_table=de_table,
    )
    overlap.to_csv(
        output_dir / "program_PAH_DE_overlap.tsv",
        sep="\t",
        index=False,
    )

    summary = coverage.merge(
        score_effects,
        on="program",
        how="left",
    )
    if not gsea.empty:
        gsea_columns = [
            column
            for column in [
                "program",
                "ES",
                "NES",
                "NOM_p_value",
                "GSEA_FDR",
                "FWER_p_value",
                "leading_edge_genes",
                "tag_fraction",
                "gene_fraction",
            ]
            if column in gsea.columns
        ]
        summary = summary.merge(
            gsea[gsea_columns],
            on="program",
            how="left",
        )
    else:
        summary["NES"] = np.nan
        summary["GSEA_FDR"] = np.nan

    up_overlap = overlap.loc[
        overlap["direction"] == "PAH_up"
    ].drop(columns="direction")
    up_overlap = up_overlap.add_prefix("PAH_up_").rename(
        columns={"PAH_up_program": "program"}
    )
    down_overlap = overlap.loc[
        overlap["direction"] == "PAH_down"
    ].drop(columns="direction")
    down_overlap = down_overlap.add_prefix("PAH_down_").rename(
        columns={"PAH_down_program": "program"}
    )
    summary = summary.merge(up_overlap, on="program", how="left")
    summary = summary.merge(down_overlap, on="program", how="left")

    summary["direction_concordant"] = (
        np.sign(pd.to_numeric(summary["hedges_g"], errors="coerce"))
        == np.sign(pd.to_numeric(summary["NES"], errors="coerce"))
    )
    summary.loc[
        summary["hedges_g"].isna() | summary["NES"].isna(),
        "direction_concordant",
    ] = np.nan
    summary["K"] = int(args.k)
    summary["density_threshold"] = float(args.threshold)
    summary["model"] = (
        f"K{args.k}_dt{str(float(args.threshold)).replace('.', '_')}"
    )

    summary.to_csv(
        output_dir / "external_validation_summary.tsv",
        sep="\t",
        index=False,
    )

    model_title = f"K={args.k}, density threshold={args.threshold:g}"
    plot_model_results(
        scores=scores,
        summary=summary,
        coverage=coverage,
        metadata=metadata,
        output_dir=output_dir,
        model_title=model_title,
    )

    metadata_json = {
        "K": int(args.k),
        "density_threshold": float(args.threshold),
        "spectra": str(spectra_path),
        "human_dataset": "GSE185479",
        "human_donors": metadata[
            ["donor", "condition", "gsm"]
        ].to_dict(orient="records"),
        "ortholog_source": str(ortholog_path),
        "ortholog_policy": "MGI one-to-one protein-coding orthologs",
        "top_n_mouse_genes": int(args.top_n),
        "min_human_genes": int(args.min_genes),
        "gsea_permutations": int(args.gsea_permutations),
        "experimentally_targeted_mouse_genes_removed": mouse_targets,
        "experimentally_targeted_human_genes_removed": sorted(excluded_human),
        "n_programs": int(len(spectra)),
        "n_evaluable_programs": int(summary["evaluable"].fillna(False).sum()),
    }
    (output_dir / "model_validation_metadata.json").write_text(
        json.dumps(metadata_json, indent=2) + "\n"
    )

    print(summary.to_string(index=False))


def threshold_tag(value: float) -> str:
    return str(float(value)).replace(".", "_")


def save_metric_plot(
    model_summary: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for threshold, group in model_summary.groupby(
        "density_threshold",
        sort=True,
    ):
        group = group.sort_values("K")
        ax.plot(
            group["K"],
            group[y],
            marker="o",
            label=f"density threshold {threshold:g}",
        )
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def summarize_models(args: argparse.Namespace) -> None:
    validation_root = Path(args.validation_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    program_tables: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []

    for k in args.k_values:
        for threshold in args.thresholds:
            tag = threshold_tag(threshold)
            model_name = f"K{k}_dt{tag}"
            model_dir = validation_root / model_name
            summary_path = model_dir / "external_validation_summary.tsv"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)

            table = pd.read_csv(summary_path, sep="\t")
            table["K"] = int(k)
            table["density_threshold"] = float(threshold)
            table["model"] = model_name
            program_tables.append(table)

            gsea_fdr = pd.to_numeric(
                table.get("GSEA_FDR"),
                errors="coerce",
            )
            nes = pd.to_numeric(table.get("NES"), errors="coerce")
            hedges = pd.to_numeric(
                table.get("hedges_g"),
                errors="coerce",
            )
            def parse_bool_series(series: pd.Series) -> pd.Series:
                if pd.api.types.is_bool_dtype(series):
                    return series.astype("boolean")
                normalized = series.astype(str).str.strip().str.lower()
                output = pd.Series(pd.NA, index=series.index, dtype="boolean")
                output.loc[normalized.isin({"true", "1", "yes"})] = True
                output.loc[normalized.isin({"false", "0", "no"})] = False
                return output

            evaluable_bool = parse_bool_series(table["evaluable"])
            complete_bool = parse_bool_series(table["complete_separation"])
            concordance = parse_bool_series(
                table["direction_concordant"]
            ).dropna()
            evaluable = table.loc[evaluable_bool.fillna(False)]

            model_rows.append(
                {
                    "K": int(k),
                    "density_threshold": float(threshold),
                    "model": model_name,
                    "n_programs": int(len(table)),
                    "n_evaluable_programs": int(len(evaluable)),
                    "median_human_expression_coverage": float(
                        pd.to_numeric(
                            table["human_expression_coverage"],
                            errors="coerce",
                        ).median()
                    ),
                    "median_abs_Hedges_g": float(
                        hedges.abs().median()
                    ),
                    "median_abs_GSEA_NES": float(
                        nes.abs().median()
                    ),
                    "n_GSEA_FDR_lt_0.05": int(
                        (gsea_fdr < 0.05).sum()
                    ),
                    "n_GSEA_FDR_lt_0.25": int(
                        (gsea_fdr < 0.25).sum()
                    ),
                    "fraction_GSEA_FDR_lt_0.25": float(
                        (gsea_fdr < 0.25).sum() / len(table)
                    ),
                    "direction_concordance_fraction": (
                        float(concordance.astype(bool).mean())
                        if len(concordance)
                        else np.nan
                    ),
                    "n_complete_donor_separation": int(
                        complete_bool.fillna(False).sum()
                    ),
                }
            )

    program_all = pd.concat(program_tables, ignore_index=True)
    model_summary = pd.DataFrame(model_rows).sort_values(
        ["K", "density_threshold"]
    )

    program_all.to_csv(
        output_dir / "external_validation_all_programs.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    model_summary.to_csv(
        output_dir / "external_validation_across_models.tsv",
        sep="\t",
        index=False,
    )

    top_programs = (
        program_all.assign(
            abs_NES=pd.to_numeric(
                program_all.get("NES"),
                errors="coerce",
            ).abs(),
            abs_Hedges_g=pd.to_numeric(
                program_all.get("hedges_g"),
                errors="coerce",
            ).abs(),
        )
        .sort_values(
            ["model", "abs_NES", "abs_Hedges_g"],
            ascending=[True, False, False],
        )
        .groupby("model", observed=True)
        .head(args.top_programs_per_model)
    )
    top_programs.to_csv(
        output_dir / "top_external_programs_per_model.tsv",
        sep="\t",
        index=False,
    )

    with pd.ExcelWriter(
        output_dir / "human_PAH_external_validation_review.xlsx",
        engine="openpyxl",
    ) as writer:
        model_summary.to_excel(
            writer,
            sheet_name="model_summary",
            index=False,
        )
        top_programs.to_excel(
            writer,
            sheet_name="top_programs",
            index=False,
        )
        selected_columns = [
            column
            for column in [
                "model",
                "program",
                "K",
                "density_threshold",
                "n_present_in_human_data",
                "human_expression_coverage",
                "hedges_g",
                "exact_permutation_p",
                "NES",
                "GSEA_FDR",
                "direction_concordant",
                "complete_separation",
                "leading_edge_genes",
            ]
            if column in program_all.columns
        ]
        program_all[selected_columns].to_excel(
            writer,
            sheet_name="all_programs",
            index=False,
        )

    metrics = [
        (
            "median_abs_GSEA_NES",
            "Median absolute GSEA NES",
            "Human PAH transcriptional enrichment",
            "median_abs_GSEA_NES_across_K",
        ),
        (
            "fraction_GSEA_FDR_lt_0.25",
            "Fraction of programs",
            "Programs with GSEA FDR < 0.25",
            "GSEA_fraction_across_K",
        ),
        (
            "direction_concordance_fraction",
            "Fraction of evaluable programs",
            "Concordance of donor score and GSEA direction",
            "direction_concordance_across_K",
        ),
        (
            "median_human_expression_coverage",
            "Median mapped fraction",
            "Human one-to-one ortholog expression coverage",
            "ortholog_coverage_across_K",
        ),
    ]

    review_plots: list[Path] = []
    for column, ylabel, title, stem in metrics:
        path = output_dir / stem
        save_metric_plot(
            model_summary=model_summary,
            y=column,
            ylabel=ylabel,
            title=title,
            output_path=path,
        )
        review_plots.append(path.with_suffix(".png"))

    # Compact model-by-metric heatmap after z-scoring each metric.
    heatmap_metrics = [
        "median_abs_GSEA_NES",
        "fraction_GSEA_FDR_lt_0.25",
        "direction_concordance_fraction",
        "median_human_expression_coverage",
        "median_abs_Hedges_g",
    ]
    heat = model_summary.set_index("model")[heatmap_metrics].copy()
    heat_z = heat.copy()
    for column in heat_z:
        values = pd.to_numeric(heat_z[column], errors="coerce")
        sd = values.std(ddof=1)
        heat_z[column] = (
            (values - values.mean()) / sd
            if pd.notna(sd) and sd > 0
            else 0.0
        )
    heat.to_csv(
        output_dir / "external_validation_model_metric_matrix.tsv",
        sep="\t",
    )

    fig_height = max(6, 0.35 * len(heat_z.index))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    image = ax.imshow(heat_z.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(heat_z.columns)))
    ax.set_xticklabels(heat_z.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat_z.index)))
    ax.set_yticklabels(heat_z.index)
    ax.set_title("Human PAH external-validation metrics across models")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Metric-specific z-score")
    fig.tight_layout()
    heat_png = output_dir / "external_validation_model_metrics.png"
    fig.savefig(output_dir / "external_validation_model_metrics.pdf")
    fig.savefig(heat_png, dpi=200)
    plt.close(fig)
    review_plots.append(heat_png)

    # Plot the strongest program-level GSEA signals across all models.
    top_plot = top_programs.dropna(subset=["NES"]).copy()
    top_plot["label"] = (
        top_plot["model"].astype(str)
        + " / P"
        + top_plot["program"].astype(str)
    )
    top_plot = top_plot.sort_values("NES")
    if not top_plot.empty:
        fig_height = max(7, 0.22 * len(top_plot))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.barh(
            np.arange(len(top_plot)),
            top_plot["NES"],
        )
        ax.axvline(0.0, linewidth=1)
        ax.set_yticks(np.arange(len(top_plot)))
        ax.set_yticklabels(top_plot["label"])
        ax.set_xlabel("Human PAH preranked GSEA NES")
        ax.set_title("Top externally validated programs per model")
        fig.tight_layout()
        top_png = output_dir / "top_external_programs.png"
        fig.savefig(output_dir / "top_external_programs.pdf")
        fig.savefig(top_png, dpi=200)
        plt.close(fig)
        review_plots.append(top_png)

    with PdfPages(
        output_dir / "human_PAH_external_validation_review.pdf"
    ) as pdf:
        for image_path in review_plots:
            if not image_path.is_file():
                continue
            image = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(image_path.stem)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    notes = """Human PAH external validation
=============================

This analysis transfers the top genes of each mouse cNMF program to curated
one-to-one human protein-coding orthologs and validates them in GSE185479,
which contains cultured pulmonary arterial endothelial cells from three
healthy and three PAH donors.

All experimentally targeted genes are removed from:
- mouse program gene lists;
- mapped human program gene lists;
- the human PAH differential-expression ranking and overlap universe.

Use these results as one component of K selection, not as an automatic rule.
Consider cNMF stability/error, mouse replication, biological interpretation,
and human validation jointly.

The donor-score exact permutation test has only choose(6,3)=20 labelings.
Its smallest attainable two-sided p-value is therefore limited. Effect size,
direction, donor separation, and GSEA are more informative than a nominal
donor-score p-value in this cohort.

GSE185479 uses cultured PAECs. A program that fails validation may be absent
because it is specific to another mouse cell lineage, an in-vivo niche, or a
state lost in culture. Lack of validation is not proof that the program is
irrelevant to PAH.
"""
    (output_dir / "HOW_TO_INTERPRET.txt").write_text(notes)

    required = [
        output_dir / "external_validation_across_models.tsv",
        output_dir / "human_PAH_external_validation_review.xlsx",
        output_dir / "human_PAH_external_validation_review.pdf",
        output_dir / "top_external_programs_per_model.tsv",
        output_dir / "HOW_TO_INTERPRET.txt",
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "Summary outputs are missing:\n" + "\n".join(missing)
        )

    print(model_summary.to_string(index=False))
    print(f"\nWrote external-validation review to {output_dir}")


def parse_float_list(values: Iterable[str]) -> list[float]:
    return [float(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mouse-to-human PAH cNMF external validation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-gse",
        help="Prepare GSE185479 and export donor pseudobulk counts.",
    )
    prepare.add_argument("--raw-dir", required=True)
    prepare.add_argument("--sample-metadata", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--min-genes", type=int, default=200)
    prepare.add_argument("--min-counts", type=int, default=500)
    prepare.add_argument("--max-pct-mt", type=float, default=20.0)
    prepare.add_argument("--min-cells-per-gene", type=int, default=3)
    prepare.set_defaults(func=prepare_gse)

    validate = subparsers.add_parser(
        "validate-model",
        help="Validate one K x density-threshold cNMF model.",
    )
    validate.add_argument("--spectra", required=True)
    validate.add_argument("--k", type=int, required=True)
    validate.add_argument("--threshold", type=float, required=True)
    validate.add_argument("--orthologs", required=True)
    validate.add_argument("--prepared-h5ad", required=True)
    validate.add_argument("--human-logcpm", required=True)
    validate.add_argument("--human-metadata", required=True)
    validate.add_argument("--human-de", required=True)
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--top-n", type=int, default=300)
    validate.add_argument("--min-genes", type=int, default=15)
    validate.add_argument(
        "--gsea-permutations",
        type=int,
        default=5000,
    )
    validate.add_argument("--threads", type=int, default=4)
    validate.add_argument("--seed", type=int, default=14)
    validate.set_defaults(func=validate_model)

    summarize = subparsers.add_parser(
        "summarize",
        help="Summarize external validation across all models.",
    )
    summarize.add_argument("--validation-root", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        required=True,
    )
    summarize.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        required=True,
    )
    summarize.add_argument(
        "--top-programs-per-model",
        type=int,
        default=5,
    )
    summarize.set_defaults(func=summarize_models)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
