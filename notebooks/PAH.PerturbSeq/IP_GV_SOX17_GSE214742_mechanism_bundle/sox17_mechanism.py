#!/usr/bin/env python3
"""Directional SOX17 variant-to-program-to-PAH analysis.

Subcommands
-----------
prepare
    Parse the 42 GSE214742 Salmon gene-level CSV files into count and TPM
    matrices using the supplied metadata and GENCODE v28 annotation.

analyze
    Transfer the final mouse K30 programs to one-to-one human orthologs,
    quantify program changes in human SOX17 siRNA/CRISPR perturbations,
    compare them with mouse Sox17 CRT effects and independent human PAH
    program activity, and generate a directional-chain review.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import gzip
import itertools
import json
import math
from pathlib import Path
import re
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

LOSS_CONTRASTS = [
    "siSOX17_vs_negativeSIRNA",
    "Promoter_Repression_vs_BFP_Repression",
    "Signal1_Repression_vs_BFP_Repression",
]
ACTIVATION_CONTRAST = "Promoter_Activation_vs_BFP_Activation"
TECHNICAL_CONTROL_CONTRAST = "siGAPDH_vs_negativeSIRNA"

CONTRASTS = OrderedDict(
    {
        "siSOX17_vs_negativeSIRNA": {
            "treatment": "siSOX17",
            "control": "negativeSIRNA",
            "batch": None,
            "loss_alignment_factor": 1.0,
            "primary": True,
        },
        "Promoter_Repression_vs_BFP_Repression": {
            "treatment": "Promoter_Repression",
            "control": "BFP_control_Repression",
            "batch": None,
            "loss_alignment_factor": 1.0,
            "primary": True,
        },
        "Signal1_Repression_vs_BFP_Repression": {
            "treatment": "Signal1_Repression",
            "control": "BFP_control_Repression",
            "batch": "3",
            "loss_alignment_factor": 1.0,
            "primary": True,
        },
        "Promoter_Activation_vs_BFP_Activation": {
            "treatment": "Promoter_Activation",
            "control": "BFP_control_Activation",
            "batch": None,
            "loss_alignment_factor": -1.0,
            "primary": True,
        },
        "siGAPDH_vs_negativeSIRNA": {
            "treatment": "siGAPDH",
            "control": "negativeSIRNA",
            "batch": None,
            "loss_alignment_factor": np.nan,
            "primary": False,
        },
    }
)


def normalized_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def choose_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    normalized = {
        normalized_column_name(column): str(column)
        for column in columns
    }
    for candidate in candidates:
        key = normalized_column_name(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(
            f"None of {list(candidates)} occurs in columns {list(columns)}."
        )
    return None


def parse_gencode_gtf(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attributes = fields[8]
            gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
            gene_name_match = re.search(r'gene_name "([^"]+)"', attributes)
            if gene_id_match and gene_name_match:
                gene_id = gene_id_match.group(1).split(".")[0].upper()
                gene_name = gene_name_match.group(1).upper()
                mapping[gene_id] = gene_name
    if len(mapping) < 20_000:
        raise RuntimeError(
            f"GENCODE mapping contains only {len(mapping):,} genes."
        )
    return mapping


def find_sample_file(raw_dir: Path, gsm: str) -> Path:
    matches = sorted(raw_dir.rglob(f"{gsm}_*.csv.gz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {gsm}_*.csv.gz under {raw_dir}; found {matches}"
        )
    return matches[0]


def infer_gene_column(frame: pd.DataFrame) -> str:
    candidates = [
        "gene",
        "gene_name",
        "gene_symbol",
        "symbol",
        "external_gene_name",
        "gene_id",
        "target_id",
        "name",
        "unnamed_0",
        "x",
    ]
    column = choose_column(frame.columns, candidates, required=False)
    if column is not None:
        return column

    for candidate in frame.columns:
        values = frame[candidate].astype(str)
        unique_fraction = values.nunique(dropna=True) / max(len(values), 1)
        gene_like = values.str.match(
            r"^(ENSG\d+|[A-Za-z0-9_.-]+)$",
            na=False,
        ).mean()
        if unique_fraction > 0.8 and gene_like > 0.8:
            return str(candidate)

    raise KeyError(
        f"Unable to infer a gene column from {frame.columns.tolist()}."
    )


def infer_numeric_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    exclude: set[str],
) -> str | None:
    column = choose_column(frame.columns, candidates, required=False)
    if column is not None and column not in exclude:
        return column

    numeric_candidates = []
    for candidate in frame.columns:
        if candidate in exclude:
            continue
        values = pd.to_numeric(frame[candidate], errors="coerce")
        valid_fraction = values.notna().mean()
        if valid_fraction > 0.95:
            numeric_candidates.append(
                (
                    float(values.fillna(0).max()),
                    str(candidate),
                )
            )

    if not numeric_candidates:
        return None
    numeric_candidates.sort(reverse=True)
    return numeric_candidates[0][1]


def parse_sample_csv(
    path: Path,
    gencode: dict[str, str],
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    frame = pd.read_csv(
        path,
        compression="infer",
        sep=None,
        engine="python",
    )
    if frame.empty:
        raise ValueError(f"Empty sample file: {path}")

    gene_column = infer_gene_column(frame)
    count_column = infer_numeric_column(
        frame,
        [
            "counts",
            "count",
            "est_counts",
            "estimated_counts",
            "estimated_count",
            "expected_count",
            "expected_counts",
            "numreads",
            "num_reads",
            "raw_count",
            "raw_counts",
        ],
        exclude={gene_column},
    )
    tpm_column = choose_column(
        frame.columns,
        [
            "tpm",
            "abundance",
            "estimated_tpm",
            "est_tpm",
        ],
        required=False,
    )

    if count_column is None:
        raise KeyError(
            f"{path.name}: no count/estimated-count column was found."
        )
    if tpm_column is None:
        # Try another numeric column that is not the count column.
        tpm_column = infer_numeric_column(
            frame,
            ["tpm", "abundance"],
            exclude={gene_column, count_column},
        )
    if tpm_column is None:
        raise KeyError(f"{path.name}: no TPM/abundance column was found.")

    raw_genes = frame[gene_column].astype(str).str.strip()
    stripped = raw_genes.str.replace(r"\.\d+$", "", regex=True)
    upper = stripped.str.upper()

    ensembl_fraction = upper.str.match(r"^ENSG\d+$", na=False).mean()
    if ensembl_fraction > 0.5:
        genes = upper.map(gencode)
    else:
        genes = upper

    counts = pd.to_numeric(frame[count_column], errors="coerce")
    tpm = pd.to_numeric(frame[tpm_column], errors="coerce")

    parsed = pd.DataFrame(
        {
            "gene": genes,
            "count": counts,
            "tpm": tpm,
        }
    ).dropna(subset=["gene", "count", "tpm"])
    parsed = parsed.loc[
        (parsed["gene"].astype(str) != "")
        & np.isfinite(parsed["count"])
        & np.isfinite(parsed["tpm"])
        & (parsed["count"] >= 0)
        & (parsed["tpm"] >= 0)
    ]
    parsed["gene"] = parsed["gene"].astype(str).str.upper()

    parsed = (
        parsed.groupby("gene", sort=True)[["count", "tpm"]]
        .sum()
    )

    report = {
        "file": str(path),
        "n_input_rows": int(len(frame)),
        "n_output_genes": int(len(parsed)),
        "gene_column": gene_column,
        "count_column": count_column,
        "tpm_column": tpm_column,
        "ensembl_fraction": float(ensembl_fraction),
    }
    return parsed["count"], parsed["tpm"], report


def prepare(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str)
    required = {
        "gsm",
        "sample_number",
        "title",
        "condition",
        "mode",
        "experiment_batch",
        "replicate",
        "pair_id",
        "expected_sox17_direction",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata lacks columns: {sorted(missing)}")
    if metadata["gsm"].duplicated().any():
        raise ValueError("Metadata contains duplicate GSM accessions.")
    if len(metadata) != 42:
        raise ValueError(f"Expected 42 metadata rows; found {len(metadata)}.")

    gencode = parse_gencode_gtf(args.gencode_gtf)

    counts_by_sample: dict[str, pd.Series] = {}
    tpm_by_sample: dict[str, pd.Series] = {}
    reports = []

    for row in metadata.itertuples(index=False):
        path = find_sample_file(raw_dir, str(row.gsm))
        print(f"Parsing {row.gsm}: {path.name}", flush=True)
        counts, tpm, report = parse_sample_csv(path, gencode)
        counts_by_sample[str(row.gsm)] = counts
        tpm_by_sample[str(row.gsm)] = tpm
        report["gsm"] = str(row.gsm)
        report["condition"] = str(row.condition)
        reports.append(report)

    counts = pd.DataFrame(counts_by_sample).fillna(0.0)
    tpm = pd.DataFrame(tpm_by_sample).fillna(0.0)

    counts = counts.loc[counts.sum(axis=1) > 0]
    tpm = tpm.reindex(counts.index).fillna(0.0)

    counts = np.rint(counts).astype(np.int64)
    counts.index.name = "gene"
    tpm.index.name = "gene"

    log2_tpm = np.log2(tpm + 1.0)

    counts_path = output_dir / "GSE214742_gene_counts.tsv.gz"
    tpm_path = output_dir / "GSE214742_gene_TPM.tsv.gz"
    log2_tpm_path = output_dir / "GSE214742_log2TPM.tsv.gz"
    metadata_path = output_dir / "GSE214742_metadata.tsv"
    parser_path = output_dir / "GSE214742_parser_report.tsv"
    qc_path = output_dir / "GSE214742_sample_QC.tsv"

    counts.to_csv(counts_path, sep="\t", compression="gzip")
    tpm.to_csv(tpm_path, sep="\t", compression="gzip")
    log2_tpm.to_csv(log2_tpm_path, sep="\t", compression="gzip")
    metadata.to_csv(metadata_path, sep="\t", index=False)
    pd.DataFrame(reports).to_csv(parser_path, sep="\t", index=False)

    qc_rows = []
    for gsm in metadata["gsm"]:
        qc_rows.append(
            {
                "gsm": gsm,
                "condition": metadata.set_index("gsm").loc[gsm, "condition"],
                "library_estimated_counts": int(counts[gsm].sum()),
                "detected_genes_count_gt0": int((counts[gsm] > 0).sum()),
                "SOX17_TPM": float(tpm.loc["SOX17", gsm]) if "SOX17" in tpm.index else np.nan,
                "GAPDH_TPM": float(tpm.loc["GAPDH", gsm]) if "GAPDH" in tpm.index else np.nan,
            }
        )
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(qc_path, sep="\t", index=False)

    if "SOX17" not in counts.index:
        raise KeyError(
            "SOX17 was not found after parsing. Inspect the parser report "
            "and GENCODE mapping before continuing."
        )

    report = {
        "n_samples": int(counts.shape[1]),
        "n_genes": int(counts.shape[0]),
        "counts": str(counts_path),
        "tpm": str(tpm_path),
        "log2_tpm": str(log2_tpm_path),
        "metadata": str(metadata_path),
        "parser_report": str(parser_path),
        "sample_qc": str(qc_path),
    }
    (output_dir / "GSE214742_PREPARATION_COMPLETE.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


def read_spectra(path: Path, k: int) -> pd.DataFrame:
    spectra = pd.read_csv(path, sep="\t", index_col=0)
    if spectra.shape[0] == k:
        result = spectra
    elif spectra.shape[1] == k:
        result = spectra.transpose()
    else:
        raise ValueError(
            f"Expected one spectra dimension to equal K={k}; "
            f"observed {spectra.shape}."
        )
    result.index = result.index.astype(str).str.replace(r"\.0$", "", regex=True)
    result.columns = result.columns.astype(str)
    return result.apply(pd.to_numeric, errors="coerce")


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
    ).dropna(subset=["mouse_symbol", "human_symbol"])
    table["mouse_key"] = table["mouse_symbol"].str.strip().str.upper()
    table["human_key"] = table["human_symbol"].str.strip().str.upper()

    mouse_degree = table.groupby("mouse_key")["human_key"].transform("nunique")
    human_degree = table.groupby("human_key")["mouse_key"].transform("nunique")
    table = table.loc[
        (mouse_degree == 1)
        & (human_degree == 1)
    ].drop_duplicates(["mouse_key", "human_key"])
    return table


def targeted_genes(
    prepared_h5ad: Path,
    orthologs: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    adata = ad.read_h5ad(prepared_h5ad, backed="r")
    try:
        values = [str(value) for value in np.asarray(adata.uns["guide_targets"])]
    finally:
        adata.file.close()

    mouse = {
        value.strip().upper()
        for value in values
        if value.strip().lower() not in CONTROL_TARGET_LABELS
    }
    human = set(
        orthologs.loc[
            orthologs["mouse_key"].isin(mouse),
            "human_key",
        ]
    )
    human.update(mouse)
    return mouse, human


def build_program_mapping(
    spectra: pd.DataFrame,
    orthologs: pd.DataFrame,
    excluded_mouse: set[str],
    excluded_human: set[str],
    human_genes: set[str],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ortholog_index = orthologs.set_index("mouse_key", drop=False)
    rows = []
    coverage_rows = []

    for program in spectra.index:
        ranked = spectra.loc[program].dropna().sort_values(ascending=False)
        mapped_rows = []
        non_target_rank = 0

        for mouse_gene, score in ranked.items():
            mouse_gene = str(mouse_gene)
            mouse_key = mouse_gene.upper()

            if mouse_key in excluded_mouse:
                continue
            non_target_rank += 1
            if non_target_rank > top_n:
                break
            if mouse_key not in ortholog_index.index:
                continue

            ortholog = ortholog_index.loc[mouse_key]
            if isinstance(ortholog, pd.DataFrame):
                ortholog = ortholog.iloc[0]
            human_key = str(ortholog["human_key"]).upper()

            if human_key in excluded_human:
                continue

            mapped_rows.append(
                {
                    "program": str(program),
                    "mouse_gene": mouse_gene,
                    "mouse_rank": non_target_rank,
                    "mouse_loading": float(score),
                    "human_gene": str(ortholog["human_symbol"]),
                    "human_gene_key": human_key,
                    "present_in_GSE214742": human_key in human_genes,
                    "rank_weight_raw": 1.0 / math.log2(non_target_rank + 1.0),
                }
            )

        mapped = pd.DataFrame(mapped_rows)
        if not mapped.empty:
            mapped = mapped.drop_duplicates("human_gene_key", keep="first")
            present = mapped["present_in_GSE214742"].astype(bool)
            total_weight = mapped.loc[present, "rank_weight_raw"].sum()
            mapped["rank_weight"] = 0.0
            if total_weight > 0:
                mapped.loc[present, "rank_weight"] = (
                    mapped.loc[present, "rank_weight_raw"] / total_weight
                )
            rows.extend(mapped.to_dict(orient="records"))
            n_present = int(present.sum())
            n_mapped = int(len(mapped))
        else:
            n_present = 0
            n_mapped = 0

        coverage_rows.append(
            {
                "program": str(program),
                "n_one_to_one_human": n_mapped,
                "n_present_GSE214742": n_present,
                "human_expression_coverage": n_present / top_n,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return np.nan

    observed = abs(float(differences.mean()))
    null = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(differences)):
        null.append(
            abs(
                float(
                    np.mean(
                        differences * np.asarray(signs, dtype=float)
                    )
                )
            )
        )
    null = np.asarray(null)
    return float(np.mean(null >= observed - 1e-12))


def paired_hedges_dz(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n < 2:
        return np.nan
    sd = differences.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    d = differences.mean() / sd
    correction = 1.0 - 3.0 / max(4.0 * n - 5.0, 1.0)
    return float(correction * d)


def contrast_metadata(
    metadata: pd.DataFrame,
    contrast_name: str,
) -> pd.DataFrame:
    specification = CONTRASTS[contrast_name]
    selected = metadata.loc[
        metadata["condition"].isin(
            [specification["treatment"], specification["control"]]
        )
    ].copy()

    if specification["batch"] is not None:
        selected = selected.loc[
            selected["experiment_batch"].astype(str)
            == str(specification["batch"])
        ]

    selected["treated"] = (
        selected["condition"] == specification["treatment"]
    )

    pair_counts = selected.groupby("pair_id")["treated"].agg(["size", "sum"])
    bad = pair_counts.loc[
        (pair_counts["size"] != 2)
        | (pair_counts["sum"] != 1)
    ]
    if not bad.empty:
        raise ValueError(
            f"{contrast_name}: invalid paired structure:\n{bad}"
        )
    return selected.sort_values(["pair_id", "treated"])


def score_programs(
    mapping: pd.DataFrame,
    coverage: pd.DataFrame,
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    min_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = expression.copy()
    expression.index = expression.index.astype(str).str.upper()
    metadata = metadata.set_index("gsm", drop=False)

    score_rows = []
    effect_rows = []

    for contrast_name, specification in CONTRASTS.items():
        contrast_meta = contrast_metadata(metadata.reset_index(drop=True), contrast_name)
        sample_names = contrast_meta["gsm"].tolist()

        missing_samples = set(sample_names).difference(expression.columns)
        if missing_samples:
            raise KeyError(
                f"{contrast_name}: expression matrix lacks samples "
                f"{sorted(missing_samples)}"
            )

        expression_subset = expression[sample_names]
        gene_mean = expression_subset.mean(axis=1)
        gene_sd = expression_subset.std(axis=1, ddof=1)
        valid_sd = gene_sd > 0
        z_expression = expression_subset.loc[valid_sd].sub(
            gene_mean.loc[valid_sd],
            axis=0,
        ).div(
            gene_sd.loc[valid_sd],
            axis=0,
        )

        for program in coverage["program"].astype(str):
            program_map = mapping.loc[
                (mapping["program"].astype(str) == program)
                & mapping["present_in_GSE214742"].astype(bool)
            ].copy()
            program_map = program_map.loc[
                program_map["human_gene_key"].isin(z_expression.index)
            ]

            if len(program_map) < min_genes:
                effect_rows.append(
                    {
                        "contrast": contrast_name,
                        "program": program,
                        "n_genes": int(len(program_map)),
                        "evaluable": False,
                        "mean_paired_difference": np.nan,
                        "paired_Hedges_dz": np.nan,
                        "exact_sign_flip_p": np.nan,
                        "n_pairs": int(contrast_meta["pair_id"].nunique()),
                        "loss_alignment_factor": specification["loss_alignment_factor"],
                        "loss_aligned_score_effect": np.nan,
                    }
                )
                continue

            weights = program_map.set_index("human_gene_key")["rank_weight_raw"]
            weights = weights.groupby(level=0).max()
            genes = [gene for gene in weights.index if gene in z_expression.index]
            weights = weights.loc[genes]
            weights = weights / weights.sum()

            scores = z_expression.loc[genes].transpose().dot(weights)

            score_frame = contrast_meta[
                ["gsm", "pair_id", "condition", "treated"]
            ].copy()
            score_frame["program"] = program
            score_frame["score"] = score_frame["gsm"].map(scores)
            score_rows.extend(score_frame.to_dict(orient="records"))

            pivot = score_frame.pivot(
                index="pair_id",
                columns="treated",
                values="score",
            )
            differences = (
                pivot[True] - pivot[False]
            ).to_numpy(dtype=float)

            mean_difference = float(np.mean(differences))
            alignment_factor = specification["loss_alignment_factor"]

            effect_rows.append(
                {
                    "contrast": contrast_name,
                    "program": program,
                    "n_genes": int(len(genes)),
                    "evaluable": True,
                    "mean_paired_difference": mean_difference,
                    "paired_Hedges_dz": paired_hedges_dz(differences),
                    "exact_sign_flip_p": exact_sign_flip_p(differences),
                    "n_pairs": int(len(differences)),
                    "all_pairs_same_direction": bool(
                        np.all(differences > 0)
                        or np.all(differences < 0)
                    ),
                    "loss_alignment_factor": alignment_factor,
                    "loss_aligned_score_effect": (
                        mean_difference * alignment_factor
                        if np.isfinite(alignment_factor)
                        else np.nan
                    ),
                }
            )

    effects = pd.DataFrame(effect_rows)
    effects["score_BH_FDR_within_contrast"] = np.nan
    for contrast_name, indices in effects.groupby("contrast").groups.items():
        p = effects.loc[indices, "exact_sign_flip_p"]
        valid = p.notna()
        if valid.any():
            effects.loc[p.loc[valid].index, "score_BH_FDR_within_contrast"] = (
                multipletests(
                    p.loc[valid],
                    method="fdr_bh",
                )[1]
            )

    return pd.DataFrame(score_rows), effects


def run_program_gsea(
    mapping: pd.DataFrame,
    de_dir: Path,
    *,
    min_genes: int,
    max_genes: int,
    permutations: int,
    threads: int,
    seed: int,
) -> pd.DataFrame:
    result_rows = []

    for contrast_name, specification in CONTRASTS.items():
        de_path = de_dir / f"{contrast_name}_edgeR.tsv.gz"
        if not de_path.is_file():
            raise FileNotFoundError(de_path)

        de = pd.read_csv(de_path, sep="\t", compression="infer")
        de["gene"] = de["gene"].astype(str).str.upper()
        de["signed_sqrt_F"] = pd.to_numeric(
            de["signed_sqrt_F"],
            errors="coerce",
        )
        ranking = (
            de[["gene", "signed_sqrt_F"]]
            .dropna()
            .sort_values(
                ["gene", "signed_sqrt_F"],
                ascending=[True, False],
            )
            .drop_duplicates("gene", keep="first")
            .sort_values("signed_sqrt_F", ascending=False)
        )

        ranking_genes = set(ranking["gene"])
        gene_sets = {}
        for program, group in mapping.loc[
            mapping["present_in_GSE214742"].astype(bool)
        ].groupby("program", sort=False):
            genes = sorted(
                set(group["human_gene_key"]).intersection(ranking_genes)
            )
            if min_genes <= len(genes) <= max_genes:
                gene_sets[str(program)] = genes

        if not gene_sets:
            continue

        gsea = gp.prerank(
            rnk=ranking,
            gene_sets=gene_sets,
            threads=max(1, int(threads)),
            min_size=min_genes,
            max_size=max_genes,
            permutation_num=permutations,
            outdir=None,
            seed=seed,
            ascending=False,
            verbose=False,
        ).res2d.reset_index(drop=True)

        gsea = gsea.rename(
            columns={
                "Term": "program",
                "ES": "ES",
                "NES": "NES",
                "NOM p-val": "NOM_p_value",
                "FDR q-val": "GSEA_FDR",
                "FWER p-val": "FWER_p_value",
                "Lead_genes": "leading_edge_genes",
            }
        )
        gsea["program"] = (
            gsea["program"].astype(str).str.replace(r"\.0$", "", regex=True)
        )
        for column in [
            "ES",
            "NES",
            "NOM_p_value",
            "GSEA_FDR",
            "FWER_p_value",
        ]:
            if column in gsea:
                gsea[column] = pd.to_numeric(gsea[column], errors="coerce")

        gsea["contrast"] = contrast_name
        gsea["loss_alignment_factor"] = specification[
            "loss_alignment_factor"
        ]
        gsea["loss_aligned_NES"] = (
            gsea["NES"] * gsea["loss_alignment_factor"]
        )
        result_rows.append(gsea)

    if not result_rows:
        return pd.DataFrame()
    return pd.concat(result_rows, ignore_index=True)


def read_mouse_sox17_crt(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    table["target_name"] = table["target_name"].astype(str)
    table["program_name"] = (
        table["program_name"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    subset = table.loc[
        table["target_name"].str.upper() == "SOX17"
    ].copy()
    if subset.empty:
        raise ValueError("No Sox17 target rows were found in the IP CRT table.")
    subset["beta_clr"] = pd.to_numeric(subset["beta_clr"], errors="coerce")
    return subset


def read_human_pah_programs(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    table["program"] = (
        table["program"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    for column in ["hedges_g", "NES", "GSEA_FDR"]:
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def program_gene_direction_chain(
    mapping: pd.DataFrame,
    de_dir: Path,
    pah_de_path: Path | None,
) -> pd.DataFrame:
    de_by_contrast = {}
    for contrast_name in CONTRASTS:
        path = de_dir / f"{contrast_name}_edgeR.tsv.gz"
        de = pd.read_csv(path, sep="\t", compression="infer")
        de["gene"] = de["gene"].astype(str).str.upper()
        de_by_contrast[contrast_name] = de.set_index("gene")

    pah_de = None
    if pah_de_path is not None and pah_de_path.is_file():
        pah_de = pd.read_csv(
            pah_de_path,
            sep="\t",
            compression="infer",
        )
        if "gene" not in pah_de:
            pah_de = pah_de.rename(columns={pah_de.columns[0]: "gene"})
        pah_de["gene"] = pah_de["gene"].astype(str).str.upper()
        pah_de = pah_de.set_index("gene")

    rows = []
    for row in mapping.itertuples(index=False):
        if not bool(row.present_in_GSE214742):
            continue
        output = {
            "program": str(row.program),
            "mouse_gene": str(row.mouse_gene),
            "human_gene": str(row.human_gene_key),
            "mouse_rank": int(row.mouse_rank),
            "mouse_loading": float(row.mouse_loading),
        }

        loss_signs = []
        for contrast_name, specification in CONTRASTS.items():
            de = de_by_contrast[contrast_name]
            logfc = (
                float(de.loc[row.human_gene_key, "logFC"])
                if row.human_gene_key in de.index
                else np.nan
            )
            output[f"{contrast_name}_logFC"] = logfc
            if (
                contrast_name in LOSS_CONTRASTS
                and np.isfinite(logfc)
            ):
                loss_signs.append(np.sign(logfc))

        activation = output.get(
            f"{ACTIVATION_CONTRAST}_logFC",
            np.nan,
        )
        loss_values = [
            output.get(f"{contrast}_logFC", np.nan)
            for contrast in LOSS_CONTRASTS
        ]
        loss_values = [
            value for value in loss_values if np.isfinite(value)
        ]
        median_loss = (
            float(np.median(loss_values))
            if loss_values
            else np.nan
        )
        output["median_SOX17_loss_logFC"] = median_loss
        output["n_loss_contrasts_same_sign"] = (
            int(
                sum(
                    np.sign(value) == np.sign(median_loss)
                    for value in loss_values
                )
            )
            if np.isfinite(median_loss)
            else 0
        )
        output["activation_opposes_loss"] = (
            bool(np.sign(activation) == -np.sign(median_loss))
            if np.isfinite(activation) and np.isfinite(median_loss)
            else np.nan
        )

        if pah_de is not None and row.human_gene_key in pah_de.index:
            pah_logfc = float(pah_de.loc[row.human_gene_key, "logFC"])
            pah_fdr = float(pah_de.loc[row.human_gene_key, "FDR"])
        else:
            pah_logfc = np.nan
            pah_fdr = np.nan

        output["human_PAH_logFC"] = pah_logfc
        output["human_PAH_FDR"] = pah_fdr
        output["PAH_matches_SOX17_loss"] = (
            bool(np.sign(pah_logfc) == np.sign(median_loss))
            if np.isfinite(pah_logfc) and np.isfinite(median_loss)
            else np.nan
        )
        rows.append(output)

    return pd.DataFrame(rows)


def analyze(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str)
    expression = pd.read_csv(
        args.logcpm,
        sep="\t",
        index_col=0,
        compression="infer",
    )
    expression.index = expression.index.astype(str).str.upper()

    spectra = read_spectra(args.spectra, args.k)
    orthologs = read_one_to_one_orthologs(args.mgi_report)
    excluded_mouse, excluded_human = targeted_genes(
        args.prepared_h5ad,
        orthologs,
    )

    mapping, coverage = build_program_mapping(
        spectra,
        orthologs,
        excluded_mouse,
        excluded_human,
        set(expression.index),
        args.top_n,
    )
    mapping.to_csv(
        output_dir / "program_mouse_to_human_mapping.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    coverage.to_csv(
        output_dir / "program_mapping_coverage.tsv",
        sep="\t",
        index=False,
    )

    scores, effects = score_programs(
        mapping,
        coverage,
        expression,
        metadata,
        args.min_genes,
    )
    scores.to_csv(
        output_dir / "human_SOX17_program_scores_long.tsv",
        sep="\t",
        index=False,
    )
    effects.to_csv(
        output_dir / "human_SOX17_program_score_effects.tsv",
        sep="\t",
        index=False,
    )

    gsea = run_program_gsea(
        mapping,
        args.de_dir,
        min_genes=args.min_genes,
        max_genes=args.top_n,
        permutations=args.gsea_permutations,
        threads=args.threads,
        seed=args.seed,
    )
    if gsea.empty:
        raise RuntimeError("No GSEA results were generated.")
    gsea.to_csv(
        output_dir / "human_SOX17_program_GSEA.tsv",
        sep="\t",
        index=False,
    )

    mouse = read_mouse_sox17_crt(args.mouse_crt_ip)
    pah = read_human_pah_programs(args.human_pah_programs)

    program_table = pd.DataFrame(
        {"program": [str(value) for value in spectra.index]}
    )
    program_table = program_table.merge(
        coverage,
        on="program",
        how="left",
    )
    program_table = program_table.merge(
        mouse[
            ["program_name", "beta_clr", "p_skew", "primary_q_BH"]
        ].rename(
            columns={
                "program_name": "program",
                "beta_clr": "mouse_IP_Sox17_beta_clr",
                "p_skew": "mouse_IP_Sox17_p_skew",
                "primary_q_BH": "mouse_IP_Sox17_primary_q_BH",
            }
        ),
        on="program",
        how="left",
    )

    # Wide program-score and GSEA effects.
    score_wide = effects.pivot(
        index="program",
        columns="contrast",
        values=[
            "mean_paired_difference",
            "paired_Hedges_dz",
            "exact_sign_flip_p",
            "loss_aligned_score_effect",
        ],
    )
    score_wide.columns = [
        f"{metric}__{contrast}"
        for metric, contrast in score_wide.columns
    ]
    score_wide = score_wide.reset_index()
    program_table = program_table.merge(
        score_wide,
        on="program",
        how="left",
    )

    gsea_wide = gsea.pivot(
        index="program",
        columns="contrast",
        values=[
            "NES",
            "NOM_p_value",
            "GSEA_FDR",
            "loss_aligned_NES",
        ],
    )
    gsea_wide.columns = [
        f"{metric}__{contrast}"
        for metric, contrast in gsea_wide.columns
    ]
    gsea_wide = gsea_wide.reset_index()
    program_table = program_table.merge(
        gsea_wide,
        on="program",
        how="left",
    )

    pah_columns = [
        column
        for column in [
            "program",
            "hedges_g",
            "exact_permutation_p",
            "score_BH_FDR",
            "complete_separation",
            "NES",
            "NOM_p_value",
            "GSEA_FDR",
            "direction_concordant",
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
    program_table = program_table.merge(
        pah_subset,
        on="program",
        how="left",
    )

    # Directional consensus.
    for contrast_name in LOSS_CONTRASTS:
        program_table[f"score_concordant_mouse__{contrast_name}"] = (
            np.sign(
                program_table[
                    f"loss_aligned_score_effect__{contrast_name}"
                ]
            )
            == np.sign(program_table["mouse_IP_Sox17_beta_clr"])
        )
        program_table[f"GSEA_concordant_mouse__{contrast_name}"] = (
            np.sign(
                program_table[f"loss_aligned_NES__{contrast_name}"]
            )
            == np.sign(program_table["mouse_IP_Sox17_beta_clr"])
        )

    program_table["activation_score_reverses_mouse_loss"] = (
        np.sign(
            program_table[
                f"mean_paired_difference__{ACTIVATION_CONTRAST}"
            ]
        )
        == -np.sign(program_table["mouse_IP_Sox17_beta_clr"])
    )
    program_table["activation_GSEA_reverses_mouse_loss"] = (
        np.sign(program_table[f"NES__{ACTIVATION_CONTRAST}"])
        == -np.sign(program_table["mouse_IP_Sox17_beta_clr"])
    )

    score_concordance_columns = [
        f"score_concordant_mouse__{contrast}"
        for contrast in LOSS_CONTRASTS
    ]
    gsea_concordance_columns = [
        f"GSEA_concordant_mouse__{contrast}"
        for contrast in LOSS_CONTRASTS
    ]
    program_table["n_loss_score_concordant_with_mouse"] = (
        program_table[score_concordance_columns].sum(axis=1)
    )
    program_table["n_loss_GSEA_concordant_with_mouse"] = (
        program_table[gsea_concordance_columns].sum(axis=1)
    )

    loss_score_columns = [
        f"loss_aligned_score_effect__{contrast}"
        for contrast in LOSS_CONTRASTS
    ]
    loss_gsea_columns = [
        f"loss_aligned_NES__{contrast}"
        for contrast in LOSS_CONTRASTS
    ]

    program_table["median_human_SOX17_loss_score_effect"] = (
        program_table[loss_score_columns].median(axis=1)
    )
    program_table["median_human_SOX17_loss_GSEA_NES"] = (
        program_table[loss_gsea_columns].median(axis=1)
    )

    program_table["human_loss_consensus_matches_mouse"] = (
        np.sign(
            program_table["median_human_SOX17_loss_score_effect"]
        )
        == np.sign(program_table["mouse_IP_Sox17_beta_clr"])
    )
    program_table["human_PAH_matches_mouse_Sox17_loss"] = (
        np.sign(program_table["human_PAH_NES"])
        == np.sign(program_table["mouse_IP_Sox17_beta_clr"])
    )
    program_table["full_directional_chain"] = (
        (program_table["n_loss_score_concordant_with_mouse"] >= 2)
        & (program_table["n_loss_GSEA_concordant_with_mouse"] >= 2)
        & program_table["activation_score_reverses_mouse_loss"].fillna(False)
        & program_table["human_PAH_matches_mouse_Sox17_loss"].fillna(False)
    )

    program_table["directional_chain_score"] = (
        program_table["n_loss_score_concordant_with_mouse"].fillna(0)
        + program_table["n_loss_GSEA_concordant_with_mouse"].fillna(0)
        + program_table["activation_score_reverses_mouse_loss"].fillna(False).astype(int)
        + program_table["activation_GSEA_reverses_mouse_loss"].fillna(False).astype(int)
        + program_table["human_PAH_matches_mouse_Sox17_loss"].fillna(False).astype(int)
    )

    program_table = program_table.sort_values(
        [
            "full_directional_chain",
            "directional_chain_score",
            "human_PAH_GSEA_FDR",
        ],
        ascending=[False, False, True],
    )
    program_table.to_csv(
        output_dir / "SOX17_directional_program_chain.tsv",
        sep="\t",
        index=False,
    )

    gene_chain = program_gene_direction_chain(
        mapping,
        args.de_dir,
        args.human_pah_de,
    )
    gene_chain.to_csv(
        output_dir / "SOX17_program_gene_directional_chain.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    coherent_genes = gene_chain.loc[
        (gene_chain["n_loss_contrasts_same_sign"] >= 2)
        & gene_chain["PAH_matches_SOX17_loss"].fillna(False)
    ].copy()
    coherent_genes = coherent_genes.sort_values(
        ["program", "mouse_rank"]
    )
    coherent_genes.to_csv(
        output_dir / "SOX17_coherent_program_genes.tsv",
        sep="\t",
        index=False,
    )

    # SOX17 expression check from edgeR outputs.
    sox17_rows = []
    for contrast_name in CONTRASTS:
        de = pd.read_csv(
            args.de_dir / f"{contrast_name}_edgeR.tsv.gz",
            sep="\t",
            compression="infer",
        )
        row = de.loc[de["gene"].astype(str).str.upper() == "SOX17"]
        if not row.empty:
            current = row.iloc[0].to_dict()
            current["contrast"] = contrast_name
            sox17_rows.append(current)
    sox17_qc = pd.DataFrame(sox17_rows)
    sox17_qc.to_csv(
        output_dir / "SOX17_expression_QC.tsv",
        sep="\t",
        index=False,
    )

    # Workbook.
    workbook_path = output_dir / "SOX17_variant_to_program_to_PAH_review.xlsx"
    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
    ) as writer:
        program_table.to_excel(
            writer,
            sheet_name="Directional_chain",
            index=False,
        )
        effects.to_excel(
            writer,
            sheet_name="Program_score_effects",
            index=False,
        )
        gsea.to_excel(
            writer,
            sheet_name="Program_GSEA",
            index=False,
        )
        sox17_qc.to_excel(
            writer,
            sheet_name="SOX17_expression_QC",
            index=False,
        )
        coherent_genes.to_excel(
            writer,
            sheet_name="Coherent_program_genes",
            index=False,
        )
        coverage.to_excel(
            writer,
            sheet_name="Ortholog_coverage",
            index=False,
        )
        mapping.to_excel(
            writer,
            sheet_name="Program_mapping",
            index=False,
        )
        metadata.to_excel(
            writer,
            sheet_name="Sample_metadata",
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

    # Plots.
    program_order = program_table["program"].astype(str).tolist()

    def effect_matrix(
        table: pd.DataFrame,
        value_column: str,
    ) -> pd.DataFrame:
        matrix = table.pivot(
            index="program",
            columns="contrast",
            values=value_column,
        )
        return matrix.reindex(
            index=program_order,
            columns=list(CONTRASTS),
        )

    score_matrix = effect_matrix(
        effects,
        "loss_aligned_score_effect",
    )
    gsea_matrix = effect_matrix(
        gsea,
        "loss_aligned_NES",
    )

    def heatmap(
        matrix: pd.DataFrame,
        title: str,
        colorbar_label: str,
        stem: str,
    ) -> Path:
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
        axis.set_xlabel("Human SOX17 perturbation contrast")
        axis.set_ylabel("cNMF program")
        axis.set_title(title)
        colorbar = fig.colorbar(image, ax=axis)
        colorbar.set_label(colorbar_label)
        fig.tight_layout()
        path = output_dir / stem
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=200)
        plt.close(fig)
        matrix.to_csv(path.with_suffix(".tsv"), sep="\t")
        return path.with_suffix(".png")

    plot_paths = []
    plot_paths.append(
        heatmap(
            score_matrix,
            "Human SOX17 perturbation effects on transferred K30 programs",
            "SOX17-loss-aligned paired score effect",
            "SOX17_loss_aligned_program_scores",
        )
    )
    plot_paths.append(
        heatmap(
            gsea_matrix,
            "Human SOX17 perturbation GSEA of transferred K30 programs",
            "SOX17-loss-aligned GSEA NES",
            "SOX17_loss_aligned_program_GSEA",
        )
    )

    scatter = program_table.dropna(
        subset=[
            "mouse_IP_Sox17_beta_clr",
            "median_human_SOX17_loss_score_effect",
        ]
    )
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        scatter["mouse_IP_Sox17_beta_clr"],
        scatter["median_human_SOX17_loss_score_effect"],
    )
    axis.axhline(0, linewidth=1)
    axis.axvline(0, linewidth=1)
    for row in scatter.itertuples(index=False):
        axis.annotate(
            f"P{row.program}",
            (
                row.mouse_IP_Sox17_beta_clr,
                row.median_human_SOX17_loss_score_effect,
            ),
            fontsize=7,
        )
    axis.set_xlabel("Mouse IP_GFP Sox17 CRT beta_clr")
    axis.set_ylabel("Median human SOX17-loss program-score effect")
    axis.set_title("Mouse versus human SOX17 perturbation direction")
    fig.tight_layout()
    path = output_dir / "mouse_vs_human_SOX17_program_effects"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    plot_paths.append(path.with_suffix(".png"))

    pah_scatter = program_table.dropna(
        subset=[
            "median_human_SOX17_loss_score_effect",
            "human_PAH_NES",
        ]
    )
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        pah_scatter["median_human_SOX17_loss_score_effect"],
        pah_scatter["human_PAH_NES"],
    )
    axis.axhline(0, linewidth=1)
    axis.axvline(0, linewidth=1)
    for row in pah_scatter.itertuples(index=False):
        axis.annotate(
            f"P{row.program}",
            (
                row.median_human_SOX17_loss_score_effect,
                row.human_PAH_NES,
            ),
            fontsize=7,
        )
    axis.set_xlabel("Median human SOX17-loss program-score effect")
    axis.set_ylabel("Human PAH GSEA NES")
    axis.set_title("SOX17-loss program direction versus human PAH")
    fig.tight_layout()
    path = output_dir / "human_SOX17_loss_vs_human_PAH"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    plot_paths.append(path.with_suffix(".png"))

    with PdfPages(
        output_dir / "SOX17_variant_to_program_to_PAH_review.pdf"
    ) as pdf:
        for image_path in plot_paths:
            image = plt.imread(image_path)
            fig, axis = plt.subplots(figsize=(11, 8.5))
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(image_path.stem)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    notes = """SOX17 directional mechanism analysis
===================================

Causal direction
----------------
Published functional studies support:

    PAH enhancer risk allele
        -> reduced enhancer activity
        -> reduced SOX17 expression.

The analysis therefore aligns promoter CRISPRa in the opposite direction so
that every primary human contrast is expressed as a SOX17-loss effect.

Primary human loss contrasts
----------------------------
- siSOX17 versus negative siRNA;
- promoter CRISPRi versus BFP CRISPRi control;
- signal-1 enhancer repression versus BFP CRISPRi control.

Orthogonal reversal
-------------------
Promoter CRISPRa is expected to move programs in the opposite direction from
SOX17 loss.

Full directional chain
----------------------
A high-priority program should show:

1. concordant mouse Sox17 and human SOX17-loss program direction;
2. support in at least two independent human loss contrasts;
3. reversal after promoter CRISPRa;
4. the same direction in independent human PAH PAECs.

All experimentally targeted genes, including SOX17, are removed from the
transferred program gene sets. Thus, program support is not driven by direct
SOX17 expression.

Limitations
-----------
- GSE214742 uses cultured HPAECs and small matched replicate sets.
- Sign-flip p-values are coarse for 3-6 pairs.
- Program score and GSEA results should be interpreted jointly.
- Mouse beta_clr is compositional program usage, not conventional expression
  log2 fold change.
- The analysis supports a variant-to-gene-to-program mechanism, but it does not
  prove mediation of the genetic association by any one program.
"""
    (output_dir / "HOW_TO_INTERPRET.txt").write_text(notes)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "K": int(args.k),
        "spectra": str(args.spectra),
        "GSE214742_logCPM": str(args.logcpm),
        "mouse_CRT_IP": str(args.mouse_crt_ip),
        "human_PAH_program_table": str(args.human_pah_programs),
        "n_programs": int(len(program_table)),
        "n_full_directional_chain_programs": int(
            program_table["full_directional_chain"].sum()
        ),
        "outputs": {
            "workbook": str(workbook_path),
            "review_pdf": str(
                output_dir
                / "SOX17_variant_to_program_to_PAH_review.pdf"
            ),
            "directional_table": str(
                output_dir / "SOX17_directional_program_chain.tsv"
            ),
        },
    }
    (output_dir / "ANALYSIS_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    required = [
        workbook_path,
        output_dir / "SOX17_variant_to_program_to_PAH_review.pdf",
        output_dir / "SOX17_directional_program_chain.tsv",
        output_dir / "SOX17_expression_QC.tsv",
        output_dir / "HOW_TO_INTERPRET.txt",
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "Missing SOX17 mechanism outputs:\n"
            + "\n".join(missing)
        )

    print(program_table.head(15).to_string(index=False))
    print(f"\nWrote SOX17 mechanism review to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--raw-dir", type=Path, required=True)
    prepare_parser.add_argument("--metadata", type=Path, required=True)
    prepare_parser.add_argument("--gencode-gtf", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.set_defaults(func=prepare)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--spectra", type=Path, required=True)
    analyze_parser.add_argument("--k", type=int, default=30)
    analyze_parser.add_argument("--mgi-report", type=Path, required=True)
    analyze_parser.add_argument("--prepared-h5ad", type=Path, required=True)
    analyze_parser.add_argument("--metadata", type=Path, required=True)
    analyze_parser.add_argument("--logcpm", type=Path, required=True)
    analyze_parser.add_argument("--de-dir", type=Path, required=True)
    analyze_parser.add_argument("--mouse-crt-ip", type=Path, required=True)
    analyze_parser.add_argument(
        "--human-pah-programs",
        type=Path,
        required=True,
    )
    analyze_parser.add_argument(
        "--human-pah-de",
        type=Path,
        default=None,
    )
    analyze_parser.add_argument("--top-n", type=int, default=300)
    analyze_parser.add_argument("--min-genes", type=int, default=15)
    analyze_parser.add_argument(
        "--gsea-permutations",
        type=int,
        default=5000,
    )
    analyze_parser.add_argument("--threads", type=int, default=4)
    analyze_parser.add_argument("--seed", type=int, default=14)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.set_defaults(func=analyze)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
