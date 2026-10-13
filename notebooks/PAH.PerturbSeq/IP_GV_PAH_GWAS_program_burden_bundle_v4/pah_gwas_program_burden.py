#!/usr/bin/env python3
"""PAH GWAS to K30 cNMF program-burden analysis.

Subcommands
-----------
download-gwas
    Discover and download a GWAS Catalog summary-statistics file and format it
    for MAGMA.Celltyping.

build-program-inputs
    Transfer mouse K30 program loadings to one-to-one human orthologs, remove
    all experimentally targeted genes, create continuous MAGMA gene-property
    covariates and categorical program gene sets, and run an Ota-inspired
    loading-weighted matched-permutation burden test.

summarize
    Merge MAGMA linear/gene-set results with the matched-permutation analysis,
    annotate driver genes, and create the final workbook and figures.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable, Sequence
from urllib.parse import urljoin

import anndata as ad
from bs4 import BeautifulSoup
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import yaml


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

MHC_CHROMOSOME = 6
MHC_START = 25_000_000
MHC_STOP = 34_000_000


def accession_bucket(accession: str) -> str:
    match = re.fullmatch(r"GCST(\d+)", accession.upper())
    if match is None:
        raise ValueError(f"Invalid GWAS Catalog accession: {accession}")
    digits = match.group(1)
    number = int(digits)
    start = ((number - 1) // 1000) * 1000 + 1
    stop = start + 999
    width = len(digits)
    return f"GCST{start:0{width}d}-GCST{stop:0{width}d}"


def http_get(
    session: requests.Session,
    url: str,
    *,
    timeout: int = 120,
) -> requests.Response:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response


def list_directory(
    session: requests.Session,
    url: str,
) -> list[str]:
    response = http_get(session, url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if href and href not in {"../", "./"}:
            links.append(urljoin(url, href))
    return links


def find_summary_file(
    session: requests.Session,
    accession: str,
) -> tuple[str, bool, list[str]]:
    bucket = accession_bucket(accession)
    root = (
        "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        f"{bucket}/{accession}/"
    )

    root_links = list_directory(session, root)

    candidate_directories = []
    for link in root_links:
        lowered = link.lower().rstrip("/")
        if lowered.endswith("/harmonised") or lowered.endswith("/harmonized"):
            candidate_directories.append(link)
    candidate_directories.append(root)

    all_candidates: list[str] = []
    selected_directory = root

    for directory in candidate_directories:
        try:
            links = list_directory(session, directory)
        except requests.RequestException:
            continue

        candidates = [
            link
            for link in links
            if re.search(r"\.tsv(\.gz|\.bgz)?$", link, flags=re.I)
            and "meta" not in link.lower()
            and "readme" not in link.lower()
        ]
        all_candidates.extend(candidates)

        if candidates:
            selected_directory = directory
            break

    if not all_candidates:
        raise FileNotFoundError(
            f"No TSV summary-statistics file was found under {root}."
        )

    def preference(url: str) -> tuple[int, int, str]:
        name = Path(url).name.lower()
        harmonised = int(
            "harmon" in url.lower()
            or ".h.tsv" in name
        )
        accession_match = int(accession.lower() in name)
        return (harmonised, accession_match, name)

    chosen = sorted(
        set(all_candidates),
        key=preference,
        reverse=True,
    )[0]
    is_harmonised = (
        "harmon" in chosen.lower()
        or ".h.tsv" in Path(chosen).name.lower()
    )
    return chosen, is_harmonised, sorted(set(all_candidates))


def download_stream(
    session: requests.Session,
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Reusing downloaded file: {destination}")
        return

    with session.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)
    temporary.replace(destination)


def recursively_find(
    value,
    keys: set[str],
) -> list[object]:
    matches = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in keys:
                matches.append(child)
            matches.extend(recursively_find(child, keys))
    elif isinstance(value, list):
        for child in value:
            matches.extend(recursively_find(child, keys))
    return matches


def choose_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    normalized = {
        str(column).strip().lower(): str(column)
        for column in columns
    }
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    if required:
        raise KeyError(
            f"None of {list(candidates)} occurs in columns {list(columns)}."
        )
    return None


def choose_variant_column(
    path: Path,
    columns: Sequence[str],
) -> str:
    candidates = [
        candidate
        for candidate in [
            "hm_variant_id",
            "hm_rsid",
            "rsid",
            "rs_id",
            "variant_id",
            "snp",
            "markername",
        ]
        if choose_column(columns, [candidate], required=False) is not None
    ]
    if not candidates:
        raise KeyError("No candidate SNP identifier column was found.")

    header_map = {
        candidate: choose_column(columns, [candidate])
        for candidate in candidates
    }
    sample = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        usecols=list(header_map.values()),
        nrows=50_000,
        low_memory=False,
    )

    scored = []
    for candidate, column in header_map.items():
        values = sample[column].astype(str)
        rs_fraction = float(values.str.match(r"^rs\d+$", na=False).mean())
        scored.append((rs_fraction, candidate, column))

    scored.sort(reverse=True)
    print("SNP identifier candidates:", scored)
    return scored[0][2]


def format_gwas(
    source: Path,
    output: Path,
    *,
    sample_size: int,
) -> dict[str, object]:
    header = pd.read_csv(
        source,
        sep="\t",
        compression="infer",
        nrows=0,
    )
    columns = header.columns.tolist()

    snp_column = choose_variant_column(source, columns)
    chr_column = choose_column(
        columns,
        [
            "hm_chromosome",
            "chromosome",
            "chromosome_name",
            "chr",
        ],
    )
    bp_column = choose_column(
        columns,
        [
            "hm_position",
            "base_pair_location",
            "position",
            "bp",
            "pos",
        ],
    )
    p_column = choose_column(
        columns,
        [
            "p_value",
            "pvalue",
            "p-value",
            "p",
            "pval",
        ],
    )
    beta_column = choose_column(
        columns,
        [
            "hm_beta",
            "beta",
            "effect_size",
            "log_odds",
            "log_or",
        ],
        required=False,
    )
    or_column = choose_column(
        columns,
        [
            "hm_odds_ratio",
            "odds_ratio",
            "or",
        ],
        required=False,
    )

    selected_columns = [
        snp_column,
        chr_column,
        bp_column,
        p_column,
    ]
    if beta_column is not None:
        selected_columns.append(beta_column)
    if or_column is not None and or_column not in selected_columns:
        selected_columns.append(or_column)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    total_input = 0
    total_output = 0
    rsid_output = 0
    first_chunk = True

    for chunk in pd.read_csv(
        source,
        sep="\t",
        compression="infer",
        usecols=selected_columns,
        chunksize=400_000,
        low_memory=False,
    ):
        total_input += len(chunk)

        chromosome = (
            chunk[chr_column]
            .astype(str)
            .str.replace(r"^chr", "", regex=True, case=False)
        )
        chromosome = pd.to_numeric(chromosome, errors="coerce")
        position = pd.to_numeric(chunk[bp_column], errors="coerce")
        p_value = pd.to_numeric(chunk[p_column], errors="coerce")
        snp = chunk[snp_column].astype(str).str.strip()

        valid = (
            chromosome.between(1, 22)
            & position.gt(0)
            & p_value.gt(0)
            & p_value.le(1)
            & snp.ne("")
            & snp.ne("nan")
        )

        if not valid.any():
            continue

        chromosome = chromosome.loc[valid].astype(int)
        position = position.loc[valid].astype(np.int64)
        p_value = p_value.loc[valid].clip(lower=1e-300, upper=1.0)
        snp = snp.loc[valid]

        signed_effect = pd.Series(
            np.nan,
            index=p_value.index,
            dtype=float,
        )

        if beta_column is not None:
            signed_effect = pd.to_numeric(
                chunk.loc[valid, beta_column],
                errors="coerce",
            )
        elif or_column is not None:
            odds_ratio = pd.to_numeric(
                chunk.loc[valid, or_column],
                errors="coerce",
            )
            signed_effect = np.log(
                odds_ratio.where(odds_ratio > 0)
            )

        z_abs = norm.isf(p_value.to_numpy() / 2.0)
        sign = np.sign(signed_effect.to_numpy(dtype=float))
        sign[~np.isfinite(sign) | (sign == 0)] = 1.0
        z_score = z_abs * sign

        formatted = pd.DataFrame(
            {
                "SNP": snp.to_numpy(),
                "CHR": chromosome.to_numpy(),
                "BP": position.to_numpy(),
                "P": p_value.to_numpy(),
                "N": int(sample_size),
                "Z": z_score,
            }
        )
        if signed_effect.notna().any():
            formatted["BETA_OR_LOGOR"] = (
                signed_effect.to_numpy(dtype=float)
            )

        formatted = formatted.drop_duplicates("SNP", keep="first")

        formatted.to_csv(
            temporary,
            sep="\t",
            index=False,
            mode="wt" if first_chunk else "at",
            header=first_chunk,
            compression="gzip",
        )
        first_chunk = False
        total_output += len(formatted)
        rsid_output += int(
            formatted["SNP"].astype(str).str.match(r"^rs\d+$").sum()
        )

    if first_chunk:
        raise RuntimeError("No valid autosomal GWAS rows were written.")
    if total_output < 100_000:
        raise RuntimeError(
            f"Only {total_output:,} valid variants were written."
        )

    temporary.replace(output)
    rs_fraction = rsid_output / total_output

    return {
        "input_rows": int(total_input),
        "formatted_rows": int(total_output),
        "rsid_fraction": float(rs_fraction),
        "columns": {
            "SNP": snp_column,
            "CHR": chr_column,
            "BP": bp_column,
            "P": p_column,
            "BETA": beta_column,
            "OR": or_column,
        },
    }


def download_gwas(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Stanford-PAH-program-burden/1.0 "
                "(academic research; GWAS Catalog download)"
            )
        }
    )

    source_url, harmonised, candidates = find_summary_file(
        session,
        args.accession,
    )
    source_path = output_dir / Path(source_url).name
    download_stream(session, source_url, source_path)

    # Download any nearby YAML metadata files.
    source_directory = source_url.rsplit("/", 1)[0] + "/"
    metadata_urls = []
    try:
        for link in list_directory(session, source_directory):
            if link.lower().endswith((".yaml", ".yml")):
                metadata_urls.append(link)
    except requests.RequestException:
        pass

    metadata_payload = None
    metadata_path = None
    if metadata_urls:
        metadata_url = metadata_urls[0]
        metadata_path = output_dir / Path(metadata_url).name
        download_stream(session, metadata_url, metadata_path)
        try:
            metadata_payload = yaml.safe_load(metadata_path.read_text())
        except Exception as error:
            print(f"WARNING: could not parse metadata YAML: {error}")

    detected_build = None
    detected_sample_size = None
    if metadata_payload is not None:
        builds = recursively_find(
            metadata_payload,
            {
                "genome_assembly",
                "genomeassembly",
                "genome_build",
                "genomebuild",
            },
        )
        sizes = recursively_find(
            metadata_payload,
            {
                "sample_size",
                "samplesize",
            },
        )

        if builds:
            detected_build = str(builds[0])
        if sizes:
            try:
                detected_sample_size = int(
                    str(sizes[0]).replace(",", "")
                )
            except ValueError:
                detected_sample_size = None

    if args.genome_build is not None:
        genome_build = args.genome_build
    elif detected_build is not None:
        genome_build = detected_build
    elif harmonised:
        genome_build = "GRCh38"
    else:
        genome_build = "GRCh37"

    genome_build_upper = genome_build.upper()
    if genome_build_upper in {"HG19", "GRCH37", "37"}:
        genome_build = "GRCh37"
    elif genome_build_upper in {"HG38", "GRCH38", "38"}:
        genome_build = "GRCh38"
    else:
        raise ValueError(f"Unsupported genome build: {genome_build}")

    if args.sample_size is not None:
        sample_size = args.sample_size
    elif detected_sample_size is not None:
        sample_size = detected_sample_size
    else:
        raise ValueError(
            "Sample size was not provided and could not be detected."
        )

    formatted_path = (
        output_dir
        / f"{args.accession}.MAGMA.formatted.tsv.gz"
    )
    format_summary = format_gwas(
        source_path,
        formatted_path,
        sample_size=sample_size,
    )

    sha256 = hashlib.sha256()
    with source_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(block)

    manifest = {
        "accession": args.accession,
        "label": args.label,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "source_file": str(source_path),
        "source_sha256": sha256.hexdigest(),
        "candidate_files": candidates,
        "harmonised": bool(harmonised),
        "metadata_file": (
            str(metadata_path)
            if metadata_path is not None
            else None
        ),
        "genome_build": genome_build,
        "sample_size": int(sample_size),
        "formatted_file": str(formatted_path),
        "format_summary": format_summary,
    }
    manifest_path = output_dir / "GWAS_DOWNLOAD_COMPLETE.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))


def read_spectra(path: Path, k: int) -> pd.DataFrame:
    spectra = pd.read_csv(path, sep="\t", index_col=0)

    if spectra.shape[0] == k:
        result = spectra
    elif spectra.shape[1] == k:
        result = spectra.transpose()
    else:
        raise ValueError(
            f"{path}: neither dimension equals K={k}; shape={spectra.shape}."
        )

    result.index = (
        result.index.astype(str).str.replace(r"\.0$", "", regex=True)
    )
    result.columns = result.columns.astype(str)
    result = result.apply(pd.to_numeric, errors="coerce")

    if result.isna().all(axis=None):
        raise ValueError("Spectra contain no numeric values.")
    return result


def read_mgi_orthologs(path: Path) -> pd.DataFrame:
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
    ).dropna(
        subset=[
            "mouse_symbol",
            "human_symbol",
            "human_entrez",
        ]
    )

    table["mouse_key"] = (
        table["mouse_symbol"].str.strip().str.upper()
    )
    table["human_key"] = (
        table["human_symbol"].str.strip().str.upper()
    )
    table["human_entrez"] = (
        table["human_entrez"]
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    mouse_degree = table.groupby("mouse_key")[
        "human_entrez"
    ].transform("nunique")
    human_degree = table.groupby("human_entrez")[
        "mouse_key"
    ].transform("nunique")

    table = table.loc[
        (mouse_degree == 1)
        & (human_degree == 1)
    ].drop_duplicates(
        ["mouse_key", "human_entrez"]
    )
    return table


def read_targeted_mouse_genes(path: Path) -> set[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        if "guide_targets" not in adata.uns:
            raise KeyError("Prepared H5AD lacks uns['guide_targets'].")
        targets = [
            str(value)
            for value in np.asarray(adata.uns["guide_targets"])
        ]
    finally:
        adata.file.close()

    return {
        target.strip().upper()
        for target in targets
        if target.strip().lower() not in CONTROL_TARGET_LABELS
    }


def read_genes_out(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        dtype={"GENE": str},
    )
    frame.columns = [str(column).upper() for column in frame.columns]
    required = {
        "GENE",
        "CHR",
        "START",
        "STOP",
        "NSNPS",
        "NPARAM",
        "ZSTAT",
        "P",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"MAGMA genes.out lacks columns: {sorted(missing)}"
        )

    frame["GENE"] = (
        frame["GENE"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    for column in required.difference({"GENE"}):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            "GENE",
            "CHR",
            "START",
            "STOP",
            "NSNPS",
            "NPARAM",
            "ZSTAT",
            "P",
        ]
    )
    frame = frame.loc[
        frame["CHR"].between(1, 22)
        & frame["NSNPS"].gt(0)
    ].copy()
    frame["CHR"] = frame["CHR"].astype(int)
    return frame.drop_duplicates("GENE")


def read_human_expression(path: Path) -> pd.Series:
    frame = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
        compression="infer",
    )
    frame.index = (
        frame.index.astype(str).str.strip().str.upper()
    )
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    expression = numeric.mean(axis=1)
    expression = expression.groupby(level=0).mean()
    expression.name = "mean_logCPM"
    return expression


def bin_series(
    values: pd.Series,
    q: int,
) -> pd.Series:
    result = pd.Series(-1, index=values.index, dtype=int)
    valid = values.notna()
    if valid.sum() < 2:
        return result
    ranked = values.loc[valid].rank(method="average")
    n_unique = int(ranked.nunique())
    n_bins = min(q, n_unique)
    if n_bins < 2:
        result.loc[valid] = 0
        return result
    result.loc[valid] = pd.qcut(
        ranked,
        q=n_bins,
        labels=False,
        duplicates="drop",
    ).astype(int)
    return result


def matched_permutation_burden(
    universe: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
    analysis_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = universe.copy()
    work = work.loc[
        work["eligible_null_gene"]
        & np.isfinite(work["ZSTAT"])
    ].copy()

    work["expression_bin"] = bin_series(
        work["mean_logCPM"],
        10,
    )
    work["nsnps_bin"] = bin_series(
        np.log1p(work["NSNPS"]),
        5,
    )
    work["nparam_bin"] = bin_series(
        np.log1p(work["NPARAM"]),
        5,
    )

    index_by_gene = {
        gene: index
        for index, gene in enumerate(work["GENE"])
    }
    z_values = work["ZSTAT"].to_numpy(dtype=float)

    exact_pools: dict[tuple[int, int, int], np.ndarray] = {}
    relaxed_pools: dict[tuple[int, int], np.ndarray] = {}
    expression_pools: dict[int, np.ndarray] = {}

    for key, indices in work.groupby(
        ["expression_bin", "nsnps_bin", "nparam_bin"],
        observed=True,
    ).indices.items():
        exact_pools[tuple(map(int, key))] = np.asarray(indices, dtype=int)

    for key, indices in work.groupby(
        ["expression_bin", "nsnps_bin"],
        observed=True,
    ).indices.items():
        relaxed_pools[tuple(map(int, key))] = np.asarray(indices, dtype=int)

    for key, indices in work.groupby(
        ["expression_bin"],
        observed=True,
    ).indices.items():
        expression_pools[int(key)] = np.asarray(indices, dtype=int)

    global_pool = np.arange(len(work), dtype=int)
    rng = np.random.default_rng(seed)

    result_rows = []
    null_summary_rows = []

    for program, group in mapping.groupby("program", sort=True):
        group = group.loc[
            group["human_entrez"].isin(index_by_gene)
        ].copy()
        if len(group) < 10:
            result_rows.append(
                {
                    "analysis": analysis_label,
                    "program": str(program),
                    "n_program_genes": int(len(group)),
                    "observed_weighted_Z": np.nan,
                    "null_mean": np.nan,
                    "null_sd": np.nan,
                    "empirical_p_greater": np.nan,
                    "matched_Z": np.nan,
                    "exact_pool_fraction": np.nan,
                }
            )
            continue

        group = group.sort_values("human_rank")
        weights = group["rank_weight"].to_numpy(dtype=float)
        weights = weights / weights.sum()

        program_indices = np.array(
            [index_by_gene[gene] for gene in group["human_entrez"]],
            dtype=int,
        )
        observed = float(
            np.sum(weights * z_values[program_indices])
        )

        excluded = set(program_indices.tolist())
        pools = []
        exact_pool_count = 0

        for index in program_indices:
            row = work.iloc[index]
            exact_key = (
                int(row["expression_bin"]),
                int(row["nsnps_bin"]),
                int(row["nparam_bin"]),
            )
            relaxed_key = (
                int(row["expression_bin"]),
                int(row["nsnps_bin"]),
            )
            expression_key = int(row["expression_bin"])

            pool = exact_pools.get(exact_key, np.empty(0, dtype=int))
            pool = np.array(
                [value for value in pool if value not in excluded],
                dtype=int,
            )
            if len(pool) >= 10:
                exact_pool_count += 1
                pools.append(pool)
                continue

            pool = relaxed_pools.get(
                relaxed_key,
                np.empty(0, dtype=int),
            )
            pool = np.array(
                [value for value in pool if value not in excluded],
                dtype=int,
            )
            if len(pool) >= 10:
                pools.append(pool)
                continue

            pool = expression_pools.get(
                expression_key,
                np.empty(0, dtype=int),
            )
            pool = np.array(
                [value for value in pool if value not in excluded],
                dtype=int,
            )
            if len(pool) >= 10:
                pools.append(pool)
                continue

            pool = np.array(
                [value for value in global_pool if value not in excluded],
                dtype=int,
            )
            pools.append(pool)

        null_values = np.empty(n_permutations, dtype=float)
        for permutation in range(n_permutations):
            sampled = np.array(
                [
                    rng.choice(pool)
                    for pool in pools
                ],
                dtype=int,
            )
            null_values[permutation] = float(
                np.sum(weights * z_values[sampled])
            )

        null_mean = float(null_values.mean())
        null_sd = float(null_values.std(ddof=1))
        empirical_p = float(
            (1 + np.sum(null_values >= observed))
            / (n_permutations + 1)
        )
        matched_z = (
            float((observed - null_mean) / null_sd)
            if null_sd > 0
            else np.nan
        )

        result_rows.append(
            {
                "analysis": analysis_label,
                "program": str(program),
                "n_program_genes": int(len(group)),
                "observed_weighted_Z": observed,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "empirical_p_greater": empirical_p,
                "matched_Z": matched_z,
                "exact_pool_fraction": (
                    exact_pool_count / len(program_indices)
                ),
            }
        )
        null_summary_rows.append(
            {
                "analysis": analysis_label,
                "program": str(program),
                "null_q025": float(
                    np.quantile(null_values, 0.025)
                ),
                "null_median": float(
                    np.quantile(null_values, 0.50)
                ),
                "null_q975": float(
                    np.quantile(null_values, 0.975)
                ),
            }
        )

    results = pd.DataFrame(result_rows)
    valid = results["empirical_p_greater"].notna()
    results["permutation_BH_FDR"] = np.nan
    if valid.any():
        results.loc[valid, "permutation_BH_FDR"] = multipletests(
            results.loc[valid, "empirical_p_greater"],
            method="fdr_bh",
        )[1]
    return results, pd.DataFrame(null_summary_rows)


def build_program_inputs(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra = read_spectra(args.spectra, args.k)
    orthologs = read_mgi_orthologs(args.mgi_report)
    targeted_mouse = read_targeted_mouse_genes(
        args.prepared_h5ad
    )
    genes = read_genes_out(args.genes_out)
    expression = read_human_expression(args.human_logcpm)

    genes["GENE"] = genes["GENE"].astype(str)
    genes = genes.merge(
        orthologs[
            [
                "human_entrez",
                "human_symbol",
                "human_key",
            ]
        ].drop_duplicates("human_entrez"),
        left_on="GENE",
        right_on="human_entrez",
        how="left",
    )
    genes["mean_logCPM"] = (
        genes["human_key"].map(expression)
    )
    genes["in_MHC"] = (
        genes["CHR"].eq(MHC_CHROMOSOME)
        & genes["START"].le(MHC_STOP)
        & genes["STOP"].ge(MHC_START)
    )
    genes["eligible_null_gene"] = (
        genes["human_entrez"].notna()
        & genes["mean_logCPM"].notna()
    )

    ortholog_by_mouse = orthologs.set_index(
        "mouse_key",
        drop=False,
    )

    mapped_rows = []
    excluded_rows = []

    for program in spectra.index:
        ranked = (
            spectra.loc[program]
            .dropna()
            .sort_values(ascending=False)
        )

        human_rank = 0

        for mouse_rank, (mouse_gene, score) in enumerate(
            ranked.items(),
            start=1,
        ):
            mouse_gene = str(mouse_gene)
            mouse_key = mouse_gene.upper()

            if mouse_key in targeted_mouse:
                excluded_rows.append(
                    {
                        "program": str(program),
                        "mouse_gene": mouse_gene,
                        "reason": "experimentally_targeted_gene",
                    }
                )
                continue

            if mouse_key not in ortholog_by_mouse.index:
                continue

            row = ortholog_by_mouse.loc[mouse_key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            human_entrez = str(row["human_entrez"])
            gene_match = genes.loc[
                genes["GENE"].eq(human_entrez)
            ]
            if gene_match.empty:
                continue

            gene_row = gene_match.iloc[0]
            human_rank += 1

            rank_weight_raw = 1.0 / math.log2(human_rank + 1.0)

            mapped_rows.append(
                {
                    "program": str(program),
                    "mouse_gene": mouse_gene,
                    "mouse_rank": int(mouse_rank),
                    "human_rank": int(human_rank),
                    "mouse_spectrum_score": float(score),
                    "human_symbol": str(row["human_symbol"]),
                    "human_entrez": human_entrez,
                    "mean_logCPM": (
                        float(gene_row["mean_logCPM"])
                        if pd.notna(gene_row["mean_logCPM"])
                        else np.nan
                    ),
                    "CHR": int(gene_row["CHR"]),
                    "START": int(gene_row["START"]),
                    "STOP": int(gene_row["STOP"]),
                    "NSNPS": float(gene_row["NSNPS"]),
                    "NPARAM": float(gene_row["NPARAM"]),
                    "ZSTAT": float(gene_row["ZSTAT"]),
                    "gene_P": float(gene_row["P"]),
                    "in_MHC": bool(gene_row["in_MHC"]),
                    "rank_weight_raw": rank_weight_raw,
                }
            )

            if human_rank >= args.top_n:
                break

    mapping = pd.DataFrame(mapped_rows)
    if mapping.empty:
        raise RuntimeError("No program genes mapped to MAGMA genes.")

    mapping = (
        mapping.sort_values(
            [
                "program",
                "human_rank",
            ]
        )
        .drop_duplicates(
            [
                "program",
                "human_entrez",
            ],
            keep="first",
        )
    )

    mapping["rank_weight"] = (
        mapping.groupby("program")[
            "rank_weight_raw"
        ].transform(lambda values: values / values.sum())
    )
    mapping["primary_analysis_gene"] = ~mapping["in_MHC"]

    mapping.to_csv(
        output_dir / "program_gene_mapping.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(excluded_rows).to_csv(
        output_dir / "excluded_targeted_genes.tsv",
        sep="\t",
        index=False,
    )

    program_columns = [
        f"P{program}"
        for program in spectra.index
    ]

    def write_covariates(
        label: str,
        include_mhc: bool,
    ) -> tuple[Path, Path, Path]:
        covariates = genes[
            [
                "GENE",
                "mean_logCPM",
            ]
        ].copy()

        selected_mapping = mapping.copy()
        if not include_mhc:
            selected_mapping = selected_mapping.loc[
                ~selected_mapping["in_MHC"]
            ].copy()

        for program in spectra.index:
            program_label = f"P{program}"
            values = selected_mapping.loc[
                selected_mapping["program"].eq(str(program))
            ].set_index("human_entrez")["rank_weight"]
            covariates[program_label] = (
                covariates["GENE"].map(values).fillna(0.0)
            )

        covariate_path = (
            output_dir
            / f"program_gene_covariates_{label}.txt"
        )
        covariates.to_csv(
            covariate_path,
            sep=" ",
            index=False,
            na_rep="NA",
        )

        set50_path = output_dir / f"program_top50_{label}.sets"
        set300_path = output_dir / f"program_top300_{label}.sets"

        with set50_path.open("w") as handle50, set300_path.open("w") as handle300:
            for program in spectra.index:
                program_mapping = (
                    selected_mapping.loc[
                        selected_mapping["program"].eq(str(program))
                    ]
                    .sort_values("human_rank")
                )
                genes50 = program_mapping["human_entrez"].head(50).tolist()
                genes300 = program_mapping["human_entrez"].head(300).tolist()

                if len(genes50) >= 10:
                    handle50.write(
                        f"P{program} " + " ".join(genes50) + "\n"
                    )
                if len(genes300) >= 10:
                    handle300.write(
                        f"P{program} " + " ".join(genes300) + "\n"
                    )

        return covariate_path, set50_path, set300_path

    primary_files = write_covariates(
        "target_and_MHC_excluded",
        include_mhc=False,
    )
    sensitivity_files = write_covariates(
        "target_excluded_MHC_included",
        include_mhc=True,
    )

    burden_tables = []
    null_tables = []

    for label, include_mhc in [
        ("target_and_MHC_excluded", False),
        ("target_excluded_MHC_included", True),
    ]:
        selected_mapping = mapping.copy()
        selected_genes = genes.copy()

        if not include_mhc:
            selected_mapping = selected_mapping.loc[
                ~selected_mapping["in_MHC"]
            ].copy()
            selected_genes["eligible_null_gene"] &= (
                ~selected_genes["in_MHC"]
            )

        burden, null_summary = matched_permutation_burden(
            selected_genes,
            selected_mapping,
            n_permutations=args.n_permutations,
            seed=args.seed + (0 if not include_mhc else 1000),
            analysis_label=label,
        )
        burden_tables.append(burden)
        null_tables.append(null_summary)

    burden_all = pd.concat(
        burden_tables,
        ignore_index=True,
    )
    null_all = pd.concat(
        null_tables,
        ignore_index=True,
    )
    burden_all.to_csv(
        output_dir / "matched_permutation_program_burden.tsv",
        sep="\t",
        index=False,
    )
    null_all.to_csv(
        output_dir / "matched_permutation_null_summary.tsv",
        sep="\t",
        index=False,
    )

    # Gene-level contribution score for interpretation.
    driver = mapping.copy()
    driver["weighted_gene_Z_contribution"] = (
        driver["rank_weight"] * driver["ZSTAT"]
    )
    driver = driver.sort_values(
        [
            "program",
            "weighted_gene_Z_contribution",
        ],
        ascending=[True, False],
    )
    driver.to_csv(
        output_dir / "program_driver_gene_contributions.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    coverage = (
        mapping.groupby("program", observed=True)
        .agg(
            mapped_genes=("human_entrez", "nunique"),
            mapped_non_MHC_genes=(
                "primary_analysis_gene",
                "sum",
            ),
            mean_expression=("mean_logCPM", "mean"),
            MHC_genes=("in_MHC", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(
        output_dir / "program_mapping_coverage.tsv",
        sep="\t",
        index=False,
    )

    manifest = {
        "k": int(args.k),
        "top_n": int(args.top_n),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "spectra": str(args.spectra),
        "genes_out": str(args.genes_out),
        "mgi_report": str(args.mgi_report),
        "prepared_h5ad": str(args.prepared_h5ad),
        "human_logcpm": str(args.human_logcpm),
        "targeted_mouse_genes_removed": sorted(targeted_mouse),
        "primary_MHC_exclusion": {
            "chromosome": MHC_CHROMOSOME,
            "start": MHC_START,
            "stop": MHC_STOP,
        },
        "program_columns": program_columns,
        "primary_covariates": str(primary_files[0]),
        "primary_top50_sets": str(primary_files[1]),
        "primary_top300_sets": str(primary_files[2]),
        "sensitivity_covariates": str(sensitivity_files[0]),
        "sensitivity_top50_sets": str(sensitivity_files[1]),
        "sensitivity_top300_sets": str(sensitivity_files[2]),
    }
    (
        output_dir / "PROGRAM_INPUTS_COMPLETE.json"
    ).write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    print("\nProgram mapping coverage:")
    print(coverage.to_string(index=False))
    print("\nMatched-permutation results:")
    print(
        burden_all.sort_values(
            [
                "analysis",
                "empirical_p_greater",
            ]
        ).to_string(index=False)
    )


def read_magma_gsa(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
    )
    frame.columns = [str(column).upper() for column in frame.columns]

    variable_column = None
    for candidate in ["VARIABLE", "SET", "GENESET"]:
        if candidate in frame.columns:
            variable_column = candidate
            break
    if variable_column is None:
        variable_column = frame.columns[0]

    frame = frame.rename(columns={variable_column: "program"})
    frame["program"] = (
        frame["program"]
        .astype(str)
        .str.replace(r"^P", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )

    for column in frame.columns:
        if column != "program":
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )
    return frame


def attach_fdr(
    frame: pd.DataFrame,
    p_column: str,
    output_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    p = pd.to_numeric(result[p_column], errors="coerce")
    valid = p.notna()
    result[output_column] = np.nan
    if valid.any():
        result.loc[valid, output_column] = multipletests(
            p.loc[valid],
            method="fdr_bh",
        )[1]
    return result


def summarize(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    burden = pd.read_csv(
        args.program_input_dir
        / "matched_permutation_program_burden.tsv",
        sep="\t",
    )
    coverage = pd.read_csv(
        args.program_input_dir
        / "program_mapping_coverage.tsv",
        sep="\t",
        dtype={"program": str},
    )
    driver = pd.read_csv(
        args.program_input_dir
        / "program_driver_gene_contributions.tsv.gz",
        sep="\t",
        compression="infer",
        dtype={"program": str},
    )

    continuous_primary = read_magma_gsa(
        args.magma_results_dir
        / "continuous_primary.gsa.out"
    )
    continuous_expr = read_magma_gsa(
        args.magma_results_dir
        / "continuous_primary_expr_conditioned.gsa.out"
    )
    top50_primary = read_magma_gsa(
        args.magma_results_dir
        / "top50_primary.gsa.out"
    )
    top300_primary = read_magma_gsa(
        args.magma_results_dir
        / "top300_primary.gsa.out"
    )
    continuous_mhc = read_magma_gsa(
        args.magma_results_dir
        / "continuous_MHC_included.gsa.out"
    )

    def select_columns(
        frame: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        keep = ["program"]
        rename = {}

        for source, target in [
            ("BETA", f"{prefix}_beta"),
            ("BETA_STD", f"{prefix}_beta_std"),
            ("SE", f"{prefix}_se"),
            ("P", f"{prefix}_p"),
            ("NGENES", f"{prefix}_n_genes"),
        ]:
            if source in frame.columns:
                keep.append(source)
                rename[source] = target

        result = frame[keep].rename(columns=rename)
        p_column = f"{prefix}_p"
        if p_column in result:
            result = attach_fdr(
                result,
                p_column,
                f"{prefix}_BH_FDR",
            )
        return result

    merged = coverage.copy()
    for frame, prefix in [
        (continuous_primary, "MAGMA_linear"),
        (continuous_expr, "MAGMA_expr_conditioned"),
        (top50_primary, "MAGMA_top50"),
        (top300_primary, "MAGMA_top300"),
        (continuous_mhc, "MAGMA_MHC_included"),
    ]:
        merged = merged.merge(
            select_columns(frame, prefix),
            on="program",
            how="outer",
        )

    burden_primary = burden.loc[
        burden["analysis"].eq(
            "target_and_MHC_excluded"
        )
    ].copy()
    burden_primary["program"] = (
        burden_primary["program"].astype(str)
    )
    burden_primary = burden_primary.rename(
        columns={
            "observed_weighted_Z": "matched_weighted_Z",
            "empirical_p_greater": "matched_permutation_p",
            "matched_Z": "matched_standardized_Z",
            "permutation_BH_FDR": "matched_permutation_BH_FDR",
        }
    )

    merged = merged.merge(
        burden_primary[
            [
                "program",
                "n_program_genes",
                "matched_weighted_Z",
                "null_mean",
                "null_sd",
                "matched_permutation_p",
                "matched_standardized_Z",
                "matched_permutation_BH_FDR",
                "exact_pool_fraction",
            ]
        ],
        on="program",
        how="left",
    )

    human_validation = None
    if (
        args.human_validation is not None
        and args.human_validation.is_file()
    ):
        human_validation = pd.read_csv(
            args.human_validation,
            sep="\t",
            dtype={"program": str},
        )
        useful = [
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
                "human_expression_coverage",
                "leading_edge_genes",
            ]
            if column in human_validation.columns
        ]
        merged = merged.merge(
            human_validation[useful],
            on="program",
            how="left",
        )

    merged["genetic_support_any_FDR_0.10"] = False
    for column in [
        "MAGMA_linear_BH_FDR",
        "MAGMA_expr_conditioned_BH_FDR",
        "MAGMA_top50_BH_FDR",
        "MAGMA_top300_BH_FDR",
        "matched_permutation_BH_FDR",
    ]:
        if column in merged:
            merged["genetic_support_any_FDR_0.10"] |= (
                merged[column] < 0.10
            ).fillna(False)

    merged = merged.sort_values(
        [
            "MAGMA_expr_conditioned_p",
            "matched_permutation_p",
        ],
        na_position="last",
    )
    merged.to_csv(
        output_dir / "PAH_GWAS_program_burden_results.tsv",
        sep="\t",
        index=False,
    )

    driver_top = (
        driver.loc[~driver["in_MHC"].astype(bool)]
        .sort_values(
            [
                "program",
                "weighted_gene_Z_contribution",
            ],
            ascending=[True, False],
        )
        .groupby("program", observed=True)
        .head(args.top_driver_genes)
        .copy()
    )
    driver_top.to_csv(
        output_dir / "PAH_GWAS_program_driver_genes.tsv",
        sep="\t",
        index=False,
    )

    workbook_path = (
        output_dir / "PAH_GWAS_K30_program_burden_review.xlsx"
    )
    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
    ) as writer:
        merged.to_excel(
            writer,
            sheet_name="Program_results",
            index=False,
        )
        driver_top.to_excel(
            writer,
            sheet_name="Driver_genes",
            index=False,
        )
        coverage.to_excel(
            writer,
            sheet_name="Program_coverage",
            index=False,
        )
        burden.to_excel(
            writer,
            sheet_name="Matched_permutation",
            index=False,
        )
        continuous_primary.to_excel(
            writer,
            sheet_name="MAGMA_linear_primary",
            index=False,
        )
        continuous_expr.to_excel(
            writer,
            sheet_name="MAGMA_expr_conditioned",
            index=False,
        )
        top50_primary.to_excel(
            writer,
            sheet_name="MAGMA_top50",
            index=False,
        )
        top300_primary.to_excel(
            writer,
            sheet_name="MAGMA_top300",
            index=False,
        )
        continuous_mhc.to_excel(
            writer,
            sheet_name="MAGMA_MHC_sensitivity",
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

    def plot_metric(
        p_column: str,
        title: str,
        stem: str,
    ) -> Path | None:
        if p_column not in merged:
            return None
        plot = merged[
            ["program", p_column]
        ].copy()
        plot[p_column] = pd.to_numeric(
            plot[p_column],
            errors="coerce",
        )
        plot = plot.dropna().sort_values(p_column)
        if plot.empty:
            return None

        plot["minus_log10_p"] = -np.log10(
            plot[p_column].clip(lower=1e-300)
        )
        fig, axis = plt.subplots(
            figsize=(9, max(6, 0.26 * len(plot)))
        )
        y = np.arange(len(plot))
        axis.barh(y, plot["minus_log10_p"])
        axis.set_yticks(y)
        axis.set_yticklabels(
            "P" + plot["program"].astype(str)
        )
        axis.axvline(
            -math.log10(0.05 / len(plot)),
            linestyle="--",
            color="black",
            linewidth=1,
            label="Bonferroni 0.05/30",
        )
        axis.set_xlabel("-log10(P)")
        axis.set_ylabel("cNMF program")
        axis.set_title(title)
        axis.legend(frameon=False)
        fig.tight_layout()

        path = output_dir / stem
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=200)
        plt.close(fig)
        return path.with_suffix(".png")

    figures = []
    for p_column, title, stem in [
        (
            "MAGMA_expr_conditioned_p",
            "PAH GWAS enrichment of K30 programs: MAGMA linear, PAEC-expression conditioned",
            "MAGMA_expression_conditioned_programs",
        ),
        (
            "matched_permutation_p",
            "PAH GWAS enrichment of K30 programs: loading-weighted matched permutation",
            "matched_permutation_programs",
        ),
        (
            "MAGMA_top50_p",
            "PAH GWAS enrichment of top-50 program genes",
            "MAGMA_top50_programs",
        ),
    ]:
        figure = plot_metric(p_column, title, stem)
        if figure is not None:
            figures.append(figure)

    if {
        "MAGMA_expr_conditioned_p",
        "matched_permutation_p",
    }.issubset(merged.columns):
        plot = merged.dropna(
            subset=[
                "MAGMA_expr_conditioned_p",
                "matched_permutation_p",
            ]
        )
        fig, axis = plt.subplots(figsize=(7, 6))
        x = -np.log10(
            plot["MAGMA_expr_conditioned_p"].clip(
                lower=1e-300
            )
        )
        y = -np.log10(
            plot["matched_permutation_p"].clip(
                lower=1e-300
            )
        )
        axis.scatter(x, y)
        for row, x_value, y_value in zip(
            plot.itertuples(),
            x,
            y,
        ):
            axis.annotate(
                f"P{row.program}",
                (x_value, y_value),
                fontsize=7,
            )
        axis.set_xlabel(
            "MAGMA expression-conditioned -log10(P)"
        )
        axis.set_ylabel(
            "Matched-permutation -log10(P)"
        )
        axis.set_title(
            "Agreement between two PAH program-genetic enrichment tests"
        )
        fig.tight_layout()
        scatter_path = (
            output_dir
            / "MAGMA_vs_matched_permutation.png"
        )
        fig.savefig(scatter_path, dpi=200)
        fig.savefig(scatter_path.with_suffix(".pdf"))
        plt.close(fig)
        figures.append(scatter_path)

    with PdfPages(
        output_dir / "PAH_GWAS_program_burden_review.pdf"
    ) as pdf:
        for path in figures:
            image = plt.imread(path)
            fig, axis = plt.subplots(figsize=(11, 8.5))
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(path.stem)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    notes = """PAH common-variant GWAS program-burden analysis
================================================

Primary analysis
----------------
- Clinically ascertained PAH GWAS GCST007228.
- MAGMA gene-level association using a European 1000 Genomes LD reference.
- All experimentally targeted genes removed from cNMF program gene weights.
- Extended MHC genes removed from the primary program analysis because the
  original PAH GWAS has a strong HLA-DPA1/DPB1 locus.
- cNMF mouse genes transferred through MGI one-to-one protein-coding human
  orthologs.
- Continuous program weights are rank-decay weights over the top 300 mapped
  non-target genes.
- MAGMA continuous gene-property analysis is reported with and without
  conditioning on mean human PAEC expression.
- Top-50 and top-300 competitive MAGMA gene-set tests are sensitivities.
- A separate loading-weighted program burden averages MAGMA gene Z statistics
  and uses matched gene permutations on human PAEC expression, SNP count, and
  gene-model parameter count.

Interpretation
--------------
This analysis tests whether genes belonging to a transcriptional program are
more strongly associated with PAH common-variant risk than other comparable
genes. It is unsigned. It does not establish whether increasing or decreasing
a program raises PAH risk.

This is an adaptation, not a direct replication, of Ota et al.'s program burden
effect. Ota et al. averaged directional loss-of-function burden effects of
highly loaded genes. Here MAGMA gene association Z statistics are unsigned.

The GWAS is relatively small and has two dominant loci. Interpret findings that
are robust across expression-conditioned MAGMA, matched permutations, MHC
exclusion, and the independent human PAEC program validation most strongly.
"""
    (output_dir / "HOW_TO_INTERPRET.txt").write_text(notes)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "program_results": str(
            output_dir
            / "PAH_GWAS_program_burden_results.tsv"
        ),
        "workbook": str(workbook_path),
        "review_pdf": str(
            output_dir
            / "PAH_GWAS_program_burden_review.pdf"
        ),
        "n_programs": int(len(merged)),
        "n_genetically_supported_FDR_0.10": int(
            merged["genetic_support_any_FDR_0.10"].sum()
        ),
    }
    (
        output_dir / "ANALYSIS_COMPLETE.json"
    ).write_text(json.dumps(manifest, indent=2) + "\n")

    required = [
        workbook_path,
        output_dir / "PAH_GWAS_program_burden_results.tsv",
        output_dir / "PAH_GWAS_program_burden_review.pdf",
        output_dir / "HOW_TO_INTERPRET.txt",
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file()
        or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "Missing analysis outputs:\n"
            + "\n".join(missing)
        )

    print(merged.to_string(index=False))
    print(f"\nWrote analysis to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    download = subparsers.add_parser("download-gwas")
    download.add_argument("--accession", required=True)
    download.add_argument("--label", required=True)
    download.add_argument("--sample-size", type=int, default=None)
    download.add_argument(
        "--genome-build",
        choices=["GRCh37", "GRCh38"],
        default=None,
    )
    download.add_argument("--output-dir", type=Path, required=True)
    download.set_defaults(func=download_gwas)

    build = subparsers.add_parser("build-program-inputs")
    build.add_argument("--spectra", type=Path, required=True)
    build.add_argument("--k", type=int, default=30)
    build.add_argument("--mgi-report", type=Path, required=True)
    build.add_argument("--prepared-h5ad", type=Path, required=True)
    build.add_argument("--human-logcpm", type=Path, required=True)
    build.add_argument("--genes-out", type=Path, required=True)
    build.add_argument("--top-n", type=int, default=300)
    build.add_argument("--n-permutations", type=int, default=20_000)
    build.add_argument("--seed", type=int, default=20260621)
    build.add_argument("--output-dir", type=Path, required=True)
    build.set_defaults(func=build_program_inputs)

    summary = subparsers.add_parser("summarize")
    summary.add_argument(
        "--program-input-dir",
        type=Path,
        required=True,
    )
    summary.add_argument(
        "--magma-results-dir",
        type=Path,
        required=True,
    )
    summary.add_argument(
        "--human-validation",
        type=Path,
        default=None,
    )
    summary.add_argument(
        "--top-driver-genes",
        type=int,
        default=25,
    )
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.set_defaults(func=summarize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
