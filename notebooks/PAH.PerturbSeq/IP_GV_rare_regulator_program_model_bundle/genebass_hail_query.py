#!/usr/bin/env python3
"""Discover and extract public Genebass gene-burden results with Hail.

The public Google bucket is requester-pays. Provide a Google Cloud billing
project using --gcp-project. No UK Biobank individual-level application is
needed for these public summary results, but Google may charge small egress or
query costs.

Recommended sequence
--------------------
1. discover-phenotypes
2. inspect matching_phenotypes.tsv and schema files
3. extract with explicit --phenotype-field and --phenotype-value
4. normalize the exported TSV using rare_regulator_program_model.py import-portal
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import re

import hail as hl
import pandas as pd


PHENO_HT = "gs://ukbb-exome-public/500k/results/pheno_results.ht"
RESULTS_MT = "gs://ukbb-exome-public/500k/results/results.mt"


def hail_init(args):
    # Set both current and legacy requester-pays keys because Sherlock's
    # installed GCS connector version may differ between clusters.
    spark_conf = {
        "spark.hadoop.fs.gs.requester.pays.mode": "CUSTOM",
        "spark.hadoop.fs.gs.requester.pays.buckets": "ukbb-exome-public",
        "spark.hadoop.fs.gs.requester.pays.project.id": args.gcp_project,
        "spark.hadoop.fs.gs.requester.pays.enable": "true",
        "spark.hadoop.fs.gs.requester.pays.billing.project.id": args.gcp_project,
    }
    hl.init(
        log=str(args.output_dir / "hail.log"),
        tmp_dir=str(args.tmp_dir),
        spark_conf=spark_conf,
        quiet=False,
    )


def discover(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    hail_init(args)

    table = hl.read_table(PHENO_HT)
    schema_path = args.output_dir / "phenotype_schema.txt"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        table.describe()
    schema_path.write_text(buffer.getvalue())

    flattened = table.flatten()
    pandas_table = flattened.to_pandas()
    pandas_table.to_csv(
        args.output_dir / "all_phenotypes.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    string_columns = [
        column
        for column in pandas_table.columns
        if pandas_table[column].dtype == object
    ]
    pattern = re.compile(args.keywords, flags=re.I)
    match = pd.Series(False, index=pandas_table.index)
    for column in string_columns:
        match |= pandas_table[column].astype(str).str.contains(
            pattern,
            na=False,
        )
    matching = pandas_table.loc[match].copy()
    matching.to_csv(
        args.output_dir / "matching_phenotypes.tsv",
        sep="\t",
        index=False,
    )
    print(matching.to_string(index=False))


def get_path_expression(root, path: str):
    expression = root
    for part in path.split("."):
        expression = expression[part]
    return expression


def extract(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    hail_init(args)

    genes = {
        line.strip().upper()
        for line in args.genes.read_text().splitlines()
        if line.strip()
    }
    if not genes:
        raise ValueError("Gene list is empty.")

    matrix = hl.read_matrix_table(RESULTS_MT)
    schema_buffer = io.StringIO()
    with contextlib.redirect_stdout(schema_buffer):
        matrix.describe()
    (args.output_dir / "results_matrix_schema.txt").write_text(
        schema_buffer.getvalue()
    )

    gene_expression = get_path_expression(
        matrix.row,
        args.gene_field,
    )
    phenotype_expression = get_path_expression(
        matrix.col,
        args.phenotype_field,
    )

    matrix = matrix.filter_rows(
        hl.literal(genes).contains(
            hl.upper(hl.str(gene_expression))
        )
    )
    matrix = matrix.filter_cols(
        hl.str(phenotype_expression) == args.phenotype_value
    )

    n_rows = matrix.count_rows()
    n_cols = matrix.count_cols()
    if n_rows == 0 or n_cols == 0:
        raise RuntimeError(
            f"No entries after filtering: rows={n_rows}, cols={n_cols}. "
            "Inspect schema and phenotype matches."
        )

    entries = matrix.entries().flatten()
    output = args.output_dir / "genebass_raw_entries.tsv.bgz"
    entries.export(str(output))

    manifest = {
        "genes": sorted(genes),
        "gene_field": args.gene_field,
        "phenotype_field": args.phenotype_field,
        "phenotype_value": args.phenotype_value,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "output": str(output),
    }
    (args.output_dir / "GENEBASS_EXTRACTION_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcp-project", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discovery = subparsers.add_parser("discover-phenotypes")
    discovery.add_argument(
        "--keywords",
        default=r"pulmonary hypertension|primary pulmonary|I27\.0|415\.21",
    )
    discovery.set_defaults(func=discover)

    extraction = subparsers.add_parser("extract")
    extraction.add_argument("--genes", type=Path, required=True)
    extraction.add_argument("--gene-field", required=True)
    extraction.add_argument("--phenotype-field", required=True)
    extraction.add_argument("--phenotype-value", required=True)
    extraction.set_defaults(func=extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
