#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 6) {
    stop(
        "Usage: run_magma_gene_analysis.R ",
        "<formatted_gwas> <genome_build> <sample_size> ",
        "<storage_dir> <manifest_json> <force_new>"
    )
}

formatted_gwas <- normalizePath(args[[1]])
genome_build <- args[[2]]
sample_size <- as.numeric(args[[3]])
storage_dir <- normalizePath(
    args[[4]],
    mustWork = FALSE
)
manifest_json <- args[[5]]
force_new <- tolower(args[[6]]) %in% c(
    "1",
    "true",
    "yes"
)

dir.create(
    storage_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

suppressPackageStartupMessages({
    library(MAGMA.Celltyping)
    library(jsonlite)
})

cat("Formatted GWAS:", formatted_gwas, "\n")
cat("Genome build:", genome_build, "\n")
cat("Sample size:", sample_size, "\n")
cat("MAGMA storage:", storage_dir, "\n")

genes_out <- MAGMA.Celltyping::map_snps_to_genes(
    path_formatted = formatted_gwas,
    genome_build = genome_build,
    upstream_kb = 35,
    downstream_kb = 10,
    N = sample_size,
    duplicate = "drop",
    synonym_dup = "skip",
    genome_ref_path = NULL,
    population = "eur",
    genes_only = FALSE,
    storage_dir = storage_dir,
    force_new = force_new,
    version = NULL,
    verbose = TRUE
)

genes_out <- normalizePath(genes_out)
genes_raw <- sub(
    "\\.genes\\.out$",
    ".genes.raw",
    genes_out
)

if (!file.exists(genes_out)) {
    stop("MAGMA genes.out was not created: ", genes_out)
}
if (!file.exists(genes_raw)) {
    stop(
        "MAGMA genes.raw was not created. ",
        "Ensure genes_only=FALSE. Expected: ",
        genes_raw
    )
}

payload <- list(
    formatted_gwas = formatted_gwas,
    genome_build = genome_build,
    sample_size = sample_size,
    genes_out = genes_out,
    genes_raw = normalizePath(genes_raw),
    storage_dir = storage_dir,
    upstream_kb = 35,
    downstream_kb = 10,
    population = "eur"
)

write_json(
    payload,
    manifest_json,
    pretty = TRUE,
    auto_unbox = TRUE
)

cat("MAGMA gene analysis completed.\n")
cat("genes.out:", genes_out, "\n")
cat("genes.raw:", genes_raw, "\n")
cat("manifest:", manifest_json, "\n")
