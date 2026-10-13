#!/usr/bin/env Rscript

# Run the MAGMA SNP-to-gene and gene-level steps without using
# MAGMA.Celltyping's GitHub-release/piggyback gene-location downloader.
#
# MAGMA.Celltyping 2.0.16 obtains NCBI*.gene.loc through piggyback and requires
# a GitHub PAT even though the same files are public in its pinned repository.
# This script downloads the pinned public file directly, reuses the already
# cached 1000 Genomes reference, and invokes the MAGMA executable itself.

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 6) {
    stop(
        "Usage: run_magma_gene_analysis_no_piggyback.R ",
        "<formatted_gwas> <genome_build> <sample_size> ",
        "<storage_dir> <manifest_json> <force_new>"
    )
}

formatted_gwas <- normalizePath(args[[1]])
genome_build <- toupper(args[[2]])
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

if (!is.finite(sample_size) || sample_size < 1000) {
    stop("sample_size must be a finite number >=1000.")
}

dir.create(
    storage_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

suppressPackageStartupMessages({
    library(jsonlite)
    library(MAGMA.Celltyping)
    library(R.utils)
})

magma <- Sys.which("magma")
if (magma == "") {
    # Fall back to the package's validated executable lookup.
    magma_candidates <- MAGMA.Celltyping:::magma_executable(
        return_all = TRUE,
        verbose = TRUE
    )
    if (length(magma_candidates) == 0L) {
        stop("No MAGMA executable is available.")
    }
    magma <- magma_candidates[[1]]
}
magma <- normalizePath(magma)
cat("MAGMA executable:", magma, "\n")
system2(magma, "--version")

# -------------------------------------------------------------------------
# 1. Obtain the public, build-specific MAGMA gene-location file without
#    piggyback or a GitHub PAT.
# -------------------------------------------------------------------------
build_dictionary <- c(
    GRCH36 = "NCBI36.3.gene.loc",
    GRCH37 = "NCBI37.3.gene.loc",
    HG37 = "NCBI37.3.gene.loc",
    HG19 = "NCBI37.3.gene.loc",
    GRCH38 = "NCBI38.gene.loc",
    HG38 = "NCBI38.gene.loc"
)

if (!genome_build %in% names(build_dictionary)) {
    stop(
        "Unsupported genome build: ",
        genome_build,
        ". Expected GRCh37/hg19 or GRCh38/hg38."
    )
}

gene_location_name <- unname(build_dictionary[[genome_build]])
gene_location_path <- file.path(
    storage_dir,
    gene_location_name
)

# This is the immutable permalink documented in MAGMA.Celltyping's own
# get_genomeLocFile.R source.
gene_location_commit <- "f931877dedcde103723f14242d5242fdca7b3af6"
gene_location_url <- paste0(
    "https://raw.githubusercontent.com/",
    "neurogenomics/MAGMA_Celltyping/",
    gene_location_commit,
    "/data/",
    gene_location_name
)

download_gene_location <- function() {
    temporary <- paste0(
        gene_location_path,
        ".part"
    )
    if (file.exists(temporary)) {
        file.remove(temporary)
    }

    old_timeout <- getOption("timeout")
    on.exit(
        options(timeout = old_timeout),
        add = TRUE
    )
    options(timeout = max(600, old_timeout))

    utils::download.file(
        url = gene_location_url,
        destfile = temporary,
        mode = "wb",
        quiet = FALSE
    )

    if (!file.exists(temporary)) {
        stop(
            "The gene-location download did not create: ",
            temporary
        )
    }
    if (file.info(temporary)$size < 100000) {
        stop(
            "The downloaded gene-location file is unexpectedly small: ",
            temporary
        )
    }

    preview <- utils::read.table(
        temporary,
        sep = "\t",
        header = FALSE,
        nrows = 10,
        quote = "",
        comment.char = "",
        stringsAsFactors = FALSE
    )
    if (ncol(preview) != 6L) {
        stop(
            "Expected six columns in ",
            gene_location_name,
            "; observed ",
            ncol(preview),
            "."
        )
    }

    if (!file.rename(temporary, gene_location_path)) {
        stop(
            "Could not move ",
            temporary,
            " to ",
            gene_location_path
        )
    }
}

if (
    force_new
    || !file.exists(gene_location_path)
    || file.info(gene_location_path)$size < 100000
) {
    cat(
        "Downloading public MAGMA gene-location file:\n  ",
        gene_location_url,
        "\n",
        sep = ""
    )
    download_gene_location()
} else {
    cat(
        "Reusing gene-location file:",
        gene_location_path,
        "\n"
    )
}

gene_location_path <- normalizePath(gene_location_path)

# -------------------------------------------------------------------------
# 2. Reuse the European 1000 Genomes reference downloaded by the failed run,
#    or let MAGMA.Celltyping download it from the MAGMA server if absent.
# -------------------------------------------------------------------------
genome_reference_prefix <- file.path(
    storage_dir,
    "g1000_eur",
    "g1000_eur"
)

required_reference_files <- paste0(
    genome_reference_prefix,
    c(".bed", ".bim", ".fam")
)

if (!all(file.exists(required_reference_files))) {
    cat(
        "European 1000 Genomes reference is incomplete; ",
        "downloading/recovering it.\n",
        sep = ""
    )
    genome_reference_prefix <- MAGMA.Celltyping::get_genome_ref(
        genome_ref_path = genome_reference_prefix,
        storage_dir = storage_dir,
        population = "eur",
        timeout = 60 * 30,
        verbose = TRUE
    )
}

required_reference_files <- paste0(
    genome_reference_prefix,
    c(".bed", ".bim", ".fam")
)
if (!all(file.exists(required_reference_files))) {
    stop(
        "The 1000 Genomes reference is incomplete. Missing:\n",
        paste(
            required_reference_files[
                !file.exists(required_reference_files)
            ],
            collapse = "\n"
        )
    )
}
genome_reference_prefix <- normalizePath(
    genome_reference_prefix,
    mustWork = FALSE
)

# -------------------------------------------------------------------------
# 3. MAGMA requires a decompressed SNP-location/p-value file.
# -------------------------------------------------------------------------
formatted_uncompressed <- if (
    grepl("\\.(gz|bgz)$", formatted_gwas, ignore.case = TRUE)
) {
    sub(
        "\\.(gz|bgz)$",
        "",
        file.path(
            storage_dir,
            basename(formatted_gwas)
        ),
        ignore.case = TRUE
    )
} else {
    formatted_gwas
}

if (
    grepl("\\.(gz|bgz)$", formatted_gwas, ignore.case = TRUE)
    && (
        force_new
        || !file.exists(formatted_uncompressed)
        || file.info(formatted_uncompressed)$size == 0
    )
) {
    cat(
        "Decompressing formatted GWAS to:",
        formatted_uncompressed,
        "\n"
    )
    R.utils::gunzip(
        filename = formatted_gwas,
        destname = formatted_uncompressed,
        overwrite = TRUE,
        remove = FALSE
    )
}
formatted_uncompressed <- normalizePath(formatted_uncompressed)

first_line <- readLines(
    formatted_uncompressed,
    n = 1L
)
headers <- strsplit(
    first_line,
    "\t",
    fixed = TRUE
)[[1]]

required_headers <- c(
    "SNP",
    "CHR",
    "BP",
    "P"
)
missing_headers <- setdiff(
    required_headers,
    headers
)
if (length(missing_headers)) {
    stop(
        "Formatted GWAS lacks required columns: ",
        paste(missing_headers, collapse = ", ")
    )
}

n_argument <- if ("N" %in% headers) {
    "ncol=N"
} else {
    paste0(
        "N=",
        format(
            sample_size,
            scientific = FALSE,
            trim = TRUE
        )
    )
}

# -------------------------------------------------------------------------
# 4. Run MAGMA directly. This reproduces map_snps_to_genes() but avoids its
#    PAT-dependent gene-location helper.
# -------------------------------------------------------------------------
analysis_dir <- file.path(
    storage_dir,
    "direct_magma"
)
dir.create(
    analysis_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

output_prefix <- file.path(
    analysis_dir,
    paste0(
        tools::file_path_sans_ext(
            basename(formatted_uncompressed)
        ),
        ".35UP.10DOWN"
    )
)

genes_annot <- paste0(
    output_prefix,
    ".genes.annot"
)
genes_out <- paste0(
    output_prefix,
    ".genes.out"
)
genes_raw <- paste0(
    output_prefix,
    ".genes.raw"
)

run_magma <- function(arguments, expected_output) {
    cat(
        "\nRunning MAGMA:\n  ",
        magma,
        " ",
        paste(arguments, collapse = " "),
        "\n",
        sep = ""
    )

    status <- system2(
        command = magma,
        args = arguments,
        stdout = "",
        stderr = ""
    )
    if (!identical(status, 0L)) {
        stop(
            "MAGMA exited with status ",
            status,
            "."
        )
    }
    if (!file.exists(expected_output)) {
        stop(
            "MAGMA did not create expected output: ",
            expected_output
        )
    }
}

if (
    force_new
    || !file.exists(genes_annot)
    || file.info(genes_annot)$size == 0
) {
    run_magma(
        arguments = c(
            "--annotate",
            "window=35,10",
            "--snp-loc",
            shQuote(formatted_uncompressed),
            "--gene-loc",
            shQuote(gene_location_path),
            "--out",
            shQuote(output_prefix)
        ),
        expected_output = genes_annot
    )
} else {
    cat(
        "Reusing gene annotation:",
        genes_annot,
        "\n"
    )
}

if (
    force_new
    || !file.exists(genes_out)
    || !file.exists(genes_raw)
    || file.info(genes_out)$size == 0
    || file.info(genes_raw)$size == 0
) {
    run_magma(
        arguments = c(
            "--bfile",
            shQuote(genome_reference_prefix),
            "synonym-dup=skip",
            "--pval",
            shQuote(formatted_uncompressed),
            n_argument,
            "duplicate=drop",
            "--gene-annot",
            shQuote(genes_annot),
            "--out",
            shQuote(output_prefix)
        ),
        expected_output = genes_out
    )
} else {
    cat(
        "Reusing MAGMA gene analysis:",
        genes_out,
        "\n"
    )
}

if (!file.exists(genes_out)) {
    stop("MAGMA genes.out was not created: ", genes_out)
}
if (!file.exists(genes_raw)) {
    stop("MAGMA genes.raw was not created: ", genes_raw)
}

genes_out <- normalizePath(genes_out)
genes_raw <- normalizePath(genes_raw)

payload <- list(
    formatted_gwas = formatted_gwas,
    formatted_gwas_uncompressed = formatted_uncompressed,
    genome_build = genome_build,
    sample_size = sample_size,
    genes_out = genes_out,
    genes_raw = genes_raw,
    storage_dir = storage_dir,
    upstream_kb = 35,
    downstream_kb = 10,
    population = "eur",
    genome_reference_prefix = genome_reference_prefix,
    gene_location_file = gene_location_path,
    gene_location_url = gene_location_url,
    gene_location_commit = gene_location_commit,
    implementation = "direct_MAGMA_without_piggyback"
)

write_json(
    payload,
    manifest_json,
    pretty = TRUE,
    auto_unbox = TRUE
)

cat("\nMAGMA gene analysis completed.\n")
cat("genes.out:", genes_out, "\n")
cat("genes.raw:", genes_raw, "\n")
cat("manifest:", manifest_json, "\n")
