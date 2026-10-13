#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 8) {
    stop(
        "Usage: run_magma_program_tests.R ",
        "<genes_raw> <primary_covar> <mhc_covar> ",
        "<primary_top50> <primary_top300> ",
        "<mhc_top50> <mhc_top300> <output_dir>"
    )
}

genes_raw <- normalizePath(args[[1]])
primary_covar <- normalizePath(args[[2]])
mhc_covar <- normalizePath(args[[3]])
primary_top50 <- normalizePath(args[[4]])
primary_top300 <- normalizePath(args[[5]])
mhc_top50 <- normalizePath(args[[6]])
mhc_top300 <- normalizePath(args[[7]])
output_dir <- normalizePath(
    args[[8]],
    mustWork = FALSE
)

dir.create(
    output_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

magma <- Sys.which("magma")
if (magma == "") {
    stop("magma executable is not on PATH.")
}

programs <- paste0("P", 1:30)
include_modifier <- paste0(
    "include=",
    paste(programs, collapse = ",")
)

run_magma <- function(arguments, output_prefix) {
    command <- c(
        arguments,
        "--out",
        output_prefix
    )

    cat(
        "\nRunning:\n",
        magma,
        paste(command, collapse = " "),
        "\n"
    )

    status <- system2(
        magma,
        args = command,
        stdout = "",
        stderr = ""
    )
    if (status != 0) {
        stop(
            "MAGMA exited with status ",
            status,
            " for output prefix ",
            output_prefix
        )
    }

    output_file <- paste0(
        output_prefix,
        ".gsa.out"
    )
    if (!file.exists(output_file)) {
        stop(
            "Expected MAGMA output was not created: ",
            output_file
        )
    }
}

run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        paste0("file=", primary_covar),
        include_modifier
    ),
    file.path(
        output_dir,
        "continuous_primary"
    )
)

run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        paste0("file=", primary_covar),
        include_modifier,
        "condition=mean_logCPM"
    ),
    file.path(
        output_dir,
        "continuous_primary_expr_conditioned"
    )
)

run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--set-annot",
        primary_top50
    ),
    file.path(
        output_dir,
        "top50_primary"
    )
)

run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--set-annot",
        primary_top300
    ),
    file.path(
        output_dir,
        "top300_primary"
    )
)

run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        paste0("file=", mhc_covar),
        include_modifier,
        "condition=mean_logCPM"
    ),
    file.path(
        output_dir,
        "continuous_MHC_included"
    )
)

run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--set-annot",
        mhc_top50
    ),
    file.path(
        output_dir,
        "top50_MHC_included"
    )
)

run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--set-annot",
        mhc_top300
    ),
    file.path(
        output_dir,
        "top300_MHC_included"
    )
)

cat("\nAll MAGMA program tests completed.\n")
