#!/usr/bin/env Rscript

# Correct MAGMA v1.10 gene-property syntax:
#
#   --gene-covar <COVARIATE_FILE> include=...
#
# Do not prepend "file=" to the path. MAGMA v1.10 otherwise treats the
# literal string "file=/path/..." as the filename.

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 8) {
    stop(
        "Usage: run_magma_program_tests_fixed.R ",
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

required_files <- c(
    genes_raw,
    primary_covar,
    mhc_covar,
    primary_top50,
    primary_top300,
    mhc_top50,
    mhc_top300
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files)) {
    stop(
        "Required MAGMA program-test input files are missing:\n",
        paste(missing_files, collapse = "\n")
    )
}

# Validate the two continuous covariate files before calling MAGMA.
validate_covar <- function(path) {
    header <- scan(
        path,
        what = character(),
        nlines = 1L,
        quiet = TRUE
    )
    required <- c(
        "GENE",
        "mean_logCPM",
        paste0("P", 1:30)
    )
    absent <- setdiff(required, header)
    if (length(absent)) {
        stop(
            path,
            " lacks required columns: ",
            paste(absent, collapse = ", ")
        )
    }
    cat(
        "Validated covariate file:",
        path,
        "\n  columns:",
        length(header),
        "\n"
    )
}

validate_covar(primary_covar)
validate_covar(mhc_covar)

magma <- Sys.which("magma")
if (magma == "") {
    stop("The magma executable is not on PATH.")
}
magma <- normalizePath(magma)

cat("MAGMA executable:", magma, "\n")
system2(magma, "--version")

programs <- paste0("P", 1:30)
include_modifier <- paste0(
    "include=",
    paste(programs, collapse = ",")
)

run_magma <- function(arguments, output_prefix) {
    output_prefix <- normalizePath(
        output_prefix,
        mustWork = FALSE
    )

    # Remove only outputs for this exact prefix so a previous failed attempt
    # cannot be mistaken for a completed analysis.
    stale <- Sys.glob(paste0(output_prefix, ".*"))
    if (length(stale)) {
        file.remove(stale)
    }

    command <- c(
        arguments,
        "--out",
        output_prefix
    )

    cat(
        "\nRunning:\n ",
        magma,
        " ",
        paste(command, collapse = " "),
        "\n",
        sep = ""
    )

    status <- system2(
        magma,
        args = command,
        stdout = "",
        stderr = ""
    )
    if (!identical(status, 0L)) {
        log_file <- paste0(output_prefix, ".log")
        if (file.exists(log_file)) {
            cat(
                "\n----- MAGMA log tail -----\n",
                paste(
                    tail(readLines(log_file, warn = FALSE), 120),
                    collapse = "\n"
                ),
                "\n----- end MAGMA log -----\n"
            )
        }
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
    if (!file.exists(output_file) || file.info(output_file)$size == 0) {
        stop(
            "Expected MAGMA output was not created: ",
            output_file
        )
    }

    cat("Created:", output_file, "\n")
}

# 1. Continuous loading/property test, targeted genes and MHC excluded.
run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        primary_covar,
        include_modifier
    ),
    file.path(
        output_dir,
        "continuous_primary"
    )
)

# 2. Same loading/property tests while conditioning on human PAEC expression.
run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        primary_covar,
        include_modifier,
        "condition=mean_logCPM"
    ),
    file.path(
        output_dir,
        "continuous_primary_expr_conditioned"
    )
)

# 3-4. Competitive categorical gene-set sensitivities.
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

# 5. MHC-included continuous sensitivity, PAEC-expression conditioned.
run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        mhc_covar,
        include_modifier,
        "condition=mean_logCPM"
    ),
    file.path(
        output_dir,
        "continuous_MHC_included"
    )
)

# 6-7. MHC-included categorical sensitivities.
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

required_outputs <- file.path(
    output_dir,
    paste0(
        c(
            "continuous_primary",
            "continuous_primary_expr_conditioned",
            "top50_primary",
            "top300_primary",
            "continuous_MHC_included",
            "top50_MHC_included",
            "top300_MHC_included"
        ),
        ".gsa.out"
    )
)
missing_outputs <- required_outputs[
    !file.exists(required_outputs)
    | file.info(required_outputs)$size == 0
]
if (length(missing_outputs)) {
    stop(
        "MAGMA program tests finished but outputs are missing:\n",
        paste(missing_outputs, collapse = "\n")
    )
}

cat("\nAll MAGMA continuous and categorical program tests completed.\n")
