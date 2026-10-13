#!/usr/bin/env Rscript

# MAGMA v1.10-compatible gene-property and gene-set tests.
#
# Important syntax:
#   --gene-covar <FILE> filter-read=P1,P2,...
#   --model direction-covar=greater
#
# MAGMA v1.10 rejects the older/ambiguous "include=" modifier for
# --gene-covar. The v1.09+ manual documents filter-read/filter-skip for
# selecting columns during input and condition-hide under --model.

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 8) {
    stop(
        "Usage: run_magma_program_tests_v2.R ",
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
missing_files <- required_files[
    !file.exists(required_files)
    | file.info(required_files)$size == 0
]
if (length(missing_files)) {
    stop(
        "Required MAGMA program-test inputs are missing or empty:\n",
        paste(missing_files, collapse = "\n")
    )
}

programs <- paste0("P", 1:30)
program_filter <- paste0(
    "filter-read=",
    paste(programs, collapse = ",")
)
expression_program_filter <- paste0(
    "filter-read=mean_logCPM,",
    paste(programs, collapse = ",")
)

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
        programs
    )
    absent <- setdiff(required, header)
    if (length(absent)) {
        stop(
            path,
            " lacks required columns: ",
            paste(absent, collapse = ", ")
        )
    }

    # Confirm every data row has exactly the same number of fields.
    connection <- file(path, open = "rt")
    on.exit(close(connection), add = TRUE)
    lines <- readLines(connection, warn = FALSE)
    if (length(lines) < 1000) {
        stop(path, " has unexpectedly few rows: ", length(lines))
    }
    field_counts <- lengths(strsplit(lines, "[[:space:]]+"))
    expected <- length(header)
    bad <- which(field_counts != expected)
    if (length(bad)) {
        stop(
            path,
            " contains rows with inconsistent field counts. ",
            "Expected ",
            expected,
            "; first bad line: ",
            bad[[1]],
            " has ",
            field_counts[bad[[1]]],
            "."
        )
    }

    cat(
        "Validated covariate file:",
        path,
        "\n  columns:",
        length(header),
        "\n  data rows:",
        length(lines) - 1L,
        "\n"
    )
}

validate_set_file <- function(path) {
    lines <- readLines(path, warn = FALSE)
    lines <- lines[nzchar(trimws(lines))]
    if (length(lines) < 25) {
        stop(
            path,
            " has unexpectedly few non-empty gene sets: ",
            length(lines)
        )
    }
    fields <- strsplit(lines, "[[:space:]]+")
    set_names <- vapply(fields, `[[`, character(1), 1L)
    missing_programs <- setdiff(programs, set_names)
    if (length(missing_programs)) {
        stop(
            path,
            " lacks program sets: ",
            paste(missing_programs, collapse = ", ")
        )
    }
    cat(
        "Validated set file:",
        path,
        "\n  gene sets:",
        length(lines),
        "\n"
    )
}

validate_covar(primary_covar)
validate_covar(mhc_covar)
validate_set_file(primary_top50)
validate_set_file(primary_top300)
validate_set_file(mhc_top50)
validate_set_file(mhc_top300)

magma <- Sys.which("magma")
if (magma == "") {
    stop("The magma executable is not on PATH.")
}
magma <- normalizePath(magma)

cat("MAGMA executable:", magma, "\n")
system2(magma, "--version")

log_tail <- function(prefix, n = 160L) {
    log_file <- paste0(prefix, ".log")
    if (file.exists(log_file)) {
        cat(
            "\n----- MAGMA log tail -----\n",
            paste(
                tail(
                    readLines(log_file, warn = FALSE),
                    n
                ),
                collapse = "\n"
            ),
            "\n----- end MAGMA log -----\n",
            sep = ""
        )
    }
}

remove_prefix_outputs <- function(prefix) {
    stale <- Sys.glob(paste0(prefix, ".*"))
    if (length(stale)) {
        unlink(stale, force = TRUE)
    }
}

run_magma <- function(
    arguments,
    output_prefix,
    expected_suffix = ".gsa.out"
) {
    output_prefix <- normalizePath(
        output_prefix,
        mustWork = FALSE
    )
    remove_prefix_outputs(output_prefix)

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
        log_tail(output_prefix)
        stop(
            "MAGMA exited with status ",
            status,
            " for output prefix ",
            output_prefix
        )
    }

    output_file <- paste0(
        output_prefix,
        expected_suffix
    )
    if (
        !file.exists(output_file)
        || file.info(output_file)$size == 0
    ) {
        log_tail(output_prefix)
        stop(
            "Expected MAGMA output was not created: ",
            output_file
        )
    }

    cat("Created:", output_file, "\n")
    invisible(output_file)
}

# -------------------------------------------------------------------------
# Syntax smoke tests. These are intentionally small single-program tests.
# They fail before any long analysis if the installed MAGMA syntax differs.
# -------------------------------------------------------------------------
smoke_dir <- file.path(output_dir, "_syntax_smoke_test")
dir.create(smoke_dir, recursive = TRUE, showWarnings = FALSE)

cat("\n=== MAGMA v1.10 gene-property syntax smoke tests ===\n")

run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        primary_covar,
        "filter-read=P1",
        "--model",
        "direction-covar=greater"
    ),
    file.path(smoke_dir, "unconditioned_P1")
)

run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        primary_covar,
        "filter-read=mean_logCPM,P1",
        "--model",
        "direction-covar=greater",
        "condition-hide=mean_logCPM"
    ),
    file.path(smoke_dir, "expression_conditioned_P1")
)

cat("MAGMA syntax smoke tests passed.\n")

# -------------------------------------------------------------------------
# Full analyses.
# -------------------------------------------------------------------------

# 1. Continuous loading/gene-property test, targeted genes and MHC excluded.
run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        primary_covar,
        program_filter,
        "--model",
        "direction-covar=greater"
    ),
    file.path(
        output_dir,
        "continuous_primary"
    )
)

# 2. Same program-loading tests conditioned on mean human PAEC expression.
run_magma(
    c(
        "--gene-results",
        genes_raw,
        "--gene-covar",
        primary_covar,
        expression_program_filter,
        "--model",
        "direction-covar=greater",
        "condition-hide=mean_logCPM"
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
        expression_program_filter,
        "--model",
        "direction-covar=greater",
        "condition-hide=mean_logCPM"
    ),
    file.path(
        output_dir,
        "continuous_MHC_included"
    )
)

# 6-7. MHC-included competitive gene-set sensitivities.
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

cat("\nAll MAGMA v1.10 program tests completed.\n")
