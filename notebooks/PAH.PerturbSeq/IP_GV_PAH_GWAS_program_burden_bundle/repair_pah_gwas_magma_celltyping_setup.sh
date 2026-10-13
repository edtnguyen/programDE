#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

MAMBA="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/bin/mamba"
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_gwas_program_burden_20260621"
TOOLS="/oak/stanford/groups/engreitz/Users/tri/Tools"
MAGMA_CELLTYPING_REPO="${TOOLS}/MAGMA_Celltyping"
MAGMA_INSTALL_DIR="${TOOLS}/MAGMA"

[[ -x "${MAMBA}" ]] || {
    echo "mamba was not found: ${MAMBA}" >&2
    exit 1
}
[[ -x "${ENV}/bin/R" ]] || {
    echo "The partially created environment is missing R: ${ENV}" >&2
    echo "Run the corrected full setup script instead." >&2
    exit 1
}
[[ -f "${MAGMA_CELLTYPING_REPO}/DESCRIPTION" ]] || {
    echo "MAGMA.Celltyping checkout is missing: ${MAGMA_CELLTYPING_REPO}" >&2
    exit 1
}

source "${CONDA_SH}"

echo "============================================================"
echo "Repair MAGMA.Celltyping dependencies"
echo "Environment: ${ENV}"
echo "Repository:  ${MAGMA_CELLTYPING_REPO}"
echo "============================================================"

echo
echo "=== 1. Install compiled and Bioconductor dependencies with mamba ==="
"${MAMBA}" install -y -p "${ENV}" \
    --override-channels \
    -c conda-forge \
    -c bioconda \
    r-lme4 \
    r-xml \
    r-httr \
    r-ggdendro \
    r-patchwork \
    r-reshape2 \
    r-gh \
    r-lifecycle \
    r-rlang \
    bioconductor-limma \
    bioconductor-orthogene=1.16.1 \
    bioconductor-ewce=1.18.0 \
    bioconductor-ewcedata=1.18.0

conda activate "${ENV}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true
export R_LIBS_USER="${ENV}/lib/R/library"

echo
echo "=== 2. Verify every required MAGMA.Celltyping import ==="
Rscript - "${MAGMA_CELLTYPING_REPO}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
repo <- normalizePath(args[[1]])

description <- read.dcf(file.path(repo, "DESCRIPTION"))
imports <- trimws(unlist(strsplit(description[1, "Imports"], ",")))
imports <- sub("\\s*\\(.*\\)$", "", imports)
imports <- unique(imports[nzchar(imports)])

missing <- imports[
    !vapply(
        imports,
        requireNamespace,
        logical(1),
        quietly = TRUE
    )
]

cat("Required Imports:\n")
for (package in imports) {
    status <- if (package %in% missing) "MISSING" else "OK"
    version <- if (
        package %in% missing
    ) {
        ""
    } else {
        paste0(" ", as.character(packageVersion(package)))
    }
    cat(sprintf("  %-22s %s%s\n", package, status, version))
}

if (length(missing)) {
    stop(
        "Missing MAGMA.Celltyping dependencies: ",
        paste(missing, collapse = ", ")
    )
}
RS

echo
echo "=== 3. Install MAGMA.Celltyping without dependency re-resolution ==="
Rscript - "${MAGMA_CELLTYPING_REPO}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
repo <- normalizePath(args[[1]])

if (!requireNamespace("remotes", quietly = TRUE)) {
    stop("The R package remotes is not installed.")
}

remotes::install_local(
    repo,
    dependencies = FALSE,
    upgrade = "never",
    force = TRUE,
    quiet = FALSE
)

suppressPackageStartupMessages(library(MAGMA.Celltyping))
cat(
    "MAGMA.Celltyping:",
    as.character(packageVersion("MAGMA.Celltyping")),
    "\n"
)
RS

echo
echo "=== 4. Install or reuse the official MAGMA executable ==="
mkdir -p "${MAGMA_INSTALL_DIR}"

MAGMA_PATH="$(
    Rscript - "${MAGMA_INSTALL_DIR}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
destination <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(destination, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(MAGMA.Celltyping))

path <- MAGMA.Celltyping::magma_install(
    dest_dir = destination,
    desired_version = "latest",
    upgrade = FALSE,
    verbose = TRUE
)

cat(normalizePath(path), "\n", sep = "")
RS
)"

MAGMA_PATH="$(printf '%s\n' "${MAGMA_PATH}" | tail -n 1)"
[[ -x "${MAGMA_PATH}" ]] || {
    echo "MAGMA installation did not yield an executable: ${MAGMA_PATH}" >&2
    exit 1
}

ln -sfn "${MAGMA_PATH}" "${ENV}/bin/magma"
printf '%s\n' "${MAGMA_PATH}" > "${MAGMA_INSTALL_DIR}/magma_path.txt"

echo
echo "=== 5. Final validation ==="
Rscript - <<'RS'
suppressPackageStartupMessages({
    library(MAGMA.Celltyping)
    library(MungeSumstats)
    library(EWCE)
    library(orthogene)
    library(lme4)
})

cat("R:", R.version.string, "\n")
cat("MAGMA.Celltyping:", as.character(packageVersion("MAGMA.Celltyping")), "\n")
cat("MungeSumstats:", as.character(packageVersion("MungeSumstats")), "\n")
cat("EWCE:", as.character(packageVersion("EWCE")), "\n")
cat("orthogene:", as.character(packageVersion("orthogene")), "\n")
cat("lme4:", as.character(packageVersion("lme4")), "\n")
RS

magma --version

cat > "${MAGMA_INSTALL_DIR}/INSTALLATION_COMPLETE.txt" <<EOF
PAH GWAS program-burden environment repair completed.
date=$(date --iso-8601=seconds)
environment=${ENV}
MAGMA_Celltyping_repo=${MAGMA_CELLTYPING_REPO}
MAGMA_Celltyping_commit=$(cd "${MAGMA_CELLTYPING_REPO}" && git rev-parse HEAD)
magma=${MAGMA_PATH}
EOF

echo
echo "PAH GWAS MAGMA SETUP REPAIR PASSED"
echo "Environment: ${ENV}"
echo "MAGMA:      ${MAGMA_PATH}"
