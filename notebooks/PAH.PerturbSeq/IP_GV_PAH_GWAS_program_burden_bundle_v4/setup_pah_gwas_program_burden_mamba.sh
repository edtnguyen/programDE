#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

MAMBA="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/bin/mamba"
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_gwas_program_burden_20260621"
TOOLS="/oak/stanford/groups/engreitz/Users/tri/Tools"
MAGMA_CELLTYPING_REPO="${TOOLS}/MAGMA_Celltyping"
MAGMA_CELLTYPING_URL="https://github.com/neurogenomics/MAGMA_Celltyping.git"
MAGMA_INSTALL_DIR="${TOOLS}/MAGMA"

[[ -x "${MAMBA}" ]] || {
    echo "mamba was not found: ${MAMBA}" >&2
    exit 1
}

source "${CONDA_SH}"
mkdir -p "${TOOLS}" "${MAGMA_INSTALL_DIR}"

echo "Using mamba: ${MAMBA}"
"${MAMBA}" --version

echo
echo "=== 1. Create or repair a user-owned GWAS/MAGMA environment ==="
PACKAGES=(
    python=3.10
    numpy=1.26.4
    scipy=1.12
    pandas=2.2
    scikit-learn=1.5
    statsmodels
    matplotlib
    openpyxl
    anndata=0.10.9
    h5py
    requests
    beautifulsoup4
    lxml
    pyyaml
    pyarrow
    tqdm
    pip
    setuptools
    wheel
    git
    curl
    wget
    unzip
    r-base=4.5
    r-remotes
    r-biocmanager
    r-jsonlite
    r-data.table
    r-dplyr
    r-readr
    r-tibble
    r-stringr
    r-r.utils
    r-rcurl
    r-yaml
    r-ggplot2
    r-matrix
    r-lme4
    r-xml
    r-httr
    r-ggdendro
    r-patchwork
    r-reshape2
    r-gh
    r-lifecycle
    r-rlang
    bioconductor-limma
    bioconductor-mungesumstats
    bioconductor-orthogene=1.16.1
    bioconductor-ewce=1.18.0
    bioconductor-ewcedata=1.18.0
)

if [[ ! -x "${ENV}/bin/python" ]]; then
    "${MAMBA}" create -y -p "${ENV}" \
        --override-channels \
        -c conda-forge \
        -c bioconda \
        "${PACKAGES[@]}"
else
    echo "Environment already exists: ${ENV}"
    "${MAMBA}" install -y -p "${ENV}" \
        --override-channels \
        -c conda-forge \
        -c bioconda \
        "${PACKAGES[@]}"
fi

conda activate "${ENV}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true
export R_LIBS_USER="${ENV}/lib/R/library"

echo
echo "=== 2. Clone or reuse MAGMA.Celltyping ==="
if [[ ! -d "${MAGMA_CELLTYPING_REPO}/.git" ]]; then
    if [[ -e "${MAGMA_CELLTYPING_REPO}" ]]; then
        rm -rf -- "${MAGMA_CELLTYPING_REPO}"
    fi
    git clone "${MAGMA_CELLTYPING_URL}" "${MAGMA_CELLTYPING_REPO}"
else
    echo "Reusing existing checkout: ${MAGMA_CELLTYPING_REPO}"
fi

(
    cd "${MAGMA_CELLTYPING_REPO}"
    echo "MAGMA.Celltyping commit: $(git rev-parse HEAD)"
    grep -E '^(Package|Version):' DESCRIPTION
)

echo
echo "=== 3. Verify required R imports before installing the package ==="
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
echo "=== 4. Install MAGMA.Celltyping without re-solving dependencies ==="
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
echo "=== 5. Register the official MAGMA executable ==="
SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"
bash "${SCRIPT_DIR}/repair_pah_gwas_magma_executable_registration.sh"

