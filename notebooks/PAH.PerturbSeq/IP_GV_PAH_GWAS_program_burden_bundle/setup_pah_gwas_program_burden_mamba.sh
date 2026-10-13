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
    bioconductor-mungesumstats
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
echo "=== 2. Clone and install MAGMA.Celltyping ==="
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

    # Prefer the documented 2.0.16 release tag when it exists. Otherwise,
    # retain the checked-out commit and record it below.
    if git show-ref --tags --verify --quiet refs/tags/v2.0.16; then
        git checkout --detach v2.0.16
    elif git show-ref --tags --verify --quiet refs/tags/2.0.16; then
        git checkout --detach 2.0.16
    fi

    echo "MAGMA.Celltyping commit: $(git rev-parse HEAD)"
)

Rscript - "${MAGMA_CELLTYPING_REPO}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
repo <- normalizePath(args[[1]])

if (!requireNamespace("remotes", quietly = TRUE)) {
    stop("The R package remotes is not installed.")
}

remotes::install_local(
    repo,
    dependencies = TRUE,
    upgrade = "never",
    force = TRUE,
    quiet = FALSE
)

suppressPackageStartupMessages(library(MAGMA.Celltyping))
cat("MAGMA.Celltyping:", as.character(packageVersion("MAGMA.Celltyping")), "\n")
RS

echo
echo "=== 3. Install the official MAGMA executable ==="
MAGMA_PATH="$(
    Rscript - "${MAGMA_INSTALL_DIR}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
dest <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(dest, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(MAGMA.Celltyping))

installer <- getExportedValue(
    "MAGMA.Celltyping",
    "magma_install"
)
path <- installer(
    dest_dir = dest,
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
echo "=== 4. Validate Python, R, MAGMA, and MAGMA.Celltyping ==="
python - <<'PY'
import sys
from pathlib import Path

bad = [entry for entry in sys.path if "/home/users/edtng/.local" in entry]
if bad:
    raise RuntimeError(f"User-site contamination detected: {bad}")

import anndata
import bs4
import matplotlib
import numpy
import openpyxl
import pandas
import requests
import scipy
import sklearn
import statsmodels
import yaml

print("Python:", sys.version.split()[0])
print("anndata:", anndata.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)
print("statsmodels:", statsmodels.__version__)
print("matplotlib:", matplotlib.__version__)
print("openpyxl:", openpyxl.__version__)
print("requests:", requests.__version__)
print("beautifulsoup4:", bs4.__version__)
print("PyYAML:", yaml.__version__)
PY

Rscript - <<'RS'
suppressPackageStartupMessages({
    library(MAGMA.Celltyping)
    library(MungeSumstats)
})
cat("R:", R.version.string, "\n")
cat("MAGMA.Celltyping:", as.character(packageVersion("MAGMA.Celltyping")), "\n")
cat("MungeSumstats:", as.character(packageVersion("MungeSumstats")), "\n")
RS

magma --version

cat > "${MAGMA_INSTALL_DIR}/INSTALLATION_COMPLETE.txt" <<EOF
PAH GWAS program-burden environment setup completed.
date=$(date --iso-8601=seconds)
environment=${ENV}
MAGMA_Celltyping_repo=${MAGMA_CELLTYPING_REPO}
MAGMA_Celltyping_commit=$(cd "${MAGMA_CELLTYPING_REPO}" && git rev-parse HEAD)
magma=${MAGMA_PATH}
EOF

echo
echo "PAH GWAS PROGRAM-BURDEN SETUP PASSED"
echo "Environment: ${ENV}"
echo "MAGMA:      ${MAGMA_PATH}"
