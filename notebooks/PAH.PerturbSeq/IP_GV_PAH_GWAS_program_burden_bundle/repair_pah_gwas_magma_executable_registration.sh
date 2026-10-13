#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_gwas_program_burden_20260621"
MAGMA_INSTALL_DIR="/oak/stanford/groups/engreitz/Users/tri/Tools/MAGMA"

MAGMA_VERSION="v1.10"
REGULAR_URL="https://vu.data.surf.nl/index.php/s/zkKbNeNOZAhFXZB/download"
STATIC_URL="https://vu.data.surf.nl/index.php/s/lxDgt2dNdNr6DYt/download"

REGULAR_ZIP="${MAGMA_INSTALL_DIR}/magma_v1.10.zip"
REGULAR_DIR="${MAGMA_INSTALL_DIR}/magma_v1.10"
STATIC_ZIP="${MAGMA_INSTALL_DIR}/magma_v1.10_static.zip"
STATIC_DIR="${MAGMA_INSTALL_DIR}/magma_v1.10_static"

[[ -x "${ENV}/bin/R" ]] || {
    echo "The GWAS environment is missing R: ${ENV}" >&2
    exit 1
}

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true
export R_LIBS_USER="${ENV}/lib/R/library"

mkdir -p "${MAGMA_INSTALL_DIR}"

echo "============================================================"
echo "Repair MAGMA executable registration"
echo "Environment: ${ENV}"
echo "Install dir: ${MAGMA_INSTALL_DIR}"
echo "============================================================"

echo
echo "=== 1. Confirm MAGMA.Celltyping is installed ==="
Rscript - <<'RS'
suppressPackageStartupMessages(library(MAGMA.Celltyping))
cat(
    "MAGMA.Celltyping:",
    as.character(packageVersion("MAGMA.Celltyping")),
    "\n"
)
RS

working_magma() {
    local candidate="$1"
    [[ -f "${candidate}" ]] || return 1

    chmod u+x "${candidate}" 2>/dev/null || true

    local version_output
    if ! version_output="$("${candidate}" --version 2>&1)"; then
        return 1
    fi

    if ! grep -qiE 'MAGMA.*v?1\.10|v?1\.10.*MAGMA' <<< "${version_output}"; then
        return 1
    fi

    printf '%s\n' "${candidate}"
}

find_working_magma() {
    local candidate
    while IFS= read -r -d '' candidate; do
        if working_magma "${candidate}" >/dev/null; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done < <(
        find "${MAGMA_INSTALL_DIR}" \
            -type f \
            \( -iname 'magma' -o -iname 'magma.exe' \) \
            -print0
    )
    return 1
}

extract_zip() {
    local archive="$1"
    local destination="$2"

    [[ -s "${archive}" ]] || return 1
    unzip -t "${archive}" >/dev/null
    mkdir -p "${destination}"
    unzip -o -j "${archive}" -d "${destination}" >/dev/null
}

download_zip() {
    local url="$1"
    local destination="$2"

    if [[ -s "${destination}" ]]; then
        return
    fi

    local temporary="${destination}.part"
    rm -f -- "${temporary}"

    curl \
        --location \
        --fail \
        --retry 10 \
        --retry-delay 10 \
        --connect-timeout 30 \
        --max-time 600 \
        --output "${temporary}" \
        "${url}"

    mv "${temporary}" "${destination}"
}

echo
echo "=== 2. Locate the binary already downloaded by MAGMA.Celltyping ==="
MAGMA_PATH="$(find_working_magma || true)"

if [[ -z "${MAGMA_PATH}" && -s "${REGULAR_ZIP}" ]]; then
    echo "No working binary was detected; re-extracting ${REGULAR_ZIP}"
    extract_zip "${REGULAR_ZIP}" "${REGULAR_DIR}"
    MAGMA_PATH="$(find_working_magma || true)"
fi

if [[ -z "${MAGMA_PATH}" ]]; then
    echo
    echo "=== 3. Fall back to the official statically linked Linux binary ==="
    download_zip "${STATIC_URL}" "${STATIC_ZIP}"
    extract_zip "${STATIC_ZIP}" "${STATIC_DIR}"
    MAGMA_PATH="$(find_working_magma || true)"
fi

if [[ -z "${MAGMA_PATH}" ]]; then
    echo "No working MAGMA v1.10 executable was found." >&2
    echo "Files currently under ${MAGMA_INSTALL_DIR}:" >&2
    find "${MAGMA_INSTALL_DIR}" -maxdepth 3 -type f -printf '%p\n' >&2
    exit 1
fi

MAGMA_PATH="$(readlink -f "${MAGMA_PATH}")"
echo "Working binary: ${MAGMA_PATH}"
"${MAGMA_PATH}" --version

echo
echo "=== 4. Register the binary where MAGMA.Celltyping actually searches ==="
# magma_install(dest_dir=...) downloads to the requested custom directory,
# but its final version check calls magma_executable() without that directory.
# Therefore install a copy under find_install_dir(), which is the package's
# default recursive search root.
R_CACHE_ROOT="$(
    Rscript - <<'RS'
suppressPackageStartupMessages(library(MAGMA.Celltyping))
cat(
    MAGMA.Celltyping:::find_install_dir(verbose = FALSE),
    "\n",
    sep = ""
)
RS
)"
R_CACHE_ROOT="$(printf '%s\n' "${R_CACHE_ROOT}" | tail -n 1)"

[[ -n "${R_CACHE_ROOT}" ]] || {
    echo "MAGMA.Celltyping returned an empty install/search directory." >&2
    exit 1
}

R_CACHE_BIN="${R_CACHE_ROOT}/magma_v1.10/magma"
mkdir -p "$(dirname "${R_CACHE_BIN}")"
install -m 0755 "${MAGMA_PATH}" "${R_CACHE_BIN}"
install -m 0755 "${MAGMA_PATH}" "${ENV}/bin/magma"

printf '%s\n' "${R_CACHE_BIN}" > "${MAGMA_INSTALL_DIR}/magma_path.txt"

echo "MAGMA.Celltyping search root: ${R_CACHE_ROOT}"
echo "Registered package binary:   ${R_CACHE_BIN}"
echo "Registered shell binary:     ${ENV}/bin/magma"

echo
echo "=== 5. Validate shell and MAGMA.Celltyping discovery ==="
hash -r
command -v magma
magma --version

Rscript - "${R_CACHE_ROOT}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
cache_root <- normalizePath(args[[1]])

suppressPackageStartupMessages(library(MAGMA.Celltyping))

executables <- MAGMA.Celltyping:::magma_executable(
    exec_dir = cache_root,
    return_all = TRUE,
    verbose = TRUE
)
if (length(executables) == 0L) {
    stop("MAGMA.Celltyping still cannot discover an executable.")
}

versions <- MAGMA.Celltyping:::magma_installed_version(
    magma_x = executables,
    verbose = TRUE
)
print(executables)
print(versions)

if (!any(grepl("^v?1\\.10", versions))) {
    stop(
        "MAGMA.Celltyping discovered MAGMA, but not version v1.10: ",
        paste(versions, collapse = ", ")
    )
}

selected <- MAGMA.Celltyping:::magma_check_version_match(
    desired_version = "v1.10",
    verbose = TRUE
)
if (is.null(selected) || !file.exists(selected)) {
    stop("magma_check_version_match() still returned no executable.")
}
cat("Selected executable:", normalizePath(selected), "\n")
RS

echo
echo "=== 6. Validate remaining analysis dependencies ==="
python - <<'PY'
import sys
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
    library(EWCE)
    library(orthogene)
    library(lme4)
})
cat("All R dependencies load successfully.\n")
RS

cat > "${MAGMA_INSTALL_DIR}/INSTALLATION_COMPLETE.txt" <<EOF
PAH GWAS MAGMA executable registration completed.
date=$(date --iso-8601=seconds)
environment=${ENV}
source_magma=${MAGMA_PATH}
registered_magma=${R_CACHE_BIN}
shell_magma=${ENV}/bin/magma
version=${MAGMA_VERSION}
EOF

echo
echo "PAH GWAS MAGMA EXECUTABLE REPAIR PASSED"
echo "Environment: ${ENV}"
echo "MAGMA:      ${R_CACHE_BIN}"
