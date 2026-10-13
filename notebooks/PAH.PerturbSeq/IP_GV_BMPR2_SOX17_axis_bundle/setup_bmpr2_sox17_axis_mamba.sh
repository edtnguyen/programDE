#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

MAMBA="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/bin/mamba"
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/bmpr2_sox17_axis_20260623"

[[ -x "${MAMBA}" ]] || {
    echo "mamba not found: ${MAMBA}" >&2
    exit 1
}

source "${CONDA_SH}"

PACKAGES=(
    python=3.10
    numpy=1.26.4
    scipy=1.12
    pandas=2.2
    anndata=0.10.9
    h5py
    matplotlib
    statsmodels
    scikit-learn
    gseapy
    openpyxl
    requests
    tqdm
    pyyaml
    r-base=4.3
    bioconductor-limma
    bioconductor-edger
    bioconductor-geoquery
    bioconductor-annotationdbi
    bioconductor-org.hs.eg.db
    r-jsonlite
    pip
    setuptools
    wheel
)

if [[ ! -x "${ENV}/bin/python" ]]; then
    "${MAMBA}" create -y -p "${ENV}" \
        --override-channels \
        -c conda-forge \
        -c bioconda \
        "${PACKAGES[@]}"
else
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

python - <<'PY'
import sys
import anndata
import gseapy
import matplotlib
import numpy
import openpyxl
import pandas
import scipy
import sklearn
import statsmodels

print("Python:", sys.version.split()[0])
print("anndata:", anndata.__version__)
print("gseapy:", gseapy.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)
print("statsmodels:", statsmodels.__version__)
print("matplotlib:", matplotlib.__version__)
print("openpyxl:", openpyxl.__version__)
PY

Rscript - <<'RS'
suppressPackageStartupMessages({
    library(limma)
    library(edgeR)
    library(GEOquery)
    library(AnnotationDbi)
    library(org.Hs.eg.db)
})
cat("R:", R.version.string, "\n")
cat("limma:", as.character(packageVersion("limma")), "\n")
cat("edgeR:", as.character(packageVersion("edgeR")), "\n")
cat("GEOquery:", as.character(packageVersion("GEOquery")), "\n")
RS

echo "BMPR2–SOX17 AXIS ENVIRONMENT PASSED"
echo "Environment: ${ENV}"
