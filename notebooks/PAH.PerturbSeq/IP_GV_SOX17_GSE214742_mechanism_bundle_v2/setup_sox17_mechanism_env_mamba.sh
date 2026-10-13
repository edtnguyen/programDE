#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

MAMBA="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/bin/mamba"
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/sox17_mechanism_20260622"

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
    ipykernel
    r-base=4.3
    bioconductor-edger
    bioconductor-limma
    r-jsonlite
    pip
    setuptools
    wheel
)

echo "=== Create or repair SOX17 mechanism environment ==="
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

echo
echo "=== Python validation ==="
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
print("matplotlib:", matplotlib.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("statsmodels:", statsmodels.__version__)
print("openpyxl:", openpyxl.__version__)
PY

echo
echo "=== R validation ==="
Rscript - <<'RS'
suppressPackageStartupMessages(library(edgeR))
cat("R:", R.version.string, "\n")
cat("edgeR:", as.character(packageVersion("edgeR")), "\n")
RS

echo
echo "SOX17 MECHANISM ENVIRONMENT PASSED"
echo "Environment: ${ENV}"
