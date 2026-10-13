#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

MAMBA="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/bin/mamba"
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_external_validation_20260620"

[[ -x "${MAMBA}" ]] || {
    echo "mamba not found: ${MAMBA}" >&2
    exit 1
}

source "${CONDA_SH}"

echo "Using mamba: ${MAMBA}"
"${MAMBA}" --version

if [[ ! -x "${ENV}/bin/python" ]]; then
    "${MAMBA}" create -y -p "${ENV}" \
        --override-channels \
        -c conda-forge \
        -c bioconda \
        python=3.10 \
        numpy=1.26.4 \
        scipy=1.12 \
        pandas=2.2 \
        anndata=0.10.9 \
        h5py \
        matplotlib \
        statsmodels \
        scikit-learn \
        gseapy \
        openpyxl \
        requests \
        tqdm \
        pyyaml \
        r-base=4.3 \
        bioconductor-edger \
        pip \
        setuptools \
        wheel
else
    echo "Environment already exists; repairing required packages."
    "${MAMBA}" install -y -p "${ENV}" \
        --override-channels \
        -c conda-forge \
        -c bioconda \
        numpy=1.26.4 \
        scipy=1.12 \
        pandas=2.2 \
        anndata=0.10.9 \
        h5py \
        matplotlib \
        statsmodels \
        scikit-learn \
        gseapy \
        openpyxl \
        requests \
        tqdm \
        pyyaml \
        r-base=4.3 \
        bioconductor-edger \
        pip \
        setuptools \
        wheel
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
import pandas
import scipy
import statsmodels

print("Python:", sys.version.split()[0])
print("anndata:", anndata.__version__)
print("gseapy:", gseapy.__version__)
print("matplotlib:", matplotlib.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("statsmodels:", statsmodels.__version__)
PY

echo
echo "=== R validation ==="
Rscript - <<'RS'
suppressPackageStartupMessages(library(edgeR))
cat("R:", R.version.string, "\n")
cat("edgeR:", as.character(packageVersion("edgeR")), "\n")
RS

echo
echo "EXTERNAL VALIDATION ENVIRONMENT PASSED"
echo "Environment: ${ENV}"
