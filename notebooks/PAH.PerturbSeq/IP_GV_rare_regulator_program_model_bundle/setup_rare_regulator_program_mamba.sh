#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

MAMBA="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/bin/mamba"
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/rare_regulator_program_20260623"

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
    scikit-learn=1.5
    statsmodels
    matplotlib
    openpyxl
    networkx
    pyvis
    pyyaml
    requests
    tqdm
    joblib
    jupyter_client
    ipykernel
    pip
    setuptools
    wheel
)

if [[ ! -x "${ENV}/bin/python" ]]; then
    "${MAMBA}" create -y -p "${ENV}" \
        --override-channels -c conda-forge \
        "${PACKAGES[@]}"
else
    "${MAMBA}" install -y -p "${ENV}" \
        --override-channels -c conda-forge \
        "${PACKAGES[@]}"
fi

conda activate "${ENV}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

python - <<'PY'
import sys
import joblib
import matplotlib
import networkx
import numpy
import openpyxl
import pandas
import pyvis
import scipy
import sklearn
import statsmodels
import yaml

print("Python:", sys.version.split()[0])
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)
print("statsmodels:", statsmodels.__version__)
print("networkx:", networkx.__version__)
print("pyvis:", pyvis.__version__)
print("openpyxl:", openpyxl.__version__)
print("matplotlib:", matplotlib.__version__)
print("joblib:", joblib.__version__)
print("PyYAML:", yaml.__version__)
PY

echo "RARE REGULATOR CORE ENVIRONMENT PASSED"
echo "Environment: ${ENV}"
