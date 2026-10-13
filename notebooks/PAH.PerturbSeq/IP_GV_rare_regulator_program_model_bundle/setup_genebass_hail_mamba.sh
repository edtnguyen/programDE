#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

MAMBA="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/bin/mamba"
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/genebass_hail_20260623"

source "${CONDA_SH}"

if [[ ! -x "${ENV}/bin/python" ]]; then
    "${MAMBA}" create -y -p "${ENV}" \
        --override-channels -c conda-forge \
        python=3.11 \
        openjdk=11 \
        numpy \
        pandas \
        scipy \
        google-cloud-sdk \
        pip \
        setuptools \
        wheel
fi

conda activate "${ENV}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

python -m pip install --upgrade "hail==0.2.138"

python - <<'PY'
import hail as hl
print("Hail:", hl.__version__)
PY

java -version

cat <<'EOF'

GENEBASS HAIL ENVIRONMENT PASSED

Important:
The Genebass Google bucket is requester-pays. Hail access therefore needs:
1. a Google Cloud project with billing enabled;
2. gcloud authentication;
3. --gcp-project <PROJECT_ID> when running the query script.

The browser remains available without UK Biobank individual-level access.
EOF
