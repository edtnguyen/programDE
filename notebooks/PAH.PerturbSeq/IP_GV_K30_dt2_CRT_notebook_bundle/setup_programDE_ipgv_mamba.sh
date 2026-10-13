#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

MAMBA="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/bin/mamba"
CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"

ENV="/oak/stanford/groups/engreitz/Users/tri/envs/programDE_ipgv_20260620"
TOOLS="/oak/stanford/groups/engreitz/Users/tri/Tools"
PROGRAMDE_REPO="${TOOLS}/programDE"
PROGRAMDE_URL="https://github.com/edtnguyen/programDE.git"
KERNEL_NAME="programDE-ipgv"
KERNEL_DISPLAY="Python (programDE IP-GV)"

[[ -x "${MAMBA}" ]] || {
    echo "mamba not found: ${MAMBA}" >&2
    exit 1
}

source "${CONDA_SH}"
mkdir -p "${TOOLS}"

echo "=== 1. Create or repair programDE environment ==="
if [[ ! -x "${ENV}/bin/python" ]]; then
    "${MAMBA}" create -y -p "${ENV}" \
        --override-channels -c conda-forge \
        python=3.10 \
        numpy=1.26.4 \
        scipy=1.12 \
        pandas=2.2 \
        scikit-learn=1.5 \
        anndata=0.10.9 \
        scanpy=1.10.4 \
        mudata=0.3.1 \
        muon=0.1.7 \
        numba=0.60 \
        joblib \
        matplotlib \
        statsmodels \
        pyyaml \
        h5py \
        ipykernel \
        jupyter_client \
        openpyxl \
        tqdm \
        pip \
        setuptools \
        wheel
else
    echo "Environment already exists: ${ENV}"
    "${MAMBA}" install -y -p "${ENV}" \
        --override-channels -c conda-forge \
        numpy=1.26.4 \
        scipy=1.12 \
        pandas=2.2 \
        scikit-learn=1.5 \
        anndata=0.10.9 \
        scanpy=1.10.4 \
        mudata=0.3.1 \
        muon=0.1.7 \
        numba=0.60 \
        joblib \
        matplotlib \
        statsmodels \
        pyyaml \
        h5py \
        ipykernel \
        jupyter_client \
        openpyxl \
        tqdm \
        pip \
        setuptools \
        wheel
fi

conda activate "${ENV}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

echo
echo "=== 2. Clone or reuse programDE ==="
if [[ ! -d "${PROGRAMDE_REPO}/.git" ]]; then
    if [[ -e "${PROGRAMDE_REPO}" ]]; then
        echo "Removing incomplete non-git path: ${PROGRAMDE_REPO}"
        rm -rf -- "${PROGRAMDE_REPO}"
    fi
    git clone "${PROGRAMDE_URL}" "${PROGRAMDE_REPO}"
else
    echo "Reusing existing checkout: ${PROGRAMDE_REPO}"
fi

(
    cd "${PROGRAMDE_REPO}"
    echo "programDE commit:"
    git rev-parse HEAD
)

echo
echo "=== 3. Install programDE in editable mode ==="
python -m pip install --no-deps -e "${PROGRAMDE_REPO}"

echo
echo "=== 4. Validate required CRT API ==="
python - <<'PY'
import inspect
import sys

from src.sceptre import (
    build_ntc_group_inputs,
    compute_guide_set_null_pvals,
    crt_pvals_for_ntc_groups_ensemble,
    crt_pvals_for_ntc_groups_ensemble_skew,
    limit_threading,
    make_ntc_groups_ensemble,
    prepare_crt_inputs,
    run_all_genes_union_crt,
)
from src.visualization import qq_plot_ntc_pvals

required = {
    "build_ntc_group_inputs": build_ntc_group_inputs,
    "compute_guide_set_null_pvals": compute_guide_set_null_pvals,
    "crt_pvals_for_ntc_groups_ensemble": crt_pvals_for_ntc_groups_ensemble,
    "crt_pvals_for_ntc_groups_ensemble_skew": crt_pvals_for_ntc_groups_ensemble_skew,
    "make_ntc_groups_ensemble": make_ntc_groups_ensemble,
    "prepare_crt_inputs": prepare_crt_inputs,
    "run_all_genes_union_crt": run_all_genes_union_crt,
    "qq_plot_ntc_pvals": qq_plot_ntc_pvals,
}

print("Python:", sys.version.split()[0])
for name, obj in required.items():
    print(f"{name}: {inspect.signature(obj)}")
print("programDE CRT imports: OK")
PY

echo
echo "=== 5. Register a Jupyter kernel ==="
python -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY}"

python - "${KERNEL_NAME}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

kernel_name = sys.argv[1]
specs = json.loads(
    subprocess.check_output(
        ["jupyter", "kernelspec", "list", "--json"],
        text=True,
    )
)
resource_dir = Path(specs["kernelspecs"][kernel_name]["resource_dir"])
kernel_json = resource_dir / "kernel.json"
payload = json.loads(kernel_json.read_text())
payload.setdefault("env", {})["PYTHONNOUSERSITE"] = "1"
kernel_json.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Updated kernel environment: {kernel_json}")
PY

echo
echo "PROGRAMDE SETUP PASSED"
echo "Environment: ${ENV}"
echo "Repository:  ${PROGRAMDE_REPO}"
echo "Kernel:      ${KERNEL_DISPLAY}"
