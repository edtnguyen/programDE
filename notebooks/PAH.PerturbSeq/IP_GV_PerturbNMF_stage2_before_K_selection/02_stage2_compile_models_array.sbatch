#!/usr/bin/env bash
#SBATCH --job-name=IPGV_s2_compile
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-15%4

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_stage2_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
PILOT_ROOT="${BASE}/PerturbNMF/pilot_array_n20"
RUN_DIR="${PILOT_ROOT}/merged"
STAGE2_ROOT="${PILOT_ROOT}/stage2_pre_k_selection"
PREFLIGHT_MARKER="${STAGE2_ROOT}/STAGE2_PREFLIGHT_COMPLETE.txt"

K_VALUES=(5 10 15 20 25 30 40 50)
THRESHOLDS=(0.2 2.0)

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this file as a SLURM array.}"
N_THRESH="${#THRESHOLDS[@]}"
K_INDEX=$((TASK_ID / N_THRESH))
T_INDEX=$((TASK_ID % N_THRESH))

if (( K_INDEX < 0 || K_INDEX >= ${#K_VALUES[@]} )); then
    echo "Invalid array task index: ${TASK_ID}" >&2
    exit 2
fi

K="${K_VALUES[$K_INDEX]}"
THRESHOLD="${THRESHOLDS[$T_INDEX]}"
TAG="${THRESHOLD//./_}"

H5MU="${RUN_DIR}/Inference/adata/cNMF_${K}_${TAG}.h5mu"
MARKER="${STAGE2_ROOT}/compile/K${K}_dt${TAG}_COMPLETE.txt"
VALIDATOR="${REPO}/.claude/skills/perturbNMF-runner/scripts/validate_data.py"

[[ -s "${PREFLIGHT_MARKER}" ]] || {
    echo "Preflight marker missing: ${PREFLIGHT_MARKER}" >&2
    exit 1
}

mkdir -p "$(dirname "${MARKER}")"

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
export MPLBACKEND=Agg
unset PYTHONSTARTUP || true

echo "============================================================"
echo "Compile pilot model for Stage 2"
echo "Started:   $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-interactive}"
echo "Task:      ${TASK_ID}"
echo "K:         ${K}"
echo "Threshold: ${THRESHOLD}"
echo "H5MU:      ${H5MU}"
echo "============================================================"

if [[ ! -s "${H5MU}" || "${RECOMPILE_MODELS:-0}" == "1" ]]; then
    python - \
        "${RUN_DIR}" \
        "${K}" \
        "${THRESHOLD}" <<'PY'
import sys
from pathlib import Path

from Stage1_Inference.src import compile_results

run_dir = Path(sys.argv[1])
k = int(sys.argv[2])
threshold = float(sys.argv[3])

compile_results(
    output_directory=str(run_dir),
    run_name="Inference",
    components=[k],
    sel_threshs=[threshold],
    guide_names_key="guide_names",
    guide_targets_key="guide_targets",
    categorical_key="sample",
    guide_assignment_key="guide_assignment",
    gene_names_key="symbol",
)

print(f"Compiled K={k}, threshold={threshold}")
PY
else
    echo "Reusing existing H5MU."
fi

[[ -s "${H5MU}" ]] || {
    echo "Compilation did not create ${H5MU}" >&2
    exit 1
}

python - "${H5MU}" "${K}" <<'PY'
import sys
from pathlib import Path

import muon as mu
import numpy as np
import scipy.sparse as sp

path = Path(sys.argv[1])
k = int(sys.argv[2])

mdata = mu.read(path)

if set(mdata.mod.keys()) != {"rna", "cNMF"}:
    raise ValueError(f"Unexpected modalities: {list(mdata.mod.keys())}")
if mdata["cNMF"].n_vars != k:
    raise ValueError(
        f"Expected {k} programs, found {mdata['cNMF'].n_vars}."
    )
if mdata["rna"].n_obs != mdata["cNMF"].n_obs:
    raise ValueError("RNA and cNMF modalities have different cell counts.")

prog = mdata["cNMF"]
for key in ["guide_names", "guide_targets"]:
    if key not in prog.uns:
        raise KeyError(f"cNMF.uns[{key!r}] is missing.")
if "guide_assignment" not in prog.obsm:
    raise KeyError("cNMF.obsm['guide_assignment'] is missing.")
if "sample" not in prog.obs:
    raise KeyError("cNMF.obs['sample'] is missing.")

assignment = prog.obsm["guide_assignment"]
if assignment.shape != (prog.n_obs, len(prog.uns["guide_names"])):
    raise ValueError("Compiled guide-assignment matrix has wrong shape.")

samples = sorted(prog.obs["sample"].astype(str).unique().tolist())
if samples != ["GV_GFP", "IP_GFP"]:
    raise ValueError(f"Unexpected compiled sample labels: {samples}")

print(mdata)
print("Samples:", samples)
print("Guide assignment shape:", assignment.shape)
print("Compiled model validation: OK")
PY

if [[ -r "${VALIDATOR}" ]]; then
    python "${VALIDATOR}" \
        --counts_fn "${H5MU}" \
        --data_key rna \
        --prog_key cNMF \
        --categorical_key sample \
        --guide_names_key guide_names \
        --guide_targets_key guide_targets \
        --guide_assignment_key guide_assignment
fi

cat > "${MARKER}" <<EOF
Compilation completed successfully.
date=$(date --iso-8601=seconds)
array_job_id=${SLURM_ARRAY_JOB_ID:-not_set}
array_task_id=${TASK_ID}
job_id=${SLURM_JOB_ID:-interactive}
K=${K}
threshold=${THRESHOLD}
h5mu=${H5MU}
EOF

echo "COMPILE COMPLETED: ${MARKER}"
