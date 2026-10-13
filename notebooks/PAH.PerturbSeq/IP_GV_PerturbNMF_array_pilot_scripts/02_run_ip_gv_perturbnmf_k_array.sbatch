#!/usr/bin/env bash
#SBATCH --job-name=IPGV_cNMF_K
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-7%4

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

START_SECONDS=$SECONDS

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/perturbnmf_sk_20260620"
REPO="/oak/stanford/groups/engreitz/Users/tri/Tools/PerturbNMF"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
PREPARED_H5AD="${WORK_ROOT}/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"

PILOT_ROOT="${WORK_ROOT}/pilot_array_n20"
SEED_DIR="${PILOT_ROOT}/seeds"
PER_K_OUTPUT="${PILOT_ROOT}/per_k"
PREP_MARKER="${PILOT_ROOT}/PREPARATION_COMPLETE.txt"

K_VALUES=(5 10 15 20 25 30 40 50)
NUM_ITER=20
NUM_HVG=3000
SEED=14
SEL_THRESHOLDS=(0.2 2.0)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array.}"
if (( TASK_ID < 0 || TASK_ID >= ${#K_VALUES[@]} )); then
    echo "Array task ${TASK_ID} is outside 0..$((${#K_VALUES[@]} - 1))." >&2
    exit 2
fi
K="${K_VALUES[$TASK_ID]}"

RUN_NAME="K${K}"
RUN_DIR="${PER_K_OUTPUT}/${RUN_NAME}"
SEED_FILE="${SEED_DIR}/nmf_seeds_K${K}.npy"
COMPLETE_MARKER="${RUN_DIR}/K${K}_FACTORIZE_COMPLETE.txt"

PIPELINE="${REPO}/src/Stage1_Inference/sk-cNMF/Slurm_Version/sk-cNMF_batch_inference_pipeline.py"
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"

[[ -s "${PREP_MARKER}" ]] || {
    echo "Preparation marker is missing: ${PREP_MARKER}" >&2
    exit 1
}
[[ -s "${PREPARED_H5AD}" ]] || {
    echo "Prepared H5AD is missing: ${PREPARED_H5AD}" >&2
    exit 1
}
[[ -s "${SEED_FILE}" ]] || {
    echo "K-specific seed file is missing: ${SEED_FILE}" >&2
    exit 1
}
[[ -r "${PIPELINE}" ]] || {
    echo "PerturbNMF pipeline is missing: ${PIPELINE}" >&2
    exit 1
}

mkdir -p "${PER_K_OUTPUT}" "${PILOT_ROOT}/tmp"
export TMPDIR="${PILOT_ROOT}/tmp/${SLURM_ARRAY_JOB_ID:-array}_${TASK_ID}"
mkdir -p "${TMPDIR}"

source "${CONDA_SH}"
conda activate "${ENV}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
unset PYTHONSTARTUP || true

export OMP_NUM_THREADS="${NTHREADS}"
export OPENBLAS_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export BLIS_NUM_THREADS="${NTHREADS}"
export NUMEXPR_NUM_THREADS="${NTHREADS}"

echo "============================================================"
echo "IP/GV PerturbNMF array factorization"
echo "Started:        $(date)"
echo "Array job ID:   ${SLURM_ARRAY_JOB_ID:-not_set}"
echo "Array task ID:  ${TASK_ID}"
echo "Job ID:         ${SLURM_JOB_ID:-interactive}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "K:              ${K}"
echo "Replicates:     ${NUM_ITER}"
echo "HVGs:           ${NUM_HVG}"
echo "Seed file:      ${SEED_FILE}"
echo "Run directory:  ${RUN_DIR}"
echo "============================================================"

python "${PIPELINE}" \
    --counts_fn "${PREPARED_H5AD}" \
    --output_directory "${PER_K_OUTPUT}" \
    --run_name "${RUN_NAME}" \
    --nmf_seeds_path "${SEED_FILE}" \
    --numiter "${NUM_ITER}" \
    --numhvgenes "${NUM_HVG}" \
    --seed "${SEED}" \
    --K "${K}" \
    --init random \
    --loss frobenius \
    --algo cd \
    --max_NMF_iter 1000 \
    --tol 1e-4 \
    --sel_threshs "${SEL_THRESHOLDS[@]}" \
    --species mouse \
    --categorical_key sample \
    --guide_names_key guide_names \
    --guide_targets_key guide_targets \
    --guide_assignment_key guide_assignment \
    --gene_names_key symbol \
    --run_factorize \
    --skip_existing

echo
echo "=== Verifying all factorization files for K=${K} ==="
python - \
    "${RUN_DIR}" \
    "${K}" \
    "${NUM_ITER}" <<'PY'
import sys
from pathlib import Path

import numpy as np
from cnmf import load_df_from_npz

run_dir = Path(sys.argv[1])
k = int(sys.argv[2])
n_iter = int(sys.argv[3])
tmp_dir = run_dir / "Inference" / "cnmf_tmp"

missing = []
bad = []
reference_columns = None

for iteration in range(n_iter):
    path = tmp_dir / f"Inference.spectra.k_{k}.iter_{iteration}.df.npz"
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(str(path))
        continue

    spectra = load_df_from_npz(path)
    if spectra.shape[0] != k:
        bad.append(f"{path}: expected {k} rows, found {spectra.shape[0]}")
    if not np.isfinite(spectra.to_numpy()).all():
        bad.append(f"{path}: contains NaN or infinite values")

    columns = spectra.columns.astype(str).tolist()
    if reference_columns is None:
        reference_columns = columns
    elif columns != reference_columns:
        bad.append(f"{path}: HVG column order differs from iteration 0")

if missing or bad:
    if missing:
        print("Missing files:")
        print("\n".join(missing))
    if bad:
        print("Invalid files:")
        print("\n".join(bad))
    raise SystemExit(1)

print(
    f"Verified {n_iter} spectra files for K={k}; "
    f"{len(reference_columns):,} HVGs."
)
PY

cat > "${COMPLETE_MARKER}" <<EOF
Factorization completed successfully.
date=$(date --iso-8601=seconds)
array_job_id=${SLURM_ARRAY_JOB_ID:-not_set}
array_task_id=${TASK_ID}
job_id=${SLURM_JOB_ID:-interactive}
K=${K}
numiter=${NUM_ITER}
seed_file=${SEED_FILE}
run_dir=${RUN_DIR}
EOF

ELAPSED=$((SECONDS - START_SECONDS))
echo
echo "============================================================"
echo "K=${K} COMPLETED: $(date)"
echo "Elapsed seconds: ${ELAPSED}"
echo "Marker: ${COMPLETE_MARKER}"
echo "============================================================"
