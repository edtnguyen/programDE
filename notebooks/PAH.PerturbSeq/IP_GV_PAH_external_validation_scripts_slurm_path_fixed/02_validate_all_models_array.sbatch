#!/usr/bin/env bash
#SBATCH --job-name=IPGV_ext_model
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=engreitz,owners
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-15%4

set -Eeuo pipefail
trap 'status=$?; echo "ERROR at line ${LINENO}; exit code ${status}" >&2; exit "${status}"' ERR

CONDA_SH="/oak/stanford/groups/engreitz/Users/tri/envs/mambaforge/etc/profile.d/conda.sh"
ENV="/oak/stanford/groups/engreitz/Users/tri/envs/pah_external_validation_20260620"

BASE="/oak/stanford/groups/engreitz/Users/tri/10xPerturbSeq/analysis/250507_ePAH1_NovaSeq/CRISPR_count/IP_GV_unified_sceptre_cnmf_no_celltype"
WORK_ROOT="${BASE}/PerturbNMF"
PILOT_ROOT="${WORK_ROOT}/pilot_array_n20"
RUN_DIR="${PILOT_ROOT}/merged"
INFERENCE_DIR="${RUN_DIR}/Inference"

EXTERNAL_ROOT="${PILOT_ROOT}/external_validation_human_PAH"
RESOURCE_DIR="${EXTERNAL_ROOT}/resources"
HUMAN_DIR="${EXTERNAL_ROOT}/GSE185479/processed"
VALIDATION_ROOT="${EXTERNAL_ROOT}/models"
PREP_MARKER="${HUMAN_DIR}/HUMAN_DATA_PREPARATION_COMPLETE.txt"

PREPARED_H5AD="${WORK_ROOT}/input/IP_GFP_GV_GFP_guide_positive_PerturbNMF_input.h5ad"
ORTHOLOGS="${RESOURCE_DIR}/HOM_ProteinCoding.rpt"
HUMAN_LOGCPM="${HUMAN_DIR}/GSE185479_pseudobulk_logCPM.tsv.gz"
HUMAN_METADATA="${HUMAN_DIR}/GSE185479_pseudobulk_metadata.tsv"
HUMAN_DE="${HUMAN_DIR}/GSE185479_edgeR_PAH_vs_healthy.tsv.gz"

# Slurm executes a copied batch script from /var/spool/slurmd. Support
# files therefore live in a persistent directory staged by the submit wrapper.
CODE_DIR="${PAH_EXT_CODE_DIR:-${EXTERNAL_ROOT}/code}"
PY_SCRIPT="${CODE_DIR}/pah_external_validation.py"

[[ -s "${PY_SCRIPT}" ]] || {
    echo "Missing staged support file: ${PY_SCRIPT}" >&2
    exit 1
}

K_VALUES=(5 10 15 20 25 30 40 50)
THRESHOLDS=(0.2 2.0)

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this file as an array.}"
N_THRESH="${#THRESHOLDS[@]}"
K_INDEX=$((TASK_ID / N_THRESH))
T_INDEX=$((TASK_ID % N_THRESH))

if (( K_INDEX < 0 || K_INDEX >= ${#K_VALUES[@]} )); then
    echo "Invalid task index: ${TASK_ID}" >&2
    exit 2
fi

K="${K_VALUES[$K_INDEX]}"
THRESHOLD="${THRESHOLDS[$T_INDEX]}"
TAG="${THRESHOLD//./_}"
MODEL_NAME="K${K}_dt${TAG}"

SPECTRA="${INFERENCE_DIR}/Inference.gene_spectra_score.k_${K}.dt_${TAG}.txt"
OUTPUT_DIR="${VALIDATION_ROOT}/${MODEL_NAME}"
MARKER="${OUTPUT_DIR}/MODEL_EXTERNAL_VALIDATION_COMPLETE.txt"

for required in \
    "${PREP_MARKER}" \
    "${PREPARED_H5AD}" \
    "${ORTHOLOGS}" \
    "${HUMAN_LOGCPM}" \
    "${HUMAN_METADATA}" \
    "${HUMAN_DE}" \
    "${SPECTRA}"
do
    [[ -s "${required}" ]] || {
        echo "Missing required input: ${required}" >&2
        exit 1
    }
done

mkdir -p "${OUTPUT_DIR}" "${EXTERNAL_ROOT}/tmp"
export TMPDIR="${EXTERNAL_ROOT}/tmp/model_${SLURM_ARRAY_JOB_ID:-array}_${TASK_ID}"
mkdir -p "${TMPDIR}"

source "${CONDA_SH}"
conda activate "${ENV}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONPATH || true
unset PYTHONSTARTUP || true

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "Mouse program -> human PAH external validation"
echo "Started:       $(date)"
echo "Array job ID:  ${SLURM_ARRAY_JOB_ID:-not_set}"
echo "Task ID:       ${TASK_ID}"
echo "K:             ${K}"
echo "Threshold:     ${THRESHOLD}"
echo "Output:        ${OUTPUT_DIR}"
echo "============================================================"

python "${PY_SCRIPT}" validate-model \
    --spectra "${SPECTRA}" \
    --k "${K}" \
    --threshold "${THRESHOLD}" \
    --orthologs "${ORTHOLOGS}" \
    --prepared-h5ad "${PREPARED_H5AD}" \
    --human-logcpm "${HUMAN_LOGCPM}" \
    --human-metadata "${HUMAN_METADATA}" \
    --human-de "${HUMAN_DE}" \
    --output-dir "${OUTPUT_DIR}" \
    --top-n 300 \
    --min-genes 15 \
    --gsea-permutations 5000 \
    --threads "${SLURM_CPUS_PER_TASK:-4}" \
    --seed 14

for required in \
    "${OUTPUT_DIR}/external_validation_summary.tsv" \
    "${OUTPUT_DIR}/program_mouse_to_human_mapping.tsv.gz" \
    "${OUTPUT_DIR}/program_PAH_preranked_GSEA.tsv" \
    "${OUTPUT_DIR}/program_donor_score_effects.tsv" \
    "${OUTPUT_DIR}/experimentally_targeted_genes_removed.tsv"
do
    [[ -s "${required}" ]] || {
        echo "Missing model validation output: ${required}" >&2
        exit 1
    }
done

cat > "${MARKER}" <<EOF
Model external validation completed successfully.
date=$(date --iso-8601=seconds)
array_job_id=${SLURM_ARRAY_JOB_ID:-not_set}
array_task_id=${TASK_ID}
job_id=${SLURM_JOB_ID:-interactive}
K=${K}
threshold=${THRESHOLD}
output_dir=${OUTPUT_DIR}
EOF

echo "MODEL EXTERNAL VALIDATION COMPLETED: ${MARKER}"
